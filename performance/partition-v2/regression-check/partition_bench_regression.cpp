// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cudf_test/column_wrapper.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/hashing.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/unary.hpp>
#include <optional>
#include <cstdlib>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cudf/utilities/error.hpp>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using cudf::size_type;
struct Fixture {
  std::string id, kind;
  int rows, partitions, columns, width, max_length, null_percent, mask, pattern, slice;
};
struct ResourceGuard {
  cuda::mr::any_resource<cuda::mr::device_accessible> previous;
  explicit ResourceGuard(cuda::mr::any_resource<cuda::mr::device_accessible> resource)
    : previous(cudf::set_current_device_resource(std::move(resource))) {}
  ~ResourceGuard() { cudf::set_current_device_resource(std::move(previous)); }
};
uint32_t mix(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16);
}
bool valid(Fixture const& f, int row, int col) {
  if (f.kind == "strings_uniform" || f.kind == "fixed_uniform") return col != 0;
  return mix(uint32_t(row) + uint32_t(col) * 1777u) % 100 >= unsigned(f.null_percent);
}
int assignment(Fixture const& f, int row) {
  if (f.partitions == 0) return 0;
  auto x = mix(uint32_t(row) + 717u);
  if (f.pattern == 1) return f.partitions - 1;
  if (f.pattern == 2 && x % 10 != 0) return f.partitions - 1;
  if (f.pattern == 3) return (row / 1024) % f.partitions;
  return int(x % unsigned(f.partitions));
}
std::unique_ptr<cudf::column> fixed_column(Fixture const& f, int col, cuda::stream_ref stream) {
  auto rows = f.rows + f.slice;
  auto type = f.width == 1 ? cudf::type_id::INT8 : f.width == 4 ? cudf::type_id::INT32
            : f.width == 8 ? cudf::type_id::INT64 : cudf::type_id::DECIMAL128;
  std::vector<unsigned char> host(std::size_t(rows) * f.width);
  for (int i = 0; i < rows; ++i) {
    __int128 value = __int128(i) * (37 + col * 2) + col * 11;
    std::memcpy(host.data() + std::size_t(i) * f.width, &value, f.width);
  }
  rmm::device_buffer data(host.data(), host.size(), stream);
  auto mask = cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream);
  int nulls = 0;
  if (f.mask || f.null_percent) {
    std::vector<cudf::bitmask_type> bits((rows + 31) / 32, ~0u);
    for (int i = 0; i < rows; ++i) if (!valid(f, i, col)) {
      bits[i / 32] &= ~(1u << (i % 32)); ++nulls;
    }
    mask = cudf::create_null_mask(rows, cudf::mask_state::ALL_VALID, stream);
    CUDF_CUDA_TRY(cudaMemcpyAsync(mask.data(), bits.data(), bits.size() * sizeof(bits[0]),
                                  cudaMemcpyHostToDevice, stream.get()));
    stream.sync();
  }
  stream.sync();
  return std::make_unique<cudf::column>(cudf::data_type{type}, rows,
                                       std::move(data), std::move(mask), nulls);
}
std::unique_ptr<cudf::column> string_column(Fixture const& f, int col, cuda::stream_ref stream) {
  int rows = f.rows + f.slice;
  std::vector<std::string> values(rows);
  std::vector<uint8_t> validity(rows);
  for (int i = 0; i < rows; ++i) {
    auto x = mix(uint32_t(i) + uint32_t(col) * 65537u + 1u);
    auto length = x % unsigned(f.max_length + 1);
    if (f.kind == "strings_tail") length = x % 100 ? x % 9 : unsigned(f.max_length);
    values[i].assign(length, char('a' + (i + col) % 26));
    if (length >= 8) for (int k = 0; k < 8; ++k) values[i][k] = "0123456789abcdef"[(unsigned(i) >> (k * 4)) & 15];
    validity[i] = valid(f, i, col);
  }
  if (f.mask || f.null_percent)
    return cudf::test::strings_column_wrapper(values.begin(), values.end(), validity.begin(), stream).release();
  auto result = cudf::test::strings_column_wrapper(values.begin(), values.end(), stream).release();
  if (f.kind == "strings64") {
    auto contents = result->release();
    auto offsets = cudf::cast(contents.children[0]->view(), cudf::data_type{cudf::type_id::INT64}, stream);
    result = cudf::make_strings_column(rows, std::move(offsets), std::move(*contents.data), 0,
      cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream));
  }
  return result;
}
std::unique_ptr<cudf::column> nested_column(Fixture const& f, int col, cuda::stream_ref stream) {
  Fixture child = f;
  child.rows += child.slice;
  child.slice = 0;
  auto values = fixed_column(child, col, stream);
  if (f.kind == "structs") {
    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(std::move(values));
    return cudf::make_structs_column(child.rows, std::move(children), 0,
      cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream), stream);
  }
  std::vector<int32_t> host_offsets(child.rows + 1);
  std::iota(host_offsets.begin(), host_offsets.end(), 0);
  cudf::test::fixed_width_column_wrapper<int32_t> offsets(host_offsets.begin(), host_offsets.end(), stream);
  return cudf::make_lists_column(child.rows, offsets.release(), std::move(values), 0,
    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream));
}
struct ThresholdGuard {
  std::optional<std::string> threshold, enabled;
  explicit ThresholdGuard(bool active) {
    if (auto p = std::getenv("LIBCUDF_LARGE_STRINGS_THRESHOLD")) threshold = p;
    if (auto p = std::getenv("LIBCUDF_LARGE_STRINGS_ENABLED")) enabled = p;
    if (active) {
      setenv("LIBCUDF_LARGE_STRINGS_THRESHOLD", "1024", 1);
      setenv("LIBCUDF_LARGE_STRINGS_ENABLED", "1", 1);
    }
  }
  ~ThresholdGuard() {
    if (threshold) setenv("LIBCUDF_LARGE_STRINGS_THRESHOLD", threshold->c_str(), 1);
    else unsetenv("LIBCUDF_LARGE_STRINGS_THRESHOLD");
    if (enabled) setenv("LIBCUDF_LARGE_STRINGS_ENABLED", enabled->c_str(), 1);
    else unsetenv("LIBCUDF_LARGE_STRINGS_ENABLED");
  }
};
std::vector<uint64_t> hashes(cudf::table_view input, cuda::stream_ref stream) {
  auto column = cudf::hashing::xxhash_64(input, 0x123456789ull, stream);
  std::vector<uint64_t> output(input.num_rows());
  CUDF_CUDA_TRY(cudaMemcpyAsync(output.data(), column->view().data<uint64_t>(), output.size() * sizeof(uint64_t),
                                cudaMemcpyDeviceToHost, stream.get()));
  stream.sync(); return output;
}
void verify(cudf::table_view input, cudf::table_view output, std::vector<int> const& offsets,
            std::vector<int> const& expected, Fixture const& f, cuda::stream_ref stream) {
  if (offsets != expected || output.num_rows() != (f.partitions ? f.rows : 0) || output.num_columns() != f.columns)
    throw std::runtime_error("row count, schema size, or partition boundaries differ");
  if (f.columns == 0 || f.partitions == 0 || f.rows == 0) return;
  auto source = hashes(input, stream), actual = hashes(output, stream);
  std::vector<uint64_t> ordered(source.size());
  auto cursors = expected;
  for (int i = 0; i < f.rows; ++i) ordered[cursors[assignment(f, i)]++] = source[i];
  for (int p = 0; p < f.partitions; ++p) {
    std::sort(ordered.begin() + expected[p], ordered.begin() + expected[p + 1]);
    std::sort(actual.begin() + expected[p], actual.begin() + expected[p + 1]);
  }
  if (ordered != actual) throw std::runtime_error("partition row-hash multisets differ");
}
double median(std::vector<double> x) {
  std::sort(x.begin(), x.end());
  return x.size() % 2 ? x[x.size()/2] : (x[x.size()/2-1] + x[x.size()/2]) / 2;
}
void array(std::vector<double> const& x) {
  std::cout << '['; for (std::size_t i = 0; i < x.size(); ++i) { if (i) std::cout << ','; std::cout << x[i]; } std::cout << ']';
}
uint64_t pool_value(cudaMemPool_t pool, cudaMemPoolAttr attribute) {
  uint64_t value = 0; CUDF_CUDA_TRY(cudaMemPoolGetAttribute(pool, attribute, &value)); return value;
}
void run(Fixture const& f, int samples, rmm::mr::cuda_async_memory_resource const& pool) {
  auto stream = cudf::get_default_stream();
  CUDF_CUDA_TRY(cudaMemPoolTrimTo(pool.pool_handle(), 0));
  std::vector<std::unique_ptr<cudf::column>> columns;
  for (int c = 0; c < f.columns; ++c) {
    bool text = f.kind.starts_with("strings") || (f.kind == "mixed" && c == 1);
    bool nested = f.kind == "lists" || f.kind == "structs";
    columns.push_back(text ? string_column(f, c, stream) : nested ? nested_column(f, c, stream) : fixed_column(f, c, stream));
  }
  cudf::table owner(std::move(columns), f.rows + f.slice);
  auto input = f.slice ? cudf::slice(owner.view(), {f.slice, f.rows + f.slice}, stream)[0] : owner.view();
  std::vector<int> host_map(f.rows), expected(f.partitions + 1, 0);
  for (int i = 0; i < f.rows; ++i) {
    host_map[i] = assignment(f, i);
    if (f.partitions) ++expected[host_map[i] + 1];
  }
  std::partial_sum(expected.begin(), expected.end(), expected.begin());
  cudf::test::fixed_width_column_wrapper<int> map(host_map.begin(), host_map.end(), stream);
  stream.sync();
  ThresholdGuard threshold_guard(f.kind == "strings_limit");
  uint64_t zero = 0;
  CUDF_CUDA_TRY(cudaMemPoolSetAttribute(pool.pool_handle(), cudaMemPoolAttrUsedMemHigh, &zero));
  CUDF_CUDA_TRY(cudaMemPoolSetAttribute(pool.pool_handle(), cudaMemPoolAttrReservedMemHigh, &zero));
  auto reserved_before = pool_value(pool.pool_handle(), cudaMemPoolAttrReservedMemCurrent);
  uint64_t pool_used_high, pool_reserved_high, output_bytes;
  {
    auto result = cudf::partition(input, map, f.partitions, stream);
    stream.sync(); output_bytes = result.first->alloc_size();
    pool_used_high = pool_value(pool.pool_handle(), cudaMemPoolAttrUsedMemHigh);
    pool_reserved_high = pool_value(pool.pool_handle(), cudaMemPoolAttrReservedMemHigh);
    verify(input, result.first->view(), result.second, expected, f, stream);
  }
  stream.sync();
  if (samples == 0) {
    std::cout << "{\"id\":\"" << f.id << "\",\"verified\":true}" << std::endl;
    return;
  }
  rmm::mr::statistics_resource_adaptor stats(pool);
  ResourceGuard guard(stats);
  auto warm_start = std::chrono::steady_clock::now();
  int warm_count = 0;
  do {
    { auto output = cudf::partition(input, map, f.partitions, stream, stats); }
    stream.sync(); ++warm_count;
  } while (warm_count < 3 || std::chrono::duration<double>(std::chrono::steady_clock::now()-warm_start).count() < 0.5);
  cudaEvent_t begin, end;
  CUDF_CUDA_TRY(cudaEventCreate(&begin)); CUDF_CUDA_TRY(cudaEventCreate(&end));
  std::vector<double> events, walls;
  std::size_t peak = 0, allocations = 0;
  for (int i = 0; i < samples; ++i) {
    stats.push_counters();
    auto t0 = std::chrono::steady_clock::now();
    CUDF_CUDA_TRY(cudaEventRecord(begin, stream.get()));
    { auto output = cudf::partition(input, map, f.partitions, stream, stats); }
    CUDF_CUDA_TRY(cudaEventRecord(end, stream.get()));
    CUDF_CUDA_TRY(cudaEventSynchronize(end));
    auto t1 = std::chrono::steady_clock::now();
    float elapsed = 0; CUDF_CUDA_TRY(cudaEventElapsedTime(&elapsed, begin, end));
    auto counters = stats.pop_counters();
    if (counters.first.value != 0) throw std::runtime_error("live tracked output/temporary allocation after output destruction");
    peak = std::max(peak, std::size_t(counters.first.peak));
    allocations = std::max(allocations, std::size_t(counters.second.total));
    events.push_back(elapsed); walls.push_back(std::chrono::duration<double, std::milli>(t1-t0).count());
  }
  CUDF_CUDA_TRY(cudaEventDestroy(begin)); CUDF_CUDA_TRY(cudaEventDestroy(end));
  std::vector<double> deviations; auto event_median = median(events);
  for (auto x : events) deviations.push_back(std::abs(x-event_median));
  std::cout << std::setprecision(9) << "{\"id\":\"" << f.id << "\",\"event_median_ms\":" << event_median
            << ",\"event_mad_ms\":" << median(deviations) << ",\"wall_median_ms\":" << median(walls)
            << ",\"peak_bytes\":" << peak << ",\"allocations\":" << allocations << ",\"output_bytes\":" << output_bytes
            << ",\"pool_reserved_before\":" << reserved_before << ",\"pool_used_high\":" << pool_used_high
            << ",\"pool_reserved_high\":" << pool_reserved_high << ",\"verified\":true,\"event_ms\":";
  array(events); std::cout << ",\"wall_ms\":"; array(walls); std::cout << "}" << std::endl;
}
int main(int argc, char** argv) {
  if (argc < 2) { std::cerr << "usage: partition_bench fixtures.tsv [samples]\n"; return 2; }
  CUDF_CUDA_TRY(cudaSetDevice(0));
  CUDF_CUDA_TRY(cudaFree(nullptr));
  int samples = argc > 2 ? std::stoi(argv[2]) : 9;
  Dl_info loaded{}; dladdr(reinterpret_cast<void*>(&cudf::partition), &loaded);
  std::cerr << "libcudf=" << (loaded.dli_fname ? loaded.dli_fname : "unknown") << '\n';
  rmm::mr::cuda_async_memory_resource pool{std::size_t{0}};
  ResourceGuard resource(pool);
  std::ifstream fixtures(argv[1]); if (!fixtures) throw std::runtime_error("cannot open fixture file");
  std::string line;
  while (std::getline(fixtures, line)) {
    if (line.empty() || line[0] == '#') continue;
    Fixture f{}; std::istringstream fields(line);
    if (!(fields >> f.id >> f.kind >> f.rows >> f.partitions >> f.columns >> f.width >> f.max_length
                 >> f.null_percent >> f.mask >> f.pattern >> f.slice)) throw std::runtime_error("invalid fixture row");
    std::cerr << "RUN " << f.id << std::endl;
    try { run(f, samples, pool); }
    catch (std::exception const& e) { std::cerr << "FAIL " << f.id << ": " << e.what() << std::endl; return 1; }
    cudf::get_default_stream().sync();
  }
}
