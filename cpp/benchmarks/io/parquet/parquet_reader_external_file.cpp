/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

void BM_parquet_read_external_file(nvbench::state& state)
{
  auto const* const filepath_env = std::getenv("CUDF_BENCHMARK_PARQUET_FILE");
  if (filepath_env == nullptr || filepath_env[0] == '\0') {
    state.skip("CUDF_BENCHMARK_PARQUET_FILE is not set");
    return;
  }

  auto const filepath = std::string{filepath_env};
  CUDF_EXPECTS(std::filesystem::is_regular_file(filepath),
               "CUDF_BENCHMARK_PARQUET_FILE is not a regular file");

  auto const source           = cudf::io::source_info{filepath};
  auto const metadata         = cudf::io::read_parquet_metadata(source);
  auto const expected_rows    = metadata.num_rows();
  auto const expected_columns = metadata.schema().root().children().size();
  auto const encoded_bytes    = std::filesystem::file_size(filepath);
  auto const chunk_read_limit = static_cast<std::size_t>(state.get_int64("chunk_read_limit"));
  auto const pass_read_limit  = static_cast<std::size_t>(state.get_int64("pass_read_limit"));

  auto const read_options = cudf::io::parquet_reader_options::builder(source).build();
  auto mem_stats_logger   = cudf::memory_stats_logger();
  std::size_t num_chunks  = 0;

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch&, auto& timer) {
      drop_page_cache_if_enabled(read_options.get_source().filepaths());

      timer.start();
      auto reader =
        cudf::io::chunked_parquet_reader(chunk_read_limit, pass_read_limit, read_options);
      int64_t num_rows_read   = 0;
      std::size_t chunks_read = 0;
      do {
        auto const result = reader.read_chunk();
        num_rows_read += result.tbl->num_rows();
        ++chunks_read;
        CUDF_EXPECTS(static_cast<std::size_t>(result.tbl->num_columns()) == expected_columns,
                     "Unexpected number of columns");
      } while (reader.has_next());
      timer.stop();

      CUDF_EXPECTS(num_rows_read == expected_rows, "Benchmark did not read the entire file");
      num_chunks = chunks_read;
    });

  auto const elapsed_time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(encoded_bytes) / elapsed_time,
                          "encoded_bytes_per_second");
  state.add_element_count(static_cast<double>(expected_rows) / elapsed_time, "rows_per_second");
  state.add_element_count(num_chunks, "chunks_per_iteration");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(encoded_bytes, "encoded_file_size", "encoded_file_size");
}

}  // namespace

NVBENCH_BENCH(BM_parquet_read_external_file)
  .set_name("parquet_read_external_file")
  .set_min_samples(10)
  .add_int64_axis("chunk_read_limit", {256 << 20})
  .add_int64_axis("pass_read_limit", {0});
