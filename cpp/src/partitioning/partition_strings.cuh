// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "partition_string_copy.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/utilities.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>

#include <cub/device/device_scan.cuh>
#include <cuda/buffer>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

namespace cudf::detail {
// A complete permutation preserves the sliced input's total character count.
CUDF_KERNEL void partition_string_sizes(partition_string_column const* columns,
                                        size_type count,
                                        size_type rows,
                                        size_type* sizes)
{
  auto const c = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (c < count) sizes[c] = columns[c].input_offsets[rows] - columns[c].input_offsets[0];
}
CUDF_KERNEL void scatter_partition_string_lengths(partition_string_column const* columns,
                                                  size_type const* map,
                                                  size_type rows)
{
  auto const col = columns[blockIdx.y];
  auto const row = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (row < rows)
    col.output_offsets[map[row]] = col.input_offsets[row + 1] - col.input_offsets[row];
  if (row == 0) col.output_offsets[rows] = 0;
}

// CPU-only admission: amortize preparation over multiple columns and at least
// one tile. Other schemas and configured reduced large-STRING thresholds retain
// generic scatter.
inline bool use_batched_partition_strings(table_view const& input, size_type partitions)
{
  // Conservative standalone admission: bound the full-width row map to 16 MiB.
  constexpr size_type max_rows = (16 * 1024 * 1024) / sizeof(size_type);
  return input.num_rows() >= 1024 && input.num_rows() <= max_rows && partitions >= 2 &&
         partitions <= 256 && input.num_columns() >= 2 && input.num_columns() <= 65'535 &&
         std::all_of(input.begin(),
                     input.end(),
                     [](auto const& col) {
                       return col.type().id() == type_id::STRING && col.null_count() == 0 &&
                              strings_column_view(col).offsets().type().id() == type_id::INT32;
                     }) &&
         cudf::strings::get_offset64_threshold() == std::numeric_limits<size_type>::max();
}

// map[input] = output, independent of how the partition map was constructed.
inline std::unique_ptr<table> scatter_partition_strings(table_view const& input,
                                                        device_span<size_type const> map,
                                                        size_type partitions,
                                                        cuda::stream_ref stream,
                                                        rmm::device_async_resource_ref mr)
{
  if (!use_batched_partition_strings(input, partitions)) return nullptr;
  auto const rows    = input.num_rows();
  auto const count   = input.num_columns();
  auto const temp_mr = cudf::get_current_device_resource_ref();
  std::vector<partition_string_column> descriptors;
  for (auto const& col : input) {
    auto const strings = strings_column_view(col);
    descriptors.push_back({strings.offsets().data<size_type>() + col.offset(),
                           strings.chars_begin(stream),
                           nullptr,
                           nullptr});
  }
  auto device_columns = make_device_buffer(descriptors, stream, temp_mr);
  cuda::device_buffer<size_type> sizes(stream, temp_mr, count, cuda::no_init);
  partition_string_sizes<<<(count + 255) / 256, 256, 0, stream.get()>>>(
    device_columns.data(), count, rows, sizes.data());
  CUDF_CUDA_TRY(cudaPeekAtLastError());
  auto const char_sizes = make_pinned_vector(sizes, stream);
  // This mandatory allocation-bound readback is not used to tune the heuristic.
  if (*std::max_element(char_sizes.begin(), char_sizes.end()) >=
      cudf::strings::get_offset64_threshold())
    return nullptr;

  std::vector<std::unique_ptr<column>> offsets;
  std::vector<rmm::device_buffer> chars;
  for (size_type c = 0; c < count; ++c) {
    offsets.push_back(make_numeric_column(
      data_type{type_id::INT32}, rows + 1, mask_state::UNALLOCATED, stream, mr));
    chars.emplace_back(char_sizes[c], stream, mr);
    descriptors[c].output_offsets = offsets.back()->mutable_view().data<size_type>();
    descriptors[c].output_chars   = static_cast<char*>(chars.back().data());
  }
  device_columns    = make_device_buffer(descriptors, stream, temp_mr);
  auto const blocks = (static_cast<std::size_t>(rows) + 255) / 256;
  scatter_partition_string_lengths<<<dim3(blocks, count), 256, 0, stream.get()>>>(
    device_columns.data(), map.data(), rows);
  CUDF_CUDA_TRY(cudaPeekAtLastError());
  std::size_t scan_bytes{};
  CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(nullptr,
                                              scan_bytes,
                                              descriptors[0].output_offsets,
                                              descriptors[0].output_offsets,
                                              rows + 1,
                                              stream.get()));
  cuda::device_buffer<std::byte> scratch(stream, temp_mr, scan_bytes, cuda::no_init);
  for (auto const& col : descriptors) {
    CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
      scratch.data(), scan_bytes, col.output_offsets, col.output_offsets, rows + 1, stream.get()));
  }
  partition_strings_plan::copy_characters<true>
    <<<dim3(std::min<std::size_t>(65'535, (static_cast<std::size_t>(rows) + 63) / 64), count),
       256,
       0,
       stream.get()>>>(device_columns.data(), map.data(), rows);
  CUDF_CUDA_TRY(cudaPeekAtLastError());
  std::vector<std::unique_ptr<column>> output;
  for (size_type c = 0; c < count; ++c) {
    output.push_back(make_strings_column(rows,
                                         std::move(offsets[c]),
                                         std::move(chars[c]),
                                         0,
                                         create_null_mask(0, mask_state::UNALLOCATED, stream, mr)));
  }
  return std::make_unique<table>(std::move(output));
}
}  // namespace cudf::detail
