// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <cub/device/device_scan.cuh>
#include <cuda/buffer>
#include <cuda/std/bit>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cudf::detail::partition_rank_scatter {
constexpr int threads = 256, tile_rows = 1024, warps = 8, bins = 256;

// Amortize preparation and avoid raising peak memory for narrow/boundary-only tables.
inline bool use_rank_scatter(table_view const& input, size_type partitions)
{
  if (input.num_rows() < 1'000'000 || partitions < 2 || partitions > bins) return false;
  std::size_t row_storage = 0;
  for (auto const& col : input) {
    if (col.type().id() == type_id::EMPTY) continue;
    if (cudf::is_fixed_width(col.type())) {
      row_storage += cudf::size_of(col.type());
    } else if (col.type().id() == type_id::STRING || col.type().id() == type_id::LIST ||
               col.type().id() == type_id::DICTIONARY32) {
      row_storage += sizeof(size_type);  // Offsets or indices; never inspect device data.
    }
    if (row_storage >= sizeof(size_type)) return true;
  }
  return false;
}

template <typename MapType>
__device__ void local_tile(MapType const* map,
                           int n,
                           int start,
                           int p,
                           int (&hist)[warps][bins],
                           int (&rank)[4],
                           int (&keys)[4])
{
  int lane = threadIdx.x % 32, w = threadIdx.x / 32;
  for (int q = lane; q < p; q += 32)
    hist[w][q] = 0;
  __syncwarp();
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int row         = start + w * 128 + j * 32 + lane;
    bool valid      = row < n;
    unsigned active = __ballot_sync(0xffffffffu, valid);
    if (valid) {
      int q          = map[row];
      unsigned peers = __match_any_sync(active, q);
      int base       = hist[w][q];
      keys[j]        = q;
      rank[j]        = base + cuda::std::popcount(peers & ((1u << lane) - 1u));
      __syncwarp(active);
      if (lane == __ffs(peers) - 1) hist[w][q] = base + cuda::std::popcount(peers);
    }
    __syncwarp();
  }
  __syncthreads();
}

template <typename MapType>
CUDF_KERNEL void count_tiles(MapType const* map, int n, int p, uint16_t* saved, int32_t* counts)
{
  __shared__ int hist[warps][bins];
  int rank[4], keys[4];
  int const start = blockIdx.x * tile_rows;
  local_tile(map, n, start, p, hist, rank, keys);
  if (threadIdx.x < p) {
    int const q = threadIdx.x;
    int sum     = 0;
#pragma unroll
    for (int w = 0; w < warps; ++w) {
      int const c = hist[w][q];
      hist[w][q]  = sum;
      sum += c;
    }
    counts[std::size_t{threadIdx.x} * gridDim.x + blockIdx.x] = sum;
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int const i = start + (threadIdx.x / 32) * 128 + j * 32 + threadIdx.x % 32;
    if (i < n) saved[i] = hist[threadIdx.x / 32][keys[j]] + rank[j];
  }
}

CUDF_KERNEL void get_offsets(int32_t const* counts, int blocks, int p, int n, int32_t* offsets)
{
  if (threadIdx.x < p) offsets[threadIdx.x] = counts[size_t(threadIdx.x) * blocks];
  if (threadIdx.x == 0) offsets[p] = n;
}

template <typename Sink, typename MapType>
CUDF_KERNEL void scatter_saved(MapType const* map,
                               uint16_t const* ranks,
                               int32_t const* offsets,
                               int n,
                               int blocks,
                               int p,
                               Sink sink)
{
  __shared__ int base[bins];
  if (threadIdx.x < p) base[threadIdx.x] = offsets[size_t(threadIdx.x) * blocks + blockIdx.x];
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int i = blockIdx.x * tile_rows + j * threads + threadIdx.x;
    if (i < n) sink(i, base[map[i]] + ranks[i]);
  }
}

struct scatter_sink {
  size_type* map;
  __device__ void operator()(size_type input, size_type output) const { map[input] = output; }
};

template <typename MapType>
std::vector<size_type> run(
  MapType const* map, size_type n, size_type p, size_type* output, cuda::stream_ref stream)
{
  auto const blocks  = (n - 1) / tile_rows + 1;
  auto const temp_mr = cudf::get_current_device_resource_ref();
  cuda::device_buffer<uint16_t> ranks(stream, temp_mr, n, cuda::no_init);
  cuda::device_buffer<int32_t> counts(
    stream, temp_mr, static_cast<std::size_t>(p) * blocks, cuda::no_init);
  cuda::device_buffer<int32_t> offsets(stream, temp_mr, p + 1, cuda::no_init);
  count_tiles<<<blocks, threads, 0, stream.get()>>>(map, n, p, ranks.data(), counts.data());
  CUDF_CUDA_TRY(cudaPeekAtLastError());
  // Partition-major scan supplies each partition's base and all preceding-block
  // counts.
  std::size_t scan_bytes{};
  CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
    nullptr, scan_bytes, counts.data(), counts.data(), counts.size(), stream.get()));
  cuda::device_buffer<std::byte> scratch(stream, temp_mr, scan_bytes, cuda::no_init);
  CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
    scratch.data(), scan_bytes, counts.data(), counts.data(), counts.size(), stream.get()));
  get_offsets<<<1, threads, 0, stream.get()>>>(counts.data(), blocks, p, n, offsets.data());
  scatter_saved<<<blocks, threads, 0, stream.get()>>>(
    map, ranks.data(), counts.data(), n, blocks, p, scatter_sink{output});
  CUDF_CUDA_TRY(cudaPeekAtLastError());
  auto const host_offsets = cudf::detail::make_pinned_vector(offsets, stream);
  return {host_offsets.begin(), host_offsets.end()};
}
}  // namespace cudf::detail::partition_rank_scatter
