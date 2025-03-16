/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/lists/detail/lists_column_factories.hpp>
#include <cudf/lists/filling.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/tabulate.h>

#include <limits>
#include <optional>
#include <stdexcept>

namespace cudf::lists {
namespace detail {
namespace {

template <typename T, typename Enable = void>
struct sequences_functor {
  template <typename... Args>
  static std::unique_ptr<column> invoke(Args&&...)
  {
    CUDF_FAIL("Unsupported per-list sequence type-agg combination.");
  }
};

struct sequences_dispatcher {
  template <typename T>
  std::unique_ptr<column> operator()(size_type n_lists,
                                     size_type n_elements,
                                     column_view const& starts,
                                     column_view const& sizes,
                                     std::optional<column_view> const& steps,
                                     size_type const* offsets,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  {
    return sequences_functor<T>::invoke(
      n_lists, n_elements, starts, sizes, steps, offsets, stream, mr);
  }
};

template <typename T>
static constexpr bool is_supported()
{
  return (cudf::is_numeric<T>() && !cudf::is_boolean<T>()) || cudf::is_duration<T>();
}

template <typename T>
void fill(T const* starts,
          size_type const* sizes,
          size_type const* offsets,
          size_type n_lists,
          size_type n_elements,
          T* result,
          rmm::cuda_stream_view stream,
          rmm::device_async_resource_ref mr)
{
  auto labels = make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, n_elements, mask_state::UNALLOCATED, stream, mr);
  auto labels_begin = labels->mutable_view().begin<int32_t>();
  cudf::detail::label_segments(
    offsets, offsets + n_lists + 1, labels_begin, labels_begin + n_elements, stream);

  thrust::tabulate(
    rmm::exec_policy(stream),
    result,
    result + n_elements,
    [starts, sizes, offsets, labels = labels->view().begin<size_type>()] __device__(size_type i) {
      size_type const list_idx    = labels[i];
      size_type const list_offset = offsets[list_idx];
      size_type const idx_in_list = i - list_offset;
      T const start               = starts[list_idx];
      return start + static_cast<T>(idx_in_list);
    });
}

template <typename T>
struct sequences_functor<T, std::enable_if_t<is_supported<T>()>> {
  static std::unique_ptr<column> invoke(size_type n_lists,
                                        size_type n_elements,
                                        column_view const& starts,
                                        column_view const& sizes,
                                        std::optional<column_view> const& steps,
                                        size_type const* offsets,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
  {
    auto result =
      make_fixed_width_column(starts.type(), n_elements, mask_state::UNALLOCATED, stream, mr);
    if (starts.is_empty()) { return result; }

    T* result_begin = result->mutable_view().template begin<T>();

    // method 1: use list parallelism
    // Use pointers instead of column_device_view to access start and step values should be enough.
    // This is because we don't need to check for nulls and only support numeric and duration types.
    auto const* starts_begin = starts.template begin<T>();
    auto const* sizes_begin  = sizes.template begin<size_type>();
    // thrust::for_each(rmm::exec_policy(stream),
    //   thrust::make_counting_iterator<cudf::size_type>(0),
    //   thrust::make_counting_iterator<cudf::size_type>(n_lists),
    //   [starts_begin, sizes_begin, offsets, result_begin] __device__(auto const list_idx) {
    //     T start = starts_begin[list_idx];
    //     size_type size  = sizes_begin[list_idx];
    //     size_type offset = offsets[list_idx];
    //     for (size_type i = 0; i < size; i++)
    //     {
    //       result_begin[offset + i] = start + static_cast<T>(i);
    //     }
    //   });

    // method 2: use label_segments
    fill(starts_begin, sizes_begin, offsets, n_lists, n_elements, result_begin, stream, mr);
    return result;
  }
};

std::unique_ptr<column> sequences(column_view const& starts,
                                  std::optional<column_view> const& steps,
                                  column_view const& sizes,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!starts.has_nulls() && !sizes.has_nulls(),
               "starts and sizes input columns must not have nulls.");
  CUDF_EXPECTS(starts.size() == sizes.size(),
               "starts and sizes input columns must have the same number of rows.");
  CUDF_EXPECTS(cudf::is_index_type(sizes.type()),
               "Input sizes column must be of integer types.",
               cudf::data_type_error);

  if (steps) {
    // TODO: support steps.
    CUDF_FAIL("Unsupported per-list sequence type-agg combination.");
  }

  auto const n_lists = starts.size();
  if (n_lists == 0) { return cudf::make_empty_lists_column(starts.type(), stream, mr); }

  // Generate list offsets for the output.
  auto list_offsets = make_numeric_column(
    data_type(type_to_id<size_type>()), n_lists + 1, mask_state::UNALLOCATED, stream, mr);
  auto const offsets_begin  = list_offsets->mutable_view().template begin<size_type>();
  auto const sizes_input_it = cudf::detail::indexalator_factory::make_input_iterator(sizes);
  // First copy the sizes since the exclusive_scan tries to read (n_lists+1) values
  thrust::copy_n(rmm::exec_policy(stream), sizes_input_it, sizes.size(), offsets_begin);
  auto const n_elements = cudf::detail::sizes_to_offsets(
    offsets_begin, offsets_begin + list_offsets->size(), offsets_begin, 0, stream);
  CUDF_EXPECTS(n_elements <= std::numeric_limits<size_type>::max(),
               "Size of output exceeds the column size limit",
               std::overflow_error);

  auto child = type_dispatcher(starts.type(),
                               sequences_dispatcher{},
                               n_lists,
                               static_cast<size_type>(n_elements),
                               starts,
                               sizes,
                               steps,
                               offsets_begin,
                               stream,
                               mr);

  return make_lists_column(n_lists,
                           std::move(list_offsets),
                           std::move(child),
                           0,
                           rmm::device_buffer(0, stream, mr),
                           stream,
                           mr);
}

}  // anonymous namespace

std::unique_ptr<column> sequences(column_view const& starts,
                                  column_view const& sizes,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  return sequences(starts, std::nullopt, sizes, stream, mr);
}

std::unique_ptr<column> sequences(column_view const& starts,
                                  column_view const& steps,
                                  column_view const& sizes,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  return sequences(starts, std::optional<column_view>{steps}, sizes, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> sequences(column_view const& starts,
                                  column_view const& sizes,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::sequences(starts, sizes, stream, mr);
}

std::unique_ptr<column> sequences(column_view const& starts,
                                  column_view const& steps,
                                  column_view const& sizes,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::sequences(starts, steps, sizes, stream, mr);
}

}  // namespace cudf::lists
