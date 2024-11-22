/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

/**
 * @brief A host-based UDF implementation.
 *
 * The aggregations perform the following computation:
 *  - For reduction: compute `sum(value^2, for value in group)` (this is sum of squared).
 *  - For segmented reduction: compute `segment_size * sum(value^2, for value in group)`.
 *  - For groupby: compute `(group_idx + 1) * sum(value^2, for value in group)`.
 *
 * In addition, for segmented reduction, if null_policy is set to `INCLUDE`, the null values are
 * replaced with an initial value if it is provided.
 */
template <typename cudf_aggregation>
class test_udf_simple_type : public cudf::host_udf_base {
  static_assert(std::is_same_v<cudf_aggregation, cudf::reduce_aggregation> ||
                std::is_same_v<cudf_aggregation, cudf::segmented_reduce_aggregation> ||
                std::is_same_v<cudf_aggregation, cudf::groupby_aggregation>);

 public:
  test_udf_simple_type() = default;

  [[nodiscard]] std::unordered_set<input_kind> const& get_required_data_kinds() const override
  {
    static std::unordered_set<input_kind> const required_data_kinds =
      [&] -> std::unordered_set<input_kind> {
      if constexpr (std::is_same_v<cudf_aggregation, cudf::reduce_aggregation>) {
        return {input_kind::INPUT_VALUES, input_kind::OUTPUT_DTYPE, input_kind::INIT_VALUE};
      } else if constexpr (std::is_same_v<cudf_aggregation, cudf::segmented_reduce_aggregation>) {
        return {input_kind::INPUT_VALUES,
                input_kind::OUTPUT_DTYPE,
                input_kind::INIT_VALUE,
                input_kind::NULL_POLICY,
                input_kind::OFFSETS};
      } else {
        return {input_kind::OFFSETS, input_kind::GROUP_LABELS, input_kind::GROUPED_VALUES};
      }
    }();

    return required_data_kinds;
  }

  [[nodiscard]] output_type operator()(std::unordered_map<input_kind, input_data> const& input,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr) override
  {
    if constexpr (std::is_same_v<cudf_aggregation, cudf::reduce_aggregation>) {
      auto const& values      = std::get<cudf::column_view>(input.at(input_kind::INPUT_VALUES));
      auto const output_dtype = std::get<cudf::data_type>(input.at(input_kind::OUTPUT_DTYPE));
      return cudf::double_type_dispatcher(
        values.type(), output_dtype, reduce_fn{}, input, stream, mr);
    } else if constexpr (std::is_same_v<cudf_aggregation, cudf::segmented_reduce_aggregation>) {
      auto const& values      = std::get<cudf::column_view>(input.at(input_kind::INPUT_VALUES));
      auto const output_dtype = std::get<cudf::data_type>(input.at(input_kind::OUTPUT_DTYPE));
      return cudf::double_type_dispatcher(
        values.type(), output_dtype, segmented_reduce_fn{}, input, stream, mr);
    } else {
      auto const& values = std::get<cudf::column_view>(input.at(input_kind::GROUPED_VALUES));
      return cudf::type_dispatcher(values.type(), groupby_fn{}, input, stream, mr);
    }
  }

  [[nodiscard]] output_type get_empty_output(
    [[maybe_unused]] std::optional<cudf::data_type> output_dtype,
    [[maybe_unused]] std::optional<std::reference_wrapper<cudf::scalar const>> init,
    [[maybe_unused]] rmm::cuda_stream_view stream,
    [[maybe_unused]] rmm::mr::device_memory_resource* mr) const override
  {
    if constexpr (std::is_same_v<cudf_aggregation, cudf::reduce_aggregation> ||
                  std::is_same_v<cudf_aggregation, cudf::segmented_reduce_aggregation>) {
      CUDF_EXPECTS(output_dtype.has_value(),
                   "Data type for the reduction result must be specified.");
      if (init.has_value() && init.value().get().is_valid(stream)) {
        CUDF_EXPECTS(output_dtype.value() == init.value().get().type(),
                     "Data type for reduction result must be the same as init value.");
        return std::make_unique<cudf::scalar>(init.value().get());
      }
      return cudf::make_default_constructed_scalar(output_dtype.value(), stream, mr);
    } else {
      return cudf::make_empty_column(
        cudf::data_type{cudf::type_to_id<typename groupby_fn::OutputType>()});
    }
  }

  [[nodiscard]] bool is_equal(host_udf_base const& other) const override
  {
    // Just check if the other object is also instance of the same derived class.
    return dynamic_cast<test_udf_simple_type const*>(&other) != nullptr;
  }

  [[nodiscard]] std::size_t do_hash() const override
  {
    return std::hash<std::string>{}({"test_udf_simple_type"});
  }

  [[nodiscard]] std::unique_ptr<host_udf_base> clone() const override
  {
    return std::make_unique<host_udf_base>();
  }

 private:
  struct reduce_fn {
    template <typename InputType,
              typename OutputType,
              typename... Args,
              CUDF_ENABLE_IF(!cudf::is_numeric<InputType>() || !cudf::is_numeric<OutputType>())>
    output_type operator()(Args...) const
    {
      CUDF_FAIL("Unsupported input type.");
    }

    template <typename InputType,
              typename OutputType,
              CUDF_ENABLE_IF(cudf::is_numeric<InputType>() && cudf::is_numeric<OutputType>())>
    output_type operator()(std::unordered_map<input_kind, input_data> const& input,
                           rmm::cuda_stream_view stream,
                           rmm::mr::device_memory_resource* mr) const
    {
      auto const& values      = std::get<cudf::column_view>(input.at(input_kind::INPUT_VALUES));
      auto const output_dtype = std::get<cudf::data_type>(input.at(input_kind::OUTPUT_DTYPE));
      auto const input_init_value =
        std::get<std::optional<std::reference_wrapper<cudf::scalar const>>>(
          input.at(input_kind::INIT_VALUE));

      if (values.size() == 0) {
        return get_empty_output(output_dtype, input_init_value, stream, mr);
      }

      auto const init_value = [&] -> OutputType {
        if (input_init_value.has_value() && input_init_value.value().get().is_valid(stream)) {
          CUDF_EXPECTS(output_dtype == input_init_value.value().get().type(),
                       "Data type for reduction result must be the same as init value.");
          auto const numeric_init_scalar =
            dynamic_cast<cudf::numeric_scalar const*>(&input_init_value.value().get());
          CUDF_EXPECTS(numeric_init_scalar != nullptr, "Invalid init scalar for reduction.");
          return static_cast<OutputType>(numeric_init_scalar->value(stream));
        }
        return OutputType{0};
      }();

      auto const values_dv_ptr = cudf::column_device_view::create(values, stream);
      auto const result        = thrust::transform_reduce(
        rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(values.size()),
        [values = *values_dv_ptr] __device__(cudf::size_type idx) -> OutputType {
          if (values.is_null(idx)) { return OutputType{0}; }
          auto const val = static_cast<OutputType>(values.element<InputType>(idx));
          return val * val;
        },
        init_value,
        thrust::plus<>{});

      auto output = cudf::make_numeric_scalar(output_dtype, stream, mr);
      static_cast<cudf::scalar_type_t<OutputType>*>(output.get())->set_value(result, stream);
      return output;
    }
  };

  struct segmented_reduce_fn {
    template <typename InputType,
              typename OutputType,
              typename... Args,
              CUDF_ENABLE_IF(!cudf::is_numeric<InputType>() || !cudf::is_numeric<OutputType>())>
    output_type operator()(Args...) const
    {
      CUDF_FAIL("Unsupported input type.");
    }

    template <typename InputType,
              typename OutputType,
              CUDF_ENABLE_IF(cudf::is_numeric<InputType>() && cudf::is_numeric<OutputType>())>
    output_type operator()(std::unordered_map<input_kind, input_data> const& input,
                           rmm::cuda_stream_view stream,
                           rmm::mr::device_memory_resource* mr) const
    {
      auto const& values      = std::get<cudf::column_view>(input.at(input_kind::INPUT_VALUES));
      auto const output_dtype = std::get<cudf::data_type>(input.at(input_kind::OUTPUT_DTYPE));
      auto const input_init_value =
        std::get<std::optional<std::reference_wrapper<cudf::scalar const>>>(
          input.at(input_kind::INIT_VALUE));

      if (values.size() == 0) {
        return get_empty_output(output_dtype, input_init_value, stream, mr);
      }

      auto const init_value = [&] -> OutputType {
        if (input_init_value.has_value() && input_init_value.value().get().is_valid(stream)) {
          CUDF_EXPECTS(output_dtype == input_init_value.value().get().type(),
                       "Data type for reduction result must be the same as init value.");
          auto const numeric_init_scalar =
            dynamic_cast<cudf::numeric_scalar const*>(&input_init_value.value().get());
          CUDF_EXPECTS(numeric_init_scalar != nullptr, "Invalid init scalar for reduction.");
          return static_cast<OutputType>(numeric_init_scalar->value(stream));
        }
        return OutputType{0};
      }();

      auto const null_handling = std::get<cudf::null_policy>(input.at(input_kind::NULL_POLICY));
      auto const offsets =
        std::get<cudf::device_span<cudf::size_type const>>(input.at(input_kind::OFFSETS));
      CUDF_EXPECTS(offsets.size() > 0, "Invalid offsets.");
      auto const num_segments = offsets.size() - 1;

      auto const values_dv_ptr = cudf::column_device_view::create(values, stream);
      auto output              = cudf::make_numeric_column(
        output_dtype, num_segments, cudf::mask_state::UNALLOCATED, stream);
      rmm::device_uvector<bool> validity(num_segments, stream);

      auto const result = thrust::transform(
        rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_segments),
        thrust::make_zip_iterator(output->mutable_view().begin<OutputType>(), validity.begin()),
        [values = *values_dv_ptr, init_value, null_handling, offsets] __device__(
          cudf::size_type idx) -> thrust::tuple<OutputType, bool> {
          auto const start = offsets[idx];
          auto const end   = offsets[idx + 1];
          if (start == end) { return {OutputType{0}, false}; }

          auto sum = init_value;
          for (auto i = start; i < end; ++i) {
            if (values.is_null(i)) {
              if (null_handling == cudf::null_policy::INCLUDE) { sum += init_value * init_value; }
              continue;
            }
            auto const val = static_cast<OutputType>(values.element<InputType>(i));
            sum += val * val;
          }
          auto const segment_size = end - start;
          return {segment_size * sum, true};
        });
      auto [null_mask, null_count] =
        cudf::detail::valid_if(validity.begin(), validity.end(), thrust::identity<>{}, stream, mr);
      if (null_count > 0) { output->set_null_mask(std::move(null_mask), null_count); }
      return output;
    }
  };

  struct groupby_fn {
    using OutputType = double;

    template <typename InputType, typename... Args, CUDF_ENABLE_IF(!cudf::is_numeric<InputType>())>
    output_type operator()(Args...) const
    {
      CUDF_FAIL("Unsupported input type.");
    }

    template <typename InputType, CUDF_ENABLE_IF(cudf::is_numeric<InputType>())>
    output_type operator()(std::unordered_map<input_kind, input_data> const& input,
                           rmm::cuda_stream_view stream,
                           rmm::mr::device_memory_resource* mr) const
    {
      auto const& values = std::get<cudf::column_view>(input.at(input_kind::GROUPED_VALUES));
      if (values.size() == 0) { return get_empty_output(std::nullopt, std::nullopt, stream, mr); }

      auto const offsets =
        std::get<cudf::device_span<cudf::size_type const>>(input.at(input_kind::OFFSETS));
      CUDF_EXPECTS(offsets.size() > 0, "Invalid offsets.");
      auto const num_groups = offsets.size() - 1;
      auto const group_indices =
        std::get<cudf::device_span<cudf::size_type const>>(input.at(input_kind::GROUP_LABELS));

      auto const values_dv_ptr = cudf::column_device_view::create(values, stream);
      auto output = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<OutputType>()},
                                              num_groups,
                                              cudf::mask_state::UNALLOCATED,
                                              stream);
      rmm::device_uvector<bool> validity(num_groups, stream);

      auto const result = thrust::transform(
        rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_groups),
        thrust::make_zip_iterator(output->mutable_view().begin<OutputType>(), validity.begin()),
        [values = *values_dv_ptr, offsets, group_indices] __device__(
          cudf::size_type idx) -> thrust::tuple<OutputType, bool> {
          auto const start = offsets[idx];
          auto const end   = offsets[idx + 1];
          if (start == end) { return {OutputType{0}, false}; }

          auto sum = OutputType{0};
          for (auto i = start; i < end; ++i) {
            if (values.is_null(i)) { continue; }
            auto const val = static_cast<OutputType>(values.element<InputType>(i));
            sum += val * val;
          }
          return {(group_indices[idx] + 1) * sum, true};
        });
      auto [null_mask, null_count] =
        cudf::detail::valid_if(validity.begin(), validity.end(), thrust::identity<>{}, stream, mr);
      if (null_count > 0) { output->set_null_mask(std::move(null_mask), null_count); }
      return output;
    }
  };
};

// using namespace cudf::test::iterators;
using int32s_col = cudf::test::fixed_width_column_wrapper<int32_t>;

struct HostUDFReductionTest : cudf::test::BaseFixture {};

TEST_F(HostUDFReductionTest, SimpleInput)
{
  int32s_col vals{0, 1, 2, 3, 4, 5};

  auto agg = cudf::make_host_udf_aggregation<cudf::groupby_aggregation>(
    std::make_unique<test_udf_simple_type<cudf::reduce_aggregation>>());
  auto const reduced = cudf::reduce(vals, agg, cudf::data_type{cudf::type_id::INT64});
  auto const result =
    static_cast<cudf::scalar_type_t<int64_t>*>(reduced.get())->value(cudf::get_default_stream());
  printf("Result: %ld\n", result);
}
