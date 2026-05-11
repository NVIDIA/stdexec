/*
 * Copyright (c) 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "__config.hpp"

#include <algorithm>
#include <ranges>

#include "__prologue.hpp"

namespace STDEXEC  // NOLINT(modernize-concat-nested-namespaces)
{
  namespace [[deprecated("Use std::ranges instead")]] ranges
  {
    using std::ranges::begin;
    using std::ranges::end;

    using std::ranges::range_value_t;
    using std::ranges::range_reference_t;
    using std::ranges::iterator_t;
    using std::ranges::sentinel_t;
  }  // namespace ranges

  namespace __ranges
  {
    // Define `std::from_range` if the standard library doesn't provide it.
#if __cpp_lib_containers_ranges >= 202202L
    using std::from_range_t;
    using std::from_range;
#else
    struct from_range_t
    {
      explicit from_range_t() = default;
    };
    inline constexpr from_range_t from_range{};
#endif

    // Define `ranges::contains` if the standard library doesn't provide it.
#if __cpp_lib_ranges_contains >= 202207L
    using std::ranges::contains;
#else
    inline constexpr auto contains =
      []<std::ranges::input_range _Range,
         class _Proj  = std::identity,
         class _Value = std::ranges::range_value_t<_Range>>(_Range&&       __rng,
                                                            _Value const & __value,
                                                            _Proj          __proj = {}) -> bool
    {
      return std::ranges::find(__rng, __value, __proj) != std::ranges::end(__rng);
    };
#endif
  }  // namespace __ranges
}  // namespace STDEXEC

#include "__epilogue.hpp"
