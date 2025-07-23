/*
 * Copyright (c) 2025 Lucian Radu Teodorescu
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

#if STDEXEC_HAS_EXECUTION_POLICY()
#  include <execution>
#endif

namespace stdexec {

#if STDEXEC_HAS_EXECUTION_POLICY()

  using sequenced_policy = std::execution::sequenced_policy;
  using parallel_policy = std::execution::parallel_policy;
  using parallel_unsequenced_policy = std::execution::parallel_unsequenced_policy;

  constexpr auto seq = std::execution::seq;
  constexpr auto par = std::execution::par;
  constexpr auto par_unseq = std::execution::par_unseq;

  using std::is_execution_policy_v;
  using std::is_execution_policy;

#else

  struct __hidden_construction { };

  struct sequenced_policy {
    constexpr explicit sequenced_policy(__hidden_construction) { };
    sequenced_policy(const sequenced_policy&) = delete;
    sequenced_policy& operator=(const sequenced_policy&) = delete;
  };

  struct parallel_policy {
    constexpr explicit parallel_policy(__hidden_construction) { };
    parallel_policy(const parallel_policy&) = delete;
    parallel_policy& operator=(const parallel_policy&) = delete;
  };

  struct parallel_unsequenced_policy {
    constexpr explicit parallel_unsequenced_policy(__hidden_construction) { };
    parallel_unsequenced_policy(const parallel_unsequenced_policy&) = delete;
    parallel_unsequenced_policy& operator=(const parallel_unsequenced_policy&) = delete;
  };

  inline constexpr sequenced_policy seq{__hidden_construction{}};
  inline constexpr parallel_policy par{__hidden_construction{}};
  inline constexpr parallel_unsequenced_policy par_unseq{__hidden_construction{}};

  template <typename>
  inline constexpr bool is_execution_policy_v = false;

  template <>
  inline constexpr bool is_execution_policy_v<sequenced_policy> = true;

  template <>
  inline constexpr bool is_execution_policy_v<parallel_policy> = true;

  template <>
  inline constexpr bool is_execution_policy_v<parallel_unsequenced_policy> = true;

  template <class _T>
  struct is_execution_policy : std::bool_constant<is_execution_policy_v<_T>> { };

#endif

#if STDEXEC_HAS_UNSEQUENCED_EXECUTION_POLICY()

  using unsequenced_policy = std::execution::unsequenced_policy;

  constexpr auto unseq = std::execution::unseq;

#else

#  if STDEXEC_HAS_EXECUTION_POLICY()
  // already defined above
  struct __hidden_construction { };
#  endif

  struct unsequenced_policy {
    constexpr explicit unsequenced_policy(__hidden_construction) { };
    unsequenced_policy(const unsequenced_policy&) = delete;
    unsequenced_policy& operator=(const unsequenced_policy&) = delete;
  };

  inline constexpr unsequenced_policy unseq{__hidden_construction{}};

  template <>
  inline constexpr bool is_execution_policy_v<unsequenced_policy> = true;

#endif

} // namespace stdexec