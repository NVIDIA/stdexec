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

namespace STDEXEC {

#if STDEXEC_HAS_EXECUTION_POLICY()

  // Import the execution policies from std::execution. The __policy namespace is used to
  // avoid name clashes if the macro STDEXEC expands to std::execution.
  namespace __policy {
    using std::execution::sequenced_policy;
    using std::execution::parallel_policy;
    using std::execution::parallel_unsequenced_policy;

    using std::execution::seq;
    using std::execution::par;
    using std::execution::par_unseq;

    using std::is_execution_policy_v;
    using std::is_execution_policy;
  } // namespace __policy

  using namespace __policy;

#else

  struct sequenced_policy { };
  struct parallel_policy { };
  struct parallel_unsequenced_policy { };

  inline constexpr sequenced_policy seq{};
  inline constexpr parallel_policy par{};
  inline constexpr parallel_unsequenced_policy par_unseq{};

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

  namespace __policy {
    using std::execution::unsequenced_policy;
    using std::execution::unseq;
  } // namespace __policy

  using namespace __policy;

#else

  struct unsequenced_policy { };

  inline constexpr unsequenced_policy unseq{};

  template <>
  inline constexpr bool is_execution_policy_v<unsequenced_policy> = true;

#endif

} // namespace STDEXEC
