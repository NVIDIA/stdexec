/*
 * Copyright (c) 2026 NVIDIA Corporation
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

#include "../stdexec/__detail/__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "../stdexec/__detail/__completion_behavior.hpp"

namespace experimental::execution
{
  struct completion_behavior
  {
    using unknown_t             = STDEXEC::__completion_behavior::__unknown_t;
    using asynchronous_t        = STDEXEC::__completion_behavior::__asynchronous_t;
    using asynchronous_affine_t = STDEXEC::__completion_behavior::__asynchronous_affine_t;
    using inline_completion_t   = STDEXEC::__completion_behavior::__inline_completion_t;
    using weakest_t             = STDEXEC::__completion_behavior::__weakest_t;

    static constexpr auto const &unknown      = STDEXEC::__completion_behavior::__unknown;
    static constexpr auto const &asynchronous = STDEXEC::__completion_behavior::__asynchronous;
    static constexpr auto const &asynchronous_affine =
      STDEXEC::__completion_behavior::__asynchronous_affine;
    static constexpr auto const &inline_completion =
      STDEXEC::__completion_behavior::__inline_completion;
    static constexpr auto const &weakest = STDEXEC::__completion_behavior::__weakest;
  };

  template <STDEXEC::__completion_tag _Tag>
  using get_completion_behavior_t = STDEXEC::__get_completion_behavior_t<_Tag>;

  template <class _Tag, class _Sndr, class... _Env>
  STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
  constexpr auto get_completion_behavior() noexcept
  {
    return STDEXEC::__get_completion_behavior<_Tag, _Sndr, _Env...>();
  }
}  // namespace experimental::execution

namespace exec = experimental::execution;
