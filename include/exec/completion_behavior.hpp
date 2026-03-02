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
   private:
    using __cb_t = STDEXEC::__completion_behavior;

   public:
    using unknown_t             = __cb_t::__unknown_t;
    using asynchronous_t        = __cb_t::__asynchronous_t;
    using asynchronous_affine_t = __cb_t::__asynchronous_affine_t;
    using inline_completion_t   = __cb_t::__inline_completion_t;
    using common_t              = __cb_t::__common_t;

    static constexpr auto const &unknown             = __cb_t::__unknown;
    static constexpr auto const &asynchronous        = __cb_t::__asynchronous;
    static constexpr auto const &asynchronous_affine = __cb_t::__asynchronous_affine;
    static constexpr auto const &inline_completion   = __cb_t::__inline_completion;
    static constexpr auto const &common              = __cb_t::__common;

    template <class _CB>
    static constexpr bool is_affine(_CB cb) noexcept
    {
      return __cb_t::__is_affine(cb);
    }

    template <class _CB>
    static constexpr bool is_always_asynchronous(_CB cb) noexcept
    {
      return __cb_t::__is_always_asynchronous(cb);
    }

    template <class _CB>
    static constexpr bool may_be_asynchronous(_CB cb) noexcept
    {
      return __cb_t::__may_be_asynchronous(cb);
    }
  };

  template <STDEXEC::__completion_tag _Tag>
  using get_completion_behavior_t = STDEXEC::__get_completion_behavior_t<_Tag>;

  template <class _Tag, class _Sndr, class... _Env>
  STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
  constexpr auto get_completion_behavior() noexcept
  {
    return STDEXEC::__completion_behavior_of_v<_Tag, STDEXEC::env_of_t<_Sndr>, _Env...>();
  }
}  // namespace experimental::execution

namespace exec = experimental::execution;
