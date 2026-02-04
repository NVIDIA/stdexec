/*
 * Copyright (c) 2021-2024 NVIDIA Corporation
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

#include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "__concepts.hpp"
#include "__tag_invoke.hpp"

#include <type_traits>

namespace STDEXEC {
  // operation state tag type
  struct operation_state_t { };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.op_state]
  namespace __start {
    template <class _Op>
    concept __has_start = requires(_Op &__op) { __op.start(); };

    struct start_t {
      template <class _Op>
        requires __has_start<_Op>
      STDEXEC_ATTRIBUTE(always_inline)
      constexpr void operator()(_Op &__op) const noexcept {
        static_assert(noexcept(__op.start()), "start() members must be noexcept");
        static_assert(__same_as<decltype(__op.start()), void>, "start() members must return void");
        __op.start();
      }

      template <class _Op>
        requires __has_start<_Op> || __tag_invocable<start_t, _Op &>
      [[deprecated("the use of tag_invoke for start is deprecated")]]
      STDEXEC_ATTRIBUTE(always_inline) //
        constexpr void operator()(_Op &__op) const noexcept {
        static_assert(__nothrow_tag_invocable<start_t, _Op &>);
        (void) __tag_invoke(start_t{}, __op);
      }
    };
  } // namespace __start

  using __start::start_t;
  inline constexpr start_t start{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.op_state]
  template <class _Op>
  concept operation_state = __std::destructible<_Op> && std::is_object_v<_Op>
                         && requires(_Op &__op) { STDEXEC::start(__op); };
} // namespace STDEXEC
