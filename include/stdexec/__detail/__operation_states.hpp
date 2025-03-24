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

namespace stdexec {
  // operation state tag type
  struct operation_state_t { };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.op_state]
  namespace __start {
    struct start_t {
      template <__same_as<start_t> _Self, class _OpState>
      STDEXEC_ATTRIBUTE((always_inline)) friend auto tag_invoke(_Self, _OpState& __op) noexcept -> decltype(__op.start()) {
        static_assert(noexcept(__op.start()), "start() members must be noexcept");
        static_assert(__same_as<decltype(__op.start()), void>, "start() members must return void");
        __op.start();
      }

      template <class _Op>
        requires tag_invocable<start_t, _Op&>
      STDEXEC_ATTRIBUTE((always_inline)) void operator()(_Op& __op) const noexcept {
        static_assert(nothrow_tag_invocable<start_t, _Op&>);
        (void) tag_invoke(start_t{}, __op);
      }
    };
  } // namespace __start

  using __start::start_t;
  inline constexpr start_t start{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.op_state]
  template <class _Op>
  concept operation_state =  //
    destructible<_Op> &&     //
    std::is_object_v<_Op> && //
    requires(_Op& __op) {    //
      stdexec::start(__op);
    };
} // namespace stdexec
