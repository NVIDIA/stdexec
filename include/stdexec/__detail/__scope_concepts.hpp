/*
 * Copyright (c) 2025 Ian Petersen
 * Copyright (c) 2025 NVIDIA Corporation
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

#include "__completion_signatures.hpp"
#include "__concepts.hpp"
#include "__env.hpp"
#include "__operation_states.hpp"
#include "__receivers.hpp"
#include "__senders_core.hpp"

#include <exception>

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [exec.scope.concepts]
  template <class _Assoc>
  concept scope_association = movable<_Assoc> && __nothrow_move_constructible<_Assoc>
                           && __nothrow_move_assignable<_Assoc> && default_initializable<_Assoc>
                           && requires(const _Assoc assoc) {
                                { static_cast<bool>(assoc) } noexcept;
                                { assoc.try_associate() } -> same_as<_Assoc>;
                              };

  namespace __scope_concepts {
    struct __test_sender {
      using sender_concept = stdexec::sender_t;

      using completion_signatures = stdexec::completion_signatures<
        stdexec::set_value_t(int),
        stdexec::set_error_t(std::exception_ptr),
        stdexec::set_stopped_t()
      >;

      struct __op {
        using operation_state_concept = stdexec::operation_state_t;

        __op() = default;
        __op(__op&&) = delete;

        void start() & noexcept {
        }
      };

      template <class _Receiver>
      __op connect(_Receiver) {
        return {};
      }
    };
  } // namespace __scope_concepts

  template <class _Token>
  concept scope_token = copyable<_Token> && requires(const _Token token) {
    { token.try_associate() } -> scope_association;
    { token.wrap(__declval<__scope_concepts::__test_sender>()) } -> sender_in<stdexec::env<>>;
  };
} // namespace stdexec
