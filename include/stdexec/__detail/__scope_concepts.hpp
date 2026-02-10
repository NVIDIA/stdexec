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
#include "__operation_states.hpp"
#include "__receivers.hpp"
#include "__sender_concepts.hpp"

#include <exception>

namespace STDEXEC {
  /////////////////////////////////////////////////////////////////////////////
  // [exec.scope.concepts]
  template <class _Assoc>
  concept scope_association = __std::movable<_Assoc> && __nothrow_move_constructible<_Assoc>
                           && __nothrow_move_assignable<_Assoc>
                           && __std::default_initializable<_Assoc> && requires(const _Assoc __assoc) {
                                { static_cast<bool>(__assoc) } noexcept;
                                { __assoc.try_associate() } -> __std::same_as<_Assoc>;
                              };

  namespace __scope_concepts {
    struct __test_sender {
      using sender_concept = STDEXEC::sender_t;

      using completion_signatures = STDEXEC::completion_signatures<
        STDEXEC::set_value_t(int),
        STDEXEC::set_error_t(std::exception_ptr),
        STDEXEC::set_stopped_t()
      >;

      struct __op {
        using operation_state_concept = STDEXEC::operation_state_t;

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
  concept scope_token = __std::copyable<_Token> && requires(const _Token __token) {
    { __token.try_associate() } -> scope_association;
    { __token.wrap(__declval<__scope_concepts::__test_sender>()) } -> sender_in<STDEXEC::env<>>;
  };
} // namespace STDEXEC
