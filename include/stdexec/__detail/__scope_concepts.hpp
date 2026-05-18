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

#include "__prologue.hpp"

namespace STDEXEC
{
  /////////////////////////////////////////////////////////////////////////////
  // [exec.scope.concepts]

  //! @brief A movable handle representing successful (or failed)
  //!        registration of an operation with an async scope.
  //!
  //! Implementations of @c stdexec::scope_token return a
  //! @c scope_association from <tt>try_associate()</tt>. It is a small,
  //! movable value that:
  //!
  //! - Contextually converts to @c bool to indicate whether the
  //!   association succeeded (the scope is still open and the operation
  //!   was registered) or failed (the scope is shutting down and the
  //!   operation must not start).
  //! - Holds onto whatever state the scope needs to track the operation
  //!   so that the operation can be deregistered at completion.
  //! - Can produce a fresh, equivalent association via
  //!   <tt>try_associate()</tt> — used to model "re-associate with the
  //!   same scope".
  //!
  //! Concretely, @c scope_association requires the type to be movable,
  //! nothrow move-constructible and assignable, default-initializable,
  //! and to support both the @c bool conversion and the
  //! @c try_associate() member.
  //!
  //! User code typically does not interact with @c scope_association
  //! directly — it is the *return type* of @c scope_token::try_associate
  //! and is used internally by @c stdexec::spawn and
  //! @c stdexec::spawn_future.
  //!
  //! See [exec.scope.concepts] in the C++26 working draft.
  //!
  //! @see stdexec::scope_token
  //! @see stdexec::spawn
  //! @see stdexec::spawn_future
  template <class _Assoc>
  concept scope_association = __std::movable<_Assoc> && __nothrow_move_constructible<_Assoc>
                           && __nothrow_move_assignable<_Assoc>
                           && __std::default_initializable<_Assoc>
                           && requires(_Assoc const __assoc) {
                                { static_cast<bool>(__assoc) } noexcept;
                                { __assoc.try_associate() } -> __std::same_as<_Assoc>;
                              };

  namespace __scope_concepts
  {
    struct __test_sender
    {
      using sender_concept = STDEXEC::sender_tag;

      using completion_signatures =
        STDEXEC::completion_signatures<STDEXEC::set_value_t(int),
                                       STDEXEC::set_error_t(std::exception_ptr),
                                       STDEXEC::set_stopped_t()>;

      struct __op
      {
        using operation_state_concept = STDEXEC::operation_state_tag;

        __op()       = default;
        __op(__op&&) = delete;

        void start() & noexcept {}
      };

      template <class _Receiver>
      __op connect(_Receiver)
      {
        return {};
      }
    };
  }  // namespace __scope_concepts

  //! @brief A copyable handle to an *async scope* — the owner of lifetime
  //!        for spawned operations.
  //!
  //! An async scope is a logical container for fire-and-forget operations
  //! launched via @c stdexec::spawn or @c stdexec::spawn_future. It tracks
  //! every operation associated with it so that, at shutdown, it can
  //! block until every spawned operation has completed (typically via a
  //! @c .join() member that returns a sender).
  //!
  //! A @c scope_token is a small, copyable handle to such a scope. User
  //! code typically obtains a token from an @c exec::async_scope via
  //! <tt>scope.get_token()</tt> and passes it into @c spawn or
  //! @c spawn_future.
  //!
  //! Concretely, a type @c T satisfies @c scope_token if it is copyable
  //! and provides two members:
  //!
  //! 1. <tt>token.try_associate()</tt> — attempts to register a new
  //!    operation with the scope, returning a @c scope_association whose
  //!    boolean conversion indicates success. Fails (returns "false")
  //!    when the scope has already begun shutting down.
  //! 2. <tt>token.wrap(sndr)</tt> — wraps a sender so that, when started
  //!    via @c spawn, its lifetime is tied to the scope.
  //!
  //! See [exec.scope.concepts] in the C++26 working draft.
  //!
  //! @see stdexec::scope_association  — the return type of @c try_associate
  //! @see stdexec::spawn              — fire-and-forget into a scope
  //! @see stdexec::spawn_future       — spawn into a scope and observe via a sender
  template <class _Token>
  concept scope_token = __std::copyable<_Token> && requires(_Token const __token) {
    { __token.try_associate() } -> scope_association;
    { __token.wrap(__declval<__scope_concepts::__test_sender>()) } -> sender_in<STDEXEC::env<>>;
  };
}  // namespace STDEXEC

#include "__epilogue.hpp"
