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

#include "__prologue.hpp"

namespace STDEXEC
{
  //! @brief Tag type used to opt a class into the @c stdexec::operation_state
  //!        concept.
  //!
  //! A user-defined operation-state type satisfies
  //! @c stdexec::operation_state by exposing a public
  //! @c operation_state_concept type alias whose type derives from
  //! @c operation_state_tag:
  //!
  //! @code{.cpp}
  //! struct my_opstate {
  //!   using operation_state_concept = stdexec::operation_state_tag;
  //!
  //!   void start() noexcept { ... }
  //! };
  //! @endcode
  //!
  //! @see stdexec::operation_state
  //! @see stdexec::sender_tag
  //! @see stdexec::receiver_tag
  struct operation_state_tag
  {};

  /////////////////////////////////////////////////////////////////////////////
  // [exec.opstate]
  template <class _Op>
  concept __has_start_member = requires(_Op &__op) { __op.start(); };

  //! @brief Customization point object that begins the execution of an
  //!        operation state.
  //!
  //! @c start is the trigger that turns a connected sender into a running
  //! asynchronous operation. The operation state returned by
  //! @c stdexec::connect does nothing until it is passed to @c start; from
  //! that moment on, it is running and will eventually deliver exactly one
  //! completion signal to the receiver it was connected with.
  //!
  //! See [exec.opstate.start] in the C++26 working draft.
  //!
  //! **Lifetime contract.**
  //!
  //! The operation state passed to @c start must:
  //!
  //! 1. Be an *lvalue* — @c start(op) with @c op an rvalue is ill-formed.
  //!    The reason: the caller is responsible for keeping the operation
  //!    state alive until completion, which only makes sense for objects
  //!    with stable identity.
  //! 2. Remain alive until the receiver has been completed. The operation
  //!    state itself typically holds child operation states (for sender
  //!    adaptors) and references to the receiver's storage — destroying
  //!    it early would invalidate those.
  //!
  //! Once @c start returns, the operation is running but may or may not
  //! have already completed (an inline-completing operation may have
  //! completed synchronously before @c start returned; an async operation
  //! may complete arbitrarily later).
  //!
  //! **Customization.**
  //!
  //! An operation state opts in by exposing a @c noexcept,
  //! @c void-returning `.start()` member:
  //!
  //! @code{.cpp}
  //! struct my_opstate {
  //!   using operation_state_concept = stdexec::operation_state_tag;
  //!
  //!   // Immovable — once constructed, must stay put:
  //!   my_opstate(my_opstate&&) = delete;
  //!
  //!   void start() noexcept {
  //!     // Begin async work; eventually call set_value/set_error/set_stopped
  //!     // on the receiver stored in this op-state.
  //!   }
  //! };
  //! @endcode
  //!
  //! `start()` *must* be @c noexcept — there's nowhere for it to throw to,
  //! since the caller is typically the runtime, not user code that can
  //! handle exceptions. It must also return @c void. The dispatch site
  //! enforces both with static asserts.
  //!
  //! @c tag_invoke-based customization is supported via a deprecated
  //! overload, retained for backwards compatibility.
  //!
  //! @see stdexec::connect           — the CPO that produces operation states
  //! @see stdexec::operation_state   — the concept this CPO drives
  //! @see stdexec::set_value         — one of the completions @c start eventually triggers
  struct start_t
  {
    //! @brief Begin execution of @c __op.
    //!
    //! Dispatches to <tt>__op.start()</tt>. Statically asserts both
    //! @c noexcept and @c void-returning.
    //!
    //! @tparam _Op    A type satisfying @c stdexec::operation_state.
    //! @param  __op   An *lvalue* reference to the operation state.
    //!                Passing an rvalue is ill-formed.
    template <class _Op>
      requires __has_start_member<_Op>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr void operator()(_Op &__op) const noexcept
    {
      static_assert(noexcept(__op.start()), "start() members must be noexcept");
      static_assert(__same_as<decltype(__op.start()), void>, "start() members must return void");
      __op.start();
    }

    template <class _Op>
      requires __has_start_member<_Op> || __tag_invocable<start_t, _Op &>
    [[deprecated("the use of tag_invoke for start is deprecated")]]
    STDEXEC_ATTRIBUTE(always_inline)  //
      constexpr void operator()(_Op &__op) const noexcept
    {
      static_assert(__nothrow_tag_invocable<start_t, _Op &>);
      (void) __tag_invoke(start_t{}, __op);
    }
  };

  //! @brief The customization point object for starting an operation state.
  //!
  //! @c start is an instance of @ref start_t. See @ref start_t for the
  //! full description, the lifetime contract, and customization examples.
  //!
  //! @hideinitializer
  inline constexpr start_t start{};

  /////////////////////////////////////////////////////////////////////////////
  // [exec.opstate]

  //! @brief An in-progress, *immovable*, *startable* representation of an
  //!        asynchronous operation — the result of connecting a sender to
  //!        a receiver.
  //!
  //! An *operation state* is what you get from @c stdexec::connect: it is the
  //! concrete, type-erased-by-construction record of one specific sender
  //! being driven into one specific receiver. Calling @c stdexec::start on
  //! it begins the work; the operation runs until it eventually invokes
  //! one of @c set_value / @c set_error / @c set_stopped on the receiver
  //! it was constructed with.
  //!
  //! Three properties define the concept @c operation_state:
  //!
  //! 1. The type is destructible — operations clean up cleanly when their
  //!    storage goes away (typically after they have completed).
  //! 2. The type is an object type (not a reference, not a function) —
  //!    operation states are *stored*, not passed by handle.
  //! 3. @c stdexec::start(op) is well-formed for any lvalue @c op of the
  //!    type.
  //!
  //! What the concept does *not* require (but the rules of the sender
  //! model do):
  //!
  //! - **Immovability after construction.** Once an operation state has
  //!   been constructed, it must not be moved or copied — child operations
  //!   inside it typically hold pointers into its storage, which would
  //!   dangle on move. The natural way to satisfy this is to delete the
  //!   move and copy constructors, which is what most operation states do.
  //!   The concept itself doesn't check this; the rule is part of the
  //!   sender-model contract.
  //! - **Lifetime guarantee until completion.** Once @c start has been
  //!   called, the operation state must remain alive until the receiver
  //!   has been completed. This is the caller's responsibility (the
  //!   caller of @c start, typically a sender adaptor or a consumer).
  //!
  //! Like receivers, operation states are usually an implementation detail
  //! of sender adaptors and consumers — most user code never names a
  //! specific operation-state type.
  //!
  //! See [exec.opstate] in the C++26 working draft.
  //!
  //! @see stdexec::connect              — the customization point that produces operation states
  //! @see stdexec::start                — the customization point this concept depends on
  //! @see stdexec::operation_state_tag  — the tag type that opts a class into this concept
  template <class _Op>
  concept operation_state = __std::destructible<_Op> && std::is_object_v<_Op>
                         && requires(_Op &__op) { STDEXEC::start(__op); };
}  // namespace STDEXEC

#include "__epilogue.hpp"
