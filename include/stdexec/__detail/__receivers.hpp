/*
 * Copyright (c) 2022-2024 NVIDIA Corporation
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

#include "__concepts.hpp"
#include "__diagnostics.hpp"
#include "__env.hpp"
#include "__tag_invoke.hpp"

#include "../functional.hpp"

#include <exception>

#include "__prologue.hpp"

namespace STDEXEC
{
  namespace __detail
  {
    template <__disposition _Disposition>
    struct __completion_tag
    {
      static constexpr STDEXEC::__disposition __disposition = _Disposition;

      template <STDEXEC::__disposition _OtherDisposition>
      constexpr bool operator==(__completion_tag<_OtherDisposition>) const noexcept
      {
        return _Disposition == _OtherDisposition;
      }
    };
  }  // namespace __detail

  /////////////////////////////////////////////////////////////////////////////
  // [exec.recv]
  template <class _Receiver, class... _As>
  concept __set_value_member = requires(_Receiver &&__rcvr, _As &&...__args) {
    static_cast<_Receiver &&>(__rcvr).set_value(static_cast<_As &&>(__args)...);
  };

  //! @brief Customization point object for the *value* completion signal of
  //!        the sender/receiver protocol.
  //!
  //! `set_value(rcvr, vs...)` is the call an operation state makes on its
  //! receiver to deliver a successful asynchronous result. It is one of
  //! three terminal completion signals (alongside @ref set_error_t and
  //! @ref set_stopped_t) that exactly one of will be called on a receiver
  //! once a connected operation has started.
  //!
  //! User code rarely calls `set_value` directly — it is invoked from
  //! inside operation-state implementations. Sender authors writing new
  //! adaptors do call it, and *receiver* authors provide the matching
  //! member that this CPO dispatches to.
  //!
  //! **Customization.**
  //!
  //! A receiver opts into receiving value completions by defining a
  //! @c noexcept, @c void-returning member:
  //!
  //! @code{.cpp}
  //! struct my_receiver {
  //!   using receiver_concept = stdexec::receiver_tag;
  //!   void set_value(int v) noexcept { ... }   // overload set per value type
  //! };
  //! @endcode
  //!
  //! At the call site, <tt>stdexec::set_value(rcvr, vs...)</tt> dispatches
  //! to <tt>rcvr.set_value(vs...)</tt>, statically asserting both that the
  //! member is @c noexcept and that it returns @c void.
  //!
  //! See [exec.recv] in the C++26 working draft.
  //!
  //! @see stdexec::set_error    — the error-completion CPO
  //! @see stdexec::set_stopped  — the stopped-completion CPO
  //! @see stdexec::receiver     — the receiver concept this CPO drives
  //! @see stdexec::receiver_of  — receiver plus specific completion signatures
  struct set_value_t : __detail::__completion_tag<__disposition::__value>
  {
    template <class _Fn, class... _As>
    using __f = __minvoke<_Fn, _As...>;

    //! @brief Deliver a value completion to @c __rcvr.
    //!
    //! Dispatches to <tt>__rcvr.set_value(__as...)</tt>. The static
    //! asserts inside enforce that the member is @c noexcept and that it
    //! returns @c void — the two non-negotiable properties of every
    //! completion signal.
    //!
    //! @tparam _Receiver A type whose decayed form satisfies
    //!                   @c stdexec::receiver and has a matching
    //!                   `.set_value(_As...)` member.
    //! @tparam _As       The value-datum argument types.
    template <class _Receiver, class... _As>
      requires __set_value_member<_Receiver, _As...>
    STDEXEC_ATTRIBUTE(host, device, always_inline)
    constexpr void operator()(_Receiver &&__rcvr, _As &&...__as) const noexcept
    {
      static_assert(noexcept(
                      static_cast<_Receiver &&>(__rcvr).set_value(static_cast<_As &&>(__as)...)),
                    "set_value member functions must be noexcept");
      static_assert(__same_as<decltype(static_cast<_Receiver &&>(__rcvr).set_value(
                                static_cast<_As &&>(__as)...)),
                              void>,
                    "set_value member functions must return void");
      static_cast<_Receiver &&>(__rcvr).set_value(static_cast<_As &&>(__as)...);
    }

    template <class _Receiver, class... _As>
      requires __set_value_member<_Receiver, _As...>
            || __tag_invocable<set_value_t, _Receiver, _As...>
    [[deprecated("the use of tag_invoke for set_value is deprecated")]]
    STDEXEC_ATTRIBUTE(host, device, always_inline)  //
      constexpr void operator()(_Receiver &&__rcvr, _As &&...__as) const noexcept
    {
      static_assert(__nothrow_tag_invocable<set_value_t, _Receiver, _As...>);
      (void) __tag_invoke(*this, static_cast<_Receiver &&>(__rcvr), static_cast<_As &&>(__as)...);
    }
  };

  template <class _Receiver, class _Error>
  concept __set_error_member = requires(_Receiver &&__rcvr, _Error &&__err) {
    static_cast<_Receiver &&>(__rcvr).set_error(static_cast<_Error &&>(__err));
  };

  //! @brief Customization point object for the *error* completion signal of
  //!        the sender/receiver protocol.
  //!
  //! `set_error(rcvr, e)` is the call an operation state makes on its
  //! receiver to deliver a failure. Unlike a thrown exception, the error
  //! is a *typed datum* — receivers may distinguish, say,
  //! @c std::exception_ptr from @c std::error_code from a domain-specific
  //! error enum by overloading on the argument type.
  //!
  //! **Customization.**
  //!
  //! A receiver opts into receiving error completions of a given type @c E
  //! by defining a @c noexcept, @c void-returning member:
  //!
  //! @code{.cpp}
  //! struct my_receiver {
  //!   using receiver_concept = stdexec::receiver_tag;
  //!   void set_error(std::exception_ptr e) noexcept { ... }
  //!   void set_error(std::error_code e) noexcept { ... }   // multiple OK
  //! };
  //! @endcode
  //!
  //! Like @c set_value, the dispatch site enforces @c noexcept and
  //! @c void return via static asserts.
  //!
  //! **Receivers MUST accept exactly one error completion at runtime.**
  //! That is: at most one of @c set_value, @c set_error, @c set_stopped
  //! is ever called on a given receiver, exactly once.
  //!
  //! See [exec.recv] in the C++26 working draft.
  //!
  //! @see stdexec::set_value
  //! @see stdexec::set_stopped
  //! @see stdexec::receiver
  struct set_error_t : __detail::__completion_tag<__disposition::__error>
  {
    template <class _Fn, class... _Args>
      requires(sizeof...(_Args) == 1)
    using __f = __minvoke<_Fn, _Args...>;

    //! @brief Deliver an error completion to @c __rcvr.
    //!
    //! Dispatches to <tt>__rcvr.set_error(__err)</tt>. Statically asserts
    //! both @c noexcept and @c void-returning.
    //!
    //! @tparam _Receiver A type with a matching `.set_error(_Error)` member.
    //! @tparam _Error    The error datum type.
    template <class _Receiver, class _Error>
      requires __set_error_member<_Receiver, _Error>
    STDEXEC_ATTRIBUTE(host, device, always_inline)
    constexpr void operator()(_Receiver &&__rcvr, _Error &&__err) const noexcept
    {
      static_assert(noexcept(
                      static_cast<_Receiver &&>(__rcvr).set_error(static_cast<_Error &&>(__err))),
                    "set_error member functions must be noexcept");
      static_assert(__same_as<decltype(static_cast<_Receiver &&>(__rcvr).set_error(
                                static_cast<_Error &&>(__err))),
                              void>,
                    "set_error member functions must return void");
      static_cast<_Receiver &&>(__rcvr).set_error(static_cast<_Error &&>(__err));
    }

    template <class _Receiver, class _Error>
      requires __set_error_member<_Receiver, _Error>
            || __tag_invocable<set_error_t, _Receiver, _Error>
    [[deprecated("the use of tag_invoke for set_error is deprecated")]]
    STDEXEC_ATTRIBUTE(host, device, always_inline)  //
      constexpr void operator()(_Receiver &&__rcvr, _Error &&__err) const noexcept
    {
      static_assert(__nothrow_tag_invocable<set_error_t, _Receiver, _Error>);
      (void) __tag_invoke(*this, static_cast<_Receiver &&>(__rcvr), static_cast<_Error &&>(__err));
    }
  };

  template <class _Receiver>
  concept __set_stopped_member = requires(_Receiver &&__rcvr) {
    static_cast<_Receiver &&>(__rcvr).set_stopped();
  };

  //! @brief Customization point object for the *stopped* completion signal
  //!        of the sender/receiver protocol.
  //!
  //! `set_stopped(rcvr)` is the call an operation state makes on its
  //! receiver to report that the operation was cancelled. It carries
  //! *no datum* — the stopped channel is informational only ("we are
  //! ending early because cancellation was requested or because no result
  //! is needed any more").
  //!
  //! Cancellation is *cooperative*: receivers can request stop via the
  //! stop token in their environment (see @c stdexec::get_stop_token);
  //! senders that observe such a request may complete with
  //! @c set_stopped instead of @c set_value or @c set_error.
  //!
  //! **Customization.**
  //!
  //! A receiver opts into receiving stopped completions by defining a
  //! @c noexcept, @c void-returning *nullary* member:
  //!
  //! @code{.cpp}
  //! struct my_receiver {
  //!   using receiver_concept = stdexec::receiver_tag;
  //!   void set_stopped() noexcept { ... }
  //! };
  //! @endcode
  //!
  //! See [exec.recv] in the C++26 working draft.
  //!
  //! @see stdexec::set_value
  //! @see stdexec::set_error
  //! @see stdexec::get_stop_token  — the receiver-environment query for the stop token
  struct set_stopped_t : __detail::__completion_tag<__disposition::__stopped>
  {
    template <class _Fn, class... _Args>
      requires(sizeof...(_Args) == 0)
    using __f = __minvoke<_Fn, _Args...>;

    //! @brief Deliver a stopped completion to @c __rcvr.
    //!
    //! Dispatches to <tt>__rcvr.set_stopped()</tt>. Statically asserts both
    //! @c noexcept and @c void-returning.
    //!
    //! @tparam _Receiver A type with a matching nullary
    //!                   `.set_stopped()` member.
    template <class _Receiver>
      requires __set_stopped_member<_Receiver>
    STDEXEC_ATTRIBUTE(host, device, always_inline)
    constexpr void operator()(_Receiver &&__rcvr) const noexcept
    {
      static_assert(noexcept(static_cast<_Receiver &&>(__rcvr).set_stopped()),
                    "set_stopped member functions must be noexcept");
      static_assert(__same_as<decltype(static_cast<_Receiver &&>(__rcvr).set_stopped()), void>,
                    "set_stopped member functions must return void");
      static_cast<_Receiver &&>(__rcvr).set_stopped();
    }

    template <class _Receiver>
      requires __set_stopped_member<_Receiver> || __tag_invocable<set_stopped_t, _Receiver>
    [[deprecated("the use of tag_invoke for set_stopped is deprecated")]]
    STDEXEC_ATTRIBUTE(host, device, always_inline)  //
      constexpr void operator()(_Receiver &&__rcvr) const noexcept
    {
      static_assert(__nothrow_tag_invocable<set_stopped_t, _Receiver>);
      (void) __tag_invoke(*this, static_cast<_Receiver &&>(__rcvr));
    }
  };

  //! @brief The customization point object for delivering a value completion.
  //!
  //! @c set_value is an instance of @ref set_value_t. See @ref set_value_t
  //! for the full description and customization rules.
  //!
  //! @hideinitializer
  inline constexpr set_value_t set_value{};

  //! @brief The customization point object for delivering an error completion.
  //!
  //! @c set_error is an instance of @ref set_error_t. See @ref set_error_t
  //! for the full description and customization rules.
  //!
  //! @hideinitializer
  inline constexpr set_error_t set_error{};

  //! @brief The customization point object for delivering a stopped completion.
  //!
  //! @c set_stopped is an instance of @ref set_stopped_t. See
  //! @ref set_stopped_t for the full description and customization rules.
  //!
  //! @hideinitializer
  inline constexpr set_stopped_t set_stopped{};

  //! @brief Tag type used to opt a class into the @c stdexec::receiver concept.
  //!
  //! A user-defined type satisfies @c stdexec::receiver by exposing a public
  //! @c receiver_concept type alias whose type derives from @c receiver_tag:
  //!
  //! @code{.cpp}
  //! struct my_receiver {
  //!   using receiver_concept = stdexec::receiver_tag;
  //!
  //!   void set_value(int v) noexcept                { ... }
  //!   void set_error(std::exception_ptr e) noexcept { ... }
  //!   void set_stopped() noexcept                   { ... }
  //! };
  //! @endcode
  //!
  //! @see stdexec::receiver
  //! @see stdexec::sender_tag
  //! @see stdexec::operation_state_tag
  struct receiver_tag
  {
    using receiver_concept = receiver_tag;  // NOT TO SPEC
  };

  namespace __detail
  {
    template <class _Receiver>
    concept __enable_receiver = (STDEXEC_PP_WHEN(
      STDEXEC_EDG(),
      requires {
        typename _Receiver::receiver_concept;
      } &&) __std::derived_from<typename _Receiver::receiver_concept, receiver_tag>);
  }  // namespace __detail

  //! @brief The fundamental concept of the receiver model: a callback-shaped
  //!        object that consumes the result of an asynchronous operation.
  //!
  //! A *receiver* is the destination half of a sender/receiver pair. It is
  //! the object on which a started operation eventually invokes one of
  //! @c set_value, @c set_error, or @c set_stopped to deliver its
  //! completion. Receivers are typically synthesized by sender consumers
  //! and adaptors — most user code never writes a receiver by hand; it
  //! writes senders and composes them.
  //!
  //! Concretely, a type @c R satisfies @c receiver if:
  //!
  //! 1. @c R is opted into the concept via a @c receiver_concept type
  //!    alias deriving from @c stdexec::receiver_tag.
  //! 2. @c R provides an environment via @c stdexec::get_env (so child
  //!    operations can query for the stop token, allocator, scheduler,
  //!    etc.).
  //! 3. @c R's decayed type is @em nothrow move-constructible — receivers
  //!    are moved into operation states by sender adaptors, and that move
  //!    must not throw.
  //! 4. @c R's decayed type is constructible from an @c R.
  //!
  //! Note that this concept alone does *not* require @c R to accept any
  //! particular completion signals — for that, see @c receiver_of, which
  //! takes a @c completion_signatures pack and validates that the receiver
  //! has matching @c set_value / @c set_error / @c set_stopped members.
  //!
  //! See [exec.recv.concepts] in the C++26 working draft.
  //!
  //! @see stdexec::receiver_of   — receiver plus specific completion signatures
  //! @see stdexec::receiver_tag  — the tag type that opts a class into this concept
  //! @see stdexec::set_value
  //! @see stdexec::set_error
  //! @see stdexec::set_stopped
  template <class _Receiver>
  concept receiver = __detail::__enable_receiver<__decay_t<_Receiver>>
                  && __environment_provider<__cref_t<_Receiver>>
                  && __nothrow_move_constructible<__decay_t<_Receiver>>
                  && __std::constructible_from<__decay_t<_Receiver>, _Receiver>;

  struct _THE_RECEIVER_DOES_NOT_ACCEPT_ALL_OF_THE_SENDERS_COMPLETION_SIGNALS_
  {};

  namespace __detail
  {
    template <class _Receiver, class _Tag, class... _Args>
    constexpr auto __try_completion(_Tag (*)(_Args...))
      -> __mexception<_WHAT_(_CONCEPT_CHECK_FAILURE_),
                      _WHY_(_THE_RECEIVER_DOES_NOT_ACCEPT_ALL_OF_THE_SENDERS_COMPLETION_SIGNALS_),
                      _UNHANDLED_COMPLETION_SIGNAL_<_Tag(_Args...)>,
                      _WITH_RECEIVER_(_Receiver)>;

    template <class _Receiver, class _Tag, class... _Args>
      requires __callable<_Tag, _Receiver, _Args...>
    auto __try_completion(_Tag (*)(_Args...)) -> __msuccess;

    template <class _Receiver, class... _Sigs>
    constexpr auto __try_completions(completion_signatures<_Sigs...> *) -> decltype((
      __msuccess(),
      ...,
      __detail::__try_completion<__decay_t<_Receiver>>(static_cast<_Sigs *>(nullptr))));
  }  // namespace __detail

  //! @brief A @c receiver that accepts a specific set of completion
  //!        signatures.
  //!
  //! `receiver_of<R, Sigs>` says: "R is a receiver, and for every
  //! @c set_xxx_t(Args...) signature in @c Sigs (a
  //! @c stdexec::completion_signatures pack), R has a matching member that
  //! is callable with @c Args... ". This is the constraint that ensures a
  //! sender's completion signals can actually be delivered to the
  //! receiver — sender adaptors typically express their compatibility
  //! requirements in terms of @c receiver_of, not bare @c receiver.
  //!
  //! When this concept fails, stdexec produces a focused error message
  //! naming the @em specific completion signal the receiver doesn't
  //! accept (e.g. "the receiver does not accept set_value_t(int)") —
  //! this is the main reason to use @c receiver_of over manually checking
  //! each member's callability.
  //!
  //! See [exec.recv.concepts] in the C++26 working draft.
  //!
  //! @see stdexec::receiver              — without the signature check
  //! @see stdexec::sender_to             — the sender-side mirror of this concept
  //! @see stdexec::completion_signatures — the signature pack this concept consumes
  template <class _Receiver, class _Completions>
  concept receiver_of = receiver<_Receiver> && requires(_Completions *__completions) {
    { __detail::__try_completions<_Receiver>(__completions) } -> __ok;
  };

  /// A utility for calling set_value with the result of a function invocation:
  template <class _Receiver, class _Fun, class... _As>
  STDEXEC_ATTRIBUTE(host, device)
  constexpr void __set_value_from(_Receiver &&__rcvr, _Fun &&__fun, _As &&...__as) noexcept
  {
    STDEXEC_TRY
    {
      if constexpr (__std::same_as<void, __invoke_result_t<_Fun, _As...>>)
      {
        __invoke(static_cast<_Fun &&>(__fun), static_cast<_As &&>(__as)...);
        STDEXEC::set_value(static_cast<_Receiver &&>(__rcvr));
      }
      else
      {
        STDEXEC::set_value(static_cast<_Receiver &&>(__rcvr),
                           __invoke(static_cast<_Fun &&>(__fun), static_cast<_As &&>(__as)...));
      }
    }
    STDEXEC_CATCH_ALL
    {
      if constexpr (!__nothrow_invocable<_Fun, _As...>)
      {
        STDEXEC::set_error(static_cast<_Receiver &&>(__rcvr), std::current_exception());
      }
    }
  }

  template <class _Tag, class _Receiver>
  struct __completion_fn
  {
    _Receiver &__rcvr_;

    template <class... _Args>
    constexpr void operator()(_Args &&...__args) const noexcept
    {
      _Tag()(static_cast<_Receiver &&>(__rcvr_), static_cast<_Args &&>(__args)...);
    }
  };

  template <class _Tag, class _Receiver>
  constexpr auto __mk_completion_fn(_Tag, _Receiver &__rcvr) noexcept
  {
    return __completion_fn<_Tag, _Receiver>{__rcvr};
  }

  // Used to test whether a sender has a nothrow connect to a receiver whose environment
  // is _Env..., or if _Env... is empty (indicating that the sender is non-dependent), to
  // a receiver with an arbitrary environment.
  struct __receiver_archetype_base
  {
    using receiver_concept = receiver_tag;

    template <class... _Args>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr void set_value(_Args &&...) noexcept
    {}

    template <class _Error>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr void set_error(_Error &&) noexcept
    {}

    STDEXEC_ATTRIBUTE(host, device)
    constexpr void set_stopped() noexcept {}
  };

  template <class _Env>
  struct __receiver_archetype : __receiver_archetype_base
  {
    STDEXEC_ATTRIBUTE(nodiscard, noreturn, host, device)
    auto get_env() const noexcept -> _Env
    {
      STDEXEC_ASSERT(false);
      STDEXEC_TERMINATE();
    }
  };
}  // namespace STDEXEC

#include "__epilogue.hpp"
