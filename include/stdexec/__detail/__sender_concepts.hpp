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
#include "__awaitable.hpp"
#include "__concepts.hpp"
#include "__diagnostics.hpp"
#include "__env.hpp"
#include "__get_completion_signatures.hpp"
#include "__meta.hpp"
#include "__receivers.hpp"
#include "__type_traits.hpp"

#include "__prologue.hpp"

namespace STDEXEC
{
  /////////////////////////////////////////////////////////////////////////////
  // [exec.snd]

  //! @brief Tag type used to opt a class into the @c stdexec::sender concept.
  //!
  //! A user-defined type satisfies @c stdexec::sender by exposing a public
  //! @c sender_concept type alias whose type derives from @c sender_tag:
  //!
  //! @code{.cpp}
  //! struct my_sender {
  //!   using sender_concept = stdexec::sender_tag;
  //!
  //!   // ... usual sender machinery: completion signatures, connect ...
  //! };
  //! @endcode
  //!
  //! @see stdexec::sender
  //! @see stdexec::receiver_tag
  //! @see stdexec::operation_state_tag
  struct sender_tag
  {
    // NOT TO SPEC:
    using sender_concept = sender_tag;
  };

  namespace __detail
  {
    template <class _Sender>
    concept __enable_sender = __std::derived_from<typename _Sender::sender_concept, sender_tag>
                           || __awaitable<_Sender, __detail::__promise<env<>>>;
  }  // namespace __detail

  //! @brief A variable template that opts a class into the @c stdexec::sender
  //!        concept by an alternative path.
  //!
  //! Specialize `enable_sender<MySender>` to @c true to declare that
  //! @c MySender is a sender, *without* having to define a
  //! @c sender_concept type alias on the class itself. This is useful
  //! when the class cannot be modified (e.g. third-party types) or when
  //! the class is a coroutine awaitable type.
  //!
  //! @code{.cpp}
  //! struct legacy_sender { };  // cannot be modified
  //!
  //! template <>
  //! inline constexpr bool stdexec::enable_sender<legacy_sender> = true;
  //! @endcode
  //!
  //! By default, `enable_sender<S>` is @c true when @c S has a
  //! @c sender_concept alias deriving from @c sender_tag, *or* when @c S
  //! is awaitable in stdexec's coroutine promise type.
  template <class _Sender>
  inline constexpr bool enable_sender = __detail::__enable_sender<_Sender>;

  // [exec.snd.concepts]

  //! @brief The fundamental concept of the sender model: a type that
  //!        describes (but does not yet execute) an asynchronous operation.
  //!
  //! A @c sender is the basic unit of composition in stdexec. It is a value
  //! type that *describes* an async computation; the work it describes
  //! does not start until the sender is *connected* to a receiver (via
  //! @c stdexec::connect) and the resulting *operation state* is started
  //! (via @c stdexec::start).
  //!
  //! Concretely, a type @c S satisfies @c sender if:
  //!
  //! 1. @c S has been opted into the concept — either by exposing a
  //!    @c sender_concept type alias derived from @c stdexec::sender_tag,
  //!    or by specializing `stdexec::enable_sender<S>` to @c true, or by
  //!    being an awaitable in stdexec's coroutine promise type.
  //! 2. @c S provides a queryable set of attributes via @c stdexec::get_env.
  //!    Every sender has a (possibly empty) set of attributes.
  //! 3. @c S's decayed type is move-constructible and constructible from
  //!    an @c S (this is what allows senders to be stored and forwarded
  //!    by value).
  //!
  //! Note that @c sender by itself does *not* require the sender's
  //! completion signatures to be computable. That is the additional
  //! constraint of @c sender_in. Generic sender-adaptor code that needs to
  //! know "in which ways can this sender complete?" uses `sender_in<S, Env>`,
  //! not @c sender (alone).
  //!
  //! See [exec.snd.concepts] in the C++26 working draft.
  //!
  //! @see stdexec::sender_in    — sender plus a specific environment, with computable signatures
  //! @see stdexec::sender_to    — sender plus a specific receiver, with compatible signatures
  //! @see stdexec::sender_tag   — the tag type that opts a class into this concept
  //! @see stdexec::enable_sender — alternative opt-in path
  template <class _Sender>
  concept sender = enable_sender<__decay_t<_Sender>>          //
                && __environment_provider<__cref_t<_Sender>>  //
                && __std::move_constructible<__decay_t<_Sender>>
                && __std::constructible_from<__decay_t<_Sender>, _Sender>;

#if STDEXEC_GCC() && STDEXEC_GCC_VERSION < 1300
  template <auto _Completions>
  inline constexpr bool __constant_completion_signatures_v =
    __valid_completion_signatures<std::remove_const_t<decltype(_Completions)>>;
#else
  template <auto _Completions>
  inline constexpr bool __constant_completion_signatures_v =
    __valid_completion_signatures<decltype(_Completions)>;
#endif

  //! @brief A @c sender whose *completion signatures* can be computed in a
  //!        given environment.
  //!
  //! @c sender_in is the form of the sender concept that generic adaptor
  //! code actually uses. Where @c sender just asks \"is this a sender at
  //! all?\", `sender_in<S, Env>` asks \"is @c S a sender whose completion
  //! signatures can be computed when connected to a receiver with
  //! environment @c Env?\" — that information is what every adaptor needs
  //! to type-check itself.
  //!
  //! Concretely, `sender_in<S, Env>` requires:
  //!
  //! 1. @c S satisfies @c sender.
  //! 2. `get_completion_signatures<S, Env>()` is a constant
  //!    expression whose value is a valid
  //!    @c completion_signatures specialization.
  //!
  //! The @c Env parameter is optional (the variadic accepts zero or one
  //! environment). When no environment is supplied, the sender must have
  //! a *non-dependent* set of completion signatures — i.e. its signatures
  //! are the same regardless of the environment. In that case,
  //! @c sender_in requires that `get_completion_signatures<S>()` is a
  //! constant expression whose value is a valid @c completion_signatures
  //! specialization.
  //!
  //! See [exec.snd.concepts] in the C++26 working draft.
  //!
  //! @see stdexec::sender                    — the base concept
  //! @see stdexec::sender_to                 — adds a specific receiver
  //! @see stdexec::get_completion_signatures — the customization point this concept depends on
  template <class _Sender, class... _Env>
  concept sender_in =
    (sizeof...(_Env) <= 1)  //
    && sender<_Sender>      //
    && __constant_completion_signatures_v<STDEXEC::get_completion_signatures<_Sender, _Env...>()>;

  template <class _Receiver, class _Sender>
  concept __receiver_from =
    receiver_of<_Receiver, __completion_signatures_of_t<_Sender, env_of_t<_Receiver>>>;

  /////////////////////////////////////////////////////////////////////////////
  // [exec.snd]
  template <class _Sender, class _Receiver>
  concept __sender_to = receiver<_Receiver>                      //
                     && sender_in<_Sender, env_of_t<_Receiver>>  //
                     && __receiver_from<_Receiver, _Sender>;

  //! @brief A @c sender that can be connected to a specific @c receiver.
  //!
  //! `sender_to<S, R>` is the strongest form of the sender concept: it
  //! requires that @c S is a sender whose completion signatures can be
  //! computed in @c R's environment, that @c R is a receiver that accepts
  //! all of those signatures, *and* that @c connect(S, R) is well-formed.
  //!
  //! This is the constraint a sender consumer or scheduler implementation
  //! uses just before actually calling @c connect — it's the strongest
  //! way to say "yes, this pair is wired up correctly."
  //!
  //! See [exec.snd.concepts] in the C++26 working draft.
  //!
  //! @see stdexec::sender_in     — without the receiver-compatibility check
  //! @see stdexec::receiver_of   — the receiver-side mirror of this concept
  //! @see stdexec::connect       — the operation @c sender_to validates
  template <class _Sender, class _Receiver>
  concept sender_to = __sender_to<_Sender, _Receiver>  //
                   && requires(_Sender &&__sndr, _Receiver &&__rcvr) {
                        connect(static_cast<_Sender &&>(__sndr), static_cast<_Receiver &&>(__rcvr));
                      };

  template <class _Sender>
  concept dependent_sender = sender<_Sender> && __is_dependent_sender<_Sender>;

  template <class _Sender, class... _Env>
  using __single_sender_value_t = __value_types_t<__completion_signatures_of_t<_Sender, _Env...>,
                                                  __qq<__msingle>,
                                                  __qq<__msingle>>;

  template <class _Sender, class... _Env>
  using __single_value_variant_sender_t =
    __value_types_t<__completion_signatures_of_t<_Sender, _Env...>, __qq<__mlist>, __qq<__msingle>>;

  template <class _Tag, class _Sender, class... _Env>
  concept __sends = sender_in<_Sender, _Env...>  //
                 && __count_of<_Tag, _Sender, _Env...>::value != 0;

  template <class _Tag, class _Sender, class... _Env>
  concept __never_sends = sender_in<_Sender, _Env...>  //
                       && __count_of<_Tag, _Sender, _Env...>::value == 0;

  template <class _Tag, class _Sender, class... _Env>
  using __never_sends_t = __mbool<__never_sends<_Tag, _Sender, _Env...>>;

  template <class _Error>
  using __is_eptr = __mbool<__decays_to<_Error, std::exception_ptr>>;

  template <class _Sender, class... _Env>
  concept __has_eptr_completion = sender_in<_Sender, _Env...>  //
                               && __error_types_t<__completion_signatures_of_t<_Sender, _Env...>,
                                                  __q1<__is_eptr>,
                                                  __qq<__mor_t>>::value;

  template <class _Sender, class... _Env>
  concept __single_value_sender = sender_in<_Sender, _Env...>  //
                               && requires { typename __single_sender_value_t<_Sender, _Env...>; };

  template <class _Sender, class... _Env>
  concept __single_value_variant_sender =
    sender_in<_Sender, _Env...>  //
    && requires { typename __single_value_variant_sender_t<_Sender, _Env...>; };

  namespace __detail
  {
    template <class _SenderName, class _Sender, class... _Env>
    constexpr auto __diagnose_sender_concept_failure() noexcept
    {
      if constexpr (!enable_sender<__decay_t<_Sender>>)
      {
        static_assert(enable_sender<_Sender>, STDEXEC_ERROR_ENABLE_SENDER_IS_FALSE);
      }
      else if constexpr (!__std::move_constructible<__decay_t<_Sender>>)
      {
        static_assert(__std::move_constructible<__decay_t<_Sender>>,
                      "The sender type is not move-constructible.");
      }
      else if constexpr (!__decay_copyable<_Sender>)
      {
        static_assert(__decay_copyable<_Sender>,
                      "The sender cannot be decay-copied. Did you forget a std::move?");
      }
      else
      {
        using _Completions = __completion_signatures_of_t<_Sender, _Env...>;
        if constexpr (__same_as<_Completions, __unrecognized_sender_error_t<_Sender, _Env...>>)
        {
          static_assert(__mnever<_Completions>, STDEXEC_ERROR_CANNOT_COMPUTE_COMPLETION_SIGNATURES);
        }
        else if constexpr (__merror<_Completions>)
        {
          static_assert(!__merror<_Completions>,
                        STDEXEC_ERROR_GET_COMPLETION_SIGNATURES_RETURNED_AN_ERROR);
        }
        else
        {
          static_assert(__valid_completion_signatures<_Completions>,
                        STDEXEC_ERROR_GET_COMPLETION_SIGNATURES_HAS_INVALID_RETURN_TYPE);
        }
#if STDEXEC_MSVC() || STDEXEC_NVHPC()
        // MSVC and NVHPC need more encouragement to print the type of the
        // error.
        _Completions __what = 0;
#endif
      }
    }
  }  // namespace __detail

  // Used to report a meaningful error message when the sender_in<Sndr, Env>
  // concept check fails.
  template <class _Sender, class... _Env>
  constexpr auto __diagnose_sender_concept_failure() noexcept
  {
    return __detail::__diagnose_sender_concept_failure<_WITH_PRETTY_SENDER_<_Sender>,
                                                       _Sender,
                                                       _Env...>();
  }
}  // namespace STDEXEC

#include "__epilogue.hpp"
