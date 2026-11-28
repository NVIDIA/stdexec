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

#include "__config.hpp"

#if STDEXEC_USE_MODULES() && !defined(STDEXEC_IN_MODULE_PURVIEW)

import stdexec;

#else

#  include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#  include "__completion_signatures_of.hpp"
#  include "__concepts.hpp"
#  include "__diagnostics.hpp"
#  include "__receivers.hpp"
#  include "__sender_adaptor_closure.hpp"
#  include "__sender_facade.hpp"
#  include "__senders.hpp"

#  if !STDEXEC_USE_MODULES()
#    include <optional>
#  endif

#  include "__prologue.hpp"

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  //////////////////////////////////////////////////////////////////////////////
  // [exec.stopped.opt]
  namespace __sao
  {
    struct _SENDER_MUST_HAVE_EXACTLY_ONE_VALUE_COMPLETION_WITH_AT_LEAST_ONE_ARGUMENT_;

    // The type of the child's value: the single value it sends, or a tuple of its values
    template <class _Completions>
    using __value_of_t = __value_types_t<_Completions, __as_single_value, __qq<__msingle>>;

    template <class _Receiver, class _Value>
    struct __state
    {
      using __value_t = _Value;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Receiver __rcvr_;
    };

    // stopped_as_optional(sndr): the child's value becomes an engaged optional, and its
    // stopped completion an empty one. Errors are forwarded.
    struct __algo : __facade::__defaults
    {
      template <class _Sender, class _Receiver>
      using __state_t =
        __state<_Receiver,
                __value_of_t<
                  __facade::__completions_of_t<__algo, __facade::__child<0>, _Sender, _Receiver>>>;

      // The child must complete with a value in exactly one way.
      template <class _Sender, class _Receiver>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __get_state(_Sender &&, _Receiver __rcvr) noexcept
        -> __state_t<_Sender, _Receiver>
      {
        return __state_t<_Sender, _Receiver>{static_cast<_Receiver &&>(__rcvr)};
      }

      template <class _Sender>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __get_state(_Sender &&, __ignore) noexcept
      {
        return __mexception<
          _WHAT_(_SENDER_MUST_HAVE_EXACTLY_ONE_VALUE_COMPLETION_WITH_AT_LEAST_ONE_ARGUMENT_),
          _WHERE_(_IN_ALGORITHM_, stopped_as_optional_t),
          _WITH_SENDER_<__facade::__nth_child_of_t<_Sender, 0>>>();
      }

      template <class _State, class _Tag, class... _Args>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __complete(__facade::__child<0>,
                                       _State &__state,
                                       _Tag    __tag,
                                       _Args &&...__args) noexcept
      {
        using __optional_t = std::optional<typename _State::__value_t>;
        if constexpr (__same_as<_Tag, set_value_t>)
        {
          // (Only decay-copying the values can throw. The constructor of optional that
          // constructs its value in place need not say that it is noexcept, and libc++'s
          // does not.)
          return __facade::__set_value_from<stopped_as_optional_t>(
            __state.__rcvr_,
            [&]() noexcept(__nothrow_decay_copyable<_Args...>)
            { return __optional_t(std::in_place, static_cast<_Args &&>(__args)...); });
        }
        else if constexpr (__same_as<_Tag, set_stopped_t>)
        {
          return __facade::__send(__state.__rcvr_, set_value_t(), __optional_t(std::nullopt));
        }
        else
        {
          return __facade::__send(__state.__rcvr_, __tag, static_cast<_Args &&>(__args)...);
        }
      }
    };
  }  // namespace __sao

  //! @brief A pipeable sender adaptor that converts a predecessor's stopped
  //!        completion into a value-channel @c std::nullopt, wrapping the
  //!        value-completion datum in a @c std::optional.
  //!
  //! @c stopped_as_optional is the value-channel mirror of
  //! @ref stopped_as_error_t. Where @c stopped_as_error turns cancellation
  //! into an error, @c stopped_as_optional turns cancellation into a
  //! "no value" signal *on the value channel*. The resulting sender
  //! value-completes with a @c std::optional<T>: engaged if the
  //! predecessor produced a value, disengaged if the predecessor was
  //! stopped.
  //!
  //! Since it takes only one argument, @c stopped_as_optional is a sender
  //! adaptor closure object ([exec.adapt.obj]/4), so it can be piped
  //! directly:
  //!
  //! @code{.cpp}
  //! auto s1 = stdexec::stopped_as_optional(sndr);
  //! auto s2 = sndr | stdexec::stopped_as_optional;
  //! auto s3 = sndr | stdexec::stopped_as_optional(); // also accepted
  //! @endcode
  //!
  //! Use this when downstream code prefers branching on a @c std::optional
  //! (a familiar idiom) over branching on an empty
  //! `std::optional<std::tuple<...>>` from @c sync_wait or
  //! handling the @c set_stopped channel via an adaptor.
  //!
  //! **Single value-completion requirement.**
  //!
  //! @c stopped_as_optional requires the predecessor to have exactly one
  //! value-completion signature with exactly one argument. (How would we
  //! wrap multiple values in a *single* @c std::optional?) If the
  //! predecessor has multiple value completions, or zero/multiple value
  //! arguments, the program is ill-formed with a focused diagnostic
  //! ("the sender must have exactly one value completion with one
  //! argument").
  //!
  //! **Completion signatures.**
  //!
  //! Given a predecessor sender @c sndr with completion signatures
  //!
  //! @code{.cpp}
  //! set_value_t(T)        // exactly one value-completion with one argument
  //! set_error_t(Es)...    // zero or more
  //! set_stopped_t()       // consumed
  //! @endcode
  //!
  //! the sender produced by `stopped_as_optional(sndr)` has
  //! completion signatures
  //!
  //! @code{.cpp}
  //! set_value_t(std::optional<std::decay_t<T>>)
  //! set_error_t(Es)...                  // forwarded unchanged
  //! set_error_t(std::exception_ptr)     // added if wrapping may throw
  //!                                     // (no set_stopped_t in the output)
  //! @endcode
  //!
  //! The original @c set_stopped_t completion is consumed: the resulting
  //! sender will never deliver @c set_stopped.
  //!
  //! **Exception behavior.**
  //!
  //! If constructing the @c std::optional from the predecessor's value
  //! throws (e.g., the value type's copy constructor throws), the
  //! exception is delivered through
  //! @c set_error_t(std::exception_ptr).
  //!
  //! **Example.**
  //!
  //! @code{.cpp}
  //! using namespace stdexec;
  //!
  //! auto sndr = just(42) | stopped_as_optional;
  //! auto [opt] = sync_wait(std::move(sndr)).value();
  //! // opt == std::optional<int>{42}
  //!
  //! auto sndr2 = just_stopped() | stopped_as_optional();
  //! // ...but to make stopped_as_optional well-formed here we need to give
  //! // the predecessor a value-shape; in practice you compose it on a
  //! // sender that may either succeed or be stopped:
  //! auto pipeline =
  //!   /* some sender that produces an int or is stopped */
  //!   | stopped_as_optional();
  //! @endcode
  //!
  //! @see stdexec::stopped_as_error — convert stopped into an error
  //! @see stdexec::upon_stopped     — handle stopped synchronously
  //! @see stdexec::let_stopped      — handle stopped with a sender-returning
  //!                                  callback
  //! @see stdexec::sync_wait        — also uses an outer @c std::optional to
  //!                                  signal stop
  struct stopped_as_optional_t : sender_adaptor_closure<stopped_as_optional_t>
  {
    //! @brief Construct a sender that wraps @c __sndr's value completion in
    //!        a @c std::optional and reroutes @c set_stopped to a disengaged
    //!        optional on the value channel.
    //!
    //! @tparam _Sender A type satisfying @c stdexec::sender whose
    //!                 completion signatures include exactly one
    //!                 @c set_value_t(T) signature.
    //!
    //! @param __sndr   The predecessor sender. Forwarded into the result.
    template <sender _Sender>
    constexpr auto operator()(_Sender &&__sndr) const -> auto
    {
      return STDEXEC::__check_well_formed(
        __facade::__sender{stopped_as_optional_t(), __(), static_cast<_Sender &&>(__sndr)});
    }

    //! @brief Returns @c *this, so that the older spelling
    //!        `sndr | stopped_as_optional()` keeps working.
    //!
    //! @returns A copy of this sender adaptor closure object.
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto operator()() const noexcept -> stopped_as_optional_t
    {
      return *this;
    }
  };

  //! @brief The customization point object for the @c stopped_as_optional
  //! sender adaptor.
  //!
  //! @c stopped_as_optional is an instance of @c stopped_as_optional_t.
  //! See @ref stopped_as_optional_t for the full description and a usage
  //! example.
  //!
  //! @hideinitializer
  inline constexpr stopped_as_optional_t stopped_as_optional{};

  template <>
  struct __facade::__algorithm<stopped_as_optional_t> : __sao::__algo
  {};
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
