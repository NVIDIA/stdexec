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
#  include "__basic_sender.hpp"
#  include "__completion_signatures.hpp"
#  include "__completion_signatures_of.hpp"
#  include "__concepts.hpp"
#  include "__diagnostics.hpp"
#  include "__receivers.hpp"
#  include "__remap_attrs.hpp"
#  include "__sender_adaptor_closure.hpp"
#  include "__senders.hpp"
#  include "__transform_completion_signatures.hpp"

#  if !STDEXEC_USE_MODULES()
#    include <exception>
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

    template <class _Receiver, class _Value>
    struct __state
    {
      using __receiver_t = _Receiver;
      using __value_t    = _Value;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Receiver __rcvr_;
    };

    // A stopped completion of the child becomes a value completion (an empty
    // optional). Value completions stay value completions, but they can turn
    // into an error completion if decay-copying the value into the optional
    // throws.
    struct __channel_map
    {
      template <class _Tag, class _From, class _Child, class... _Env>
      [[nodiscard]]
      static consteval auto __feeds() noexcept -> bool
      {
        if constexpr (__same_as<_From, set_stopped_t>)
        {
          return __same_as<_Tag, set_value_t>;
        }
        else if constexpr (__same_as<_From, set_value_t>)
        {
          if constexpr (__same_as<_Tag, set_error_t>)
          {
            return !STDEXEC::__all_completions_satisfy<set_value_t,
                                                       _Child,
                                                       __q<__nothrow_decay_copyable_t>,
                                                       _Env...>();
          }
          else
          {
            return __same_as<_Tag, set_value_t>;
          }
        }
        else
        {
          return __same_as<_Tag, _From>;
        }
      }
    };

    struct __stopped_as_optional_impl : __sexpr_defaults
    {
      static constexpr auto __get_attrs =
        []<class _Child>(__ignore, __ignore, _Child const & __child) noexcept
      {
        return __remap_attrs<__channel_map, _Child>{__child};
      };

      static consteval auto __get_value_transform_fn() noexcept
      {
        return []<class... _Ts>()
        {
          using __value_t = __mcall<__as_single_value, _Ts...>;
          return STDEXEC::__concat_completion_signatures(
            completion_signatures<set_value_t(std::optional<__value_t>)>(),
            __eptr_completion_unless_t<__nothrow_decay_copyable_t<_Ts...>>());
        };
      }

      template <class _Self, class... _Env>
      static consteval auto __get_completion_signatures()
      {
        static_assert(__sender_for<_Self, stopped_as_optional_t>);
        using __cv_sndr_t  = __child_of<_Self>;
        auto __completions = STDEXEC::get_completion_signatures<__cv_sndr_t, _Env...>();
        STDEXEC_IF_OK(__completions)
        {
          if constexpr (!__single_value_sender<__cv_sndr_t, _Env...>)
          {
            return STDEXEC::__throw_compile_time_error<
              _WHAT_(_SENDER_MUST_HAVE_EXACTLY_ONE_VALUE_COMPLETION_WITH_AT_LEAST_ONE_ARGUMENT_),
              _WHERE_(_IN_ALGORITHM_, stopped_as_optional_t),
              _WITH_SENDER_<__cv_sndr_t>>();
          }
          else
          {
            return STDEXEC::__transform_completion_signatures(__completions,
                                                              __get_value_transform_fn(),
                                                              {},
                                                              __ignore_completion());
          }
        }
      }

      static constexpr auto __get_state =
        []<class _Self, class _Receiver>(_Self&&, _Receiver&& __rcvr) noexcept
        -> __state<_Receiver, __single_sender_value_t<__child_of<_Self>, env_of_t<_Receiver>>>
      {
        static_assert(__sender_for<_Self, stopped_as_optional_t>);
        return {static_cast<_Receiver&&>(__rcvr)};
      };

      static constexpr auto __complete =
        []<class _State, class _Tag, class... _Args>(__ignore,
                                                     _State& __state,
                                                     _Tag,
                                                     _Args&&... __args) noexcept -> void
      {
        using __value_t = _State::__value_t;
        if constexpr (__same_as<_Tag, set_value_t>)
        {
          STDEXEC_TRY
          {
            static_assert(__std::constructible_from<__value_t, _Args...>);
            STDEXEC::set_value(static_cast<_State&&>(__state).__rcvr_,
                               std::optional<__value_t>{std::in_place,
                                                        static_cast<_Args&&>(__args)...});
          }
          STDEXEC_CATCH_ALL
          {
            if constexpr (!__nothrow_decay_copyable<_Args...>)
            {
              STDEXEC::set_error(static_cast<_State&&>(__state).__rcvr_, std::current_exception());
            }
          }
        }
        else if constexpr (__same_as<_Tag, set_error_t>)
        {
          STDEXEC::set_error(static_cast<_State&&>(__state).__rcvr_,
                             static_cast<_Args&&>(__args)...);
        }
        else
        {
          STDEXEC::set_value(static_cast<_State&&>(__state).__rcvr_,
                             std::optional<__value_t>{std::nullopt});
        }
      };
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
    constexpr auto operator()(_Sender&& __sndr) const -> __well_formed_sender auto
    {
      return __sexpr{stopped_as_optional_t(), __(), static_cast<_Sender&&>(__sndr)};
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
  struct __sexpr_impl<stopped_as_optional_t> : __sao::__stopped_as_optional_impl
  {};
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
