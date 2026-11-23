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
#  include "../stop_token.hpp"
#  include "__completion_behavior.hpp"
#  include "__completion_signatures.hpp"
#  include "__concepts.hpp"
#  include "__diagnostics.hpp"
#  include "__env.hpp"
#  include "__meta.hpp"
#  include "__optional.hpp"
#  include "__queries.hpp"
#  include "__receivers.hpp"
#  include "__schedulers.hpp"
#  include "__sender_facade.hpp"
#  include "__submit.hpp"  // IWYU pragma: keep

#  if !STDEXEC_USE_MODULES()
#    include <exception>
#    include <type_traits>
#  endif

#  include "__prologue.hpp"

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  namespace __read_env
  {
    struct _THE_CURRENT_EXECUTION_ENVIRONMENT_DOESNT_HAVE_A_VALUE_FOR_THE_GIVEN_QUERY_;

    // The operation state, when the query returns a value: the result is stored here, and
    // the value sent is an rvalue reference to it. (A CUDA stream algorithm can pass the
    // value to a kernel by reference. A temporary on the host's stack would be gone by
    // then, and the device may not be able to read it at all.)
    template <class _Receiver, class _Query, class _Result>
    struct __state : __facade::__basic_state<_Receiver, _Query>
    {
      using __facade::__basic_state<_Receiver, _Query>::__basic_state;

      __optional<_Result> __result_{__nullopt};
    };

    template <class _Receiver, class _Query, class _Env = env_of_t<_Receiver>>
    struct __state_of
    {
      using __t = __facade::__basic_state<_Receiver, _Query>;
    };

    template <class _Receiver, class _Query, class _Env>
      requires __callable<_Query, _Env> && (!std::is_reference_v<__call_result_t<_Query, _Env>>)
    struct __state_of<_Receiver, _Query, _Env>
    {
      using __t = __state<_Receiver, _Query, __call_result_t<_Query, _Env>>;
    };

    // read_env(query) completes inline with the result of the query on the receiver's
    // environment. Its completions depend on that environment.
    struct __algo : __facade::__defaults
    {
      template <class _Sender>
      static constexpr bool __depends_on_env = true;

      template <class _Sender, class _Receiver>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __get_state(_Sender&& __sndr, _Receiver __rcvr) noexcept
        -> __state_of<_Receiver, __decay_t<__facade::__data_of_t<_Sender>>>::__t
      {
        using __state_t = __state_of<_Receiver, __decay_t<__facade::__data_of_t<_Sender>>>::__t;
        return __state_t{static_cast<_Receiver&&>(__rcvr),
                         __facade::__get_data(static_cast<_Sender&&>(__sndr))};
      }

      template <class _State, class _Children>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __start(_State& __state, _Children&) noexcept
      {
        using __query_t = __decay_t<decltype(__state.__data_)>;
        using __env_t   = decltype(STDEXEC::get_env(__state.__rcvr_));
        if constexpr (!__callable<__query_t, __env_t>)
        {
          return __facade::__type_error(
            __mexception<
              _THE_CURRENT_EXECUTION_ENVIRONMENT_DOESNT_HAVE_A_VALUE_FOR_THE_GIVEN_QUERY_,
              _WHERE_(_IN_ALGORITHM_, __read_env_t),
              _WITH_QUERY_(__query_t),
              _WITH_ENVIRONMENT_(__env_t)>());
        }
        else if constexpr (requires { __state.__result_; })
        {
          using __rcvr_t   = decltype(__state.__rcvr_);
          using __result_t = __call_result_t<__query_t, __env_t>;
          STDEXEC::__set_value_from(
            static_cast<__rcvr_t&&>(__state.__rcvr_),
            [&]() noexcept(__nothrow_callable<__query_t, __env_t>) -> __result_t&&
            {
              return static_cast<__result_t&&>(
                __state.__result_.__emplace_from(__query_t(), STDEXEC::get_env(__state.__rcvr_)));
            });
          return __facade::__set_value_from_token<__query_t, __env_t>();
        }
        else
        {
          return __facade::__set_value_from<__read_env_t>(__state.__rcvr_,
                                                          __query_t(),
                                                          STDEXEC::get_env(__state.__rcvr_));
        }
      }
    };
  }  // namespace __read_env

  struct __read_env_t
  {
    template <class _Query>
    constexpr auto operator()(_Query) const noexcept
    {
      return __facade::__sender{__read_env_t(), _Query()};
    }
  };

  //! @brief A sender factory that produces a sender whose value completion is
  //!        the result of querying the receiver's environment.
  //!
  //! @c read_env reaches *into* the receiver to read a value associated with
  //! a *query CPO* — things like the receiver's stop token, its associated
  //! allocator, or its preferred scheduler. The resulting sender, when
  //! connected and started, evaluates @c q(get_env(rcvr)) and delivers the
  //! result via @c set_value to the connected receiver.
  //!
  //! It is the primitive used by the standard environment-query helpers
  //! such as @c get_stop_token(), @c get_allocator(), @c get_scheduler(),
  //! and @c get_delegation_scheduler() — each of those is simply
  //! @c read_env applied to the corresponding query CPO.
  //!
  //! The call form takes a *query CPO* (not a value):
  //!
  //! @code{.cpp}
  //! auto sndr = stdexec::read_env(stdexec::get_stop_token);
  //! @endcode
  //!
  //! See [exec.read.env] in the C++26 working draft for the normative
  //! specification.
  //!
  //! **Completion signatures.**
  //!
  //! Given `read_env(q)` and an environment type @c Env (taken from
  //! the connected receiver), the resulting sender has completion signatures:
  //!
  //! @code{.cpp}
  //! set_value_t(decltype(q(declval<Env>())))    // always present
  //! set_error_t(std::exception_ptr)             // present if q(env) may throw
  //! @endcode
  //!
  //! The query result type is taken from the *actual* environment at
  //! connect time, so the same @c read_env sender may have different
  //! concrete completion signatures depending on which receiver it is
  //! connected to.
  //!
  //! If the environment does not provide a value for @c q (i.e.
  //! `q(env)` is ill-formed or returns @c void), the program is
  //! ill-formed at the point where the sender is connected, with a
  //! diagnostic that names the offending query.
  //!
  //! **Exception behavior.**
  //!
  //! If invoking @c q on the receiver's environment throws, the exception
  //! is delivered through @c set_error_t(std::exception_ptr). If @c q is
  //! @c noexcept (typical for query CPOs), no @c std::exception_ptr error
  //! completion is added.
  //!
  //! **Cancellation.**
  //!
  //! @c read_env does not consult the receiver's stop token; it completes
  //! synchronously in its @c start.
  //!
  //! **Example.**
  //!
  //! @code{.cpp}
  //! #include <stdexec/execution.hpp>
  //! using namespace stdexec;
  //!
  //! // Lift the current stop token into the pipeline so a downstream
  //! // algorithm can inspect it:
  //! auto sndr =
  //!   read_env(get_stop_token)
  //!   | then([](auto tok) {
  //!       return tok.stop_requested();
  //!     });
  //! @endcode
  //!
  //! @see stdexec::just           — synchronously complete with literal values
  //! @see stdexec::get_stop_token — equivalent to `read_env(get_stop_token)`
  //! @see stdexec::get_scheduler  — equivalent to `read_env(get_scheduler)`
  //!
  //! @hideinitializer
  inline constexpr __read_env_t read_env{};

  template <>
  struct __facade::__algorithm<__read_env_t> : __read_env::__algo
  {};

  STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
  constexpr auto get_scheduler_t::operator()() const noexcept
  {
    return read_env(get_scheduler);
  }

  STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
  constexpr auto get_start_scheduler_t::operator()() const noexcept
  {
    return read_env(get_start_scheduler);
  }

  STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
  constexpr auto get_delegation_scheduler_t::operator()() const noexcept
  {
    return read_env(get_delegation_scheduler);
  }
}  // namespace STDEXEC

STDEXEC_P2300_NAMESPACE_BEGIN()
  STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
  constexpr auto get_allocator_t::operator()() const noexcept
  {
    return STDEXEC::read_env(get_allocator);
  }

  STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
  constexpr auto get_stop_token_t::operator()() const noexcept
  {
    return STDEXEC::read_env(get_stop_token);
  }
STDEXEC_P2300_NAMESPACE_END()

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
