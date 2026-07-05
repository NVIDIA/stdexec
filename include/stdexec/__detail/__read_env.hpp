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
#include "../stop_token.hpp"
#include "__basic_sender.hpp"
#include "__completion_behavior.hpp"
#include "__completion_signatures.hpp"
#include "__concepts.hpp"
#include "__diagnostics.hpp"
#include "__env.hpp"
#include "__meta.hpp"
#include "__optional.hpp"
#include "__queries.hpp"
#include "__receivers.hpp"
#include "__schedulers.hpp"
#include "__submit.hpp"  // IWYU pragma: keep

#include <exception>

#include "__prologue.hpp"

namespace STDEXEC
{
  namespace __read_env
  {
    struct _THE_CURRENT_EXECUTION_ENVIRONMENT_DOESNT_HAVE_A_VALUE_FOR_THE_GIVEN_QUERY_;

    template <class _Receiver,
              class _Query,
              class _Ty = __call_result_t<_Query, env_of_t<_Receiver>>>
    struct __opstate
    {
      constexpr void start() noexcept
      {
        constexpr bool _Nothrow   = __nothrow_callable<_Query, env_of_t<_Receiver>>;
        auto           __query_fn = [&]() noexcept(_Nothrow) -> _Ty&&
        {
          auto& __result = __result_.__emplace_from(_Query(), STDEXEC::get_env(__rcvr_));
          return static_cast<_Ty&&>(__result);
        };
        STDEXEC::__set_value_from(static_cast<_Receiver&&>(__rcvr_), __query_fn);
      }

      _Receiver       __rcvr_;
      __optional<_Ty> __result_ = __nullopt;
    };

    template <class _Receiver, class _Query, class _Ty>
      requires __same_as<_Ty, _Ty&&>
    struct __opstate<_Receiver, _Query, _Ty>
    {
      constexpr void start() noexcept
      {
        // The query returns a reference type; pass it straight through to the receiver.
        STDEXEC::__set_value_from(static_cast<_Receiver&&>(__rcvr_),
                                  _Query(),
                                  STDEXEC::get_env(__rcvr_));
      }

      _Receiver __rcvr_;
    };

    template <class _Query>
    struct __attrs
    {
      STDEXEC_ATTRIBUTE(nodiscard)
      constexpr auto query(__get_completion_behavior_t<set_value_t>) const noexcept
      {
        return __completion_behavior::__inline_completion;
      }

      template <class _Env>
        requires(!__nothrow_callable<_Query, _Env>)
      STDEXEC_ATTRIBUTE(nodiscard)
      constexpr auto query(__get_completion_behavior_t<set_error_t>, _Env const &) const noexcept
      {
        return __completion_behavior::__inline_completion;
      }
    };

    struct __read_env_impl : __sexpr_defaults
    {
      static constexpr auto __get_attrs = []<class _Query>(__ignore, _Query) noexcept
      {
        return __attrs<_Query>{};
      };

      template <class _Self, class _Env>
      static consteval auto __get_completion_signatures()
      {
        using __query_t = __data_of<_Self>;
        if constexpr (__callable<__query_t, _Env>)
        {
          using __result_t = __call_result_t<__query_t, _Env>;
          if constexpr (__nothrow_callable<__query_t, _Env>)
          {
            return completion_signatures<set_value_t(__result_t)>();
          }
          else
          {
            return completion_signatures<set_value_t(__result_t),
                                         set_error_t(std::exception_ptr)>();
          }
        }
        else
        {
          return STDEXEC::__throw_compile_time_error<
            _THE_CURRENT_EXECUTION_ENVIRONMENT_DOESNT_HAVE_A_VALUE_FOR_THE_GIVEN_QUERY_,
            _WHERE_(_IN_ALGORITHM_, __read_env_t),
            _WITH_QUERY_(__query_t),
            _WITH_ENVIRONMENT_(_Env)>();
        }
      };

      static constexpr auto __connect =
        []<class _Self, class _Receiver>(_Self const &, _Receiver&& __rcvr) noexcept
      {
        using __query_t = __data_of<_Self>;
        return __opstate<_Receiver, __query_t>{static_cast<_Receiver&&>(__rcvr)};
      };

      static constexpr auto __submit =
        []<class _Sender, class _Receiver>(_Sender const &, _Receiver&& __rcvr) noexcept
        requires std::is_reference_v<__call_result_t<__data_of<_Sender>, env_of_t<_Receiver>>>
      {
        static_assert(__sender_for<_Sender, __read_env_t>);
        using __query_t = __data_of<_Sender>;
        STDEXEC::__set_value_from(static_cast<_Receiver&&>(__rcvr),
                                  __query_t(),
                                  STDEXEC::get_env(__rcvr));
      };
    };
  }  // namespace __read_env

  struct __read_env_t
  {
    template <class _Query>
    constexpr auto operator()(_Query) const noexcept
    {
      return __make_sexpr<__read_env_t>(_Query());
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
  //! Given <tt>read_env(q)</tt> and an environment type @c Env (taken from
  //! the connected receiver), the resulting sender has completion signatures:
  //!
  //! @code{.cpp}
  //! set_value_t(decltype(q(declval<Env>())))    // always present
  //! set_error_t(std::exception_ptr)             // present iff q(env) may throw
  //! @endcode
  //!
  //! The query result type is taken from the *actual* environment at
  //! connect time, so the same @c read_env sender may have different
  //! concrete completion signatures depending on which receiver it is
  //! connected to.
  //!
  //! If the environment does not provide a value for @c q (i.e.
  //! <tt>q(env)</tt> is ill-formed or returns @c void), the program is
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
  //! @see stdexec::just          — synchronously complete with literal values
  //! @see stdexec::get_stop_token  — equivalent to <tt>read_env(get_stop_token)</tt>
  //! @see stdexec::get_scheduler   — equivalent to <tt>read_env(get_scheduler)</tt>
  //!
  //! @hideinitializer
  inline constexpr __read_env_t read_env{};

  template <>
  struct __sexpr_impl<__read_env_t> : __read_env::__read_env_impl
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

#include "__epilogue.hpp"
