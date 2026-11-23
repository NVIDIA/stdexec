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

#  include "__completion_behavior.hpp"
#  include "__sender_facade.hpp"
#  include "__tuple.hpp"

#  include "__prologue.hpp"

STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  //////////////////////////////////////////////////////////////////////////////
  // [exec.factories]
  namespace __just
  {
    template <class _SetTag>
    struct __attrs
    {
      static constexpr auto query(__get_completion_behavior_t<_SetTag>) noexcept
      {
        return __completion_behavior::__inline_completion;
      }
    };

    // just(vs...), just_error(err), and just_stopped() complete inline, from start()
    template <class _SetTag>
    struct __algo : __facade::__defaults
    {
      template <class _State, class _Children>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __start(_State& __state, _Children&) noexcept
      {
        return STDEXEC::__apply(
          [&]<class... _Ts>(_Ts&&... __vs) noexcept
          { return __facade::__send(__state.__rcvr_, _SetTag(), static_cast<_Ts&&>(__vs)...); },
          static_cast<decltype(__state.__data_)&&>(__state.__data_));
      }
    };
  }  // namespace __just

  //! @brief A sender factory that produces a sender which completes
  //!        synchronously with the given values on the value channel.
  //!
  //! @c just is the simplest sender factory: it captures zero or more values
  //! and produces a sender that, when connected and started, immediately
  //! delivers those values via @c set_value to the connected receiver — all
  //! within the receiver's @c start() call, without any context transition.
  //! It is the canonical way to inject literal values into a sender pipeline.
  //!
  //! @code{.cpp}
  //! auto s0 = stdexec::just();              // value-completes with no datums
  //! auto s1 = stdexec::just(42);            // value-completes with one int
  //! auto s2 = stdexec::just(1, 2, 3);       // value-completes with three ints
  //! auto s3 = stdexec::just(std::string{"x"}, 7);  // mixed types are fine
  //! @endcode
  //!
  //! See [exec.just] in the C++26 working draft for the normative
  //! specification.
  //!
  //! **Completion signatures.**
  //!
  //! Given @c just(ts...) with argument-pack types @c Ts..., the resulting
  //! sender has the single completion signature:
  //!
  //! @code{.cpp}
  //! set_value_t(std::decay_t<Ts>...)
  //! @endcode
  //!
  //! Each argument is decay-copied into the resulting sender. The error and
  //! stopped channels are empty — @c just never completes with @c set_error
  //! or @c set_stopped.
  //!
  //! **Exception behavior.**
  //!
  //! The factory call itself is @c noexcept when every decay-copy is
  //! @c noexcept. The produced sender's @c start() is @c noexcept by
  //! construction.
  //!
  //! **Cancellation.**
  //!
  //! @c just does not consult the receiver's stop token; it always
  //! synchronously completes with @c set_value.
  //!
  //! **Example.**
  //!
  //! @code{.cpp}
  //! #include <stdexec/execution.hpp>
  //! #include <cassert>
  //!
  //! int main() {
  //!   using namespace stdexec;
  //!
  //!   auto sndr = just(21) | then([](int x) { return x * 2; });
  //!   auto [v]  = sync_wait(std::move(sndr)).value();
  //!   assert(v == 42);
  //! }
  //! @endcode
  //!
  //! @see stdexec::just_error    — synchronously complete with an error
  //! @see stdexec::just_stopped  — synchronously complete with stopped
  //! @see stdexec::read_env      — synchronously complete with a value read
  //!                               from the receiver's environment
  struct just_t
  {
    using __tag_t = set_value_t;

    template <class... _Ts>
    struct __sender;

    template <class... _Ts>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __sender(just_t, __tuple<_Ts...>) -> __sender<_Ts...>;

    template <__movable_value... _Ts>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto operator()(_Ts&&... __ts) const noexcept(__nothrow_decay_copyable<_Ts...>);
  };

  template <>
  struct __facade::__algorithm<just_t> : __just::__algo<set_value_t>
  {};

  template <class... _Ts>
  struct just_t::__sender : __facade::__sender<just_t, __tuple<_Ts...>>
  {};

  //! @brief Construct a sender that synchronously value-completes with the
  //!        decay-copies of @c __ts....
  //!
  //! @tparam _Ts  Zero or more types each satisfying the internal
  //!              `__movable_value` concept.
  //! @param __ts  The values to deliver. Each is decay-copied into the
  //!              resulting sender.
  //!
  //! @returns A sender with the single completion signature
  //!          `set_value_t(std::decay_t<_Ts>...)`.
  template <__movable_value... _Ts>
  STDEXEC_ATTRIBUTE(host, device)
  inline constexpr auto just_t::operator()(_Ts&&... __ts) const
    noexcept(__nothrow_decay_copyable<_Ts...>)
  {
    return just_t::__sender{just_t(), __tuple{static_cast<_Ts&&>(__ts)...}};
  }

  //! @brief A sender factory that produces a sender which completes
  //!        synchronously with the given error on the error channel.
  //!
  //! @c just_error is the error-channel analogue of @c just: it captures a
  //! single error datum and produces a sender that, when connected and
  //! started, immediately delivers that error via @c set_error to the
  //! connected receiver. It is the canonical way to inject a literal error
  //! into a sender pipeline — useful for testing error-handling adaptors
  //! such as @c upon_error and @c let_error.
  //!
  //! @code{.cpp}
  //! auto s1 = stdexec::just_error(std::error_code{ENOENT, std::system_category()});
  //! auto s2 = stdexec::just_error(std::make_exception_ptr(std::runtime_error{"boom"}));
  //! @endcode
  //!
  //! See [exec.just] in the C++26 working draft for the normative
  //! specification (@c just_error is specified alongside @c just and
  //! @c just_stopped).
  //!
  //! **Completion signatures.**
  //!
  //! Given `just_error(e)` with @c E = `decltype((e))`, the
  //! resulting sender has the single completion signature:
  //!
  //! @code{.cpp}
  //! set_error_t(std::decay_t<E>)
  //! @endcode
  //!
  //! The error is decay-copied into the resulting sender. The value and
  //! stopped channels are empty.
  //!
  //! **Cancellation.**
  //!
  //! @c just_error does not consult the receiver's stop token; it always
  //! synchronously completes with @c set_error.
  //!
  //! @see stdexec::just          — synchronously complete with values
  //! @see stdexec::just_stopped  — synchronously complete with stopped
  //! @see stdexec::upon_error    — handle the error channel
  //! @see stdexec::let_error     — handle the error channel with a
  //!                               sender-returning function
  struct just_error_t
  {
    using __tag_t = set_error_t;

    template <class _Error>
    struct __sender;

    template <class _Error>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __sender(just_error_t, __tuple<_Error>) -> __sender<_Error>;

    template <__movable_value _Error>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto operator()(_Error&& __err) const noexcept(__nothrow_decay_copyable<_Error>);
  };

  template <>
  struct __facade::__algorithm<just_error_t> : __just::__algo<set_error_t>
  {};

  template <class _Error>
  struct just_error_t::__sender : __facade::__sender<just_error_t, __tuple<_Error>>
  {};

  //! @brief Construct a sender that synchronously error-completes with the
  //!        decay-copy of @c __err.
  //!
  //! @tparam _Error A type satisfying the internal `__movable_value` concept.
  //! @param __err   The error datum to deliver. Decay-copied into the sender.
  //!
  //! @returns A sender with the single completion signature
  //!          `set_error_t(std::decay_t<_Error>)`.
  template <__movable_value _Error>
  STDEXEC_ATTRIBUTE(host, device)
  inline constexpr auto just_error_t::operator()(_Error && __err) const
    noexcept(__nothrow_decay_copyable<_Error>)
  {
    return just_error_t::__sender{just_error_t(), __tuple{static_cast<_Error&&>(__err)}};
  }

  //! @brief A sender factory that produces a sender which completes
  //!        synchronously on the stopped channel.
  //!
  //! @c just_stopped is the stopped-channel analogue of @c just: it produces a
  //! sender that, when connected and started, immediately invokes
  //! @c set_stopped on the connected receiver. It carries no datum (the
  //! stopped channel has none). It is the canonical way to inject a literal
  //! cancellation into a sender pipeline — useful for testing cancellation
  //! handling adaptors such as @c upon_stopped and @c let_stopped.
  //!
  //! @code{.cpp}
  //! auto s = stdexec::just_stopped();
  //! @endcode
  //!
  //! See [exec.just] in the C++26 working draft for the normative
  //! specification (@c just_stopped is specified alongside @c just and
  //! @c just_error).
  //!
  //! **Completion signatures.**
  //!
  //! The resulting sender has the single completion signature:
  //!
  //! @code{.cpp}
  //! set_stopped_t()
  //! @endcode
  //!
  //! The value and error channels are empty.
  //!
  //! **Cancellation.**
  //!
  //! @c just_stopped does not consult the receiver's stop token (the
  //! cancellation it delivers is unconditional, not a response to a request).
  //!
  //! @see stdexec::just          — synchronously complete with values
  //! @see stdexec::just_error    — synchronously complete with an error
  //! @see stdexec::upon_stopped  — handle the stopped channel
  //! @see stdexec::let_stopped   — handle the stopped channel with a
  //!                               sender-returning function
  struct just_stopped_t
  {
    using __tag_t = set_stopped_t;
    struct __sender;

    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto operator()() const noexcept;
  };

  template <>
  struct __facade::__algorithm<just_stopped_t> : __just::__algo<set_stopped_t>
  {};

  struct just_stopped_t::__sender : __facade::__sender<just_stopped_t, __tuple<>>
  {};

  //! @brief Construct a sender that synchronously stops-completes.
  //!
  //! @returns A sender with the single completion signature
  //!          `set_stopped_t()`.
  STDEXEC_ATTRIBUTE(host, device)
  inline constexpr auto just_stopped_t::operator()() const noexcept
  {
    return just_stopped_t::__sender{just_stopped_t(), __tuple{}};
  }

  //! @brief The customization point object for the @c just sender factory.
  //!
  //! @c just is an instance of @c just_t. See @ref just_t for the full
  //! description, completion signatures, and a usage example.
  //!
  //! @hideinitializer
  inline constexpr just_t just{};

  //! @brief The customization point object for the @c just_error sender
  //! factory.
  //!
  //! @c just_error is an instance of @c just_error_t. See @ref just_error_t
  //! for the full description, completion signatures, and a usage example.
  //!
  //! @hideinitializer
  inline constexpr just_error_t just_error{};

  //! @brief The customization point object for the @c just_stopped sender
  //! factory.
  //!
  //! @c just_stopped is an instance of @c just_stopped_t. See
  //! @ref just_stopped_t for the full description, completion signatures,
  //! and a usage example.
  //!
  //! @hideinitializer
  inline constexpr just_stopped_t just_stopped{};
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
