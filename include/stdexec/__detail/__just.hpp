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

#include "__basic_sender.hpp"
#include "__completion_behavior.hpp"
#include "__completion_signatures.hpp"
#include "__meta.hpp"
#include "__receivers.hpp"
#include "__sender_introspection.hpp"
#include "__type_traits.hpp"

#include "__prologue.hpp"

STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace STDEXEC
{
  /////////////////////////////////////////////////////////////////////////////
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

    template <class _SetTag, class _Tuple, class _Receiver>
    struct __opstate
    {
      constexpr void start() noexcept
      {
        __apply(_SetTag(), static_cast<_Tuple&&>(__data_), static_cast<_Receiver&&>(__rcvr_));
      }

      _Receiver __rcvr_;
      _Tuple    __data_;
    };

    template <class _JustTag>
    struct __impl : __sexpr_defaults
    {
      using __set_tag_t = _JustTag::__tag_t;

      static constexpr auto __get_attrs = [](__ignore, __ignore) noexcept -> __attrs<__set_tag_t>
      {
        return {};
      };

      template <class _Sender, class... _Env>
      static consteval auto __get_completion_signatures()
      {
        static_assert(__sender_for<_Sender, _JustTag>);
        return completion_signatures<__mapply<__qf<__set_tag_t>, __decay_t<__data_of<_Sender>>>>{};
      }

      static constexpr auto __connect =
        []<class _Sender, class _Receiver>(_Sender&& __sndr, _Receiver&& __rcvr) noexcept(
          __nothrow_decay_copyable<_Sender>)
      {
        auto& [__tag, __data] = __sndr;
        return __opstate<__set_tag_t, decltype(__data), _Receiver>{static_cast<_Receiver&&>(__rcvr),
                                                                   STDEXEC::__forward_like<_Sender>(
                                                                     __data)};
      };
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
  //! See [exec.just] in the C++26 working draft for the normative specification.
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
  //! @see stdexec::read_env      — synchronously complete with a value read from the environment
  struct just_t
  {
    using __tag_t = set_value_t;

    //! @brief Construct a sender that synchronously value-completes with the
    //!        decay-copies of @c __ts....
    //!
    //! @tparam _Ts  Zero or more types each satisfying the internal
    //!              <tt>__movable_value</tt> concept.
    //! @param __ts  The values to deliver. Each is decay-copied into the
    //!              resulting sender.
    //!
    //! @returns A sender with the single completion signature
    //!          <tt>set_value_t(std::decay_t<_Ts>...)</tt>.
    template <__movable_value... _Ts>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto operator()(_Ts&&... __ts) const noexcept(__nothrow_decay_copyable<_Ts...>)
    {
      return __make_sexpr<just_t>(__tuple{static_cast<_Ts&&>(__ts)...});
    }
  };

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
  //! Given <tt>just_error(e)</tt> with @c E = <tt>decltype((e))</tt>, the
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
  //! @see stdexec::let_error     — handle the error channel with a sender-returning function
  struct just_error_t
  {
    using __tag_t = set_error_t;

    //! @brief Construct a sender that synchronously error-completes with the
    //!        decay-copy of @c __err.
    //!
    //! @tparam _Error A type satisfying the internal <tt>__movable_value</tt> concept.
    //! @param __err   The error datum to deliver. Decay-copied into the sender.
    //!
    //! @returns A sender with the single completion signature
    //!          <tt>set_error_t(std::decay_t<_Error>)</tt>.
    template <__movable_value _Error>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto operator()(_Error&& __err) const noexcept(__nothrow_decay_copyable<_Error>)
    {
      return __make_sexpr<just_error_t>(__tuple{static_cast<_Error&&>(__err)});
    }
  };

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
  //! @see stdexec::let_stopped   — handle the stopped channel with a sender-returning function
  struct just_stopped_t
  {
    using __tag_t = set_stopped_t;

    //! @brief Construct a sender that synchronously stops-completes.
    //!
    //! @returns A sender with the single completion signature
    //!          <tt>set_stopped_t()</tt>.
    template <class _Tag = just_stopped_t>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto operator()() const noexcept
    {
      return __make_sexpr<_Tag>(__tuple{});
    }
  };

  template <>
  struct __sexpr_impl<just_t> : __just::__impl<just_t>
  {};

  template <>
  struct __sexpr_impl<just_error_t> : __just::__impl<just_error_t>
  {};

  template <>
  struct __sexpr_impl<just_stopped_t> : __just::__impl<just_stopped_t>
  {};

  //! @brief The customization point object for the @c just sender factory.
  //!
  //! @c just is an instance of @ref just_t. See @ref just_t for the full
  //! description, completion signatures, and a usage example.
  //!
  //! @hideinitializer
  inline constexpr just_t just{};

  //! @brief The customization point object for the @c just_error sender factory.
  //!
  //! @c just_error is an instance of @ref just_error_t. See @ref just_error_t
  //! for the full description, completion signatures, and a usage example.
  //!
  //! @hideinitializer
  inline constexpr just_error_t just_error{};

  //! @brief The customization point object for the @c just_stopped sender factory.
  //!
  //! @c just_stopped is an instance of @ref just_stopped_t. See
  //! @ref just_stopped_t for the full description, completion signatures,
  //! and a usage example.
  //!
  //! @hideinitializer
  inline constexpr just_stopped_t just_stopped{};
}  // namespace STDEXEC

#include "__epilogue.hpp"
