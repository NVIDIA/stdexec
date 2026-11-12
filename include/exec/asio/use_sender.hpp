/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *                         Copyright (c) 2025 Robert Leahy. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <exec/asio/asio_config.hpp>

#include "../../stdexec/__detail/__config.hpp"

#include "../completion_signatures.hpp"
#include "as_default_on.hpp"
#include "completion_token.hpp"

#if STDEXEC_USE_MODULES()
import std;
import stdexec;
#else
#  include "../../stdexec/__detail/__execution_fwd.hpp"

#  include "../../stdexec/__detail/__completion_signatures_of.hpp"
#  include "../../stdexec/__detail/__receivers.hpp"
#  include "../../stdexec/__detail/__senders.hpp"
#  include "../../stdexec/__detail/__type_traits.hpp"

#  include <exception>
#  include <mutex>
#  include <system_error>
#  include <type_traits>
#endif

namespace experimental::execution::asio
{
  namespace detail::use_sender
  {
    template <class Ty>
    concept error_code_like =
      ::STDEXEC::__one_of<std::remove_cvref_t<Ty>, error_code, std::error_code>;

    inline bool is_cancelled(std::error_code ec) noexcept
    {
      return ec == std::errc::operation_canceled;
    }

    inline bool is_cancelled(error_code ec) noexcept
    {
      return ec == asio_impl::error::operation_aborted || ec == errc::operation_canceled;
    }

    template <error_code_like Ty>
    std::exception_ptr to_exception_ptr(Ty ty) noexcept
    {
      using exception_type =
        std::conditional_t<std::is_same_v<Ty, std::error_code>, std::system_error, system_error>;

      STDEXEC_TRY
      {
        return std::make_exception_ptr(exception_type(static_cast<Ty &&>(ty)));
      }
      STDEXEC_CATCH_ALL
      {
        return std::current_exception();
      }
    }

    template <class Receiver>
    struct receiver : Receiver
    {
      template <error_code_like Ty, class... Args>
      void set_value(Ty &&ty, Args &&...args) && noexcept
      {
        if (!ty)
        {
          ::STDEXEC::set_value(static_cast<Receiver &&>(*this), static_cast<Args &&>(args)...);
        }
        else if (use_sender::is_cancelled(ty))
        {
          ::STDEXEC::set_stopped(static_cast<Receiver &&>(*this));
        }
        else
        {
          ::STDEXEC::set_error(static_cast<Receiver &&>(*this),
                               use_sender::to_exception_ptr(static_cast<Ty &&>(ty)));
        }
      }

      template <class... Args>
      constexpr void set_value(Args &&...args) && noexcept
      {
        ::STDEXEC::set_value(static_cast<Receiver &&>(*this), static_cast<Args &&>(args)...);
      }
    };

    template <class Receiver>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE receiver(Receiver) -> receiver<Receiver>;

    inline constexpr auto _value_sig_transform = []<class Ty = void, class... Args>() noexcept
    {
      if constexpr (std::is_same_v<Ty, void>)
      {
        return ::STDEXEC::completion_signatures<::STDEXEC::set_value_t()>();
      }
      else if constexpr (error_code_like<Ty>)
      {
        return ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(Args...)>();
      }
      else
      {
        return ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(Ty, Args...)>();
      }
    };

    template <class Sender>
    struct sender
    {
      using sender_concept = ::STDEXEC::sender_tag;

      template <class Self, class... Env>
      static consteval auto get_completion_signatures()
      {
        return exec::transform_completion_signatures(
          ::STDEXEC::get_completion_signatures<::STDEXEC::__copy_cvref_t<Self, Sender>, Env...>(),
          _value_sig_transform);
      }

      template <::STDEXEC::receiver Receiver>
      constexpr auto connect(Receiver rcvr) const & noexcept(
        ::STDEXEC::__nothrow_connectable<Sender const &, receiver<Receiver>>)
      {
        return ::STDEXEC::connect(sndr_, receiver{static_cast<Receiver &&>(rcvr)});
      }

      template <::STDEXEC::receiver Receiver>
      constexpr auto connect(Receiver rcvr) && noexcept(
        ::STDEXEC::__nothrow_connectable<Sender, receiver<Receiver>>)
      {
        return ::STDEXEC::connect(static_cast<Sender &&>(sndr_),
                                  receiver{static_cast<Receiver &&>(rcvr)});
      }

      Sender sndr_;
    };

    template <class Sender>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE explicit sender(Sender) -> sender<Sender>;

    template <typename Mutex>
    struct token
    {
      static constexpr auto as_default_on = asio::as_default_on<token>;
      template <typename IoObject>
      using as_default_on_t = asio::as_default_on_t<token, IoObject>;
    };
  }  // namespace detail::use_sender

  using use_sender_t = detail::use_sender::token<std::recursive_mutex>;

  inline use_sender_t const use_sender{};

  using thread_unsafe_use_sender_t =
    detail::use_sender::token<detail::completion_token::null_basic_lockable>;

  inline thread_unsafe_use_sender_t const thread_unsafe_use_sender{};

}  // namespace experimental::execution::asio

namespace exec = experimental::execution;

namespace ASIOEXEC_ASIO_NAMESPACE
{
  template <typename Mutex, typename... Signatures>
  struct async_result<::exec::asio::detail::use_sender::token<Mutex>, Signatures...>
  {
    template <typename Initiation, ::STDEXEC::__decay_copyable... Args>
    static constexpr auto
    initiate(Initiation &&i, ::exec::asio::detail::use_sender::token<Mutex> const &, Args &&...args)
    {
      return ::exec::asio::detail::use_sender::sender(
        async_result<::exec::asio::detail::completion_token::token<Mutex>, Signatures...>::initiate(
          static_cast<Initiation &&>(i),
          ::exec::asio::detail::completion_token::token<Mutex>{},
          static_cast<Args &&>(args)...));
    }
  };
}  // namespace ASIOEXEC_ASIO_NAMESPACE
