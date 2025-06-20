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

#include <concepts>
#include <exception>
#include <system_error>
#include <type_traits>
#include <utility>
#include <asioexec/asio_config.hpp>
#include <asioexec/completion_token.hpp>
#include <stdexec/execution.hpp>

namespace asioexec {

  namespace detail::use_sender {

    template <typename T>
    concept is_error_code = std::is_same_v<std::remove_cvref_t<T>, error_code>
                         || std::is_same_v<std::remove_cvref_t<T>, std::error_code>;

    template <is_error_code T>
    std::exception_ptr to_exception_ptr(T t) noexcept {
      using exception_type =
        std::conditional_t<std::is_same_v<T, std::error_code>, std::system_error, system_error>;
      STDEXEC_TRY {
        return std::make_exception_ptr(exception_type(static_cast<T&&>(t)));
      }
      STDEXEC_CATCH_ALL {
        return std::current_exception();
      }
    }

    template <typename Receiver>
    struct receiver {
      template <typename T>
        requires std::constructible_from<Receiver, T>
      constexpr explicit receiver(T&& t) noexcept(std::is_nothrow_constructible_v<Receiver, T>)
        : r_(static_cast<T&&>(t)) {
      }

      using receiver_concept = ::stdexec::receiver_t;

      constexpr void set_stopped() && noexcept
        requires ::stdexec::receiver_of<
          Receiver,
          ::stdexec::completion_signatures<::stdexec::set_stopped_t()>
        >
      {
        ::stdexec::set_stopped(static_cast<Receiver&&>(r_));
      }

      void set_error(std::exception_ptr ex) && noexcept
        requires ::stdexec::receiver_of<
          Receiver,
          ::stdexec::completion_signatures<::stdexec::set_error_t(std::exception_ptr)>
        >
      {
        ::stdexec::set_error(static_cast<Receiver&&>(r_), std::move(ex));
      }

      template <typename T, typename... Args>
        requires is_error_code<T>
              && ::stdexec::receiver_of<
                   Receiver,
                   ::stdexec::completion_signatures<
                     ::stdexec::set_value_t(Args...),
                     ::stdexec::set_error_t(std::exception_ptr),
                     ::stdexec::set_stopped_t()
                   >
              >
      void set_value(T&& t, Args&&... args) && noexcept {
        if (!t) {
          ::stdexec::set_value(static_cast<Receiver&&>(r_), static_cast<Args&&>(args)...);
          return;
        }
        if ([&]() noexcept {
              using type = std::remove_cvref_t<T>;
              if constexpr (std::is_same_v<type, error_code>) {
                if (t == asio_impl::error::operation_aborted) {
                  return true;
                }
              }
              if constexpr (std::is_same_v<std::error_code, std::remove_cvref_t<T>>) {
                return t == std::errc::operation_canceled;
              } else {
                return t == errc::operation_canceled;
              }
            }()) {
          ::stdexec::set_stopped(static_cast<Receiver&&>(r_));
          return;
        }
        ::stdexec::set_error(
          static_cast<Receiver&&>(r_), use_sender::to_exception_ptr(static_cast<T&&>(t)));
      }

      template <typename... Args>
        requires ::stdexec::receiver_of<
          Receiver,
          ::stdexec::completion_signatures<::stdexec::set_value_t(Args...)>
        >
      constexpr void set_value(Args&&... args) && noexcept {
        ::stdexec::set_value(static_cast<Receiver&&>(r_), static_cast<Args&&>(args)...);
      }

      constexpr decltype(auto) get_env() const noexcept {
        return ::stdexec::get_env(r_);
      }
     private:
      Receiver r_;
    };

    template <typename... Args>
    struct transform_set_value {
      using type = ::stdexec::completion_signatures<::stdexec::set_value_t(Args...)>;
    };

    template <typename T, typename... Args>
      requires is_error_code<T>
    struct transform_set_value<T, Args...> {
      using type = ::stdexec::completion_signatures<::stdexec::set_value_t(Args...)>;
    };

    template <typename... Args>
    using transform_set_value_t = typename transform_set_value<Args...>::type;

    template <typename Signatures>
    using completion_signatures = ::stdexec::transform_completion_signatures<
      Signatures,
      ::stdexec::completion_signatures<>,
      transform_set_value_t
    >;

    template <typename Sender>
    struct sender {
      template <typename T>
        requires std::constructible_from<Sender, T>
      constexpr explicit sender(T&& t) noexcept(std::is_nothrow_constructible_v<Sender, T>)
        : s_(static_cast<T&&>(t)) {
      }

      using sender_concept = ::stdexec::sender_t;

      template <typename Env>
      constexpr completion_signatures<::stdexec::completion_signatures_of_t<Sender, Env>>
        get_completion_signatures(const Env&) && noexcept {
        return {};
      }

      template <typename Env>
      constexpr completion_signatures<::stdexec::completion_signatures_of_t<const Sender&, Env>>
        get_completion_signatures(const Env&) const & noexcept {
        return {};
      }

      template <typename Receiver>
        requires ::stdexec::sender_to<const Sender&, receiver<Receiver>>
      constexpr auto connect(Receiver r) const & noexcept(
        std::is_nothrow_constructible_v<receiver<Receiver>, Receiver>
        && noexcept(
          ::stdexec::connect(std::declval<const Sender&>(), std::declval<receiver<Receiver>>()))) {
        return ::stdexec::connect(s_, receiver<Receiver>(static_cast<Receiver&&>(r)));
      }

      template <typename Receiver>
        requires ::stdexec::sender_to<Sender, receiver<Receiver>>
      constexpr auto connect(Receiver r) && noexcept(
        std::is_nothrow_constructible_v<receiver<Receiver>, Receiver>
        && noexcept(
          ::stdexec::connect(std::declval<Sender>(), std::declval<receiver<Receiver>>()))) {
        return ::stdexec::connect(
          static_cast<Sender&&>(s_), receiver<Receiver>(static_cast<Receiver&&>(r)));
      }
     private:
      Sender s_;
    };

    template <typename Sender>
    explicit sender(Sender) -> sender<Sender>;

  } // namespace detail::use_sender

  struct use_sender_t { };

  inline const use_sender_t use_sender{};

} // namespace asioexec

namespace ASIOEXEC_ASIO_NAMESPACE {

  template <typename... Signatures>
  struct async_result<::asioexec::use_sender_t, Signatures...> {
    template <typename Initiation, typename... Args>
    static constexpr auto
      initiate(Initiation&& i, const ::asioexec::use_sender_t&, Args&&... args) {
      return ::asioexec::detail::use_sender::sender(
        async_result<::asioexec::completion_token_t, Signatures...>::initiate(
          static_cast<Initiation&&>(i),
          ::asioexec::completion_token,
          static_cast<Args&&>(args)...));
    }
  };

} // namespace ASIOEXEC_ASIO_NAMESPACE
