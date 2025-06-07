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

#include <asioexec/use_sender.hpp>

#include <chrono>
#include <exception>
#include <functional>
#include <system_error>
#include <utility>
#include <asioexec/asio_config.hpp>
#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>

using namespace stdexec;
using namespace asioexec;

namespace {

  static_assert(
    set_equivalent<
      detail::use_sender::completion_signatures<completion_signatures<
        set_value_t(std::error_code),
        set_stopped_t(),
        set_error_t(std::exception_ptr)
      >>,
      completion_signatures<set_value_t(), set_stopped_t(), set_error_t(std::exception_ptr)>
    >);
  static_assert(
    set_equivalent<
      detail::use_sender::completion_signatures<completion_signatures<
        set_value_t(error_code),
        set_stopped_t(),
        set_error_t(std::exception_ptr)
      >>,
      completion_signatures<set_value_t(), set_stopped_t(), set_error_t(std::exception_ptr)>
    >);
  static_assert(
    set_equivalent<
      detail::use_sender::completion_signatures<completion_signatures<
        set_value_t(error_code, int),
        set_value_t(int),
        set_stopped_t(),
        set_error_t(std::exception_ptr)
      >>,
      completion_signatures<set_value_t(int), set_stopped_t(), set_error_t(std::exception_ptr)>
    >);

  TEST_CASE(
    "Asio-based asynchronous operation ends with set_stopped when "
    "cancellation occurs",
    "[asioexec][use_sender]") {
    bool stopped = false;
    inplace_stop_source source;

    const struct {
      auto query(const get_stop_token_t&) const noexcept {
        return source_.get_token();
      }

      inplace_stop_source& source_;
    } e{source};

    asio_impl::io_context ctx;
    asio_impl::system_timer t(ctx);
    t.expires_after(std::chrono::years(1));
    auto sender = t.async_wait(use_sender);
    static_assert(
      set_equivalent<
        ::stdexec::completion_signatures_of_t<decltype(sender), ::stdexec::env<>>,
        completion_signatures<set_value_t(), set_stopped_t(), set_error_t(std::exception_ptr)>
      >);
    static_assert(
      set_equivalent<
        ::stdexec::completion_signatures_of_t<const decltype(sender)&, ::stdexec::env<>>,
        completion_signatures<set_value_t(), set_stopped_t(), set_error_t(std::exception_ptr)>
      >);
    CHECK(!ctx.poll());
    CHECK(ctx.stopped());
    ctx.restart();
    auto op = ::stdexec::connect(std::move(sender), expect_stopped_receiver_ex(e, stopped));
    CHECK(!ctx.poll());
    CHECK(ctx.stopped());
    ctx.restart();
    source.request_stop();
    start(op);
    CHECK(!stopped);
    CHECK(ctx.poll());
    CHECK(ctx.stopped());
    CHECK(stopped);
  }

  TEST_CASE(
    "std::errc::operation_canceled causes the operation to end with "
    "set_stopped",
    "[asioexec][use_sender]") {
    const auto initiating_function = [](auto&& token) {
      return asio_impl::async_initiate<decltype(token), void(std::error_code)>(
        [](auto&& h) {
          std::invoke(std::forward<decltype(h)>(h), make_error_code(std::errc::operation_canceled));
        },
        token);
    };
    bool stopped = false;
    auto op =
      ::stdexec::connect(initiating_function(use_sender), expect_stopped_receiver_ex(stopped));
    CHECK(!stopped);
    start(op);
    CHECK(stopped);
  }

  TEST_CASE(
    "errc::operation_canceled (note in the case of Boost.Asio this is "
    "boost::system::errc::operation_canceled) causes the operation to end with "
    "set_stopped",
    "[asioexec][use_sender]") {
    const auto initiating_function = [](auto&& token) {
      return asio_impl::async_initiate<decltype(token), void(error_code)>(
        [](auto&& h) {
          std::invoke(std::forward<decltype(h)>(h), make_error_code(errc::operation_canceled));
        },
        token);
    };
    bool stopped = false;
    auto op =
      ::stdexec::connect(initiating_function(use_sender), expect_stopped_receiver_ex(stopped));
    CHECK(!stopped);
    start(op);
    CHECK(stopped);
  }

  TEST_CASE(
    "When an Asio-based asynchronous operation which could fail "
    "completes successfully the success is reported via a value completion "
    "signal",
    "[asioexec][use_sender]") {
    bool invoked = false;
    asio_impl::io_context ctx;
    asio_impl::system_timer t(ctx);
    t.expires_after(std::chrono::milliseconds(1));
    auto sender = t.async_wait(use_sender);
    CHECK(!ctx.poll());
    CHECK(ctx.stopped());
    ctx.restart();
    auto op = ::stdexec::connect(std::move(sender), expect_void_receiver_ex(invoked));
    CHECK(!ctx.poll());
    CHECK(ctx.stopped());
    ctx.restart();
    start(op);
    CHECK(!invoked);
    CHECK(ctx.run());
    CHECK(invoked);
  }

  TEST_CASE("Post works with use_sender", "[asioexec][use_sender]") {
    bool invoked = false;
    asio_impl::io_context ctx;
    auto sender = asio_impl::post(ctx, use_sender);
    CHECK(!ctx.poll());
    CHECK(ctx.stopped());
    ctx.restart();
    auto op = ::stdexec::connect(std::move(sender), expect_void_receiver_ex(invoked));
    CHECK(!ctx.poll());
    CHECK(ctx.stopped());
    ctx.restart();
    start(op);
    CHECK(!invoked);
    CHECK(ctx.poll());
    CHECK(ctx.stopped());
    CHECK(invoked);
  }

  template <typename CompletionToken>
  decltype(auto) async_error_code(CompletionToken&& token) {
    using signature_type = void(error_code);
    return asio_impl::async_initiate<CompletionToken, signature_type>(
      [](auto&& h) {
        std::invoke(
          std::forward<decltype(h)>(h), error_code(make_error_code(std::errc::not_enough_memory)));
      },
      token);
  }

  TEST_CASE(
    "Error codes native to the version of Asio used are transformed "
    "into a system_error",
    "[asioexec][use_sender]") {
    std::exception_ptr ex;
    auto op = ::stdexec::connect(async_error_code(use_sender), expect_error_receiver_ex(ex));
    CHECK(!ex);
    start(op);
    REQUIRE(ex);
    CHECK_THROWS_AS(std::rethrow_exception(std::move(ex)), system_error);
  }

  template <typename CompletionToken>
  decltype(auto) async_std_error_code(CompletionToken&& token) {
    using signature_type = void(std::error_code);
    return asio_impl::async_initiate<CompletionToken, signature_type>(
      [](auto&& h) {
        std::invoke(std::forward<decltype(h)>(h), make_error_code(std::errc::not_enough_memory));
      },
      token);
  }

  TEST_CASE(
    "Standard error codes are transformed into a std::system_error "
    "(note that in the case of standalone Asio the error code native to that "
    "version of Asio a std::error_code are the same and therefore this is a "
    "duplicate of another test)",
    "[asioexec][use_sender]") {
    std::exception_ptr ex;
    auto op = ::stdexec::connect(async_std_error_code(use_sender), expect_error_receiver_ex(ex));
    CHECK(!ex);
    start(op);
    REQUIRE(ex);
    CHECK_THROWS_AS(std::rethrow_exception(std::move(ex)), std::system_error);
  }

} // namespace
