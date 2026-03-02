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

#include <exec/nest_scopes.hpp>

#include <catch2/catch.hpp>
#include <exec/enter_scope_sender.hpp>
#include <stdexec/execution.hpp>

#include <cstddef>
#include <optional>
#include <exception>
#include <stdexcept>
#include <utility>

#include "../test_common/receivers.hpp"
#include "../test_common/type_helpers.hpp"

namespace {

static_assert(
  exec::exit_scope_sender_in<
    decltype(
      exec::detail::nest_scopes::make_exit_scope_sender(
        ::STDEXEC::just())),
    ::STDEXEC::env<>>);
static_assert(
  exec::exit_scope_sender_in<
    decltype(
      exec::detail::nest_scopes::make_exit_scope_sender(
        ::STDEXEC::just(),
        ::STDEXEC::just())),
    ::STDEXEC::env<>>);

TEST_CASE("nest_scopes exit sender exits scopes in reverse order", "[nest_scopes]") {
  std::size_t n = 0;
  std::optional<std::size_t> a;
  std::optional<std::size_t> b;
  auto sender = exec::detail::nest_scopes::make_exit_scope_sender(
    ::STDEXEC::just() | ::STDEXEC::then([&]() {
      CHECK(!a);
      a = ++n;
    }),
    ::STDEXEC::just() | ::STDEXEC::then([&]() {
      CHECK(!b);
      b = ++n;
    }));
  auto op = ::STDEXEC::connect(
    std::move(sender),
    expect_void_receiver{});
  CHECK(!a);
  CHECK(!b);
  ::STDEXEC::start(op);
  REQUIRE(a);
  CHECK(*a == 2);
  REQUIRE(b);
  CHECK(*b == 1);
}

TEST_CASE("Scopes can be nested", "[nest_scopes]") {
  std::size_t n = 0;
  std::optional<std::size_t> a_entered;
  std::optional<std::size_t> a_exited;
  auto a = ::STDEXEC::just() | ::STDEXEC::then([&]() noexcept {
    CHECK(!a_entered);
    a_entered = ++n;
    return ::STDEXEC::just() | ::STDEXEC::then([&]() noexcept {
      CHECK(!a_exited);
      a_exited = ++n;
    });
  });
  static_assert(
    exec::detail::nest_scopes::nothrow_connect_sender<
      decltype(a),
      ::STDEXEC::env<>>);
  static_assert(
    !all_contained_in<
      ::STDEXEC::completion_signatures<
        ::STDEXEC::set_error_t(std::exception_ptr)>,
      ::STDEXEC::completion_signatures_of_t<
        decltype(a),
        ::STDEXEC::env<>>>);
  std::optional<std::size_t> b_entered;
  std::optional<std::size_t> b_exited;
  auto b = ::STDEXEC::just() | ::STDEXEC::then([&]() noexcept {
    CHECK(!b_entered);
    b_entered = ++n;
    return ::STDEXEC::just() | ::STDEXEC::then([&]() noexcept {
      CHECK(!b_exited);
      b_exited = ++n;
    });
  });
  static_assert(
    exec::detail::nest_scopes::nothrow_connect_sender<
      decltype(b),
      ::STDEXEC::env<>>);
  static_assert(
    !all_contained_in<
      ::STDEXEC::completion_signatures<
        ::STDEXEC::set_error_t(std::exception_ptr)>,
      ::STDEXEC::completion_signatures_of_t<
        decltype(b),
        ::STDEXEC::env<>>>);
  auto sender = exec::nest_scopes(std::move(a), std::move(b));
  static_assert(exec::enter_scope_sender<decltype(sender)>);
  static_assert(exec::enter_scope_sender_in<
    decltype(sender),
    ::STDEXEC::env<>>);
  static_assert(
    !all_contained_in<
      ::STDEXEC::completion_signatures<
        ::STDEXEC::set_error_t(std::exception_ptr)>,
      ::STDEXEC::completion_signatures_of_t<
        decltype(sender),
        ::STDEXEC::env<>>>);
  auto op = ::STDEXEC::connect(
    std::move(sender) | ::STDEXEC::let_value([&](auto exit) noexcept {
      ++n;
      return exit;
    }),
    expect_void_receiver{});
  CHECK(!a_entered);
  CHECK(!a_exited);
  CHECK(!b_entered);
  CHECK(!b_exited);
  ::STDEXEC::start(op);
  REQUIRE(a_entered);
  CHECK(*a_entered == 1);
  REQUIRE(a_exited);
  CHECK(*a_exited == 5);
  REQUIRE(b_entered);
  CHECK(*b_entered == 2);
  REQUIRE(b_exited);
  CHECK(*b_exited == 4);
}

TEST_CASE("Nested scopes can fail thereby exiting all entered scopes", "[nest_scopes]") {
  std::size_t n = 0;
  std::optional<std::size_t> a_entered;
  std::optional<std::size_t> a_exited;
  auto a = ::STDEXEC::just() | ::STDEXEC::then([&]() noexcept {
    CHECK(!a_entered);
    a_entered = ++n;
    return ::STDEXEC::just() | ::STDEXEC::then([&]() noexcept {
      CHECK(!a_exited);
      a_exited = ++n;
    });
  });
  std::optional<std::size_t> b_entered;
  std::optional<std::size_t> b_exited;
  auto b = ::STDEXEC::just() | ::STDEXEC::then([&]() noexcept {
    CHECK(!b_entered);
    b_entered = ++n;
    return ::STDEXEC::just() | ::STDEXEC::then([&]() noexcept {
      CHECK(!b_exited);
      b_exited = ++n;
    });
  });
  auto sender = exec::nest_scopes(
    std::move(a),
    std::move(b),
    ::STDEXEC::just() | ::STDEXEC::then([&]() -> decltype(::STDEXEC::just()) {
      throw std::logic_error("TEST");
    }));
  static_assert(exec::enter_scope_sender<decltype(sender)>);
  static_assert(exec::enter_scope_sender_in<
    decltype(sender),
    ::STDEXEC::env<>>);
  static_assert(
    all_contained_in<
      ::STDEXEC::completion_signatures<
        ::STDEXEC::set_error_t(std::exception_ptr)>,
      ::STDEXEC::completion_signatures_of_t<
        decltype(sender),
        ::STDEXEC::env<>>>);
  auto op = ::STDEXEC::connect(
    std::move(sender) | ::STDEXEC::then([&](auto&&) noexcept {
      FAIL_CHECK("Unexpected invocation!");
    }),
    expect_error_receiver{});
  CHECK(!a_entered);
  CHECK(!a_exited);
  CHECK(!b_entered);
  CHECK(!b_exited);
  ::STDEXEC::start(op);
  REQUIRE(a_entered);
  CHECK(*a_entered == 1);
  REQUIRE(a_exited);
  CHECK(*a_exited == 4);
  REQUIRE(b_entered);
  CHECK(*b_entered == 2);
  REQUIRE(b_exited);
  CHECK(*b_exited == 3);
}

} // unnamed namespace
