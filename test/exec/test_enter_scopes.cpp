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

#include <exec/enter_scopes.hpp>

#include <catch2/catch.hpp>
#include <exec/enter_scope_sender.hpp>
#include <exec/exit_scope_sender.hpp>
#include <exec/variant_sender.hpp>
#include <stdexec/execution.hpp>

#include <cstddef>
#include <exception>
#include <stdexcept>
#include <utility>

#include "../test_common/receivers.hpp"
#include "../test_common/type_helpers.hpp"

namespace {

static_assert(
  ::exec::enter_scope_sender_in<
    decltype(::exec::enter_scopes()),
    ::STDEXEC::env<>>);
static_assert(
  ::exec::enter_scope_sender_in<
    decltype(::exec::enter_scopes(::STDEXEC::just(::STDEXEC::just()))),
    ::STDEXEC::env<>>);

TEST_CASE("No scopes may be entered", "[enter_scopes]") {
  bool invoked = false;
  auto sender = ::exec::enter_scopes();
  static_assert(
    ::exec::enter_scope_sender_in<
      decltype(sender),
      ::STDEXEC::env<>>);
  auto op = ::STDEXEC::connect(
    std::move(sender),
    make_fun_receiver([&](auto&& sender) noexcept {
      static_assert(::exec::exit_scope_sender<decltype(sender)>);
      CHECK(!invoked);
      invoked = true;
      auto op = ::STDEXEC::connect(
        std::forward<decltype(sender)>(sender),
        expect_void_receiver{});
      ::STDEXEC::start(op);
    }));
  CHECK(!invoked);
  ::STDEXEC::start(op);
  CHECK(invoked);
}

TEST_CASE("One scope may be entered", "[enter_scopes]") {
  std::size_t done = 0;
  std::size_t undone = 0;
  auto undo = ::STDEXEC::just() | ::STDEXEC::then([&]() noexcept {
    ++undone;
  });
  auto succeed = ::STDEXEC::just() | ::STDEXEC::then([&]() noexcept {
    ++done;
    return undo;
  });
  auto sender = ::exec::enter_scopes(succeed);
  static_assert(
    ::exec::enter_scope_sender_in<
      decltype(sender),
      ::STDEXEC::env<>>);
  auto op = ::STDEXEC::connect(
    std::move(sender),
    make_fun_receiver([&](auto&& sender) noexcept {
      static_assert(::exec::exit_scope_sender<decltype(sender)>);
      CHECK(done == 1);
      CHECK(!undone);
      auto op = ::STDEXEC::connect(
        std::forward<decltype(sender)>(sender),
        expect_void_receiver{});
      CHECK(!undone);
      ::STDEXEC::start(op);
      CHECK(undone == 1);
    }));
  CHECK(!done);
  CHECK(!undone);
  ::STDEXEC::start(op);
  CHECK(done == 1);
  CHECK(undone == 1);
}

TEST_CASE("Multiple scopes may be entered", "[enter_scopes]") {
  std::size_t done = 0;
  std::size_t undone = 0;
  auto undo = ::STDEXEC::just() | ::STDEXEC::then([&]() noexcept {
    ++undone;
  });
  auto succeed = ::STDEXEC::just() | ::STDEXEC::then([&]() noexcept {
    ++done;
    return undo;
  });
  auto sender = ::exec::enter_scopes(succeed, succeed);
  static_assert(
    ::exec::enter_scope_sender_in<
      decltype(sender),
      ::STDEXEC::env<>>);
  auto op = ::STDEXEC::connect(
    std::move(sender),
    make_fun_receiver([&](auto&& sender) noexcept {
      static_assert(::exec::exit_scope_sender<decltype(sender)>);
      CHECK(done == 2);
      CHECK(!undone);
      auto op = ::STDEXEC::connect(
        std::forward<decltype(sender)>(sender),
        expect_void_receiver{});
      CHECK(!undone);
      ::STDEXEC::start(op);
      CHECK(undone == 2);
    }));
  CHECK(!done);
  CHECK(!undone);
  ::STDEXEC::start(op);
  CHECK(done == 2);
  CHECK(undone == 2);
}

TEST_CASE("When an attempt is made to enter multiple scopes and entering one of them fails the effects of entering the other are undone", "[construct]") {
  bool undone = false;
  auto undo = ::STDEXEC::just() | ::STDEXEC::then([&]() noexcept {
    CHECK(!undone);
    undone = true;
  });
  auto succeed = ::STDEXEC::just(undo);
  auto fail = ::STDEXEC::just_error(
    std::make_exception_ptr(
      std::logic_error("TEST")));
  using enter_scope_sender_type = ::exec::variant_sender<
    decltype(succeed),
    decltype(fail)>;
  static_assert(
    ::exec::enter_scope_sender_in<
      enter_scope_sender_type,
      ::STDEXEC::env<>>);
  auto sender = ::exec::enter_scopes(
    enter_scope_sender_type(succeed),
    enter_scope_sender_type(fail));
  static_assert(
    ::exec::enter_scope_sender_in<
      decltype(sender),
      ::STDEXEC::env<>>);
  static_assert(
    set_equivalent<
      ::STDEXEC::completion_signatures_of_t<
        decltype(sender),
        ::STDEXEC::env<>>,
      ::STDEXEC::completion_signatures<
        ::STDEXEC::set_value_t(
          ::exec::exit_scope_sender_of_t<
            decltype(sender),
            ::STDEXEC::env<>>),
        ::STDEXEC::set_error_t(std::exception_ptr)>>);
  auto op = ::STDEXEC::connect(std::move(sender), expect_error_receiver{});
  CHECK(!undone);
  ::STDEXEC::start(op);
  CHECK(undone);
}

} // unnamed namespace
