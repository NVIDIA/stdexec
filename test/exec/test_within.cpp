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

#include <exec/within.hpp>

#include <catch2/catch.hpp>
#include <exec/enter_scope_sender.hpp>
#include <stdexec/execution.hpp>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "../test_common/receivers.hpp"

namespace {

TEST_CASE("The scope is entered, the wrapped sender is run, and then the scope is exited", "[within]") {
  std::size_t n{};
  std::optional<std::size_t> entered;
  std::optional<std::size_t> executed;
  std::optional<std::size_t> exited;
  auto enter =
    ::STDEXEC::just() |
    ::STDEXEC::then([&]() noexcept {
      CHECK(!entered);
      entered = n;
      ++n;
      return
        ::STDEXEC::just() |
        ::STDEXEC::then([&]() noexcept {
          CHECK(!exited);
          exited = n;
          ++n;
        });
    });
  auto sender =
    ::STDEXEC::just() |
    ::STDEXEC::then([&]() noexcept {
      CHECK(!executed);
      executed = n;
      ++n;
    });
  auto within = ::exec::within(enter, sender);
  static_assert(
    std::is_same_v<
      ::STDEXEC::completion_signatures_of_t<
        decltype(within),
        ::STDEXEC::env<>>,
      ::STDEXEC::completion_signatures<
        ::STDEXEC::set_value_t()>>);
  auto op = ::STDEXEC::connect(
    std::move(within),
    expect_void_receiver{});
  CHECK(!n);
  ::STDEXEC::start(op);
  CHECK(entered == 0);
  CHECK(executed == 1);
  CHECK(exited == 2);
}

TEST_CASE("If the work throws the scope is still exited", "[within]") {
  std::size_t n{};
  std::optional<std::size_t> entered;
  std::optional<std::size_t> executed;
  std::optional<std::size_t> exited;
  auto enter =
    ::STDEXEC::just() |
    ::STDEXEC::then([&]() noexcept {
      CHECK(!entered);
      entered = n;
      ++n;
      return
        ::STDEXEC::just() |
        ::STDEXEC::then([&]() noexcept {
          CHECK(!exited);
          exited = n;
          ++n;
        });
    });
  auto sender =
    ::STDEXEC::just() |
    ::STDEXEC::then([&]() {
      CHECK(!executed);
      executed = n;
      ++n;
      throw std::logic_error("TEST");
    });
  auto op = ::STDEXEC::connect(
    ::exec::within(
      enter,
      sender),
    expect_error_receiver{});
  CHECK(!n);
  ::STDEXEC::start(op);
  CHECK(entered == 0);
  CHECK(executed == 1);
  CHECK(exited == 2);
}

} // unnamed namespace
