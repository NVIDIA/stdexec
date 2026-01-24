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

#include <exec/lifetime.hpp>

#include <catch2/catch.hpp>
#include <exec/enter_scope_sender.hpp>
#include <exec/enter_scopes.hpp>
#include <stdexec/execution.hpp>

#include <cstddef>
#include <new>
#include <optional>
#include <tuple>
#include <utility>

#include "../test_common/receivers.hpp"

namespace {

struct state {
  std::size_t& n;
  std::optional<std::size_t> constructed;
  std::optional<std::size_t> destroyed;
};

struct object {
  state& s;
  using type = int;
  ::exec::enter_scope_sender auto operator()(int* storage) const noexcept {
    return
      ::STDEXEC::just() |
      ::STDEXEC::then([&s = s, storage]() noexcept {
        new(storage) int(5);
        CHECK(!s.constructed);
        s.constructed = s.n;
        ++s.n;
        return
          ::STDEXEC::just() |
          ::STDEXEC::then([&s]() noexcept {
            CHECK(!s.destroyed);
            s.destroyed = s.n;
            ++s.n;
          });
      });
  }
};

TEST_CASE("Zero async objects work", "[lifetime]") {
  bool invoked = false;
  auto sender = ::exec::lifetime(
    [&]() {
      CHECK(!invoked);
      invoked = true;
      return ::STDEXEC::just();
    });
  static_assert(::STDEXEC::__well_formed_sender<decltype(sender)>);
  auto op = ::STDEXEC::connect(
    std::move(sender),
    expect_void_receiver{});
  CHECK(!invoked);
  ::STDEXEC::start(op);
  CHECK(invoked);
}

TEST_CASE("Single async object works", "[lifetime]") {
  std::size_t n{};
  state s{n};
  auto sender = ::exec::lifetime(
    [&](int& i) {
      CHECK(i == 5);
      return ::STDEXEC::just();
    },
    object{s});
  auto op = ::STDEXEC::connect(
    std::move(sender),
    expect_void_receiver{});
  CHECK(!n);
  ::STDEXEC::start(op);
  CHECK(n == 2);
  CHECK(s.constructed == 0);
  CHECK(s.destroyed == 1);
}

TEST_CASE("Multiple async objects work", "[lifetime]") {
  std::size_t n{};
  state a{n};
  state b{n};
  auto sender = ::exec::lifetime(
    [&](int& a, int& b) {
      CHECK(a == 5);
      CHECK(b == 5);
      CHECK(&a != &b);
      return ::STDEXEC::just();
    },
    object{a},
    object{b});
  auto op = ::STDEXEC::connect(
    std::move(sender),
    expect_void_receiver{});
  CHECK(!n);
  ::STDEXEC::start(op);
  CHECK(n == 4);
  CHECK(a.constructed == 0);
  CHECK(b.constructed == 1);
  CHECK(a.destroyed == 2);
  CHECK(b.destroyed == 3);
}

TEST_CASE("A custom function may be used to decide how the scopes controlling object lifetime are entered and exited", "[lifetime]") {
  std::size_t n{};
  state a{n};
  state b{n};
  auto sender = ::exec::lifetime(
    [](
      ::exec::enter_scope_sender auto&& a,
      ::exec::enter_scope_sender auto&& b)
    {
      return ::exec::enter_scopes(
        std::forward<decltype(b)>(b),
        std::forward<decltype(a)>(a));
    },
    [&](int& a, int& b) {
      CHECK(a == 5);
      CHECK(b == 5);
      CHECK(&a != &b);
      return ::STDEXEC::just();
    },
    object{a},
    object{b});
  auto op = ::STDEXEC::connect(
    std::move(sender),
    expect_void_receiver{});
  CHECK(!n);
  ::STDEXEC::start(op);
  CHECK(n == 4);
  CHECK(a.constructed == 1);
  CHECK(b.constructed == 0);
  CHECK(a.destroyed == 3);
  CHECK(b.destroyed == 2);
}

} // unnamed namespace
