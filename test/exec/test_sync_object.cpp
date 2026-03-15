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

#include <exec/sync_object.hpp>

#include <cstddef>
#include <functional>
#include <utility>

#include <catch2/catch.hpp>

#include <exec/lifetime.hpp>
#include <stdexec/execution.hpp>

#include "../test_common/receivers.hpp"

namespace {

struct state {
  std::size_t constructed{0};
  std::size_t destroyed{0};
};

class object {
  state& s_;
public:
  int i;
  explicit constexpr object(state& s, int i) noexcept : s_(s), i(i) {
    ++s_.constructed;
  }
  object(const object&) = delete;
  object& operator=(const object&) = delete;
  constexpr ~object() noexcept {
    ++s_.destroyed;
  }
};

//  GCC 14 complains about an object being used outside its lifetime trying to
//  build this, but doesn't really give any clues about which object so it's
//  difficult to address
#ifdef __clang__
static_assert([]() {
  state s;
  struct receiver {
    using receiver_concept = ::STDEXEC::receiver_t;
    bool& b_;
    constexpr void set_value(const int i) && noexcept {
      b_ = i == 5;
    }
  };
  auto sender = ::exec::lifetime(
    [&](object& o) noexcept {
      return ::STDEXEC::just(o.i);
    },
    ::exec::make_sync_object<object>(
      std::ref(s),
      5));
  bool success = false;
  auto op = ::STDEXEC::connect(
    std::move(sender),
    receiver{success});
  ::STDEXEC::start(op);
  return success;
}());
#endif

TEST_CASE("Synchronous object may be adapted into asynchronous objects", "[sync_object]") {
  state s;
  auto sender = ::exec::lifetime(
    [&](object& o) {
      CHECK(s.constructed == 1);
      CHECK(s.destroyed == 0);
      return ::STDEXEC::just(o.i);
    },
    ::exec::make_sync_object<object>(
      std::ref(s),
      5));
  auto op = ::STDEXEC::connect(
    std::move(sender),
    expect_value_receiver(5));
  CHECK(s.constructed == 0);
  CHECK(s.destroyed == 0);
  ::STDEXEC::start(op);
  CHECK(s.constructed == 1);
  CHECK(s.destroyed == 1);
}

} // unnamed namespace
