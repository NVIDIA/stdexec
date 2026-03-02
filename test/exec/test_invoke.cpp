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

#include <exec/invoke.hpp>

#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#include "../test_common/receivers.hpp"
#include "../test_common/type_helpers.hpp"

namespace {

TEST_CASE("Values from a predecessor are used to predicate the successor", "[invoke]") {
  auto f = [](int&& i) noexcept {
    return ::STDEXEC::just(i * 2);
  };
  auto predecessor = ::STDEXEC::just(5);
  auto sender = predecessor | ::exec::invoke(f);
  static_assert(
    std::is_same_v<
      ::exec::detail::invoke::completions<
        decltype(predecessor),
        decltype(f),
        ::STDEXEC::env<>>,
      ::STDEXEC::completion_signatures<
        ::STDEXEC::set_value_t(int)>>);
  static_assert(
    set_equivalent<
      ::exec::detail::invoke::variant_for_operation_states<
        decltype(f),
        expect_value_receiver<::STDEXEC::env<>, int>,
        std::tuple<>,
        ::STDEXEC::completion_signatures<
          ::STDEXEC::set_value_t(int)>>::type,
      std::variant<
        ::STDEXEC::connect_result_t<
          decltype(predecessor),
          ::exec::detail::invoke::receiver_ref<
            expect_value_receiver<::STDEXEC::env<>, int>>>>>);
  static_assert(
    std::is_same_v<
      ::STDEXEC::completion_signatures_of_t<
        decltype(sender),
        ::STDEXEC::env<>>,
      ::STDEXEC::completion_signatures<
        ::STDEXEC::set_value_t(int)>>);
  auto op = ::STDEXEC::connect(
    std::move(sender),
    expect_value_receiver(10));
  ::STDEXEC::start(op);
}

TEST_CASE("If the predecessor never completes successfully then the invocable is never invoked", "[invoke]") {
  auto op = ::STDEXEC::connect(
    //  Notably 5 is not invocable
    ::STDEXEC::just_stopped() | ::exec::invoke(5),
    expect_stopped_receiver{});
  ::STDEXEC::start(op);
}

TEST_CASE("When the invocable throws that exception is passed on", "[invoke]") {
  auto op = ::STDEXEC::connect(
    ::STDEXEC::just() | ::exec::invoke([]() -> decltype(::STDEXEC::just()) {
      throw std::logic_error("TEST");
    }),
    expect_error_receiver{});
  ::STDEXEC::start(op);
}

} // unnamed namespace
