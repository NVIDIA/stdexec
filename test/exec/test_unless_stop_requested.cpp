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

#include <exec/unless_stop_requested.hpp>

#include <exception>

#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

#include "../test_common/receivers.hpp"
#include "../test_common/type_helpers.hpp"

namespace {

  TEST_CASE(
    "When stop has not been requested the child operation runs normally",
    "[unless_stop_requested]") {
    ::STDEXEC::inplace_stop_source source;
    auto env = ::STDEXEC::prop(::STDEXEC::get_stop_token, source.get_token());
    static_assert(!::exec::__unless_stop_requested::__unstoppable_env<decltype(env)>);
    {
      auto sender = ::STDEXEC::just() | ::exec::unless_stop_requested;
      static_assert(
        set_equivalent<
          ::STDEXEC::completion_signatures_of_t<decltype(sender), decltype(env)>,
          ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(), ::STDEXEC::set_stopped_t()>
        >);
      auto op = ::STDEXEC::connect(sender, expect_void_receiver(env));
      ::STDEXEC::start(op);
    }
    {
      auto sender = ::STDEXEC::just(5) | ::exec::unless_stop_requested();
      static_assert(
        set_equivalent<
          ::STDEXEC::completion_signatures_of_t<decltype(sender), decltype(env)>,
          ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(int), ::STDEXEC::set_stopped_t()>
        >);
      auto op = ::STDEXEC::connect(sender, expect_value_receiver(env_tag{}, env, 5));
      ::STDEXEC::start(op);
    }
  }

  TEST_CASE(
    "When stop has been requested the child operation is not started",
    "[unless_stop_requested]") {
    ::STDEXEC::inplace_stop_source source;
    source.request_stop();
    auto env = ::STDEXEC::prop(::STDEXEC::get_stop_token, source.get_token());
    static_assert(!::exec::__unless_stop_requested::__unstoppable_env<decltype(env)>);
    auto sender = ::STDEXEC::just()
                | ::STDEXEC::then([&]() { FAIL_CHECK("Operation should not have been started"); })
                | ::exec::unless_stop_requested();
    static_assert(set_equivalent<
                  ::STDEXEC::completion_signatures_of_t<decltype(sender), decltype(env)>,
                  ::STDEXEC::completion_signatures<
                    ::STDEXEC::set_value_t(),
                    ::STDEXEC::set_error_t(std::exception_ptr),
                    ::STDEXEC::set_stopped_t()
                  >
    >);
    auto op = ::STDEXEC::connect(sender, expect_stopped_receiver(env));
    ::STDEXEC::start(op);
  }

  TEST_CASE("No op when the associated stop token is unstoppable", "[unless_stop_requested]") {
    static_assert(::exec::__unless_stop_requested::__unstoppable_env<::STDEXEC::env<>>);
    auto sender = ::STDEXEC::just() | ::exec::unless_stop_requested;
    static_assert(set_equivalent<
                  ::STDEXEC::completion_signatures_of_t<decltype(sender), ::STDEXEC::env<>>,
                  ::STDEXEC::completion_signatures<::STDEXEC::set_value_t()>
    >);
    auto op = ::STDEXEC::connect(sender, expect_void_receiver{});
    ::STDEXEC::start(op);
  }

} // unnamed namespace
