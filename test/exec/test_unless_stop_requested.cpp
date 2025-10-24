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
    ::stdexec::inplace_stop_source source;
    auto env = ::stdexec::prop(::stdexec::get_stop_token, source.get_token());
    static_assert(!::exec::__unless_stop_requested::__unstoppable_env<decltype(env)>);
    {
      auto sender = ::stdexec::just() | ::exec::unless_stop_requested;
      static_assert(
        set_equivalent<
          ::stdexec::completion_signatures_of_t<decltype(sender), decltype(env)>,
          ::stdexec::completion_signatures<::stdexec::set_value_t(), ::stdexec::set_stopped_t()>>);
      auto op = ::stdexec::connect(sender, expect_void_receiver(env));
      ::stdexec::start(op);
    }
    {
      auto sender = ::stdexec::just(5) | ::exec::unless_stop_requested();
      static_assert(set_equivalent<
                    ::stdexec::completion_signatures_of_t<decltype(sender), decltype(env)>,
                    ::stdexec::completion_signatures<
                      ::stdexec::set_value_t(int),
                      ::stdexec::set_stopped_t()>>);
      auto op = ::stdexec::connect(sender, expect_value_receiver(env_tag{}, env, 5));
      ::stdexec::start(op);
    }
  }

  TEST_CASE(
    "When stop has been requested the child operation is not started",
    "[unless_stop_requested]") {
    ::stdexec::inplace_stop_source source;
    source.request_stop();
    auto env = ::stdexec::prop(::stdexec::get_stop_token, source.get_token());
    static_assert(!::exec::__unless_stop_requested::__unstoppable_env<decltype(env)>);
    auto sender = ::stdexec::just()
                | ::stdexec::then([&]() { FAIL_CHECK("Operation should not have been started"); })
                | ::exec::unless_stop_requested();
    static_assert(set_equivalent<
                  ::stdexec::completion_signatures_of_t<decltype(sender), decltype(env)>,
                  ::stdexec::completion_signatures<
                    ::stdexec::set_value_t(),
                    ::stdexec::set_error_t(std::exception_ptr),
                    ::stdexec::set_stopped_t()>>);
    auto op = ::stdexec::connect(sender, expect_stopped_receiver(env));
    ::stdexec::start(op);
  }

  TEST_CASE("No op when the associated stop token is unstoppable", "[unless_stop_requested]") {
    static_assert(::exec::__unless_stop_requested::__unstoppable_env<::stdexec::env<>>);
    auto sender = ::stdexec::just() | ::exec::unless_stop_requested;
    static_assert(set_equivalent<
                  ::stdexec::completion_signatures_of_t<decltype(sender), ::stdexec::env<>>,
                  ::stdexec::completion_signatures<::stdexec::set_value_t()>>);
    auto op = ::stdexec::connect(sender, expect_void_receiver{});
    ::stdexec::start(op);
  }

} // unnamed namespace
