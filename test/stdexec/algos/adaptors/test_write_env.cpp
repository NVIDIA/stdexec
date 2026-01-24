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

#include <catch2/catch.hpp>

#include <type_traits>
#include <utility>

#include <stdexec/execution.hpp>
#include <test_common/receivers.hpp>

namespace {

  template <typename T>
  struct receiver : expect_void_receiver<> {
    [[nodiscard]]
    constexpr ::STDEXEC::env<> get_env() const noexcept {
      return state_->get_env();
    }
    T* state_;
  };

  struct state {
    [[nodiscard]]
    constexpr ::STDEXEC::env<> get_env() const noexcept {
      return {};
    }
  };

  static_assert(!std::is_same_v<
                void,
                decltype(::STDEXEC::connect(
                  ::STDEXEC::just()
                    | ::STDEXEC::write_env(
                      ::STDEXEC::prop{
                        ::STDEXEC::get_stop_token,
                        std::declval<::STDEXEC::inplace_stop_source&>().get_token()}),
                  receiver<state>{{}, nullptr}))
  >);

  TEST_CASE(
    "write_env works when the actual environment is sourced from a type which was initially "
    "incomplete but has since been completed",
    "[adaptors][write_env]") {
    ::STDEXEC::inplace_stop_source source;
    state s;
    auto op = ::STDEXEC::connect(
      ::STDEXEC::just()
        | ::STDEXEC::write_env(::STDEXEC::prop{::STDEXEC::get_stop_token, source.get_token()}),
      receiver<state>{{}, &s});
    ::STDEXEC::start(op);
  }
} // namespace
