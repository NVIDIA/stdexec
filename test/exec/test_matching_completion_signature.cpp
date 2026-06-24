/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *                         Copyright (c) 2026 Robert Leahy. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 * https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <exec/matching_completion_signature.hpp>

#include <catch2/catch_all.hpp>
#include <stdexec/execution.hpp>

#include <exception>
#include <type_traits>

using namespace exec;

namespace {

struct base {
};

struct derived : base {
};

using signatures = ::STDEXEC::completion_signatures<
  ::STDEXEC::set_value_t(int),
  ::STDEXEC::set_value_t(base),
  ::STDEXEC::set_value_t(derived&),
  ::STDEXEC::set_error_t(std::exception_ptr),
  ::STDEXEC::set_stopped_t()>;

static_assert(
  std::is_same_v<
    matching_completion_signature_t<signatures, ::STDEXEC::set_value_t, int>,
    ::STDEXEC::set_value_t(int)>);

static_assert(
  std::is_same_v<
    matching_completion_signature_t<
      signatures,
      ::STDEXEC::set_error_t,
      std::exception_ptr>,
    ::STDEXEC::set_error_t(std::exception_ptr)>);

static_assert(
  std::is_same_v<
    matching_completion_signature_t<signatures, ::STDEXEC::set_stopped_t>,
    ::STDEXEC::set_stopped_t()>);

static_assert(
  std::is_same_v<
    matching_completion_signature_t<
      signatures,
      ::STDEXEC::set_value_t,
      derived&>,
    ::STDEXEC::set_value_t(derived&)>);

static_assert(
  has_matching_completion_signature_v<
    signatures,
    ::STDEXEC::set_value_t,
    int>);
static_assert(
  has_matching_completion_signature<
    signatures,
    ::STDEXEC::set_value_t,
    int>::value);

static_assert(
  !has_matching_completion_signature_v<
    signatures,
    ::STDEXEC::set_value_t,
    int,
    int>);
static_assert(
  !has_matching_completion_signature<
    signatures,
    ::STDEXEC::set_value_t,
    int,
    int>::value);

static_assert(
  !has_matching_completion_signature_v<
    signatures,
    ::STDEXEC::set_error_t,
    int>);

static_assert(
  !has_matching_completion_signature_v<
    signatures,
    ::STDEXEC::set_value_t,
    derived>);

TEST_CASE("matching_completion_signature compile-time checks", "[matching_completion_signature]") {
  CHECK(true);
}

} // unnamed namespace
