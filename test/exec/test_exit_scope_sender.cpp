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

#include <exec/exit_scope_sender.hpp>

#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

namespace {

static_assert(
  ::exec::exit_scope_sender_in<
    decltype(::STDEXEC::just()),
    ::STDEXEC::env<>>);
static_assert(
  !::exec::exit_scope_sender_in<
    decltype(::STDEXEC::just(5)),
    ::STDEXEC::env<>>);

} // unnamed namespace
