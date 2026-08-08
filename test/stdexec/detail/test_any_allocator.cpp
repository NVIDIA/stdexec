/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <stdexec/execution.hpp>

#include <test_common/catch2.hpp>

#include <utility>

// The converting constructor __any_allocator(_Uy) reads the private member of
// another specialization of the same class template. This is legal since C++17
// ([class.access]) but MSVC rejects it with C2248; the friend declaration on
// __any_allocator makes the conversion compile everywhere. The conversion is
// instantiated whenever task_scheduler's type-erased backend needs to copy an
// allocator on the heap-allocation fallback path (e.g. with an asio-based
// scheduler), so this test guards the fix for NVIDIA/stdexec#2158.
TEST_CASE("__any_allocator cross-specialization converting constructor compiles",
          "[detail][allocator]")
{
  STDEXEC::__any_allocator<int> src;
  STDEXEC::__any_allocator<std::byte> dst(std::move(src)); // instantiates the converting ctor
  CHECK(dst.has_value() == false);
}
