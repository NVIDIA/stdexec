/*
 * Copyright (c) 2024 Lauri Vasama
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <catch2/catch.hpp>

#include <tuple>

// Workaround for https://github.com/llvm/llvm-project/issues/113087
#if defined(__clang__) && defined(__cpp_lib_tuple_like)
#  define CHECK_TUPLE(...) CHECK((__VA_ARGS__))
#else
#  define CHECK_TUPLE CHECK
#endif
