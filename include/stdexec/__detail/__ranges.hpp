/*
 * Copyright (c) 2023 NVIDIA Corporation
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

#include "__config.hpp"

#include <ranges>

#include "__prologue.hpp"

namespace STDEXEC
{
  namespace [[deprecated("use std::ranges directly")]] ranges
  {
    using std::ranges::begin;
    using std::ranges::end;

    using std::ranges::range_value_t;
    using std::ranges::range_reference_t;
    using std::ranges::iterator_t;
    using std::ranges::sentinel_t;
  }  // namespace ranges
}  // namespace STDEXEC

#include "__epilogue.hpp"
