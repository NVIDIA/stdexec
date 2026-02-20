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

#pragma once

#if STDEXEC_MSVC()
#  pragma message(                                                                                 \
    "WARNING: The header <exec/asio/completion_token.hpp> is deprecated. Please include <exec/asio/completion_token.hpp> instead.")
#else
#  warning                                                                                         \
    "The header <exec/asio/completion_token.hpp> is deprecated. Please include <exec/asio/completion_token.hpp> instead."
#endif

#include "../exec/asio/completion_token.hpp" // IWYU pragma: export

namespace asioexec {
  using completion_token_t [[deprecated(
    "asioexec::completion_token_t is deprecated. Please use exec::asio::completion_token_t "
    "instead.")]] = exec::asio::completion_token_t;

  inline constexpr const auto& completion_token [[deprecated(
    "asioexec::completion_token is deprecated. Please use exec::asio::completion_token instead.")]]
  = exec::asio::completion_token;
} // namespace asioexec
