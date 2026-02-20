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
    "WARNING: The header <exec/asio/use_sender.hpp> is deprecated. Please include <exec/asio/use_sender.hpp> instead.")
#else
#  warning                                                                                         \
    "The header <exec/asio/use_sender.hpp> is deprecated. Please include <exec/asio/use_sender.hpp> instead."
#endif

#include "../exec/asio/use_sender.hpp" // IWYU pragma: export

namespace asioexec {
  using use_sender_t [[deprecated(
    "asioexec::use_sender_t is deprecated. Please use exec::asio::use_sender_t "
    "instead.")]] = exec::asio::use_sender_t;

  inline constexpr const auto& use_sender
    [[deprecated("asioexec::use_sender is deprecated. Please use exec::asio::use_sender instead.")]]
    = exec::asio::use_sender;
} // namespace asioexec
