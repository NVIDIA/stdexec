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

#include <stdexec/execution.hpp>

#include <type_traits>

namespace {

  struct on_stop_request {
    void operator()() && noexcept {
    }
  };

  static_assert(::STDEXEC::stoppable_token<::STDEXEC::never_stop_token>);
  static_assert(::STDEXEC::unstoppable_token<::STDEXEC::never_stop_token>);
  static_assert(::STDEXEC::stoppable_token<::STDEXEC::inplace_stop_token>);
  static_assert(!::STDEXEC::unstoppable_token<::STDEXEC::inplace_stop_token>);
  static_assert(std::is_same_v<
                ::STDEXEC::stop_callback_for_t<::STDEXEC::never_stop_token, on_stop_request>,
                ::STDEXEC::never_stop_token::callback_type<on_stop_request>
  >);
  static_assert(::STDEXEC::stoppable_token<std::stop_token>);
  static_assert(std::is_same_v<
                ::STDEXEC::stop_callback_for_t<std::stop_token, on_stop_request>,
                std::stop_callback<on_stop_request>
  >);

} // namespace
