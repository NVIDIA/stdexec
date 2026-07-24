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

#include <type_traits>

#include "../stdexec/execution.hpp"

namespace experimental::execution {

template<typename Sender>
concept exit_scope_sender =
  ::STDEXEC::sender<Sender> &&
  std::is_nothrow_constructible_v<
    std::remove_cvref_t<Sender>,
    Sender> &&
  std::is_nothrow_move_constructible_v<
    std::remove_cvref_t<Sender>>;

template<typename Sender, typename Env>
concept exit_scope_sender_in =
  exit_scope_sender<Sender> &&
  ::STDEXEC::sender_in<Sender, Env> &&
  ::STDEXEC::__nothrow_connectable<
    Sender,
    ::STDEXEC::__receiver_archetype<Env>> &&
  std::is_same_v<
    ::STDEXEC::completion_signatures<
      ::STDEXEC::set_value_t()>,
    ::STDEXEC::completion_signatures_of_t<
      Sender,
      Env>>;

}  // namespace exec
