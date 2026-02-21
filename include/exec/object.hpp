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

#include <concepts>
#include <functional>
#include <type_traits>

#include "enter_scope_sender.hpp"

namespace experimental::execution {

template<typename Object>
concept object =
  std::constructible_from<
    std::remove_cvref_t<Object>,
    Object> &&
  std::move_constructible<
    std::remove_cvref_t<Object>> &&
  std::is_object_v<
    typename std::remove_cvref_t<Object>::type> &&
  requires(Object o) {
    {
      std::invoke(
        (Object&&)o,
        (typename std::remove_cvref_t<Object>::type*)nullptr) }
          -> enter_scope_sender;
  };

template<typename Object, typename Env>
concept object_in =
  object<Object> &&
  requires(Object o) {
    {
      std::invoke(
        (Object&&)o,
        (typename std::remove_cvref_t<Object>::type*)nullptr) }
          -> enter_scope_sender_in<Env>;
  };

template<object Object>
using type_of_object_t = std::remove_cvref_t<Object>::type;

template<object Object>
using enter_scope_sender_of_object_t = std::invoke_result_t<
  Object,
  type_of_object_t<Object>*>;

}  // namespace exec
