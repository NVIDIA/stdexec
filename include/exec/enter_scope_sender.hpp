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
#include <type_traits>
#include <utility>

#include "exit_scope_sender.hpp"
#include "../stdexec/execution.hpp"

namespace experimental::execution {

template<typename Sender>
concept enter_scope_sender = ::STDEXEC::sender<Sender>;

namespace detail::exit_scope_sender_of {

template<typename Env, typename... Args>
struct transform_set_value_impl;
template<typename Env, ::exec::exit_scope_sender_in<Env> Sender>
struct transform_set_value_impl<Env, Sender> {
  using type = ::STDEXEC::completion_signatures<
    ::STDEXEC::set_value_t(Sender)>;
};

template<typename Env>
struct transform_set_value {
  template<typename... Args>
  using fn = transform_set_value_impl<Env, Args...>::type;
};

template<typename T>
using transform_set_error = ::STDEXEC::completion_signatures<>;

template<typename Signatures>
struct impl;
template<typename Sender>
struct impl<
  ::STDEXEC::completion_signatures<
    ::STDEXEC::set_value_t(Sender)>>
{
  using type = Sender;
};

}

template<enter_scope_sender Constructor, typename Env>
using exit_scope_sender_of_t = detail::exit_scope_sender_of::impl<
  ::STDEXEC::transform_completion_signatures<
    ::STDEXEC::completion_signatures_of_t<Constructor, Env>,
    ::STDEXEC::completion_signatures<>,
    detail::exit_scope_sender_of::transform_set_value<Env>::template fn,
    detail::exit_scope_sender_of::transform_set_error,
    ::STDEXEC::completion_signatures<>>>::type;

template<typename Sender, typename Env>
concept enter_scope_sender_in =
  enter_scope_sender<Sender> &&
  ::STDEXEC::sender_in<Sender, Env> &&
  requires {
    typename exit_scope_sender_of_t<Sender, Env>;
  };

}  // namespace exec
