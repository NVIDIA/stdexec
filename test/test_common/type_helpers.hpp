/*
 * Copyright (c) Lucian Radu Teodorescu
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

#include <execution.hpp>

namespace ex = std::execution;

//! Used for debugging, to generate errors to the console
template <typename T>
struct type_printer;

//! Used in various sender types queries
template <typename... Ts>
struct type_array {};

//! Used as a default empty context
struct empty_env {
  friend void tag_invoke(ex::set_error_t, empty_env, std::exception_ptr) noexcept;
  friend void tag_invoke(ex::set_done_t, empty_env) noexcept;
};

//! Check that the value_types of a sender matches the expected type
template <typename ExpectedValType, typename Context = empty_env, typename S>
inline void check_val_types(S snd) {
  using t = typename ex::sender_traits_t<S, Context>::template value_types<type_array, type_array>;
  static_assert(std::is_same<t, ExpectedValType>::value);
}

//! Check that the error_types of a sender matches the expected type
template <typename ExpectedValType, typename Context = empty_env, typename S>
inline void check_err_types(S snd) {
  using t = typename ex::sender_traits_t<S, Context>::template error_types<type_array>;
  static_assert(std::is_same<t, ExpectedValType>::value);
}

//! Check that the send_done of a sender matches the expected value
template <bool Expected, typename Context = empty_env, typename S>
inline void check_sends_done(S snd) {
  constexpr bool val = ex::sender_traits_t<S, Context>::sends_done;
  static_assert(val == Expected);
}
