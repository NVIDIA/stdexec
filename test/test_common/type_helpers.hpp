/*
 * Copyright (c) 2022 Lucian Radu Teodorescu
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

#include <stdexec/execution.hpp>

namespace ex = stdexec;

//! Used for to make a class non-movable without giving up aggregate initialization
struct immovable {
  immovable() = default;

 private:
  STDEXEC_IMMOVABLE(immovable);
};

//! A move-only type
struct movable {
  movable(int value)
    : value_(value) {
  }

  movable(movable&&) = default;
  bool operator==(const movable&) const noexcept = default;

  int value() {
    return value_;
  } // silence warning of unused private field
 private:
  int value_;
};

//! A type with potentially throwing move/copy constructors
struct potentially_throwing {
  potentially_throwing() = default;

  potentially_throwing(potentially_throwing&&) noexcept(false) {
  }

  potentially_throwing(const potentially_throwing&) noexcept(false) {
  }

  potentially_throwing& operator=(potentially_throwing&&) noexcept(false) {
    return *this;
  }

  potentially_throwing& operator=(const potentially_throwing&) noexcept(false) {
    return *this;
  }
};

//! Used for debugging, to generate errors to the console
template <typename T>
struct type_printer;

//! Used in various sender types queries
template <typename... Ts>
struct type_array { };

//! Used as a default empty context
using ex::empty_env;

//! Check that the value_types of a sender matches the expected type
template <typename ExpectedValType, typename Env = empty_env, typename S>
inline void check_val_types(S snd) {
  using t = typename ex::value_types_of_t<S, Env, type_array, type_array>;
  static_assert(std::same_as<t, ExpectedValType>);
}

//! Check that the env of a sender matches the expected type
template <typename ExpectedEnvType, typename S>
inline void check_env_type(S snd) {
  using t = decltype(ex::get_env(snd));
  static_assert(std::same_as<t, ExpectedEnvType>);
}

//! Check that the error_types of a sender matches the expected type
template <typename ExpectedValType, typename Env = empty_env, typename S>
inline void check_err_types(S snd) {
  using t = ex::error_types_of_t<S, Env, type_array>;
  static_assert(std::same_as<t, ExpectedValType>);
}

//! Check that the sends_stopped of a sender matches the expected value
template <bool Expected, typename Env = empty_env, typename S>
inline void check_sends_stopped(S snd) {
  constexpr bool val = ex::sends_stopped<S, Env>;
  static_assert(val == Expected);
}
