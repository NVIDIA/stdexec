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

namespace {

  template <class Haystack>
  struct mall_contained_in {
    template <class... Needles>
    using __f = ex::__mand<ex::__mapply<ex::__mcontains<Needles>, Haystack>...>;
  };

  template <class Needles, class Haystack>
  concept all_contained_in = ex::__v<ex::__mapply<mall_contained_in<Haystack>, Needles>>;

  template <class Needles, class Haystack>
  concept set_equivalent =
    ex::same_as<ex::__mapply<ex::__msize, Needles>, ex::__mapply<ex::__msize, Haystack>>
    && all_contained_in<Needles, Haystack>;

  template <const auto& Tag, class... Args>
  using result_of_t = ex::__result_of<Tag, Args...>;

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
    auto operator==(const movable&) const noexcept -> bool = default;

    auto value() -> int {
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

    potentially_throwing(const potentially_throwing&) noexcept(false) = default;

    auto operator=(potentially_throwing&&) noexcept(false) -> potentially_throwing& {
      return *this;
    }

    auto operator=(const potentially_throwing&) noexcept(false) -> potentially_throwing& = default;
  };

  //! Used for debugging, to generate errors to the console
  template <class T>
  struct type_printer;

  //! Used in various sender types queries
  template <class... Ts>
  struct pack { };

  //! Check that the value_types of a sender matches the expected type
  template <class ExpectedValType, class Env = ex::env<>, class S>
  inline void check_val_types(S) {
    using actual_t = ex::value_types_of_t<S, Env, pack, ex::__mset>;
    static_assert(ex::__mset_eq<actual_t, ExpectedValType>);
  }

  //! Check that the env of a sender matches the expected type
  template <class ExpectedEnvType, class S>
  inline void check_env_type(S snd) {
    using actual_t = decltype(ex::get_env(snd));
    static_assert(ex::same_as<actual_t, ExpectedEnvType>);
  }

  //! Check that the error_types of a sender matches the expected type
  template <class ExpectedValType, class Env = ex::env<>, class S>
  inline void check_err_types(S) {
    using actual_t = ex::error_types_of_t<S, Env, ex::__mset>;
    static_assert(ex::__mset_eq<actual_t, ExpectedValType>);
  }

  //! Check that the sends_stopped of a sender matches the expected value
  template <bool Expected, class Env = ex::env<>, class S>
  inline void check_sends_stopped(S) {
    constexpr bool actual = ex::sends_stopped<S, Env>;
    static_assert(actual == Expected);
  }
} // namespace
