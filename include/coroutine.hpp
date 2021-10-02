/*
 * Copyright (c) NVIDIA
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

#include <concepts.hpp>

#if __has_include(<coroutine>)
#include <coroutine>
namespace coro = std;
#else
#include <experimental/coroutine>
namespace coro = std::experimental;
#endif

namespace std {
  // Defined some concepts and utilities for working with awaitables
  template <class A>
    concept __awaiter =
      requires (A&& a) {
        {((A&&) a).await_ready()} -> same_as<bool>;
        ((A&&) a).await_resume();
      };

  template <class T>
  decltype(auto) __get_awaiter(T&& t) {
    if constexpr (requires { ((T&&) t).operator co_await(); }) {
      return ((T&&) t).operator co_await();
    }
    else if constexpr (requires { operator co_await((T&&) t); }) {
      return operator co_await((T&&) t);
    }
    else {
      return (T&&) t;
    }
  }

  template <class A>
    concept __awaitable =
      requires (A&& a) {
        {std::__get_awaiter((A&&) a)} -> __awaiter;
      };

  template <__awaitable A>
    using __await_result_t = decltype(std::__get_awaiter(std::declval<A>()).await_resume());
}
