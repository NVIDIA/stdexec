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

#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include "elide.hpp"
#include "enter_scope_sender.hpp"
#include "../stdexec/execution.hpp"

namespace experimental::execution {

template<typename T, typename... Args>
  requires
    std::is_constructible_v<T, Args...> &&
    std::is_destructible_v<T>
struct sync_object {
  using type = T;
  template<typename... Ts>
    requires (std::is_constructible_v<Args, Ts> && ...)
  constexpr explicit sync_object(Ts&&... ts) noexcept(
    (std::is_nothrow_constructible_v<Args, Ts> && ...))
    : args_((Ts&&)ts...)
  {}
  constexpr enter_scope_sender auto operator()(type* storage) &
    noexcept(noexcept(make_sender(*this, storage)))
  {
    return make_sender(*this, storage);
  }
  constexpr enter_scope_sender auto operator()(type* storage) const &
    noexcept(noexcept(make_sender(*this, storage)))
  {
    return make_sender(*this, storage);
  }
  constexpr enter_scope_sender auto operator()(type* storage) &&
    noexcept(noexcept(make_sender(std::move(*this), storage)))
  {
    return make_sender(std::move(*this), storage);
  }
  constexpr enter_scope_sender auto operator()(type* storage) const &&
    noexcept(noexcept(make_sender(std::move(*this), storage)))
  {
    return make_sender(std::move(*this), storage);
  }
private:
  template<typename Self>
  static constexpr enter_scope_sender auto make_sender(Self&& self, type* storage)
    noexcept(
      std::is_nothrow_constructible_v<
        std::tuple<Args...>,
        ::STDEXEC::__copy_cvref_t<Self, std::tuple<Args...>>>)
  {
    constexpr auto nothrow = std::is_nothrow_constructible_v<T, Args...>;
    return
      ::STDEXEC::just(std::forward<Self>(self).args_) |
      ::STDEXEC::then([storage](std::tuple<Args...>&& tuple) noexcept(nothrow) {
        const auto ptr = std::construct_at(
          storage,
          ::exec::elide([&]() noexcept(nothrow) {
            return std::make_from_tuple<T>(std::move(tuple));
          }));
        return
          ::STDEXEC::just() |
          //  It's important we capture ptr not storage because storage just
          //  points to storage where ptr actually points to an object
          ::STDEXEC::then([ptr]() noexcept {
            ptr->~T();
          });
      });
  }
  std::tuple<Args...> args_;
};

template<typename T, typename... Args>
constexpr sync_object<T, std::decay_t<Args>...> make_sync_object(Args&&... args)
  noexcept((std::is_nothrow_constructible_v<std::decay_t<Args>, Args> && ...))
{
  return sync_object<T, std::decay_t<Args>...>((Args&&)args...);
}

}  // namespace exec
