/*
 * Copyright (c) 2022 NVIDIA Corporation
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

#include <algorithm>
#include <type_traits>
#include <stdexec/concepts.hpp>

namespace nvexec {
  namespace detail {
    template <std::size_t I, class... Ts>
      struct tuple_impl_t;

    template <std::size_t I>
      struct tuple_impl_t<I> {};

    template <std::size_t I, class Head, class... Tail>
      struct tuple_impl_t<I, Head, Tail...> : tuple_impl_t<I + 1, Tail...> {
        Head value_{};

        tuple_impl_t() = default;

        template <class H, class... T>
        tuple_impl_t(H &&head, T&&...tail) 
            : tuple_impl_t<I + 1, Tail...>(std::forward<T>(tail)...)
            , value_(std::forward<H>(head)) {}

        void operator=(const tuple_impl_t<I, Head, Tail...> & tpl) {
          value_ = tpl.value_;

          tuple_impl_t<I + 1, Tail...>::operator=(tpl);
        }
      };

    template <class F, class T, std::size_t... Is>
      decltype(auto) apply_impl(F&& f, T&& t, std::index_sequence<Is...>) {
        return ((F&&)f)(get<Is>(t)...);
      }
  }

  template <class... Ts>
      // requires (std::is_trivial_v<Ts> && ...)
    struct tuple_t : detail::tuple_impl_t<0, Ts...> {
      static constexpr std::size_t size = sizeof...(Ts);

      tuple_t() = default;

      template <class... Us>
      tuple_t(Us&&... us)
        requires(sizeof...(Ts) == sizeof...(Us))
        : detail::tuple_impl_t<0, Ts...>(std::forward<Us>(us)...) {}

      tuple_t &operator=(const tuple_t& tpl) noexcept {
        detail::tuple_impl_t<0, Ts...>::operator=(tpl);
        return *this;
      }

      tuple_t &operator=(tuple_t&& tpl) noexcept {
        detail::tuple_impl_t<0, Ts...>::operator=(std::move(tpl));
        return *this;
      }
    };

  template <class F, class T>
    decltype(auto) apply(F &&f, T &&tpl) {
      return detail::apply_impl(
          (F&&)f, (T&&)tpl, std::make_index_sequence<std::decay_t<T>::size>{});
    }

  template <std::size_t I, class Head, class... Tail>
    Head &get(detail::tuple_impl_t<I, Head, Tail...> &t) {
      return t.template tuple_impl_t<I, Head, Tail...>::value_;
    }
}

