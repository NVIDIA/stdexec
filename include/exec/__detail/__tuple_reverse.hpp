/*
 * Copyright (c) 2024 Kirk Shoop
 * Copyright (c) 2023 NVIDIA Corporation
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

#include "../../stdexec/concepts.hpp"
#include "../../stdexec/__detail/__tuple.hpp"

#include <type_traits>
#include <tuple>
#include <utility>

namespace exec {

template<template<class...> class R, class I, class... Tn>
struct __apply_reverse_impl;

template<template<class...> class R, std::size_t... In, class... Tn>
struct __apply_reverse_impl<R, std::index_sequence<In...>, Tn...> {
  using tn_t = std::tuple<Tn...>;
  using type = R<typename std::tuple_element<sizeof...(In) - 1 - In, tn_t>::type...>;
};

template<template<class...> class R, class... Tn>
using __apply_reverse = typename __apply_reverse_impl<R, std::make_index_sequence<sizeof...(Tn)>, Tn...>::type;

struct __tuple_reverse_t {
  template<typename T, size_t... I>
  static auto reverse(T&& t, std::index_sequence<I...>) {
    return std::make_tuple(std::get<sizeof...(I) - 1 - I>(std::forward<T>(t))...);
  }

  template<typename T>
  auto operator()(T&& t) const {
    return __tuple_reverse_t::reverse(std::forward<T>(t), std::make_index_sequence<std::tuple_size<T>::value>());
  }
};
constexpr inline static __tuple_reverse_t __tuple_reverse;

} // namespace exec
