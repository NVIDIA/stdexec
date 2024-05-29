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

struct __tuple_index_pack_t {
  template<typename Fn, typename T, typename... Tn>
  auto operator()(Fn&& fn, T&& t, Tn&&... tn) const {
    return fn(std::make_index_sequence<std::tuple_size<std::remove_cvref_t<T>>::value>(), std::forward<T&&>(t), std::forward<Tn&&>(tn)...);
  }
};
constexpr inline static __tuple_index_pack_t __tuple_index_pack;

} // namespace exec
