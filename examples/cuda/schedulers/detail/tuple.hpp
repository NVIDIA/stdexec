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
#include <type_traits>
#include <utility>

namespace example::cuda
{

template <std::size_t I, class... Ts>
struct tuple_impl_t;

template <std::size_t I>
struct tuple_impl_t<I>
{};

template <std::size_t I, class Head, class... Tail>
struct tuple_impl_t<I, Head, Tail...> : tuple_impl_t<I + 1, Tail...>
{
  Head value;

  __host__ __device__ tuple_impl_t() = default;

  template <class H, class... T>
    requires std::is_nothrow_move_constructible_v<Head>
  __host__ __device__ tuple_impl_t(H &&head, T&&...tail) noexcept
      : tuple_impl_t<I + 1, Tail...>(std::forward<T>(tail)...)
      , value(std::forward<H>(head))
  {}

  __host__ __device__ void operator=(const tuple_impl_t<I, Head, Tail...> & tpl) noexcept
    requires std::is_nothrow_copy_assignable_v<Head>
  {
    value = tpl.value;

    tuple_impl_t<I + 1, Tail...>::operator=(tpl);
  }

  __host__ __device__ void operator=(tuple_impl_t<I, Head, Tail...>&& tpl) noexcept
    requires std::is_nothrow_move_assignable_v<Head>
  {
    value = std::move(tpl.value);

    tuple_impl_t<I + 1, Tail...>::operator=(std::move(tpl));
  }
};

template <std::size_t I, class Head, class... Tail>
__host__ __device__ Head &get(tuple_impl_t<I, Head, Tail...> &t)
{
  return t.template tuple_impl_t<I, Head, Tail...>::value;
}
template <std::size_t I, class Head, class... Tail>
__host__ __device__ const Head &get(const tuple_impl_t<I, Head, Tail...> &t)
{
  return t.template tuple_impl_t<I, Head, Tail...>::value;
}

struct tuple_base
{};

template <class... Ts>
struct tuple : tuple_base
             , tuple_impl_t<0, Ts...>
{
  template <class Fn>
  using result_t_ = std::invoke_result_t<Fn, Ts...>;

  template <class Fn>
  using result_t = std::conditional_t<std::is_same_v<result_t_<Fn>, void>,
                                      tuple<>,
                                      tuple<result_t_<Fn>>>;

  constexpr static bool empty = sizeof...(Ts) == 0;

  __host__ __device__ tuple() = default;

  template <class... Us>
  __host__ __device__ tuple(Us&&... us)
    requires(sizeof...(Ts) == sizeof...(Us))
    : tuple_impl_t<0, Ts...>(std::forward<Us>(us)...)
  {
  }

  __host__ __device__ tuple &operator=(const tuple& tpl) noexcept
  {
    tuple_impl_t<0, Ts...>::operator=(tpl);
    return *this;
  }

  __host__ __device__ tuple &operator=(tuple&& tpl) noexcept
  {
    tuple_impl_t<0, Ts...>::operator=(std::move(tpl));
    return *this;
  }

  __host__ __device__ constexpr static std::size_t size()
  {
    return sizeof...(Ts);
  }
};

template <class T>
concept tuple_specialization = std::derived_from<std::decay_t<T>, tuple_base>;

namespace detail
{

template <class F, tuple_specialization T, std::size_t... Is>
__host__ __device__ auto apply_impl(F &&f, T &&t, std::index_sequence<Is...>)
{
  return f(get<Is>(t)...);
}

} // namespace detail

template <class F, tuple_specialization T>
__host__ __device__ auto apply(F &&f, T &&tpl)
{
  return detail::apply_impl(std::forward<F>(f),
                            std::forward<T>(tpl),
                            std::make_index_sequence<std::decay_t<T>::size()>{});
}

} // namespace example::cuda
