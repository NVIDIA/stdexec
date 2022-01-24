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

#include <concepts>
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

template <std::size_t Offset, class... Ts>
struct sub_tuple;

template <class... Ts>
struct tuple;

struct tuple_base
{};

template <class... Ts>
struct tuple
    : tuple_base
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

  __host__ __device__ tuple(Ts... ts) requires(sizeof...(Ts) > 0)
  {
    init<0>(ts...);
  }

  template <std::size_t Offset, class... STs>
  __host__ __device__ tuple &operator=(sub_tuple<Offset, STs...> st)
  {
    copy_tuple_helper<Offset>(st, std::make_index_sequence<sizeof...(STs)>{});

    return *this;
  }

  __host__ __device__ constexpr static std::size_t size()
  {
    return sizeof...(Ts);
  }

protected:
  template <std::size_t I>
  __host__ __device__ void init()
  {}

  template <std::size_t I, class Head, class... Tail>
  __host__ __device__ void init(Head h, Tail... t)
  {
    get<I>(*this) = h;
    init<I + 1>(t...);
  }

  template <std::size_t Offset, class SubTupleT, std::size_t... Is>
  __host__ __device__ void copy_tuple_helper(const SubTupleT &src,
                                             std::index_sequence<Is...>)
  {
    ((cuda::get<Offset + Is>(*this) = cuda::get<Is>(src)), ...);
  }
};

template <class T>
concept tuple_specialization = std::derived_from<std::decay_t<T>, tuple_base>;

template <std::size_t Offset, class... Ts>
struct sub_tuple : tuple<Ts...>
{
  using tuple<Ts...>::tuple;

  __host__ __device__ explicit sub_tuple(tuple<Ts...> src)
  {
    tuple_init_helper(src, std::make_index_sequence<sizeof...(Ts)>{});
  }

  constexpr static std::size_t offset = Offset;

private:
  template <std::size_t... Is>
  __host__ __device__ void tuple_init_helper(const cuda::tuple<Ts...> &src,
                                             std::index_sequence<Is...>)
  {
    ((cuda::get<Is>(*this) = cuda::get<Is>(src)), ...);
  }
};

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
