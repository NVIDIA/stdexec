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

#include <type_traits>

#include <execution.hpp>
#include <schedulers/detail/storage.hpp>
#include <schedulers/detail/tuple.hpp>

namespace example::cuda
{

namespace detail
{

template <class... Ts>
constexpr std::size_t variadic_max(Ts... as)
{
  std::size_t val = 0;
  ((val = std::max(as, val)), ...);
  return val;
}

template <std::size_t I, class... Ts>
using nth_type = typename std::tuple_element<I, std::tuple<Ts...>>::type;

} // namespace detail

class variant_base
{};

template <class... Ts>
class variant : public variant_base
{
  using storage_t =
    static_storage_t<detail::variadic_max(std::size_t{1},
                                          std::alignment_of_v<Ts>...),
                     detail::variadic_max(sizeof(Ts)...)>;

  storage_t storage_;
  std::size_t index_{std::numeric_limits<std::size_t>::max()};

public:
  template <class Fn>
  using result_t = variant<typename Ts::template result_t<Fn>...>;

  constexpr static bool empty = sizeof...(Ts) == 0;

  __host__ __device__ variant() = default;

  template <std::__one_of<Ts...> T>
  __host__ __device__ variant(T val)
  {}

  constexpr static __host__ __device__ std::size_t size() { return sizeof...(Ts); }

  [[nodiscard]] __host__ __device__ std::size_t index() const { return index_; }

  [[nodiscard]] __host__ __device__ std::byte *data() { return storage_.data; }

  template <std::size_t I>
  __host__ __device__ void emplace(const detail::nth_type<I, Ts...> &val)
  {
    memcpy(data(), &val, sizeof(val));
    index_ = I;
  }
};

template <class T>
class variant<T> : public variant_base
{
  T storage_;

public:
  constexpr static bool empty = false;

  template <class Fn>
  using result_t = variant<typename T::template result_t<Fn>>;

  __host__ __device__ variant() = default;

  template <class... U>
  __host__ __device__ variant(U&&... val) noexcept
      : storage_(std::forward<U>(val)...)
  {}

  template <class... U>
  __host__ __device__ variant &operator=(U&&... val) noexcept
  {
    storage_ = T{std::forward<U>(val)...};
    return *this;
  }

  __host__ __device__ constexpr static std::size_t size() { return 1; }

  [[nodiscard]] __host__ __device__ std::size_t index() const { return 0; }

  [[nodiscard]] __host__ __device__ T &data() { return storage_; }

  template <std::size_t I>
  __host__ __device__ void emplace(T val)
  {
    static_assert(I == 0, "Out of bounds");
    storage_ = val;
  }
};

template <class T>
concept variant_specialization =
  std::derived_from<std::decay_t<T>, variant_base>;

template <std::size_t I, class... Ts>
__host__ __device__ auto get(variant<Ts...> &v) -> detail::nth_type<I, Ts...> &
{
  assert(I == v.index());
  static_assert(I < sizeof...(Ts), "Out of bounds");
  using target_t = detail::nth_type<I, Ts...>;
  return *reinterpret_cast<target_t *>(v.data());
}

template <std::size_t I, class T>
__host__ __device__ T &get(variant<T> &v) noexcept
{
  static_assert(I == 0, "Out of bounds");
  return v.data();
}

namespace detail
{

template <class F, variant_specialization T>
__host__ __device__ void visit_impl(std::integral_constant<std::size_t, 0>, F &&f, T &&v)
{
  if (0 == v.index())
  {
    f(get<0>(v));
  }
}

template <std::size_t I, class F, variant_specialization T>
__host__ __device__ void visit_impl(std::integral_constant<std::size_t, I>, F &&f, T &&v)
{
  if (I == v.index())
  {
    f(get<I>(v));

    return;
  }

  visit_impl(std::integral_constant<std::size_t, I - 1>{},
             std::forward<F>(f),
             v);
}

} // namespace detail

template <class F, variant_specialization T>
__host__ __device__ void visit(F &&f, T &&v)
{
  constexpr std::size_t size = std::decay_t<T>::size();

  return detail::visit_impl(std::integral_constant < std::size_t,
                            (size > 0) ? size - 1 : 0 > {},
                            std::forward<F>(f),
                            std::forward<T>(v));
}

template <class Fn, class Variant>
using visit_t = typename Variant::template result_t<Fn>;

template <class Fn, class Variant>
using apply_t = visit_t<Fn, Variant>;

template <class F, class... Ts>
__host__ __device__ void invoke(F f, cuda::variant<Ts...> &storage)
{
  if constexpr (cuda::variant<Ts...>::empty)
  {
    f();
  }
  else
  {
    cuda::visit(
      [f](auto &&tpl) { cuda::apply(f, std::forward<decltype(tpl)>(tpl)); },
      storage);
  }
}

} // namespace example::cuda
