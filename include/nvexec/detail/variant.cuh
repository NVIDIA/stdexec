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
#include "../../stdexec/concepts.hpp"
#include "../../stdexec/execution.hpp"

#include "config.cuh"

#include <cuda/std/tuple>

namespace nvexec {

  namespace detail {
    template <class T, class... As>
      concept one_of =
        (std::same_as<T, As> ||...);

    template <class... As>
      struct front_;
    template <class A, class... As>
      struct front_<A, As...> {
        using type = A;
      };

    template <class... As>
        requires(sizeof...(As) > 0)
      using front = typename front_<As...>::type;

    template <std::size_t I, typename... T>
      struct nth_type_;

    template <typename T0, typename... T>
      struct nth_type_<0, T0, T...> {
          using type = T0;
      };

    template <std::size_t I, typename T0, typename... T>
      struct nth_type_<I, T0, T...> {
          using type = typename nth_type_<I - 1, T...>::type;
      };

    template <std::size_t I, typename... T>
      using nth_type = typename nth_type_<I, T...>::type;

    template <class... Ts>
    constexpr std::size_t variadic_max(Ts... as) {
      std::size_t val = 0;
      ((val = std::max(as, val)), ...);
      return val;
    }

    template <std::unsigned_integral IndexT>
    constexpr IndexT not_found() {
      return ~IndexT(0);
    }

    template <std::unsigned_integral IndexT>
    constexpr IndexT npos() {
      return not_found<IndexT>();
    }

    template <std::unsigned_integral IndexT>
    constexpr IndexT ambiguous() {
      return not_found<IndexT>() - 1;
    }

    template <std::unsigned_integral IndexT, class T, class... Ts>
    constexpr IndexT find_index() {
      constexpr bool matches[] = {std::is_same_v<T, Ts>...};
      IndexT result = not_found<IndexT>();
      for (IndexT i = 0; i < sizeof...(Ts); ++i) {
        if (matches[i]) {
          if (result != not_found<IndexT>()) {
            return ambiguous<IndexT>();
          }
          result = i;
        }
      }
      return result;
    }

    template <std::size_t Alignment, std::size_t Size>
    struct alignas(Alignment) static_storage_t {
      std::byte data_[Size];
    };

    template <std::size_t Alignment>
    struct alignas(Alignment) static_storage_t<Alignment, 0> {
      std::byte* data_ = nullptr;
    };

    template <class VisitorT, class V>
      void visit_impl(std::integral_constant<std::size_t, 0>, VisitorT&& visitor, V&& v, std::size_t index) {
        if (0 == index) {
          ((VisitorT&&)visitor)(v.template get<0>());
        }
      }

    template <std::size_t I, class VisitorT, class V>
      void visit_impl(std::integral_constant<std::size_t, I>, VisitorT&& visitor, V&& v, std::size_t index) {
        if (I == index) {
          ((VisitorT&&)visitor)(v.template get<I>());
          return;
        }

        visit_impl(std::integral_constant<std::size_t, I - 1>{}, (VisitorT&&)visitor, (V&&)v, index);
      }
  }

  template <class VisitorT, class V>
    void visit(VisitorT&& visitor, V&& v) {
      detail::visit_impl(
          std::integral_constant<std::size_t, std::decay_t<V>::size - 1>{}, 
          (VisitorT&&)visitor, 
          (V&&)v,
          v.index_);
    }

  template <class VisitorT, class V>
    void visit(VisitorT&& visitor, V&& v, std::size_t index) {
      detail::visit_impl(
          std::integral_constant<std::size_t, std::decay_t<V>::size - 1>{}, 
          (VisitorT&&)visitor, 
          (V&&)v,
          index);
    }

  template <class... Ts>
    // requires (sizeof...(Ts) > 0) && (std::is_trivial_v<Ts> && ...)
  struct variant_t {
    static constexpr std::size_t size = sizeof...(Ts);
    static constexpr std::size_t max_size = std::max({sizeof(Ts)...});
    static constexpr std::size_t max_alignment = std::max({std::alignment_of_v<Ts>...});

    using index_t = unsigned int;
    using union_t = detail::static_storage_t<max_alignment, max_size>;

    template <detail::one_of<Ts...> T>
      using index_of = 
        std::integral_constant<
          index_t, 
          detail::find_index<index_t, T, Ts...>()>;

    template <detail::one_of<Ts...> T>
      T& get() {
        return *reinterpret_cast<T*>(storage_.data_);
      }

    template <std::size_t I>
      detail::nth_type<I, Ts...>& get() {
        return get<detail::nth_type<I, Ts...>>();
      }

    variant_t() {
      emplace<detail::front<Ts...>>();
    }

    ~variant_t() {
      destroy();
    }

    bool holds_alternative() const {
      return index_ != detail::npos<index_t>();
    }

    template <detail::one_of<Ts...> T, class... As>
      void emplace(As&&... as) {
        destroy();
        construct<T>((As&&)as...);
      }
      
    template <detail::one_of<Ts...> T, class... As>
      void construct(As&&... as) {
        ::new(storage_.data_) T((As&&)as...);
        index_ = index_of<T>();
      }

    void destroy() {
      if (holds_alternative()) {
        visit([](auto& val) noexcept {
          using val_t = std::decay_t<decltype(val)>;
          if constexpr(std::is_same_v<val_t, ::cuda::std::tuple<stdexec::set_error_t, std::exception_ptr>>) {
            // TODO Not quite possible at the moment
          } else {
            val.~val_t();
          }
        }, *this);
      }
      index_ = detail::npos<index_t>();
    }

    union_t storage_;
    index_t index_;
  };

}
