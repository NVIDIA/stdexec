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

// clang-format Language: Cpp

#pragma once

#include "../../stdexec/execution.hpp"

#include <concepts>
#include <cstddef>
#include <exception> // IWYU pragma: keep
#include <type_traits>

#include "config.cuh"

#include <cuda/std/tuple>

namespace nvexec {

  namespace detail {
    template <std::unsigned_integral IndexT>
    constexpr auto npos() -> IndexT {
      return ~IndexT(0);
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
    STDEXEC_ATTRIBUTE(host, device)
    void visit_impl(
      std::integral_constant<std::size_t, 0>,
      VisitorT&& visitor,
      V&& v,
      std::size_t index) {
      if (0 == index) {
        static_cast<VisitorT&&>(visitor)(static_cast<V&&>(v).template get<0>());
      }
    }

    template <std::size_t I, class VisitorT, class V>
    STDEXEC_ATTRIBUTE(host, device)
    void visit_impl(
      std::integral_constant<std::size_t, I>,
      VisitorT&& visitor,
      V&& v,
      std::size_t index) {
      if (I == index) {
        static_cast<VisitorT&&>(visitor)(static_cast<V&&>(v).template get<I>());
        return;
      }

      visit_impl(
        std::integral_constant<std::size_t, I - 1>{},
        static_cast<VisitorT&&>(visitor),
        static_cast<V&&>(v),
        index);
    }
  } // namespace detail

  template <class VisitorT, class V>
  STDEXEC_ATTRIBUTE(host, device)
  void visit(VisitorT&& visitor, V&& v) {
    detail::visit_impl(
      std::integral_constant<std::size_t, STDEXEC::__decay_t<V>::size - 1>{},
      static_cast<VisitorT&&>(visitor),
      static_cast<V&&>(v),
      v.index_);
  }

  template <class VisitorT, class V>
  STDEXEC_ATTRIBUTE(host, device)
  void visit(VisitorT&& visitor, V&& v, std::size_t index) {
    detail::visit_impl(
      std::integral_constant<std::size_t, STDEXEC::__decay_t<V>::size - 1>{},
      static_cast<VisitorT&&>(visitor),
      static_cast<V&&>(v),
      index);
  }

  template <class... Ts>
  // requires (sizeof...(Ts) > 0) && (std::is_trivial_v<Ts> && ...)
  struct variant_t {
    static constexpr std::size_t size = sizeof...(Ts);
    static constexpr std::size_t max_size = STDEXEC::__umax({sizeof(Ts)...});
    static constexpr std::size_t max_alignment = STDEXEC::__umax({std::alignment_of_v<Ts>...});

    using index_t = unsigned int;
    using union_t = detail::static_storage_t<max_alignment, max_size>;
    using front_t = STDEXEC::__mfront<Ts...>;

    template <STDEXEC::__one_of<Ts...> Type>
    STDEXEC_ATTRIBUTE(host, device)
    auto get() noexcept -> Type& {
      void* data = storage_.data_;
      return *static_cast<Type*>(data);
    }

    template <std::size_t I>
    STDEXEC_ATTRIBUTE(host, device)
    auto get() noexcept -> STDEXEC::__m_at_c<I, Ts...>& {
      return get<STDEXEC::__m_at_c<I, Ts...>>();
    }

    STDEXEC_ATTRIBUTE(host, device)
    variant_t()
      requires std::default_initializable<front_t>
    {
      emplace<front_t>();
    }

    STDEXEC_ATTRIBUTE(host, device)
    ~variant_t() {
      destroy();
    }

    STDEXEC_ATTRIBUTE(host, device) auto holds_alternative() const -> bool {
      return index_ != detail::npos<index_t>();
    }

    template <STDEXEC::__one_of<Ts...> Type, class... Args>
    STDEXEC_ATTRIBUTE(host, device)
    void emplace(Args&&... args) {
      destroy();
      construct<Type>(static_cast<Args&&>(args)...);
    }

    template <STDEXEC::__one_of<Ts...> Type, class... Args>
    STDEXEC_ATTRIBUTE(host, device)
    void construct(Args&&... args) {
      ::new (storage_.data_) Type(static_cast<Args&&>(args)...);
      index_ = STDEXEC::__mcall<STDEXEC::__mfind_i<Type>, Ts...>::value;
    }

    STDEXEC_ATTRIBUTE(host, device)
    void destroy() {
      if (holds_alternative()) {
        nvexec::visit(
          [](auto& val) noexcept {
            using val_t = STDEXEC::__decay_t<decltype(val)>;
            if constexpr (std::is_same_v<
                            val_t,
                            ::cuda::std::tuple<STDEXEC::set_error_t, std::exception_ptr>
                          >) {
              // TODO Not quite possible at the moment
            } else {
              val.~val_t();
            }
          },
          *this);
      }
      index_ = detail::npos<index_t>();
    }

    union_t storage_;
    index_t index_;
  };
} // namespace nvexec
