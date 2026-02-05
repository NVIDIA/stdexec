/*
 * Copyright (c) 2021-2024 NVIDIA Corporation
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

#include "__execution_fwd.hpp"
#include "__meta.hpp"
#include "__tuple.hpp"
#include "__type_traits.hpp"

#include <cstddef>
#include <exception> // IWYU pragma: keep for std::terminate

namespace STDEXEC {
#if defined(STDEXEC_DEMANGLE_SENDER_NAMES)
  template <auto _Descriptor>
  struct __sexpr;
#else
  namespace {
    template <auto _Descriptor>
    struct __sexpr;
  } // namespace
#endif

  namespace __detail {
    // A type that describes a sender's metadata
    template <class _Tag, class _Data, class... _Child>
    struct __desc {
      using __tag = _Tag;
      using __data = _Data;
      using __indices = __make_indices<sizeof...(_Child)>;
      using __children = __mlist<_Child...>;

      constexpr auto operator()() const noexcept -> __desc {
        return __desc{};
      }

      template <class _Fn, class... _Args>
      using __f = __minvoke<_Fn, _Args..., _Tag, _Data, _Child...>;
    };

    template <class _Sender>
    using __desc_of = STDEXEC_REMOVE_REFERENCE(_Sender)::__desc_t;

    template <class _Sender>
    using __tag_of = __desc_of<_Sender>::__tag;

    template <class _Sender>
      requires __minvocable_q<__tag_of, _Sender>
    extern __tag_of<_Sender> __tag_of_v;
  } // namespace __detail

  template <class _Sender>
  using tag_of_t = decltype(__detail::__tag_of_v<_Sender>);

  template <class _Sender>
  using __data_of = __tuple_element_t<1, _Sender>;

  template <class _Sender, class _Continuation = __q<__mlist>>
  using __children_of = __mapply<
    __mtransform<__copy_cvref_fn<_Sender>, _Continuation>,
    typename __detail::__desc_of<_Sender>::__children
  >;

  template <std::size_t _Ny, class _Sender>
  using __nth_child_of_c = __tuple_element_t<_Ny + 2, _Sender>;

  template <class _Ny, class _Sender>
  using __nth_child_of = __nth_child_of_c<_Ny::value, _Sender>;

  template <class _Sender>
  using __child_of = __tuple_element_t<2, _Sender>;

  template <class _Sender>
  inline constexpr std::size_t __nbr_children_of = __tuple_size_v<_Sender> - 2;

  template <auto _Descriptor>
  struct __mfor<__sexpr<_Descriptor>> : decltype(_Descriptor()){};

  template <auto _Descriptor>
  struct __mfor<__sexpr<_Descriptor> &> : decltype(_Descriptor()){};

  template <auto _Descriptor>
  struct __mfor<__sexpr<_Descriptor> const &> : decltype(_Descriptor()){};

  template <class _Sender>
  concept sender_expr = __minvocable_q<tag_of_t, _Sender>;

  template <class _Sender, class _Tag>
  concept sender_expr_for = sender_expr<_Sender> && __std::same_as<tag_of_t<_Sender>, _Tag>;
} // namespace STDEXEC
