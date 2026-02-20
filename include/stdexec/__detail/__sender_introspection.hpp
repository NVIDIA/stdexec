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
#include "__sender_concepts.hpp"
#include "__tuple.hpp"
#include "__type_traits.hpp"

#include <cstddef>
#include <exception>  // IWYU pragma: keep for std::terminate

namespace STDEXEC
{
#if defined(STDEXEC_DEMANGLE_SENDER_NAMES)
  template <auto _Descriptor>
  struct __sexpr;
#else
  namespace
  {
    template <auto _Descriptor>
    struct __sexpr;
  }  // namespace
#endif

  // A type that describes a sender's metadata
  template <class _Tag, class _Data, class... _Child>
  struct __desc
  {
    using __tag      = _Tag;
    using __data     = _Data;
    using __indices  = __make_indices<sizeof...(_Child)>;
    using __children = __mlist<_Child...>;

    constexpr auto operator()() const noexcept -> __desc
    {
      return __desc{};
    }

    template <class _Fn, class... _Args>
    using __f = __minvoke<_Fn, _Args..., _Tag, _Data, _Child...>;
  };

  template <class _Ty>
  inline constexpr int __structured_binding_size_v = -1;

#if STDEXEC_HAS_BUILTIN(__builtin_structured_binding_size)
  template <class _Ty>
    requires(__builtin_structured_binding_size(_Ty) >= 0U)
  inline constexpr int __structured_binding_size_v<_Ty> = __builtin_structured_binding_size(_Ty);
#else
  template <__is_tuple _Ty>
  inline constexpr int __structured_binding_size_v<_Ty> = __tuple_size_v<_Ty>;
  // For types that are *not* tuples, __structured_binding_size_v must be specialized
  // explicitly.
#endif

  namespace __detail
  {
    template <class _Tag, class _Data, class... _Child>
    auto __get_desc_impl(__tuple<_Tag, _Data, _Child...> &&) -> __desc<_Tag, _Data, _Child...>;

// Can structured bindings introduce a pack?
#if defined(__cpp_structured_bindings) && __cpp_structured_bindings >= 2024'11L
    STDEXEC_PRAGMA_PUSH()
    STDEXEC_PRAGMA_IGNORE_GNU("-Wc++26-extensions")

    template <class _Sender>
      requires(!__is_tuple<_Sender>)
    auto __get_desc_impl(_Sender &&__sndr)
    {
      auto &&[__tag, __data, ... __child] = __sndr;
      return __desc<decltype(__tag), decltype(__data), decltype(__child)...>{};
    }

    STDEXEC_PRAGMA_POP()
#else
    template <std::size_t _Arity>
    extern __undefined<__msize_t<_Arity>> &__get_desc_impl_v;

#  define STDEXEC_GET_DESC_IMPL_CHILD(_NY)      , __child##_NY
#  define STDEXEC_GET_DESC_IMPL_CHILD_TYPE(_NY) , decltype(__child##_NY)
#  define STDEXEC_GET_DESC_IMPL_ITERATE(_IDX)                                                      \
    template <>                                                                                    \
    inline constexpr auto __get_desc_impl_v<_IDX> = []<class _Sender>(_Sender &&__sndr) {          \
      auto &&[__tag, __data STDEXEC_PP_REPEAT(_IDX, STDEXEC_GET_DESC_IMPL_CHILD)] = __sndr;        \
      return __desc<                                                                               \
        decltype(__tag),                                                                           \
        decltype(__data) STDEXEC_PP_REPEAT(_IDX, STDEXEC_GET_DESC_IMPL_CHILD_TYPE)                 \
      >{};                                                                                         \
    }
    STDEXEC_GET_DESC_IMPL_ITERATE(0);
    STDEXEC_GET_DESC_IMPL_ITERATE(1);
    STDEXEC_GET_DESC_IMPL_ITERATE(2);
    STDEXEC_GET_DESC_IMPL_ITERATE(3);
    STDEXEC_GET_DESC_IMPL_ITERATE(4);
    STDEXEC_GET_DESC_IMPL_ITERATE(5);
    STDEXEC_GET_DESC_IMPL_ITERATE(6);
    STDEXEC_GET_DESC_IMPL_ITERATE(7);
    STDEXEC_GET_DESC_IMPL_ITERATE(8);
    STDEXEC_GET_DESC_IMPL_ITERATE(9);
    STDEXEC_GET_DESC_IMPL_ITERATE(10);
#  undef STDEXEC_GET_DESC_IMPL_CHILD
#  undef STDEXEC_GET_DESC_IMPL_CHILD_TYPE
#  undef STDEXEC_GET_DESC_IMPL_ITERATE

    template <class _Sender>
      requires(!__is_tuple<_Sender>)
    auto __get_desc_impl(_Sender &&__sndr)
    {
      using __desc_t = decltype(__get_desc_impl_v<__structured_binding_size_v<_Sender> - 2>(
        __sndr));
      return __desc_t{};
    }
#endif

    template <class _Sender>
    using __desc_of_t = decltype(__detail::__get_desc_impl(__declval<_Sender>()));

    template <class _Sender>
    extern __undefined<_Sender> &__desc_of_v;

    template <class _Sender>
      requires(__structured_binding_size_v<_Sender> >= 2)
    extern __desc_of_t<_Sender> __desc_of_v<_Sender>;

    template <auto _Descriptor>
    extern decltype(_Descriptor()) __desc_of_v<__sexpr<_Descriptor>>;
  }  // namespace __detail

  template <class _Sender>
  using __desc_of_t = decltype(__detail::__desc_of_v<__decay_t<_Sender>>());

  template <class _Sender>
    requires enable_sender<__decay_t<_Sender>>
  using tag_of_t = __desc_of_t<_Sender>::__tag;

  template <class _Sender>
  using __data_of = __copy_cvref_t<_Sender, typename __desc_of_t<_Sender>::__data>;

  template <class _Sender, class _Continuation = __q<__mlist>>
  using __children_of = __mapply<__mtransform<__copy_cvref_fn<_Sender>, _Continuation>,
                                 typename __desc_of_t<_Sender>::__children>;

  template <class _Ny, class _Sender>
  using __nth_child_of = __children_of<_Sender, __mbind_front_q<__m_at, _Ny>>;

  template <std::size_t _Ny, class _Sender>
  using __nth_child_of_c = __nth_child_of<__msize_t<_Ny>, _Sender>;

  template <class _Sender>
  using __child_of = __children_of<_Sender, __qq<__mfront>>;

  template <class _Sender>
  inline constexpr std::size_t __nbr_children_of = __children_of<_Sender, __msize>::value;

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
}  // namespace STDEXEC
