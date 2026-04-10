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
#include "__tuple.hpp"  // IWYU pragma: keep for __is_tuple and __tuple_size_v
#include "__type_traits.hpp"

#include <cstddef>
#include <exception>  // IWYU pragma: keep for std::terminate

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wc++26-extensions")

namespace STDEXEC
{
  namespace
  {
    template <auto _Descriptor>
    struct __sexpr;
  }  // namespace

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

  namespace __detail
  {
    template <class... _Ts>
    auto __std_tuple_sizer(std::tuple<_Ts...> const &) -> __msize_t<sizeof...(_Ts)>;
  }  // namespace __detail

  template <class _Tuple>
  concept __is_std_tuple = requires(_Tuple &&__arg) { __detail::__std_tuple_sizer(__arg); };

  template <class _Ty>
  inline constexpr int __structured_binding_size_v = -1;

#if STDEXEC_HAS_BUILTIN(__builtin_structured_binding_size)
  template <class _Ty>
    requires(__builtin_structured_binding_size(_Ty) >= 0U)
  inline constexpr int __structured_binding_size_v<_Ty> = __builtin_structured_binding_size(_Ty);
#else
  template <__is_tuple _Ty>
  inline constexpr int __structured_binding_size_v<_Ty> = __tuple_size_v<_Ty>;

  template <__is_std_tuple _Ty>
  inline constexpr int __structured_binding_size_v<_Ty> = decltype(__detail::__std_tuple_sizer(
    __declval<_Ty>()))::value;

  // For types that are *not* tuples, __structured_binding_size_v must be specialized
  // explicitly.
#endif

  namespace __detail
  {
    template <auto _Apply>
    struct __static_const
    {
      using type                  = decltype(_Apply);
      static constexpr type value = _Apply;
    };

#if defined(__cpp_structured_bindings) && __cpp_structured_bindings >= 202411L

    // Structured bindings can introduce a pack, so the implementation of
    // __structured_apply is simple.
    template <bool _Nothrow>
    inline constexpr auto __structured_apply_impl =
      []<class _Fn, class _Type, class... _Us>(_Fn &&__fn, _Type &&__obj, _Us &&...__us)  //
      noexcept(_Nothrow) -> decltype(auto)
    {
      using __cpcv     = __copy_cvref_fn<_Type>;
      auto &[... __as] = __obj;
      return static_cast<_Fn &&>(__fn)(static_cast<_Us &&>(__us)...,
                                       static_cast<__mcall1<__cpcv, decltype(__as)> &&>(__as)...);
    };

    template <int _Ny, bool _Nothrow = true>
    inline constexpr auto const &__structured_apply_v = __structured_apply_impl<_Nothrow>;

#else

    // Structured bindings *cannot* introduce a pack, so we explicitly handle structures
    // with up to 10 members.
    template <int _Ny, bool _Nothrow = true>
    extern __undefined<__msize_t<_Ny>> __structured_apply_v;

    template <bool _Nothrow>
    inline constexpr auto const &__structured_apply_v<0, _Nothrow> = __static_const<(      //
      []<class _Fn, class... _Us>(_Fn &&__fn, __ignore, _Us &&...__us) noexcept(_Nothrow)  //
      -> decltype(auto)                                                                    //
      {                                                                                    //
        return static_cast<_Fn &&>(__fn)(static_cast<_Us &&>(__us)...);                    //
      })>::value;                                                                          //

#  define STDEXEC_STRUCTURED_APPLY_ELEM_ID(_NY)   STDEXEC_PP_IF(_NY, STDEXEC_PP_COMMA, STDEXEC_PP_EAT)() __a##_NY
#  define STDEXEC_STRUCTURED_APPLY_ELEM(_NY)      , static_cast<__mcall1<__cpcv, decltype(__a##_NY)> &&>(__a##_NY)
#  define STDEXEC_STRUCTURED_APPLY_ITERATE(_IDX)                                                    \
    template <bool _Nothrow>                                                                  \
    inline constexpr auto const& __structured_apply_v<_IDX, _Nothrow> = __static_const<(      \
      []<class _Fn, class _Type, class... _Us>(_Fn &&__fn, _Type &&__obj, _Us &&...__us)      \
        noexcept(_Nothrow) -> decltype(auto)                                                  \
      {                                                                                       \
        using __cpcv                                                = __copy_cvref_fn<_Type>; \
        auto &[STDEXEC_PP_REPEAT(_IDX, STDEXEC_STRUCTURED_APPLY_ELEM_ID)] = __obj;                  \
        return static_cast<_Fn &&>(__fn)(                                                     \
          static_cast<_Us &&>(__us)... STDEXEC_PP_REPEAT(_IDX, STDEXEC_STRUCTURED_APPLY_ELEM));     \
      })>::value

    STDEXEC_STRUCTURED_APPLY_ITERATE(1);
    STDEXEC_STRUCTURED_APPLY_ITERATE(2);
    STDEXEC_STRUCTURED_APPLY_ITERATE(3);
    STDEXEC_STRUCTURED_APPLY_ITERATE(4);
    STDEXEC_STRUCTURED_APPLY_ITERATE(5);
    STDEXEC_STRUCTURED_APPLY_ITERATE(6);
    STDEXEC_STRUCTURED_APPLY_ITERATE(7);
    STDEXEC_STRUCTURED_APPLY_ITERATE(8);
    STDEXEC_STRUCTURED_APPLY_ITERATE(9);
    STDEXEC_STRUCTURED_APPLY_ITERATE(10);
#  undef STDEXEC_STRUCTURED_APPLY_ELEM
#  undef STDEXEC_STRUCTURED_APPLY_ELEM_ID
#  undef STDEXEC_STRUCTURED_APPLY_ITERATE

#endif

    struct __structured_apply
    {
     private:
      template <class _Fn>
      struct __get_declfn
      {
        template <class... _As>
        constexpr auto operator()(_As &&...) const noexcept
        {
          if constexpr (__callable<_Fn, _As...>)
          {
            return __declfn<__call_result_t<_Fn, _As...>, __nothrow_callable<_Fn, _As...>>();
          }
        }
      };

#if STDEXEC_EDG()
      template <class _Fn, class _Type, class... _Us>
      static auto __declfn_fn()                                                             //
        -> decltype((__detail::__structured_apply_v<__structured_binding_size_v<_Type>>) (  //
          __get_declfn<_Fn>{},
          __declval<_Type>(),
          __declval<_Us>()...));

      template <class _Fn, class _Type, class... _Us>
        requires(__structured_binding_size_v<_Type> >= 0)
      using __declfn_t = decltype(__declfn_fn<_Fn, _Type, _Us...>());
#else
      template <class _Fn, class _Type, class... _Us>
        requires(__structured_binding_size_v<_Type> >= 0)
      using __declfn_t = __result_of<__structured_apply_v<__structured_binding_size_v<_Type>>,
                                     __get_declfn<_Fn>,
                                     _Type,
                                     _Us...>;
#endif

     public:
      template <class _Fn, class _Type, class... _Us, class _DeclFn = __declfn_t<_Fn, _Type, _Us...>>
        requires __callable<_DeclFn>
      constexpr auto operator()(_Fn &&__fn, _Type &&__obj, _Us &&...__us) const
        noexcept(__nothrow_callable<_DeclFn>) -> __call_result_t<_DeclFn>
      {
        return (__structured_apply_v<__structured_binding_size_v<_Type>,
                                     __nothrow_callable<_DeclFn>>) (static_cast<_Fn &&>(__fn),
                                                                    static_cast<_Type &&>(__obj),
                                                                    static_cast<_Us &&>(__us)...);
      }
    };

    struct __get_desc
    {
      template <class _Tag, class _Data, class... _Child>
      constexpr auto operator()(_Tag, _Data const &, _Child const &...) const noexcept
        -> __desc<_Tag, _Data, _Child...>
      {
        return __desc<_Tag, _Data, _Child...>{};
      }
    };

    template <class _Sender>
    using __desc_of_t = __call_result_t<__structured_apply, __get_desc, _Sender>;

    template <class _Sender>
    extern __undefined<_Sender> &__desc_of_v;

    template <class _Sender>
      requires(__structured_binding_size_v<_Sender> >= 2)
    extern __desc_of_t<_Sender> __desc_of_v<_Sender>;

    template <auto _Descriptor>
    extern decltype(_Descriptor()) __desc_of_v<__sexpr<_Descriptor>>;
  }  // namespace __detail

  using __structured_apply_t = __detail::__structured_apply;
  inline constexpr __structured_apply_t __structured_apply{};

  template <class _Sender>
  using __desc_of_t = decltype(__detail::__desc_of_v<__decay_t<_Sender>>());

  // NOT TO SPEC: in the specification, the tparam of tag_of_t is constrained with the
  // sender concept
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

  template <class _Sender, class... _Tag>
  concept __sender_for = sender<_Sender> && __minvocable_q<tag_of_t, _Sender>
                      && (__std::same_as<tag_of_t<_Sender>, _Tag> && ...);

  template <class _Sender>
  concept sender_expr
    STDEXEC_DEPRECATE_CONCEPT("Please use exec::sender_for from "
                              "<exec/sender_for.hpp> instead") = __sender_for<_Sender>;

  template <class _Sender, class _Tag>
  concept sender_expr_for
    STDEXEC_DEPRECATE_CONCEPT("Please use exec::sender_for from "
                              "<exec/sender_for.hpp> instead") = __sender_for<_Sender, _Tag>;
}  // namespace STDEXEC

STDEXEC_PRAGMA_POP()
