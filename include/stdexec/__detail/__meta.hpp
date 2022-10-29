/*
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#include <cassert>
#include <exception>
#include <type_traits>
#include <utility>
#include "__config.hpp"

namespace stdexec {

  struct __ {};

  struct __ignore {
    __ignore() = default;
    __ignore(auto&&) noexcept {}
  };

    // Before gcc-12, gcc really didn't like tuples or variants of immovable types
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ < 12)
#  define STDEXEC_IMMOVABLE(_X) _X(_X&&)
#else
#  define STDEXEC_IMMOVABLE(_X) _X(_X&&) = delete
#endif

    // BUG (gcc PR93711): copy elision fails when initializing a
    // [[no_unique_address]] field from a function returning an object
    // of class type by value
#if defined(__GNUC__) && !defined(__clang__)
#  define STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
#else
#  define STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

  struct __none_such {};

  template <class...>
    concept __typename = true;

  struct __immovable {
    __immovable() = default;
   private:
    STDEXEC_IMMOVABLE(__immovable);
  };

  template <class _T>
    using __t = typename _T::__t;

  // For hiding a template type parameter from ADL
  template <class _T>
    struct _X {
      using __t = struct __t_ {
        using __t = _T;
      };
    };
  template <class _T>
    using __x = __t<_X<_T>>;

  template <bool _B>
    using __bool = std::bool_constant<_B>;

  template <std::size_t _N>
    using __index = std::integral_constant<std::size_t, _N>;

  // Some utilities for manipulating lists of types at compile time
  template <class...>
    struct __types;

  template <class _T>
    using __id = _T;

  template <class _T>
    inline constexpr auto __v = _T::value;

  template <class _T, class _U>
    inline constexpr bool __v<std::is_same<_T, _U>> = false;

  template <class _T>
    inline constexpr bool __v<std::is_same<_T, _T>> = true;

  template <class _T, _T _I>
    inline constexpr _T __v<std::integral_constant<_T, _I>> = _I;

  template <bool>
    struct __i {
      template <template <class...> class _Fn, class... _Args>
        using __g = _Fn<_Args...>;
    };

  template <template <class...> class _Fn, class... _Args>
    using __meval =
      typename __i<(sizeof...(_Args) == ~0u)>::
        template __g<_Fn, _Args...>;

  template <class _Fn, class... _Args>
    using __minvoke =
      __meval<_Fn::template __f, _Args...>;

  template <class _Ty, class... _Args>
    using __make_dependent_on =
      typename __i<(sizeof...(_Args) == ~0u)>::
        template __g<__id, _Ty>;

  template <template <class...> class _Fn>
    struct __q {
      template <class... _Args>
        using __f = __meval<_Fn, _Args...>;
    };

  template <template <class...> class _Fn, class... _Front>
    struct __mbind_front_q {
      template <class... _Args>
        using __f = __meval<_Fn, _Front..., _Args...>;
    };

  template <class _Fn, class... _Front>
    using __mbind_front = __mbind_front_q<_Fn::template __f, _Front...>;

  template <template <class...> class _Fn, class... _Back>
    struct __mbind_back_q {
      template <class... _Args>
        using __f = __meval<_Fn, _Args..., _Back...>;
    };

  template <class _Fn, class... _Back>
    using __mbind_back = __mbind_back_q<_Fn::template __f, _Back...>;

  template <template <class...> class _T, class... _Args>
    concept __valid =
      requires {
        typename __meval<_T, _Args...>;
      };

  template <class _Fn, class... _Args>
    concept __minvocable =
      __valid<_Fn::template __f, _Args...>;

  template <bool>
    struct __if_ {
      template <class _True, class>
        using __f = _True;
    };
  template <>
    struct __if_<false> {
      template <class, class _False>
        using __f = _False;
    };
  #if STDEXEC_NVHPC()
  template <class _Pred, class _True, class _False>
    using __if = __minvoke<__if_<_Pred::value>, _True, _False>;
  #else
  template <class _Pred, class _True, class _False>
    using __if = __minvoke<__if_<__v<_Pred>>, _True, _False>;
  #endif
  template <bool _Pred, class _True, class _False>
    using __if_c = __minvoke<__if_<_Pred>, _True, _False>;

  template <class _T>
    struct __mconst {
      template <class...>
        using __f = _T;
    };

  template <class _Fn, class _Default>
    struct __with_default {
      template <class... _Args>
        using __f =
          __minvoke<
            __if_c<__minvocable<_Fn, _Args...>, _Fn, __mconst<_Default>>,
            _Args...>;
    };

  template <class _Fn, class _Continuation = __q<__types>>
    struct __transform {
      template <class... _Args>
        using __f = __minvoke<_Continuation, __minvoke<_Fn, _Args>...>;
    };

  template <class _Fn, class...>
    struct __fold_right_ {};
  template <class _Fn, class _State, class _Head, class... _Tail>
      requires __minvocable<_Fn, _State, _Head>
    struct __fold_right_<_Fn, _State, _Head, _Tail...>
      : __fold_right_<_Fn, __minvoke<_Fn, _State, _Head>, _Tail...> {};
  template <class _Fn, class _State>
    struct __fold_right_<_Fn, _State> {
      using __t = _State;
    };

  template <class _Init, class _Fn>
    struct __fold_right {
      template <class... _Args>
        using __f = __t<__fold_right_<_Fn, _Init, _Args...>>;
    };

  template <class _Continuation, class...>
    struct __concat_ {};
  template <class _Continuation, class... _As>
      requires (sizeof...(_As) == 0) &&
        __minvocable<_Continuation, _As...>
    struct __concat_<_Continuation, _As...> {
      using __t = __minvoke<_Continuation, _As...>;
    };
  template <class _Continuation, template <class...> class _A, class... _As>
      requires __minvocable<_Continuation, _As...>
    struct __concat_<_Continuation, _A<_As...>> {
      using __t = __minvoke<_Continuation, _As...>;
    };
  template <class _Continuation,
            template <class...> class _A, class... _As,
            template <class...> class _B, class... _Bs,
            class... _Tail>
    struct __concat_<_Continuation, _A<_As...>, _B<_Bs...>, _Tail...>
      : __concat_<_Continuation, __types<_As..., _Bs...>, _Tail...> {};
  template <class _Continuation,
            template <class...> class _A, class... _As,
            template <class...> class _B, class... _Bs,
            template <class...> class _C, class... _Cs,
            class... _Tail>
    struct __concat_<_Continuation, _A<_As...>, _B<_Bs...>, _C<_Cs...>, _Tail...>
      : __concat_<_Continuation, __types<_As..., _Bs..., _Cs...>, _Tail...> {};
  template <class _Continuation,
            template <class...> class _A, class... _As,
            template <class...> class _B, class... _Bs,
            template <class...> class _C, class... _Cs,
            template <class...> class _D, class... _Ds,
            class... _Tail>
    struct __concat_<_Continuation, _A<_As...>, _B<_Bs...>, _C<_Cs...>, _D<_Ds...>, _Tail...>
      : __concat_<_Continuation, __types<_As..., _Bs..., _Cs..., _Ds...>, _Tail...> {};

  template <class _Continuation = __q<__types>>
    struct __concat {
      template <class... _Args>
        using __f = __t<__concat_<_Continuation, _Args...>>;
    };

  template <class _Fn>
    struct __curry {
      template <class... _Ts>
        using __f = __minvoke<_Fn, _Ts...>;
    };

  template <class _Fn, class _T>
    struct __uncurry_;
  template <class _Fn, template <class...> class _A, class... _As>
      requires __minvocable<_Fn, _As...>
    struct __uncurry_<_Fn, _A<_As...>> {
      using __t = __minvoke<_Fn, _As...>;
    };
  template <class _Fn>
    struct __uncurry {
      template <class _T>
        using __f = __t<__uncurry_<_Fn, _T>>;
    };
  template <class _Fn, class _List>
    using __mapply =
      __minvoke<__uncurry<_Fn>, _List>;

  struct __mcount {
    template <class... _Ts>
      using __f = std::integral_constant<std::size_t, sizeof...(_Ts)>;
  };

  template <class _Fn>
    struct __mcount_if {
      template <class... _Ts>
        using __f =
          std::integral_constant<std::size_t, (bool(__minvoke<_Fn, _Ts>::value) + ...)>;
    };

  template <class _T>
    struct __contains {
      template <class... _Args>
        using __f = __bool<(__v<std::is_same<_T, _Args>> ||...)>;
    };

  template <class _Continuation = __q<__types>>
    struct __push_back {
      template <class _List, class _Item>
        using __f =
          __mapply<__mbind_back<_Continuation, _Item>, _List>;
    };

  template <class _Continuation = __q<__types>>
    struct __push_back_unique {
      template <class _List, class _Item>
        using __f =
          __mapply<
            __if<
              __mapply<__contains<_Item>, _List>,
              _Continuation,
              __mbind_back<_Continuation, _Item>>,
            _List>;
    };

  template <class _Continuation = __q<__types>>
    struct __munique {
      template <class... _Ts>
        using __f =
          __mapply<
            _Continuation,
            __minvoke<__fold_right<__types<>, __push_back_unique<>>, _Ts...>>;
    };

  template <class...>
    struct __mcompose {};

  template <class _First>
    struct __mcompose<_First> : _First {};

  template <class _Second, class _First>
    struct __mcompose<_Second, _First> {
      template <class... _Args>
        using __f = __minvoke<_Second, __minvoke<_First, _Args...>>;
    };

  template <class _Last, class _Penultimate, class... _Rest>
    struct __mcompose<_Last, _Penultimate, _Rest...> {
      template <class... _Args>
        using __f =
          __minvoke<_Last, __minvoke<__mcompose<_Penultimate, _Rest...>, _Args...>>;
    };

  template <class _Old, class _New, class _Continuation = __q<__types>>
    struct __replace {
      template <class... _Args>
        using __f =
          __minvoke<_Continuation, __if<std::is_same<_Args, _Old>, _New, _Args>...>;
    };

  template <class _Old, class _Continuation = __q<__types>>
    struct __remove {
      template <class... _Args>
        using __f =
          __minvoke<
            __concat<_Continuation>,
            __if<std::is_same<_Args, _Old>, __types<>, __types<_Args>>...>;
    };

  template <class _Return>
    struct __qf {
      template <class... _Args>
        using __f = _Return(_Args...);
    };

  template <class _T>
    _T&& __declval() noexcept requires true;

  template <class>
    void __declval() noexcept;

  // For copying cvref from one type to another:
  template <class _Member, class _Self>
    _Member _Self::*__memptr(const _Self&);

  template <typename _Self, typename _Member>
    using __member_t = decltype(
      (__declval<_Self>() .* __memptr<_Member>(__declval<_Self>())));

  template <class... _As>
    struct __front_;
  template <class _A, class... _As>
    struct __front_<_A, _As...> {
      using __t = _A;
    };
  template <class... _As>
      requires (sizeof...(_As) != 0)
    using __front = __t<__front_<_As...>>;
  template <class... _As>
      requires (sizeof...(_As) == 1)
    using __single = __front<_As...>;
  template <class _Ty>
    struct __single_or {
      template <class... _As>
          requires (sizeof...(_As) <= 1)
        using __f = __front<_As..., _Ty>;
    };

  template <class _Fun, class... _As>
    concept __callable =
      requires (_Fun&& __fun, _As&&... __as) {
        ((_Fun&&) __fun)((_As&&) __as...);
      };
  template <class _Fun, class... _As>
    concept __nothrow_callable =
      __callable<_Fun, _As...> &&
      requires (_Fun&& __fun, _As&&... __as) {
        { ((_Fun&&) __fun)((_As&&) __as...) } noexcept;
      };
  template <class _Fun, class... _As>
    using __call_result_t = decltype(__declval<_Fun>()(__declval<_As>()...));

  // For working around clang's lack of support for CWG#2369:
  // http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#2369
  struct __qcall_result {
    template <class _Fun, class... _As>
      using __f = __call_result_t<_Fun, _As...>;
  };
  template <bool _Enable, class _Fun, class... _As>
    using __call_result_if_t =
      typename __if<__bool<_Enable>, __qcall_result, __>
        ::template __f<_Fun, _As...>;

  // For emplacing non-movable types into optionals:
  template <class _Fn>
      requires std::is_nothrow_move_constructible_v<_Fn>
    struct __conv {
      _Fn __fn_;
      using __t = __call_result_t<_Fn>;
      operator __t() && noexcept(__nothrow_callable<_Fn>) {
        return ((_Fn&&) __fn_)();
      }
      __t operator()() && noexcept(__nothrow_callable<_Fn>) {
        return ((_Fn&&) __fn_)();
      }
    };
  template <class _Fn>
    __conv(_Fn) -> __conv<_Fn>;

  template <class _T>
    using __cref_t = const std::remove_reference_t<_T>&;

  template <class, class, class, class>
    struct __mzip_with2_;
  template <class _Fn, class _Continuation,
            template <class...> class _C, class... _Cs,
            template <class...> class _D, class... _Ds>
      requires requires {
        typename __minvoke<_Continuation, __minvoke<_Fn, _Cs, _Ds>...>;
      }
    struct __mzip_with2_<_Fn, _Continuation, _C<_Cs...>, _D<_Ds...>> {
      using __t = __minvoke<_Continuation, __minvoke<_Fn, _Cs, _Ds>...>;
    };

  template <class _Fn, class _Continuation = __q<__types>>
    struct __mzip_with2 {
      template <class _C, class _D>
        using __f = __t<__mzip_with2_<_Fn, _Continuation, _C, _D>>;
    };

  template <std::size_t... _Indices>
    auto __mconvert_indices(std::index_sequence<_Indices...>)
      -> __types<__index<_Indices>...>;
  template <std::size_t _N>
    using __mmake_index_sequence =
      decltype(__mconvert_indices(std::make_index_sequence<_N>{}));
  template <class... _Ts>
    using __mindex_sequence_for =
      __mmake_index_sequence<sizeof...(_Ts)>;

  template <class _Fn, class _Continuation, class... _Args>
    struct __mfind_if_ {
      using __t = __minvoke<_Continuation, _Args...>;
    };
  template <class _Fn, class _Continuation, class _Head, class... _Tail>
    struct __mfind_if_<_Fn, _Continuation, _Head, _Tail...>
      : __mfind_if_<_Fn, _Continuation, _Tail...>
    {};
  template <class _Fn, class _Continuation, class _Head, class... _Tail>
      requires __v<__minvoke<_Fn, _Head>>
    struct __mfind_if_<_Fn, _Continuation, _Head, _Tail...> {
      using __t = __minvoke<_Continuation, _Head, _Tail...>;
    };
  template <class _Fn, class _Continuation = __q<__types>>
    struct __mfind_if {
      template <class... _Args>
        using __f = __t<__mfind_if_<_Fn, _Continuation, _Args...>>;
    };

  template <class _Fn>
    struct __mfind_if_i {
      template <class... _Args>
        using __f =
          __index<(
            sizeof...(_Args) -
              __v<__minvoke<__mfind_if<_Fn, __mcount>, _Args...>>)>;
    };

  template <class... _Bools>
    using __mand = __bool<(__v<_Bools> &&...)>;
  template <class... _Bools>
    using __mor = __bool<(__v<_Bools> ||...)>;
  template <class _Bool>
    using __mnot = __bool<!__v<_Bool>>;

  template <class _Fn>
    struct __mall_of {
      template <class... _Args>
        using __f = __mand<__minvoke<_Fn, _Args>...>;
    };
  template <class _Fn>
    struct __mnone_of {
      template <class... _Args>
        using __f = __mand<__mnot<__minvoke<_Fn, _Args>>...>;
    };
  template <class _Fn>
    struct __many_of {
      template <class... _Args>
        using __f = __mor<__minvoke<_Fn, _Args>...>;
    };

  template <class _Ty>
    struct __mtypeof__ {
      using __t = _Ty;
    };
  template <class _Ty>
    using __mtypeof__t = __t<__mtypeof__<_Ty>>;
  template <class _Ret, class... _Args>
      requires __callable<_Ret, __mtypeof__t<_Args>...>
    struct __mtypeof__<_Ret(*)(_Args...)> {
      using __t = __call_result_t<_Ret, __mtypeof__t<_Args>...>;
    };
  template <class _Ty>
    struct __mtypeof_
    {};
  template <class _Ret, class... _Args>
      requires __callable<_Ret, __mtypeof__t<_Args>...>
    struct __mtypeof_<_Ret(_Args...)> {
      using __t = __call_result_t<_Ret, __mtypeof__t<_Args>...>;
    };
  template <class _Ty>
    using __mtypeof = __t<__mtypeof_<_Ty>>;

  template <class _Ty>
    using __mrequires =
      __bool<__valid<__mtypeof, _Ty>>;
  template <class _Ty>
    concept __mrequires_v =
      __valid<__mtypeof, _Ty>;

  template <class _Ty>
    inline constexpr bool __mnoexcept__ = true;
  template <class _Ret, class... _Args>
      requires __callable<_Ret, __mtypeof__t<_Args>...>
    inline constexpr bool __mnoexcept__<_Ret(*)(_Args...)> =
      (__mnoexcept__<_Args> &&...) &&
      __nothrow_callable<_Ret, __mtypeof__t<_Args>...>;
  template <class _Ty>
    inline constexpr bool __mnoexcept_v = false;
  template <class _Ret, class... _Args>
      requires __callable<_Ret, __mtypeof__t<_Args>...>
    inline constexpr bool __mnoexcept_v<_Ret(_Args...)> =
      (__mnoexcept__<_Args> &&...) &&
      __nothrow_callable<_Ret, __mtypeof__t<_Args>...>;
  template <class _Ty>
    using __mnoexcept = __bool<__mnoexcept_v<_Ty>>;

  template <class... _Sigs>
    struct __msignatures {
      template <class _Continuation, class... _Extra>
        using __f = __minvoke<_Continuation, _Sigs..., _Extra...>;
    };
  template <class _Signatures>
    using __many_well_formed =
      __minvoke<_Signatures, __many_of<__q<__mrequires>>>;
  template <class _Signatures, class... _Extra>
    using __mwhich_t =
      __minvoke<
        _Signatures,
        __mfind_if<__q<__mrequires>, __q<__front>>,
        _Extra...>;
  template <class _Signatures, class... _Extra>
    using __mwhich_i =
      __index<(
        __v<__minvoke<_Signatures, __mcount, _Extra...>> -
        __v<__minvoke<
          _Signatures,
          __mfind_if<__q<__mrequires>, __mcount>,
          _Extra...>>)>;
  template <class _Ty, bool _Noexcept = true>
    struct __mconstruct {
      template <class... _As>
        auto operator()(_As&&... __as) const
          noexcept(_Noexcept && noexcept(_Ty((_As&&) __as...)))
          -> decltype(_Ty((_As&&) __as...));
    };
}
