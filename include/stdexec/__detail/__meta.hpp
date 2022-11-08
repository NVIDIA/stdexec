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
    constexpr __ignore(auto&&...) noexcept {}
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

  template <bool _B>
    using __bool = std::bool_constant<_B>;

  template <class _Ty>
    struct __mtype {
      using __t = _Ty;
    };

  // Some utilities for manipulating lists of types at compile time
  template <class...>
    struct __types;

  template <class _T>
    using __midentity = _T;

  template <std::size_t _N>
    using __msize_t = char[_N+1];

  template <class _T>
    inline constexpr auto __v = _T::value;

  template <class _T, class _U>
    inline constexpr bool __v<std::is_same<_T, _U>> = false;

  template <class _T>
    inline constexpr bool __v<std::is_same<_T, _T>> = true;

  template <class _T, _T _I>
    inline constexpr _T __v<std::integral_constant<_T, _I>> = _I;

  template <std::size_t _I>
    inline constexpr std::size_t __v<char[_I]> = _I-1;

  template <bool>
    struct __i {
      template <template <class...> class _Fn, class... _Args>
        using __g = _Fn<_Args...>;
    };

  template <class...>
    concept __tru = true; // a dependent value

  template <template <class...> class _Fn, class... _Args>
    using __meval =
      typename __i<__tru<_Args...>>::
        template __g<_Fn, _Args...>;

  template <class _Fn, class... _Args>
    using __minvoke =
      __meval<_Fn::template __f, _Args...>;

  template <class _Ty, class... _Args>
    using __make_dependent_on =
      typename __i<__tru<_Args...>>::
        template __g<__midentity, _Ty>;

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

  template <class _Fn, class... _Args>
    struct __force_minvoke_ {
      using __t = __minvoke<_Fn, _Args...>;
    };
  template <class _Fn, class... _Args>
    using __force_minvoke = __t<__force_minvoke_<_Fn, _Args...>>;

  template <bool>
    struct __if_ {
      template <class _True, class...>
        using __f = _True;
    };
  template <>
    struct __if_<false> {
      template <class, class _False>
        using __f = _False;
    };
  #if STDEXEC_NVHPC()
  template <class _Pred, class _True, class... _False>
      requires (sizeof...(_False) <= 1)
    using __if = __minvoke<__if_<_Pred::value>, _True, _False...>;
  #else
  template <class _Pred, class _True, class... _False>
      requires (sizeof...(_False) <= 1)
    using __if = __minvoke<__if_<__v<_Pred>>, _True, _False...>;
  #endif
  template <bool _Pred, class _True, class... _False>
      requires (sizeof...(_False) <= 1)
    using __if_c = __minvoke<__if_<_Pred>, _True, _False...>;

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

  template <bool>
    struct __mfold_right_ {
      template <class _Fn, class _State, class _Head, class... _Tail>
        using __f =
          __minvoke<
            __mfold_right_<sizeof...(_Tail) == 0>,
            _Fn,
            __minvoke<_Fn, _State, _Head>,
            _Tail...>;
    };
  template <>
    struct __mfold_right_<true> { // empty pack
      template <class _Fn, class _State, class...>
        using __f = _State;
    };

  template <class _Init, class _Fn>
    struct __mfold_right {
      template <class... _Args>
        using __f =
          __minvoke<__mfold_right_<sizeof...(_Args) == 0>, _Fn, _Init, _Args...>;
    };

  template <class _Continuation, class...>
    struct __mconcat_ {};
  template <class _Continuation, class... _As>
      requires (sizeof...(_As) == 0) &&
        __minvocable<_Continuation, _As...>
    struct __mconcat_<_Continuation, _As...> {
      using __t = __minvoke<_Continuation, _As...>;
    };
  template <class _Continuation, template <class...> class _A, class... _As>
      requires __minvocable<_Continuation, _As...>
    struct __mconcat_<_Continuation, _A<_As...>> {
      using __t = __minvoke<_Continuation, _As...>;
    };
  template <class _Continuation,
            template <class...> class _A, class... _As,
            template <class...> class _B, class... _Bs>
        requires __minvocable<_Continuation, _As..., _Bs...>
    struct __mconcat_<_Continuation, _A<_As...>, _B<_Bs...>> {
      using __t = __minvoke<_Continuation, _As..., _Bs...>;
    };
  template <class _Continuation,
            template <class...> class _A, class... _As,
            template <class...> class _B, class... _Bs,
            template <class...> class _C, class... _Cs>
        requires __minvocable<_Continuation, _As..., _Bs..., _Cs...>
    struct __mconcat_<_Continuation, _A<_As...>, _B<_Bs...>, _C<_Cs...>> {
      using __t = __minvoke<_Continuation, _As..., _Bs..., _Cs...>;
    };
  template <class _Continuation,
            template <class...> class _A, class... _As,
            template <class...> class _B, class... _Bs,
            template <class...> class _C, class... _Cs,
            template <class...> class _D, class... _Ds,
            class... _Tail>
    struct __mconcat_<_Continuation, _A<_As...>, _B<_Bs...>, _C<_Cs...>, _D<_Ds...>, _Tail...>
      : __mconcat_<_Continuation, __types<_As..., _Bs..., _Cs..., _Ds...>, _Tail...> {};

  template <class _Continuation = __q<__types>>
    struct __mconcat {
      template <class... _Args>
        using __f = __t<__mconcat_<_Continuation, _Args...>>;
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

  struct __msize {
    template <class... _Ts>
      using __f = __msize_t<sizeof...(_Ts)>;
  };

  template <class _Ty>
    struct __mcount {
      template <class... _Ts>
        using __f = __msize_t<(__v<std::is_same<_Ts, _Ty>> + ... + 0)>;
    };

  template <class _Fn>
    struct __mcount_if {
      template <class... _Ts>
        using __f = __msize_t<(bool(__v<__minvoke<_Fn, _Ts>>) + ... + 0)>;
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
            __minvoke<__mfold_right<__types<>, __push_back_unique<>>, _Ts...>>;
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
            __mconcat<_Continuation>,
            __if<std::is_same<_Args, _Old>, __types<>, __types<_Args>>...>;
    };

  template <class _Return>
    struct __qf {
      template <class... _Args>
        using __f = _Return(_Args...);
    };

  // A very simple std::declval replacement that doesn't handle void
  template <class _T>
    _T&& __declval() noexcept;

  // For copying cvref from one type to another:
  struct __cp {
    template <class _T>
      using __f = _T;
  };
  struct __cpc {
    template <class _T>
      using __f = const _T;
  };
  struct __cplr {
    template <class _T>
      using __f = _T&;
  };
  struct __cprr {
    template <class _T>
      using __f = _T&&;
  };
  struct __cpclr {
    template <class _T>
      using __f = const _T&;
  };
  struct __cpcrr {
    template <class _T>
      using __f = const _T&&;
  };

  template <class>
    extern __cp __cpcvr;
  template <class _T>
    extern __cpc __cpcvr<const _T>;
  template <class _T>
    extern __cplr __cpcvr<_T&>;
  template <class _T>
    extern __cprr __cpcvr<_T&&>;
  template <class _T>
    extern __cpclr __cpcvr<const _T&>;
  template <class _T>
    extern __cpcrr __cpcvr<const _T&&>;
  template <class _T>
    using __copy_cvref_fn = decltype(__cpcvr<_T>);

  template <class _From, class _To>
    using __copy_cvref_t = __minvoke<__copy_cvref_fn<_From>, _To>;

  template <class _Ty, class...>
    using __mfront_ = _Ty;
  template <class... _As>
    using __mfront = __meval<__mfront_, _As...>;
  template <class... _As>
      requires (sizeof...(_As) == 1)
    using __msingle = __mfront<_As...>;
  template <class _Ty>
    using __msingle_or = __mbind_back_q<__mfront_, _Ty>;

  template <class _Continuation = __q<__types>>
    struct __pop_front {
      template <class, class... _Ts>
        using __f = __minvoke<_Continuation, _Ts...>;
    };

  // For hiding a template type parameter from ADL
  template <class _Ty>
    struct _X {
      using __t = struct _T {
        using __t = _Ty;
      };
    };
  template <class _Ty>
    using __x = __t<_X<_Ty>>;

  template <class _Ty>
    struct _Y {
      using __t = _Ty;
    };

  template <class _Ty>
    concept __has_id =
      requires {
        typename _Ty::__id;
      };
  template <bool = true>
    struct __id_ {
      template <class _Ty>
        using __f = typename _Ty::__id;
    };
  template <>
    struct __id_<false> {
      template <class _Ty>
        using __f = _Y<_Ty>;
    };
  template <class _Ty>
    using __id = __minvoke<__id_<__has_id<_Ty>>, _Ty>;

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

#if STDEXEC_GCC() && (__GNUC__ < 12)
  template <class>
    extern int __mconvert_indices;
  template <std::size_t... _Indices>
    extern __types<__msize_t<_Indices>...> __mconvert_indices<std::index_sequence<_Indices...>>;
  template <std::size_t _N>
    using __mmake_index_sequence =
      decltype(stdexec::__mconvert_indices<std::make_index_sequence<_N>>);
#else
  template <std::size_t... _Indices>
    __types<__msize_t<_Indices>...> __mconvert_indices(std::index_sequence<_Indices...>*);
  template <std::size_t _N>
    using __mmake_index_sequence =
      decltype(stdexec::__mconvert_indices((std::make_index_sequence<_N>*) nullptr));
#endif

  template <class... _Ts>
    using __mindex_sequence_for =
      __mmake_index_sequence<sizeof...(_Ts)>;

  template <bool>
    struct __mfind_if_ {
      template <class _Fn, class _Continuation, class _Head, class... _Tail>
        using __f =
          __minvoke<
            __if_c<
              __v<__minvoke<_Fn, _Head>>,
              __mbind_front<_Continuation, _Head>,
              __mbind_front<__mfind_if_<(sizeof...(_Tail) != 0)>, _Fn, _Continuation>>,
            _Tail...>;
      };
  template <>
    struct __mfind_if_<false> {
      template <class _Fn, class _Continuation>
        using __f = __minvoke<_Continuation>;
      };

  template <class _Fn, class _Continuation = __q<__types>>
    struct __mfind_if {
      template <class... _Args>
        using __f =
          __minvoke<__mfind_if_<(sizeof...(_Args) != 0)>, _Fn, _Continuation, _Args...>;
    };

  template <class _Fn>
    struct __mfind_if_i {
      template <class... _Args>
        using __f =
          __msize_t<(
            sizeof...(_Args) -
              __v<__minvoke<__mfind_if<_Fn, __msize>, _Args...>>)>;
    };

  template <class... _Booleans>
    using __mand = __bool<(__v<_Booleans> &&...)>;
  template <class... _Booleans>
    using __mor = __bool<(__v<_Booleans> ||...)>;
  template <class _Boolean>
    using __mnot = __bool<!__v<_Boolean>>;

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

#if __has_builtin(__type_pack_element)
  template <std::size_t _N, class... _Ts>
    using __m_at = __type_pack_element<_N, _Ts...>;
#else
  template <unsigned _N>
    struct __m_at_ {
      template <class, class, class, class, class... _Ts>
        using __f = __minvoke<__m_at_<_N-4>, _Ts...>;
    };
  template <>
    struct __m_at_<0> {
      template <class _Ty, class...>
        using __f = _Ty;
    };
  template <>
    struct __m_at_<1> {
      template <class, class _Ty, class...>
        using __f = _Ty;
    };
  template <>
    struct __m_at_<2> {
      template <class, class, class _Ty, class...>
        using __f = _Ty;
    };
  template <>
    struct __m_at_<3> {
      template <class, class, class, class _Ty, class...>
        using __f = _Ty;
    };

  template <unsigned _N, class... _Ts>
    using __m_at = __minvoke<__m_at_<_N>, _Ts...>;
#endif

  template <std::size_t _N>
    struct __placeholder_;
  template <std::size_t _N>
    using __placeholder = __placeholder_<_N>*;

  using __0 = __placeholder<0>;
  using __1 = __placeholder<1>;
  using __2 = __placeholder<2>;
  using __3 = __placeholder<3>;

  template <class _Ty, class _Noexcept = __bool<true>>
    struct __mconstruct {
      template <class... _As>
        auto operator()(_As&&... __as) const
          noexcept(__v<_Noexcept> && noexcept(_Ty((_As&&) __as...)))
          -> decltype(_Ty((_As&&) __as...)) {
          return _Ty((_As&&) __as...);
        }
    };

  template <template <class...> class _C, class _Noexcept = __bool<true>>
    using __mconstructor_for =
      __mcompose<__q<__mconstruct>, __q<_C>>;

  template <std::size_t>
    using __ignore_t = __ignore;

  template <std::size_t... _Is, class _Ty, class... _Us>
    _Ty&& __nth_pack_element_(__ignore_t<_Is>..., _Ty&& __t, _Us&&...) noexcept {
      return (_Ty&&) __t;
    }
  template <std::size_t _N, class... _Ts>
    constexpr decltype(auto) __nth_pack_element(_Ts&&... __ts) noexcept {
      return [&]<std::size_t... _Is>(std::index_sequence<_Is...>) noexcept -> decltype(auto) {
        return stdexec::__nth_pack_element_<_Is...>((_Ts&&) __ts...);
      }(std::make_index_sequence<_N>());
    }

  template <class _Ty>
    struct __mdispatch_ {
      template <class... _Ts>
        _Ty operator()(_Ts&&...) const noexcept(noexcept(_Ty{})) {
          return _Ty{};
        }
    };
  template <std::size_t _N>
    struct __mdispatch_<__placeholder<_N>> {
      template <class... _Ts>
        decltype(auto) operator()(_Ts&&... __ts) const noexcept {
          return stdexec::__nth_pack_element<_N>((_Ts&&) __ts...);
        }
    };
  template <std::size_t _N>
    struct __mdispatch_<__placeholder<_N>&> {
      template <class... _Ts>
        decltype(auto) operator()(_Ts&&... __ts) const noexcept {
          return stdexec::__nth_pack_element<_N>(__ts...);
        }
    };
  template <std::size_t _N>
    struct __mdispatch_<__placeholder<_N>&&> {
      template <class... _Ts>
        decltype(auto) operator()(_Ts&&... __ts) const noexcept {
          return std::move(stdexec::__nth_pack_element<_N>(__ts...));
        }
    };
  template <std::size_t _N>
    struct __mdispatch_<const __placeholder<_N>&> {
      template <class... _Ts>
        decltype(auto) operator()(_Ts&&... __ts) const noexcept {
          return std::as_const(stdexec::__nth_pack_element<_N>(__ts...));
        }
    };
  template <class _Ret, class... _Args>
    struct __mdispatch_<_Ret(*)(_Args...)> {
      template <class... _Ts>
          requires (__callable<__mdispatch_<_Args>, _Ts...> &&...) &&
            __callable<_Ret, __call_result_t<__mdispatch_<_Args>, _Ts...>...>
        auto operator()(_Ts&&... __ts) const
          noexcept(__nothrow_callable<_Ret, __call_result_t<__mdispatch_<_Args>, _Ts...>...>)
          -> __call_result_t<_Ret, __call_result_t<__mdispatch_<_Args>, _Ts...>...> {
          return _Ret{}(__mdispatch_<_Args>{}((_Ts&&) __ts...)...);
        }
    };

  template <class _Ty>
    struct __mdispatch
    {};
  template <class _Ret, class... _Args>
    struct __mdispatch<_Ret(_Args...)> {
      template <class... _Ts>
          requires (__callable<__mdispatch_<_Args>, _Ts...> &&...) &&
            __callable<_Ret, __call_result_t<__mdispatch_<_Args>, _Ts...>...>
        auto operator()(_Ts&&... __ts) const
          noexcept(__nothrow_callable<_Ret, __call_result_t<__mdispatch_<_Args>, _Ts...>...>)
          -> __call_result_t<_Ret, __call_result_t<__mdispatch_<_Args>, _Ts...>...> {
          return _Ret{}(__mdispatch_<_Args>{}((_Ts&&) __ts...)...);
        }
    };
  template <class _Ty, class... _Ts>
    concept __dispatchable = __callable<__mdispatch<_Ty>, _Ts...>;

  template <class _Ty, class... _Ts>
    concept __nothrow_dispatchable = __nothrow_callable<__mdispatch<_Ty>, _Ts...>;

  template <class _Ty, class... _Ts>
    using __dispatch_result_t = __call_result_t<__mdispatch<_Ty>, _Ts...>;

  template <class _Signature, class... _Args>
    using __try_dispatch_ =
      __bool<__dispatchable<_Signature, _Args...>>;
  template <class _Signatures, class _Continuation = __q<__mfront>>
    struct __which
    {};
  template <template <class...> class _C, class... _Signatures, class _Continuation>
    struct __which<_C<_Signatures...>, _Continuation> {
      template <class... _Args>
        using __f =
          __minvoke<
            __mfind_if<__mbind_back_q<__try_dispatch_, _Args...>, _Continuation>,
            _Signatures...>;
    };

  template <class _Signatures, class _DefaultFn, class... _Args>
    using __make_dispatcher =
      __minvoke<
        __if_c<
          __minvocable<__which<_Signatures>, _Args...>,
          __mcompose<__q<__mdispatch>, __which<_Signatures>>,
          _DefaultFn>,
        _Args...>;
} // namespace stdexec
