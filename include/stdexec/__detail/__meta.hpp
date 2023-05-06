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
#include "__type_traits.hpp"
#include "__concepts.hpp"

namespace stdexec {

  template <class...>
  struct __undefined;

  struct __ { };

  struct __ignore {
    __ignore() = default;

    constexpr __ignore(auto&&...) noexcept {
    }
  };

  struct __none_such { };

  struct __immovable {
    __immovable() = default;
   private:
    STDEXEC_IMMOVABLE(__immovable);
  };

  struct __move_only {
    __move_only() = default;

    __move_only(__move_only&&) noexcept = default;
    __move_only& operator=(__move_only&&) noexcept = default;

    __move_only(const __move_only&) = delete;
    __move_only& operator=(const __move_only&) = delete;
  };

  template <class _Tp>
  using __t = typename _Tp::__t;

  template <bool _Bp>
  using __mbool = std::bool_constant<_Bp>;

  template <class _Ty>
  struct __mtype {
    using __t = _Ty;
  };

  template <auto _Value>
  using __mtypeof = decltype(_Value);

  template <class...>
  struct __types;

  template <class _Tp>
  using __midentity = _Tp;

  template <std::size_t _Np>
  using __msize_t = char[_Np + 1];

  template <class _Tp, class _Up>
  using __mfirst = _Tp;

  template <class _Tp, class _UXp>
  using __msecond = _UXp;

  template <class _Tp>
  extern const __undefined<_Tp> __v;

  template <class _Tp>
    requires __typename<__mtypeof<_Tp::value>>
  inline constexpr auto __v<_Tp> = _Tp::value;

  template <class _Tp, class _Up>
  inline constexpr bool __v<std::is_same<_Tp, _Up>> = false;

  template <class _Tp>
  inline constexpr bool __v<std::is_same<_Tp, _Tp>> = true;

  template <class _Tp, _Tp _Ip>
  inline constexpr _Tp __v<std::integral_constant<_Tp, _Ip>> = _Ip;

  template <std::size_t _Ip>
  inline constexpr std::size_t __v<char[_Ip]> = _Ip - 1;

  template <std::size_t... _Is>
  using __indices = std::index_sequence<_Is...>*;

  template <std::size_t _Np>
  using __make_indices = std::make_index_sequence<_Np>*;

  template <class _Char>
  concept __mchar = __same_as<_Char, char>;

  template <std::size_t _Len>
  class __mstring {
    template <std::size_t... _Is>
    constexpr __mstring(const char (&__str)[_Len], __indices<_Is...>) noexcept
      : __what_{__str[_Is]...} {
    }

   public:
    constexpr __mstring(const char (&__str)[_Len]) noexcept
      : __mstring{__str, __make_indices<_Len>{}} {
    }

    template <__mchar... _Char>
      requires(sizeof...(_Char) == _Len)
    constexpr __mstring(_Char... __chars) noexcept
      : __what_{__chars...} {
    }

    static constexpr std::size_t __length() noexcept {
      return _Len;
    }

    char const __what_[_Len];
  };

#if STDEXEC_NVHPC() && (__EDG_VERSION__ < 604)
  // Use a non-standard extension for older nvc++ releases
  template <__mchar _Char, _Char... _Str>
  constexpr __mstring<sizeof...(_Str)> operator""__csz() noexcept {
    return {_Str...};
  }
#elif STDEXEC_NVHPC()
  // BUGBUG TODO This is to work around an unknown EDG bug
  template <__mstring _Str>
  constexpr auto operator""__csz() noexcept {
    return _Str;
  }
#else
  // Use a standard user-defined string literal template
  template <__mstring _Str>
  constexpr __mtypeof<_Str> operator""__csz() noexcept {
    return _Str;
  }
#endif

  using __msuccess = int;

  template <class _What, class... _With>
  struct _WARNING_ { };

  template <class _What, class... _With>
  struct _ERROR_ {
    const _ERROR_& operator,(__msuccess) const noexcept;
  };

  template <class _What, class... _With>
  using __mexception = const _ERROR_<_What, _With...>&;

  template <class>
  extern __msuccess __ok_v;

  template <class _What, class... _With>
  extern _ERROR_<_What, _With...> __ok_v<__mexception<_What, _With...>>;

  template <class _Ty>
  using __ok_t = decltype(__ok_v<_Ty>);

  template <class... _Ts>
  using __disp = const decltype((__msuccess(), ..., __ok_t<_Ts>()))&;

  template <bool _AllOK>
  struct __i {
    template <template <class...> class _Fn, class... _Args>
    using __g = _Fn<_Args...>;
  };

  template <>
  struct __i<false> {
    template <template <class...> class, class... _Args>
    using __g = __disp<_Args...>;
  };

  template <class _Arg>
  concept __ok = __same_as<__ok_t<_Arg>, __msuccess>;

  template <class _Arg>
  concept __merror = !__ok<_Arg>;

  template <class... _Args>
  concept _Ok = (__ok<_Args> && ...);

#if STDEXEC_NVHPC()
  // Most compilers memoize alias template specializations, but
  // nvc++ does not. So we memoize the type computations by
  // indirecting through a class template specialization.
  template <template <class...> class _Fn, class... _Args>
  using __meval__ = typename __i<_Ok<_Args...>>::template __g<_Fn, _Args...>;

  template <template <class...> class _Fn, class... _Args>
  struct __meval_ { };

  template <template <class...> class _Fn, class... _Args>
    requires __typename<__meval__<_Fn, _Args...>>
  struct __meval_<_Fn, _Args...> {
    using __t = __meval__<_Fn, _Args...>;
  };

  template <template <class...> class _Fn, class... _Args>
  using __meval = __t<__meval_<_Fn, _Args...>>;

#else

  template <template <class...> class _Fn, class... _Args>
  using __meval = typename __i<_Ok<_Args...>>::template __g<_Fn, _Args...>;

#endif

  template <class _Fn, class... _Args>
  using __minvoke = __meval<_Fn::template __f, _Args...>;

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

  template <template <class...> class _Tp, class... _Args>
  concept __valid = requires { typename __meval<_Tp, _Args...>; };

  template <template <class...> class _Tp, class... _Args>
  concept __invalid = !__valid<_Tp, _Args...>;

  template <class _Fn, class... _Args>
  concept __minvocable = __valid<_Fn::template __f, _Args...>;

  template <template <class...> class _Tp, class... _Args>
  concept __msucceeds = __valid<_Tp, _Args...> && __ok<__meval<_Tp, _Args...>>;

  template <class _Fn, class... _Args>
  concept __minvocable_succeeds = __minvocable<_Fn, _Args...> && __ok<__minvoke<_Fn, _Args...>>;

  template <class _Fn, class... _Args>
  struct __force_minvoke_ {
    using __t = __minvoke<_Fn, _Args...>;
  };
  template <class _Fn, class... _Args>
  using __force_minvoke = __t<__force_minvoke_<_Fn, _Args...>>;

  template <class _Fn, class... _Args>
  struct __mdefer_ { };

  template <class _Fn, class... _Args>
    requires __minvocable<_Fn, _Args...>
  struct __mdefer_<_Fn, _Args...> {
    using __t = __minvoke<_Fn, _Args...>;
  };

  template <class _Fn, class... _Args>
  struct __mdefer : __mdefer_<_Fn, _Args...> { };

  struct __if_ {
    template <bool>
    struct __ {
      template <class _True, class...>
      using __f = _True;
    };

    template <class _Pred, class _True, class... _False>
    using __f = __minvoke<__<static_cast<bool>(__v<_Pred>)>, _True, _False...>;
  };

  template <>
  struct __if_::__<false> {
    template <class, class _False>
    using __f = _False;
  };

  template <class _Pred, class _True = void, class... _False>
    requires(sizeof...(_False) <= 1)
  using __if = __minvoke<__if_, _Pred, _True, _False...>;

  template <bool _Pred, class _True = void, class... _False>
    requires(sizeof...(_False) <= 1)
  using __if_c = __minvoke<__if_::__<_Pred>, _True, _False...>;

  template <class _Tp>
  struct __mconst {
    template <class...>
    using __f = _Tp;
  };

  template <template <class...> class _Try, class _Catch>
  struct __mtry_catch_q {
    template <class... _Args>
    using __f = __minvoke< __if_c<__valid<_Try, _Args...>, __q<_Try>, _Catch>, _Args...>;
  };

  template <class _Try, class _Catch>
  struct __mtry_catch {
    template <class... _Args>
    using __f = __minvoke< __if_c<__minvocable<_Try, _Args...>, _Try, _Catch>, _Args...>;
  };

  template <class _Fn, class _Default>
  using __with_default = __mtry_catch<_Fn, __mconst<_Default>>;

  inline constexpr __mstring __mbad_substitution =
    "The specified meta-function could not be evaluated with the types provided."__csz;

  template <__mstring _Diagnostic = __mbad_substitution>
  struct _BAD_SUBSTITUTION_ { };

  template <class... _Args>
  struct _WITH_TYPES_ { };

  template <template <class...> class _Fun>
  struct _WITH_META_FUNCTION_T_ {
    template <class... _Args>
    using __f = __mexception<_BAD_SUBSTITUTION_<>, _WITH_META_FUNCTION_T_, _WITH_TYPES_<_Args...>>;
  };

  template <class _Fun>
  struct _WITH_META_FUNCTION_ {
    template <class... _Args>
    using __f = __mexception<_BAD_SUBSTITUTION_<>, _WITH_META_FUNCTION_, _WITH_TYPES_<_Args...>>;
  };

  template <template <class...> class _Fn>
  struct __mtry_eval_ {
    template <class... _Args>
    using __f = __meval<_Fn, _Args...>;
  };

  template <template <class...> class _Fn, class... _Args>
  using __mtry_eval =
    __minvoke<__mtry_catch<__mtry_eval_<_Fn>, _WITH_META_FUNCTION_T_<_Fn>>, _Args...>;

  template <class _Fn, class... _Args>
  using __mtry_invoke = __minvoke<__mtry_catch<_Fn, _WITH_META_FUNCTION_<_Fn>>, _Args...>;

  template <class _Ty, class... _Default>
  using __msuccess_or_t = __if_c<__ok<_Ty>, _Ty, _Default...>;

  template <class _Fn, class _Continuation = __q<__types>>
  struct __transform {
    template <class... _Args>
    using __f = __minvoke<_Continuation, __minvoke<_Fn, _Args>...>;
  };

  template <bool>
  struct __mfold_right_ {
    template <class _Fn, class _State, class _Head, class... _Tail>
    using __f =
      __minvoke< __mfold_right_<sizeof...(_Tail) == 0>, _Fn, __minvoke<_Fn, _State, _Head>, _Tail...>;
  };

  template <>
  struct __mfold_right_<true> { // empty pack
    template <class _Fn, class _State, class...>
    using __f = _State;
  };

  template <class _Init, class _Fn>
  struct __mfold_right {
    template <class... _Args>
    using __f = __minvoke<__mfold_right_<sizeof...(_Args) == 0>, _Fn, _Init, _Args...>;
  };

  // customization point
  template <class _Tp>
  struct __mexpand;

  template <template <class...> class _Ap, class... _As>
  struct __mexpand<_Ap<_As...>> {
    template <class _Fn, class... _Bs>
    using __f = __minvoke<_Fn, _Bs..., _As...>;
  };

  template <class _Continuation>
  struct __mconcat {
    template <class... _As>
    using __f = __minvoke<
      __minvoke<
        __transform< __q<__mexpand>, __mfold_right<__q<__minvoke>, __q<__mbind_front>>>,
        _As...>,
      _Continuation>;
  };

  template <class _Fn>
  struct __uncurry {
    template <class _Tp>
    using __f = __minvoke<__mexpand<_Tp>, _Fn>;
  };

  template <class _Fn, class _List>
  using __mapply = __minvoke<__mexpand<_List>, _Fn>;

  template <class _Fn, class _List>
  concept __mapplicable = __minvocable<__mexpand<_List>, _Fn>;

  struct __msize {
    template <class... _Ts>
    using __f = __msize_t<sizeof...(_Ts)>;
  };

  template <class _Ty>
  struct __mcount {
    template <class... _Ts>
    using __f = __msize_t<(__same_as<_Ts, _Ty> + ... + 0)>;
  };

  template <class _Fn>
  struct __mcount_if {
    template <class... _Ts>
    using __f = __msize_t<(bool(__v<__minvoke<_Fn, _Ts>>) + ... + 0)>;
  };

  template <class _Tp>
  struct __contains {
    template <class... _Args>
    using __f = __mbool<(__same_as<_Tp, _Args> || ...)>;
  };

  template <class _Continuation = __q<__types>>
  struct __push_back {
    template <class _List, class _Item>
    using __f = __mapply<__mbind_back<_Continuation, _Item>, _List>;
  };

  template <class _Continuation = __q<__types>>
  struct __push_back_unique {
    template <class _List, class _Item>
    using __f = //
      __mapply<
        __if< __mapply<__contains<_Item>, _List>, _Continuation, __mbind_back<_Continuation, _Item>>,
        _List>;
  };

  template <class _Continuation = __q<__types>>
  struct __munique {
    template <class... _Ts>
    using __f =
      __mapply< _Continuation, __minvoke<__mfold_right<__types<>, __push_back_unique<>>, _Ts...>>;
  };

  template <class...>
  struct __mcompose { };

  template <class _First>
  struct __mcompose<_First> : _First { };

  template <class _Second, class _First>
  struct __mcompose<_Second, _First> {
    template <class... _Args>
    using __f = __minvoke<_Second, __minvoke<_First, _Args...>>;
  };

  template <class _Last, class _Penultimate, class... _Rest>
  struct __mcompose<_Last, _Penultimate, _Rest...> {
    template <class... _Args>
    using __f = __minvoke<_Last, __minvoke<__mcompose<_Penultimate, _Rest...>, _Args...>>;
  };

  template <class _Old, class _New, class _Continuation = __q<__types>>
  struct __replace {
    template <class... _Args>
    using __f = __minvoke<_Continuation, __if_c<__same_as<_Args, _Old>, _New, _Args>...>;
  };

  template <class _Old, class _Continuation = __q<__types>>
  struct __remove {
    template <class... _Args>
    using __f = //
      __minvoke<
        __mconcat<_Continuation>,
        __if_c<__same_as<_Args, _Old>, __types<>, __types<_Args>>...>;
  };

  template <class _Pred, class _Continuation = __q<__types>>
  struct __remove_if {
    template <class... _Args>
    using __f = //
      __minvoke<
        __mconcat<_Continuation>,
        __if<__minvoke<_Pred, _Args>, __types<>, __types<_Args>>...>;
  };

  template <class _Return>
  struct __qf {
    template <class... _Args>
    using __f = _Return(_Args...);
  };

  template <class _Ty, class...>
  using __mfront_ = _Ty;
  template <class... _As>
  using __mfront = __meval<__mfront_, _As...>;
  template <class... _As>
    requires(sizeof...(_As) == 1)
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
  struct _Xp {
    using __t = struct _Up {
      using __t = _Ty;
    };
  };
  template <class _Ty>
  using __x = __t<_Xp<_Ty>>;

  template <class _Ty>
  concept __has_id = requires { typename _Ty::__id; };

  template <class _Ty>
  struct _Yp {
    using __t = _Ty;

    // Uncomment the line below to find any code that likely misuses the
    // ADL isolation mechanism. In particular, '__id<T>' when T is a
    // reference is a likely misuse. The static_assert below will trigger
    // when the type passed to the __id alias template is a reference to
    // a type that is setup to use ADL isolation.
    //static_assert(!__has_id<std::remove_cvref_t<_Ty>>);
  };

  template <bool = true>
  struct __id_ {
    template <class _Ty>
    using __f = typename _Ty::__id;
  };

  template <>
  struct __id_<false> {
    template <class _Ty>
    using __f = _Yp<_Ty>;
  };
  template <class _Ty>
  using __id = __minvoke<__id_<__has_id<_Ty>>, _Ty>;

  template <class _Ty>
  using __cvref_t = __copy_cvref_t<_Ty, __t<std::remove_cvref_t<_Ty>>>;

  template <class _From, class _To = __decay_t<_From>>
  using __cvref_id = __copy_cvref_t<_From, __id<_To>>;

#if 0 //STDEXEC_NVHPC()
  // nvc++ doesn't cache the results of alias template specializations.
  // To avoid repeated computation of the same function return type,
  // cache the result ourselves in a class template specialization.
  template <class _Fun, class... _As>
  using __call_result_ = decltype(__declval<_Fun>()(__declval<_As>()...));
  template <class _Fun, class... _As>
  using __call_result_t = __t<__mdefer<__q<__call_result_>, _Fun, _As...>>;
#else
  template <class _Fun, class... _As>
  using __call_result_t = decltype(__declval<_Fun>()(__declval<_As>()...));
#endif

  // For working around clang's lack of support for CWG#2369:
  // http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#2369
  struct __qcall_result {
    template <class _Fun, class... _As>
    using __f = __call_result_t<_Fun, _As...>;
  };
  template <bool _Enable, class _Fun, class... _As>
  using __call_result_if_t = __minvoke<__if<__mbool<_Enable>, __qcall_result, __>, _Fun, _As...>;

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

  // Implemented as a class instead of a free function
  // because of a bizarre nvc++ compiler bug:
  struct __cref_fn {
    template <class _Ty>
    const _Ty& operator()(const _Ty&);
  };
  template <class _Ty>
  using __cref_t = decltype(__cref_fn()(__declval<_Ty>()));

  // clang-format off
  template <class... _Lists>
    requires (__mapplicable<__q<__types>, _Lists> &&...)
  struct __mzip_with_
    : __mzip_with_<__mapply<__q<__types>, _Lists>...> { };

  template <class... _Cs>
  struct __mzip_with_<__types<_Cs...>> {
    template <class _Fn, class _Continuation>
    using __f = __minvoke<_Continuation, __minvoke<_Fn, _Cs>...>;
  };

  template <class... _Cs, class... _Ds>
  struct __mzip_with_<__types<_Cs...>, __types<_Ds...>> {
    template <class _Fn, class _Continuation>
    using __f = __minvoke<_Continuation, __minvoke<_Fn, _Cs, _Ds>...>;
  };

  template <class _Fn, class _Continuation = __q<__types>>
  struct __mzip_with {
    template <class... _Lists>
    using __f = __minvoke<__mzip_with_<_Lists...>, _Fn, _Continuation>;
  };

  // clang-format on

#if STDEXEC_GCC() && (__GNUC__ < 12)
  template <class>
  extern int __mconvert_indices;
  template <std::size_t... _Indices>
  extern __types<__msize_t<_Indices>...> __mconvert_indices<std::index_sequence<_Indices...>>;
  template <std::size_t _Np>
  using __mmake_index_sequence =
    decltype(stdexec::__mconvert_indices<std::make_index_sequence<_Np>>);
#else
  template <std::size_t... _Indices>
  __types<__msize_t<_Indices>...> __mconvert_indices(std::index_sequence<_Indices...>*);
  template <std::size_t _Np>
  using __mmake_index_sequence =
    decltype(stdexec::__mconvert_indices((std::make_index_sequence<_Np>*) nullptr));
#endif

  template <class... _Ts>
  using __mindex_sequence_for = __mmake_index_sequence<sizeof...(_Ts)>;

  template <bool>
  struct __mfind_if_ {
    template <class _Fn, class _Continuation, class _Head, class... _Tail>
    using __f = //
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
    using __f = __minvoke<__mfind_if_<(sizeof...(_Args) != 0)>, _Fn, _Continuation, _Args...>;
  };

  template <class _Fn>
  struct __mfind_if_i {
    template <class... _Args>
    using __f = __msize_t<(sizeof...(_Args) - __v<__minvoke<__mfind_if<_Fn, __msize>, _Args...>>)>;
  };

  template <class... _Booleans>
  using __mand_ = __mbool<(__v<_Booleans> && ...)>;
  template <class... _Booleans>
  using __mand = __meval<__mand_, _Booleans...>;

  template <class... _Booleans>
  using __mor_ = __mbool<(__v<_Booleans> || ...)>;
  template <class... _Booleans>
  using __mor = __meval<__mor_, _Booleans...>;

  template <class _Boolean>
  using __mnot_ = __mbool<!__v<_Boolean>>;
  template <class _Boolean>
  using __mnot = __meval<__mnot_, _Boolean>;

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
  template <std::size_t _Np, class... _Ts>
  using __m_at = __type_pack_element<_Np, _Ts...>;
#else
  template <std::size_t>
  using __void_ptr = void*;

  template <class _Ty>
  using __mtype_ptr = __mtype<_Ty>*;

  template <class _Ty>
  struct __m_at_;

  template <std::size_t... _Is>
  struct __m_at_<__indices<_Is...>> {
    template <class _Up, class... _Us>
    static _Up __f_(__void_ptr<_Is>..., _Up*, _Us*...);
    template <class... _Ts>
    using __f = __t<decltype(__m_at_::__f_(__mtype_ptr<_Ts>()...))>;
  };

  template <std::size_t _Np, class... _Ts>
  using __m_at = __minvoke<__m_at_<__make_indices<_Np>>, _Ts...>;
#endif

  template <std::size_t _Np>
  struct __placeholder_;
  template <std::size_t _Np>
  using __placeholder = __placeholder_<_Np>*;

  using __0 = __placeholder<0>;
  using __1 = __placeholder<1>;
  using __2 = __placeholder<2>;
  using __3 = __placeholder<3>;

  template <class _Ty, class _Noexcept = __mbool<true>>
  struct __mconstruct {
    template <class... _As>
    auto operator()(_As&&... __as) const noexcept(__v<_Noexcept>&& noexcept(_Ty((_As&&) __as...)))
      -> decltype(_Ty((_As&&) __as...)) {
      return _Ty((_As&&) __as...);
    }
  };

  template <template <class...> class _Cp, class _Noexcept = __mbool<true>>
  using __mconstructor_for = __mcompose<__q<__mconstruct>, __q<_Cp>>;

  template <std::size_t>
  using __ignore_t = __ignore;

  template <class _Indices>
  struct __nth_pack_element_;

  template <std::size_t... _Is>
  struct __nth_pack_element_<__indices<_Is...>> {
    template <class _Ty, class... _Us>
    constexpr _Ty&& operator()(__ignore_t<_Is>..., _Ty&& __ty, _Us&&...) const noexcept {
      return (_Ty&&) __ty;
    }
  };

  template <std::size_t _Nn>
  using __nth_pack_element = __nth_pack_element_<__make_indices<_Nn>>;

//////////////////////////////////////////////////////////
// Begin __mfill_n implementation (needed by get<_Nn>(tuple))
#if __has_builtin(__make_integer_seq)

  // Implement __mfill_n in terms of the __make_integer_seq compiler intrinsic
  template <class _Ty, class _Tail>
  struct __mfill_n_ {
    template <class, std::size_t... _Indices>
    using __idx_tuple = __minvoke<_Tail, __mfirst<_Ty, __msize_t<_Indices>>...>;
    template <class _Nn>
    using __f = __make_integer_seq<__idx_tuple, std::size_t, __v<_Nn>>;
  };
  template <std::size_t _Nn, class _Ty, class _Tail = __q<__types>>
  using __mfill_n = __minvoke<__mfill_n_<_Ty, _Tail>, __msize_t<_Nn>>;

#elif __has_builtin(__integer_pack)

  // Implement __mfill_n in terms of the __integer_pack compiler intrinsic
  template <class _Tail, class _Ty, std::size_t... Ns>
  using __mfill_n_ = __minvoke<_Tail, __mfirst<_Ty, char[Ns + 1]>...>;

  template <std::size_t _Nn, class _Ty, class _Tail = __q<__types>>
  using __mfill_n = __mfill_n_<_Tail, _Ty, __integer_pack(_Nn)...>;

#else

  // Implement __mfill_n in terms of the __make_integer_seq compiler intrinsic
  template <class _Ty, class _Tail>
  struct __mfill_n_ {
    template <class _Nn>
    using __f = __t<
      decltype([]<std::size_t... _Is>(__indices<_Is...>) //
        -> __mtype<__minvoke<_Tail, __mfirst<_Ty, __msize_t<_Is>>...>> {
        return {};
      }(__make_indices<__v<_Nn>>()))>;
  };
  template <std::size_t _Nn, class _Ty, class _Tail = __q<__types>>
  using __mfill_n = __minvoke<__mfill_n_<_Ty, _Tail>, __msize_t<_Nn>>;

#endif
  // End __mfill_n implementation
  ////////////////////////////////


  template <class _Ty>
  struct __mdispatch_ {
    template <class... _Ts>
    _Ty operator()(_Ts&&...) const noexcept(noexcept(_Ty{})) {
      return _Ty{};
    }
  };

  template <std::size_t _Np>
  struct __mdispatch_<__placeholder<_Np>> {
    template <class... _Ts>
    decltype(auto) operator()(_Ts&&... __ts) const noexcept {
      return stdexec::__nth_pack_element<_Np>()((_Ts&&) __ts...);
    }
  };

  template <std::size_t _Np>
  struct __mdispatch_<__placeholder<_Np>&> {
    template <class... _Ts>
    decltype(auto) operator()(_Ts&&... __ts) const noexcept {
      return stdexec::__nth_pack_element<_Np>()(__ts...);
    }
  };

  template <std::size_t _Np>
  struct __mdispatch_<__placeholder<_Np>&&> {
    template <class... _Ts>
    decltype(auto) operator()(_Ts&&... __ts) const noexcept {
      return std::move(stdexec::__nth_pack_element<_Np>()(__ts...));
    }
  };

  template <std::size_t _Np>
  struct __mdispatch_<const __placeholder<_Np>&> {
    template <class... _Ts>
    decltype(auto) operator()(_Ts&&... __ts) const noexcept {
      return std::as_const(stdexec::__nth_pack_element<_Np>()(__ts...));
    }
  };

  template <class _Ret, class... _Args>
  struct __mdispatch_<_Ret (*)(_Args...)> {
    template <class... _Ts>
      requires(__callable<__mdispatch_<_Args>, _Ts...> && ...)
           && __callable<_Ret, __call_result_t<__mdispatch_<_Args>, _Ts...>...>
    auto operator()(_Ts&&... __ts) const
      noexcept(__nothrow_callable<_Ret, __call_result_t<__mdispatch_<_Args>, _Ts...>...>)
        -> __call_result_t<_Ret, __call_result_t<__mdispatch_<_Args>, _Ts...>...> {
      return _Ret{}(__mdispatch_<_Args>{}((_Ts&&) __ts...)...);
    }
  };

  template <class _Ty>
  struct __mdispatch { };

  template <class _Ret, class... _Args>
  struct __mdispatch<_Ret(_Args...)> {
    template <class... _Ts>
      requires(__callable<__mdispatch_<_Args>, _Ts...> && ...)
           && __callable<_Ret, __call_result_t<__mdispatch_<_Args>, _Ts...>...>
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
  using __try_dispatch_ = __mbool<__dispatchable<_Signature, _Args...>>;

  template <class _Signatures, class _Continuation = __q<__mfront>>
  struct __which { };

  template <template <class...> class _Cp, class... _Signatures, class _Continuation>
  struct __which<_Cp<_Signatures...>, _Continuation> {
    template <class... _Args>
    using __f = //
      __minvoke<
        __mfind_if<__mbind_back_q<__try_dispatch_, _Args...>, _Continuation>,
        _Signatures...>;
  };

  template <class _Signatures, class _DefaultFn, class... _Args>
  using __make_dispatcher = //
    __minvoke<
      __if_c<
        __minvocable<__which<_Signatures>, _Args...>,
        __mcompose<__q<__mdispatch>, __which<_Signatures>>,
        _DefaultFn>,
      _Args...>;
} // namespace stdexec
