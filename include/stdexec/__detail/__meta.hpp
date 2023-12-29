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

    STDEXEC_ATTRIBUTE((always_inline)) //
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

  template <auto _Np>
  struct __mconstant_;

  template <auto _Np>
  using __mconstant = __mconstant_<_Np>*;

  template <class _Tp, class _Up>
  using __mfirst = _Tp;

  template <class _Tp, class _Up>
  using __msecond = _Up;

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

  template <auto _Np>
  inline constexpr __mtypeof<_Np> __v<__mconstant<_Np>> = _Np;

  template <std::size_t _Ip>
  inline constexpr std::size_t __v<char[_Ip]> = _Ip - 1;

  template <std::size_t... _Is>
  using __indices = std::index_sequence<_Is...>*;

  template <std::size_t _Np>
  using __make_indices = std::make_index_sequence<_Np>*;

  template <class... _Ts>
  using __indices_for = __make_indices<sizeof...(_Ts)>;

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

    template <std::size_t... _Is>
    constexpr bool __equal(__mstring __other, __indices<_Is...>) const noexcept {
      return ((__what_[_Is] == __other.__what_[_Is]) && ...);
    }

    constexpr bool operator==(__mstring __other) const noexcept {
      return __equal(__other, __make_indices<_Len>());
    }

    char const __what_[_Len];
  };

#if STDEXEC_NVHPC() && (__EDG_VERSION__ < 604)
  // Use a non-standard extension for older nvc++ releases
  template <__mchar _Char, _Char... _Str>
  constexpr __mstring<sizeof...(_Str)> operator""__csz() noexcept {
    return {_Str...};
  }
#elif STDEXEC_NVHPC() && (__EDG_VERSION__ < 605)
  // This is to work around an unfiled (by me) EDG bug that fixed in build 605
  template <__mstring _Str>
  constexpr __mtypeof<_Str> const operator""__csz() noexcept {
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
    _ERROR_ operator,(__msuccess) const noexcept;
  };

  template <__mstring _What>
  struct _WHAT_ { };

  template <class _What, class... _With>
  using __mexception = _ERROR_<_What, _With...>;

  template <class>
  extern __msuccess __ok_v;

  template <class _What, class... _With>
  extern _ERROR_<_What, _With...> __ok_v<__mexception<_What, _With...>>;

  template <class _Ty>
  using __ok_t = decltype(__ok_v<_Ty>);

  template <class... _Ts>
  using __disp = decltype((__msuccess(), ..., __ok_t<_Ts>()));

  template <class _Arg>
  concept __ok = __same_as<__ok_t<_Arg>, __msuccess>;

  template <class _Arg>
  concept __merror = !__ok<_Arg>;

  template <class... _Args>
  concept _Ok = (__ok<_Args> && ...);

  template <bool _AllOK>
  struct __i;

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

  template <class _Fn, class... _Args>
  using __minvoke__ = typename __i<_Ok<_Fn>>::template __h<_Fn, _Args...>;

  template <class _Fn, class... _Args>
  struct __minvoke_ { };

  template <class _Fn, class... _Args>
    requires __typename<__minvoke__<_Fn, _Args...>>
  struct __minvoke_<_Fn, _Args...> {
    using __t = __minvoke__<_Fn, _Args...>;
  };

  template <class _Fn, class... _Args>
  using __minvoke = __t<__minvoke_<_Fn, _Args...>>;

#else

  template <template <class...> class _Fn, class... _Args>
  using __meval = typename __i<_Ok<_Args...>>::template __g<_Fn, _Args...>;

  template <class _Fn, class... _Args>
  using __minvoke = typename __i<_Ok<_Fn>>::template __h<_Fn, _Args...>;

#endif

  template <bool _AllOK>
  struct __i {
    template <template <class...> class _Fn, class... _Args>
    using __g = _Fn<_Args...>;

    template <class _Fn, class... _Args>
    using __h = __meval<_Fn::template __f, _Args...>;
  };

  template <>
  struct __i<false> {
    template <template <class...> class, class... _Args>
    using __g = __disp<_Args...>;

    template <class _Fn, class...>
    using __h = _Fn;
  };

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
  concept __mvalid = requires { typename __meval<_Tp, _Args...>; };

  template <class _Fn, class... _Args>
  concept __minvocable = __mvalid<_Fn::template __f, _Args...>;

  template <template <class...> class _Tp, class... _Args>
  concept __msucceeds = __mvalid<_Tp, _Args...> && __ok<__meval<_Tp, _Args...>>;

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

  template <class _Fn, class... _Args>
  using __mmemoize = __t<__mdefer<_Fn, _Args...>>;

  template <template <class...> class _Fn, class... _Args>
  using __mmemoize_q = __mmemoize<__q<_Fn>, _Args...>;

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

  template <class _Pred, class _True, class... _False>
  using __minvoke_if = __minvoke<__if<_Pred, _True, _False...>>;

  template <bool _Pred, class _True, class... _False>
  using __minvoke_if_c = __minvoke<__if_c<_Pred, _True, _False...>>;

  template <class _Tp>
  struct __mconst {
    template <class...>
    using __f = _Tp;
  };

  inline constexpr __mstring __mbad_substitution =
    "The specified meta-function could not be evaluated with the types provided."__csz;

  template <__mstring _Diagnostic = __mbad_substitution>
  struct _BAD_SUBSTITUTION_ { };

  template <class... _Args>
  struct _WITH_TYPES_;

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

  template <template <class...> class _Try, class _Catch>
  struct __mtry_catch_q {
    template <class... _Args>
    using __f = __minvoke< __if_c<__mvalid<_Try, _Args...>, __q<_Try>, _Catch>, _Args...>;
  };

  template <class _Try, class _Catch>
  struct __mtry_catch {
    template <class... _Args>
    using __f = __minvoke< __if_c<__minvocable<_Try, _Args...>, _Try, _Catch>, _Args...>;
  };

  template <class _Fn, class _Default>
  using __with_default = __mtry_catch<_Fn, __mconst<_Default>>;

  template <template <class...> class _Fn, class _Default>
  using __with_default_q = __mtry_catch_q<_Fn, __mconst<_Default>>;

  template <class _Fn, class _Default, class... _Args>
  using __minvoke_or = __minvoke<__with_default<_Fn, _Default>, _Args...>;

  template <template <class...> class _Fn, class _Default, class... _Args>
  using __meval_or = __minvoke<__with_default_q<_Fn, _Default>, _Args...>;

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

  template <class _Continuation, class... _As>
  struct __mconcat_ { };

  template <class _Continuation, class... _As>
    requires(sizeof...(_As) == 0) && __minvocable<_Continuation, _As...>
  struct __mconcat_<_Continuation, _As...> {
    using __t = __minvoke<_Continuation, _As...>;
  };

  template <class _Continuation, template <class...> class _Ap, class... _As>
    requires __minvocable<_Continuation, _As...>
  struct __mconcat_<_Continuation, _Ap<_As...>> {
    using __t = __minvoke<_Continuation, _As...>;
  };

  template <             //
    class _Continuation, //
    template <class...>
    class _Ap,
    class... _As, //
    template <class...>
    class _Bp,
    class... _Bs>
    requires __minvocable<_Continuation, _As..., _Bs...>
  struct __mconcat_<_Continuation, _Ap<_As...>, _Bp<_Bs...>> {
    using __t = __minvoke<_Continuation, _As..., _Bs...>;
  };

  template <             //
    class _Continuation, //
    template <class...>
    class _Ap,
    class... _As, //
    template <class...>
    class _Bp,
    class... _Bs, //
    template <class...>
    class _Cp,
    class... _Cs>
    requires __minvocable<_Continuation, _As..., _Bs..., _Cs...>
  struct __mconcat_<_Continuation, _Ap<_As...>, _Bp<_Bs...>, _Cp<_Cs...>> {
    using __t = __minvoke<_Continuation, _As..., _Bs..., _Cs...>;
  };

  template <             //
    class _Continuation, //
    template <class...>
    class _Ap,
    class... _As, //
    template <class...>
    class _Bp,
    class... _Bs, //
    template <class...>
    class _Cp,
    class... _Cs, //
    template <class...>
    class _Dp,
    class... _Ds, //
    class... _Tail>
  struct __mconcat_<_Continuation, _Ap<_As...>, _Bp<_Bs...>, _Cp<_Cs...>, _Dp<_Ds...>, _Tail...>
    : __mconcat_<_Continuation, __types<_As..., _Bs..., _Cs..., _Ds...>, _Tail...> { };

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

  template <class _Fn, class _Tp>
  struct __uncurry_;

  template <__merror _Fn, class _Tp>
  struct __uncurry_<_Fn, _Tp> {
    using __t = _Fn;
  };

  template <class _Fn, template <class...> class _Ap, class... _As>
    requires __minvocable<_Fn, _As...>
  struct __uncurry_<_Fn, _Ap<_As...>> {
    using __t = __minvoke<_Fn, _As...>;
  };

  template <class _Fn>
  struct __uncurry {
    template <class _Tp>
    using __f = __t<__uncurry_<_Fn, _Tp>>;
  };
  template <class _Fn, class _List>
  using __mapply = __minvoke<__uncurry<_Fn>, _List>;

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
  template <class _Default, class... _As>
    requires(sizeof...(_As) <= 1)
  using __msingle_or_ = __mfront<_As..., _Default>;
  template <class _Default>
  using __msingle_or = __mbind_front_q<__msingle_or_, _Default>;

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

  template <class _From, class _To = __decay_t<_From>>
  using __cvref_t = __copy_cvref_t<_From, __t<_To>>;

  template <class _From, class _To = __decay_t<_From>>
  using __cvref_id = __copy_cvref_t<_From, __id<_To>>;

#if STDEXEC_NVHPC()
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

  template <const auto& _Fun, class... _As>
  using __result_of = __call_result_t<decltype(_Fun), _As...>;

#if STDEXEC_CLANG() && (__clang_major__ < 13)
  template <class _Ty>
  constexpr auto __hide_ = [] {
    return (__mtype<_Ty>(*)()) 0;
  };
#else
  template <class _Ty>
  extern decltype([] { return (__mtype<_Ty>(*)()) 0; }) __hide_;
#endif

  template <class _Ty>
  using __hide = decltype(__hide_<_Ty>);

  template <class _Id>
  using __unhide = __t<__call_result_t<__call_result_t<_Id>>>;

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
  using __cref_t = decltype(__cref_fn{}(__declval<_Ty>()));

  template <class, class, class, class>
  struct __mzip_with2_;

  template <             //
    class _Fn,           //
    class _Continuation, //
    template <class...>
    class _Cp,
    class... _Cs, //
    template <class...>
    class _Dp,
    class... _Ds>
    requires requires { typename __minvoke<_Continuation, __minvoke<_Fn, _Cs, _Ds>...>; }
  struct __mzip_with2_<_Fn, _Continuation, _Cp<_Cs...>, _Dp<_Ds...>> {
    using __t = __minvoke<_Continuation, __minvoke<_Fn, _Cs, _Ds>...>;
  };

  template <class _Fn, class _Continuation = __q<__types>>
  struct __mzip_with2 {
    template <class _Cp, class _Dp>
    using __f = __t<__mzip_with2_<_Fn, _Continuation, _Cp, _Dp>>;
  };

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

#if STDEXEC_HAS_BUILTIN(__type_pack_element)
  template <std::size_t _Np, class... _Ts>
  struct __m_at_ {
    using __t = __type_pack_element<_Np, _Ts...>;
  };

  template <std::size_t _Np, class... _Ts>
  using __m_at_c = __t<__m_at_<_Np, _Ts...>>;
#else
  template <std::size_t>
  using __void_ptr = void*;

  template <class _Ty>
  using __mtype_ptr = __mtype<_Ty>*;

  template <class _Ty>
  struct __m_at_;

  template <std::size_t... _Is>
  struct __m_at_<std::index_sequence<_Is...>> {
    template <class _Up, class... _Us>
    static _Up __f_(__void_ptr<_Is>..., _Up*, _Us*...);
    template <class... _Ts>
    using __f = __t<decltype(__m_at_::__f_(__mtype_ptr<_Ts>()...))>;
  };

  template <std::size_t _Np, class... _Ts>
  using __m_at_c = __minvoke<__m_at_<std::make_index_sequence<_Np>>, _Ts...>;
#endif

  template <class _Np, class... _Ts>
  using __m_at = __m_at_c<__v<_Np>, _Ts...>;

  template <class... _Ts>
  using __mback = __m_at_c<sizeof...(_Ts) - 1, _Ts...>;

  template <class _Continuation = __q<__types>>
  struct __mpop_back {
    template <class>
    struct __impl;

    template <std::size_t... _Idx>
    struct __impl<__indices<_Idx...>> {
      template <class... _Ts>
      using __f = __minvoke<_Continuation, __m_at_c<_Idx, _Ts...>...>;
    };

    template <class... _Ts>
      requires(sizeof...(_Ts) != 0)
    using __f = __minvoke<__impl<__make_indices<sizeof...(_Ts) - 1>>, _Ts...>;
  };

  template <std::size_t _Np>
  struct __placeholder {
    __placeholder() = default;

    constexpr __placeholder(void*) noexcept {
    }

    friend constexpr std::size_t __get_placeholder_offset(__placeholder) noexcept {
      return _Np;
    }
  };

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

#if STDEXEC_MSVC()
  // MSVCBUG https://developercommunity.visualstudio.com/t/Incorrect-function-template-argument-sub/10437827

  template <std::size_t>
  struct __ignore_t {
    __ignore_t() = default;

    constexpr __ignore_t(auto&&...) noexcept {
    }
  };
#else
  template <std::size_t>
  using __ignore_t = __ignore;
#endif

  template <class... _Ignore>
  struct __nth_pack_element_impl {
    template <class _Ty, class... _Us>
    STDEXEC_ATTRIBUTE((always_inline)) //
    constexpr _Ty&& operator()(_Ignore..., _Ty&& __t, _Us&&...) const noexcept {
      return (decltype(__t)&&) __t;
    }
  };

  template <std::size_t _Np>
  struct __nth_pack_element_t {
    template <std::size_t... _Is>
    STDEXEC_ATTRIBUTE((always_inline)) //
    static constexpr auto __impl(__indices<_Is...>) noexcept {
      return __nth_pack_element_impl<__ignore_t<_Is>...>();
    }

    template <class... _Ts>
    STDEXEC_ATTRIBUTE((always_inline)) //
    constexpr decltype(auto) operator()(_Ts&&... __ts) const noexcept {
      static_assert(_Np < sizeof...(_Ts));
      return __impl(__make_indices<_Np>())((_Ts&&) __ts...);
    }
  };

  template <std::size_t _Np>
  inline constexpr __nth_pack_element_t<_Np> __nth_pack_element{};

  template <auto... _Vs>
  struct __mliterals {
    template <std::size_t _Np>
    STDEXEC_ATTRIBUTE((always_inline)) //
    static constexpr auto __nth() noexcept {
      return stdexec::__nth_pack_element<_Np>(_Vs...);
    }
  };

  template <std::size_t _Np>
  struct __nth_member {
    template <class _Ty>
    STDEXEC_ATTRIBUTE((always_inline)) //
    constexpr decltype(auto) operator()(_Ty&& __ty) const noexcept {
      return ((_Ty&&) __ty).*(__ty.__mbrs_.template __nth<_Np>());
    }
  };

  template <class _Ty, std::size_t _Offset = 0>
  struct __mdispatch_ {
    template <class... _Ts>
    _Ty operator()(_Ts&&...) const noexcept(noexcept(_Ty{})) {
      return _Ty{};
    }
  };

  template <std::size_t _Np, std::size_t _Offset>
  struct __mdispatch_<__placeholder<_Np>, _Offset> {
    template <class... _Ts>
    decltype(auto) operator()(_Ts&&... __ts) const noexcept {
      return stdexec::__nth_pack_element<_Np + _Offset>((_Ts&&) __ts...);
    }
  };

  template <std::size_t _Np, std::size_t _Offset>
  struct __mdispatch_<__placeholder<_Np>&, _Offset> {
    template <class... _Ts>
    decltype(auto) operator()(_Ts&&... __ts) const noexcept {
      return stdexec::__nth_pack_element<_Np + _Offset>(__ts...);
    }
  };

  template <std::size_t _Np, std::size_t _Offset>
  struct __mdispatch_<__placeholder<_Np>&&, _Offset> {
    template <class... _Ts>
    decltype(auto) operator()(_Ts&&... __ts) const noexcept {
      return std::move(stdexec::__nth_pack_element<_Np + _Offset>(__ts...));
    }
  };

  template <std::size_t _Np, std::size_t _Offset>
  struct __mdispatch_<const __placeholder<_Np>&, _Offset> {
    template <class... _Ts>
    decltype(auto) operator()(_Ts&&... __ts) const noexcept {
      return std::as_const(stdexec::__nth_pack_element<_Np + _Offset>(__ts...));
    }
  };

  template <class _Ret, class... _Args, std::size_t _Offset>
  struct __mdispatch_<_Ret (*)(_Args...), _Offset> {
    template <class... _Ts>
      requires(__callable<__mdispatch_<_Args, _Offset>, _Ts...> && ...)
           && __callable<_Ret, __call_result_t<__mdispatch_<_Args, _Offset>, _Ts...>...>
    auto operator()(_Ts&&... __ts) const
      noexcept(__nothrow_callable<_Ret, __call_result_t<__mdispatch_<_Args, _Offset>, _Ts...>...>)
        -> __call_result_t<_Ret, __call_result_t<__mdispatch_<_Args, _Offset>, _Ts...>...> {
      return _Ret{}(__mdispatch_<_Args, _Offset>{}((_Ts&&) __ts...)...);
    }
  };

  template <class _Ret, class... _Args, std::size_t _Offset>
  struct __mdispatch_<_Ret (*)(_Args..., ...), _Offset> {
    static_assert(_Offset == 0, "nested pack expressions are not supported");
    using _Pattern = __mback<_Args...>;
    static constexpr std::size_t __offset = __get_placeholder_offset((__mtype<_Pattern>*) nullptr);

    struct __impl {
      template <std::size_t... _Idx, class... _Ts>
        requires(__callable<__mdispatch_<_Args>, _Ts...> && ...)
             && (__callable<__mdispatch_<_Pattern, _Idx + 1>, _Ts...> && ...)
             && __callable< //
                  _Ret,
                  __call_result_t<__mdispatch_<_Args>, _Ts...>...,
                  __call_result_t<__mdispatch_<_Pattern, _Idx + 1>, _Ts...>...>
      auto operator()(__indices<_Idx...>, _Ts&&... __ts) const noexcept(
        __nothrow_callable<                                                                  //
          _Ret,                                                                              //
          __call_result_t<__mdispatch_<_Args>, _Ts...>...,                                   //
          __call_result_t<__mdispatch_<_Pattern, _Idx + 1>, _Ts...>...>) -> __call_result_t< //
        _Ret,
        __call_result_t<__mdispatch_<_Args>, _Ts...>...,
        __call_result_t<__mdispatch_<_Pattern, _Idx + 1>, _Ts...>...> {
        return _Ret()(                               //
          __mdispatch_<_Args>()((_Ts&&) __ts...)..., //
          __mdispatch_<_Pattern, _Idx + 1 >()((_Ts&&) __ts...)...);
      }
    };

    template <class... _Ts>
      requires(__offset < sizeof...(_Ts))
           && __callable<__impl, __make_indices<sizeof...(_Ts) - __offset - 1>, _Ts...>
    auto operator()(_Ts&&... __ts) const
      noexcept(__nothrow_callable<__impl, __make_indices<sizeof...(_Ts) - __offset - 1>, _Ts...>)
        -> __msecond<
          __if_c<(__offset < sizeof...(_Ts))>,
          __call_result_t<__impl, __make_indices<sizeof...(_Ts) - __offset - 1>, _Ts...>> {
      return __impl()(__make_indices<sizeof...(_Ts) - __offset - 1>(), (_Ts&&) __ts...);
    }

    template <class... _Ts>
      requires(sizeof...(_Ts) == __offset)
           && __callable<__mdispatch_<__minvoke<__mpop_back<__qf<_Ret>>, _Args...>*>, _Ts...>
    auto operator()(_Ts&&... __ts) const noexcept(
      __nothrow_callable<__mdispatch_<__minvoke<__mpop_back<__qf<_Ret>>, _Args...>*>, _Ts...>)
      -> __msecond<
        __if_c<(sizeof...(_Ts) == __offset)>,
        __call_result_t<__mdispatch_<__minvoke<__mpop_back<__qf<_Ret>>, _Args...>*>, _Ts...>> {
      return __mdispatch_<__minvoke<__mpop_back<__qf<_Ret>>, _Args...>*>()((_Ts&&) __ts...);
    }
  };

  template <class _Ty>
  struct __mdispatch { };

  template <class _Ret, class... _Args>
  struct __mdispatch<_Ret(_Args...)> : __mdispatch_<_Ret (*)(_Args...)> { };

  template <class _Ret, class... _Args>
  struct __mdispatch<_Ret(_Args..., ...)> : __mdispatch_<_Ret (*)(_Args..., ...)> { };

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
      __mtry_catch<__mcompose<__q<__mdispatch>, __which<_Signatures>>, _DefaultFn>,
      _Args...>;
} // namespace stdexec
