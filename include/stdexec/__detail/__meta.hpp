/*
 * Copyright (c) 2026 NVIDIA Corporation
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

#include "__concepts.hpp"
#include "__config.hpp"
#include "__type_traits.hpp"

#include <cassert>
#include <cstddef>
#include <source_location> // IWYU pragma: keep for std::source_location::current
#include <string_view>
#include <type_traits>

namespace STDEXEC {
  //! Convenience metafunction getting the dependant type `__t` out of `_Tp`.
  //! That is, `typename _Tp::__t`.
  //! See MAINTAINERS.md#class-template-parameters for details.
  template <class _Tp>
  using __t = _Tp::__t;

  template <class _Ret, class... _Args>
  using __fn_t = _Ret(_Args...);

  template <class _Ty>
  struct __mtype {
    using __t = _Ty;
  };

  template <class...>
  concept __mnever = false;

  namespace __detail {
    // NB: This variable template is partially specialized for __type_index in __typeinfo.hpp:
    template <auto _Value>
    extern __fn_t<decltype(_Value)> *__mtypeof_v;
  } // namespace __detail

  template <auto _Value>
  using __mtypeof = decltype(__detail::__mtypeof_v<_Value>());

  template <class...>
  struct __mlist;

  template <class _Tp>
  using __midentity = _Tp;

  template <auto _Np>
  struct __mconstant {
    using type = __mconstant;
    using value_type = __mtypeof<_Np>;
    static constexpr value_type value = _Np;

    constexpr operator value_type() const noexcept {
      return value;
    }

    constexpr auto operator()() const noexcept -> value_type {
      return value;
    }
  };

  template <std::size_t _Np>
  using __msize_t = std::integral_constant<std::size_t, _Np>;

  //! Metafunction selects the first of two type arguments.
  template <class _Tp, class _Up>
  using __mfirst = _Tp;

  //! Metafunction selects the second of two type arguments.
  template <class _Tp, class _Up>
  using __msecond = _Up;

  template <class...>
  struct __undefined;

  template <std::size_t... _Is>
  struct __iota;

  template <std::size_t... _Is>
  using __indices = __iota<_Is...> *;

#if STDEXEC_MSVC()
  namespace __pack {
    template <class _Ty, _Ty... _Is>
    struct __idx;

    template <class>
    extern int __mkidx;

    template <std::size_t... _Is>
    extern __indices<_Is...> __mkidx<__idx<std::size_t, _Is...>>;
  } // namespace __pack

  template <std::size_t _Np>
  using __make_indices =
    decltype(__pack::__mkidx<__make_integer_seq<__pack::__idx, std::size_t, _Np>>);
#elif STDEXEC_HAS_BUILTIN(__make_integer_seq)
  namespace __pack {
    template <class _Ty, _Ty... _Is>
    using __idx = __indices<_Is...>;
  } // namespace __pack

  template <std::size_t _Np>
  using __make_indices = __make_integer_seq<__pack::__idx, std::size_t, _Np>;
#elif STDEXEC_HAS_BUILTIN(__integer_pack)
  namespace __pack {
    template <std::size_t _Np>
    extern __indices<__integer_pack(_Np)...> __make_indices;
  } // namespace __pack

  template <std::size_t _Np>
  using __make_indices = decltype(__pack::__make_indices<_Np>);
#else
  namespace __pack {
    template <std::size_t... _Is>
    auto __mk_indices(__indices<0, _Is...>) -> __indices<_Is...>;

    template <std::size_t _Np, class = char[_Np], std::size_t... _Is>
    auto __mk_indices(__indices<_Np, _Is...>)
      -> decltype(__mk_indices(__indices<_Np - 1, 0, (_Is + 1)...>{}));
  } // namespace __pack

  template <std::size_t _Np>
  using __make_indices = decltype(__pack::__mk_indices(__indices<_Np>{}));
#endif

  template <class... _Ts>
  using __indices_for = __make_indices<sizeof...(_Ts)>;

  struct __ignore {
    constexpr __ignore() = default;

    template <class... _Ts>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr __ignore(_Ts&&...) noexcept {
    }
  };

  using __msuccess = int;

  template <class _What, class... _With>
  struct _WARNING_ { };

  template <class... _What>
  struct _ERROR_ {
    using __t = _ERROR_;
    using __partitioned = _ERROR_;

    template <class, class>
    using __value_types = _ERROR_;

    template <class, class>
    using __error_types = _ERROR_;

    template <class, class>
    using __stopped_types = _ERROR_;

    using __decay_copyable = _ERROR_;
    using __nothrow_decay_copyable = _ERROR_;
    using __values = _ERROR_;
    using __errors = _ERROR_;
    using __all = _ERROR_;

    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto operator+() const -> _ERROR_;

    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto operator,(__ignore) const -> _ERROR_;
  };

  template <class... _What>
  using __mexception = _ERROR_<_What...>;

  template <class>
  extern __msuccess __ok_v;

  template <class... _What>
  extern _ERROR_<_What...> __ok_v<__mexception<_What...>>;

  template <class _Ty>
  using __ok_t = decltype(__ok_v<_Ty>);

  template <class... _Ts>
  using __mfind_error = decltype((__msuccess(), ..., __ok_t<_Ts>()));

  template <class _Arg>
  concept __ok = STDEXEC_IS_SAME(__ok_t<_Arg>, __msuccess);

  template <class _Arg>
  concept __merror = !STDEXEC_IS_SAME(__ok_t<_Arg>, __msuccess);

  template <class _Fn, class... _Args>
  using __mcall = _Fn::template __f<_Args...>;

  template <class _Fn, class _Arg>
  using __mcall1 = _Fn::template __f<_Arg>;

  template <class _Fn, class _First, class _Second>
  using __mcall2 = _Fn::template __f<_First, _Second>;

  template <template <class...> class _Fn, class... _Args>
  using __mcall_q = _Fn<_Args...>;

  template <template <class> class _Fn, class _Arg>
  using __mcall1_q = _Fn<_Arg>;

  template <template <class, class> class _Fn, class _First, class _Second>
  using __mcall2_q = _Fn<_First, _Second>;

  template <class _Fn, class... _Args>
  concept __mcallable = requires { typename __mcall<_Fn, _Args...>; };

  template <class _Fn, class _Arg>
  concept __mcallable1 = requires { typename __mcall1<_Fn, _Arg>; };

  template <class _Fn, class _First, class _Second>
  concept __mcallable2 = requires { typename __mcall2<_Fn, _First, _Second>; };

  template <template <class...> class _Fn, class... _Args>
  concept __mcallable_q = requires { typename __mcall_q<_Fn, _Args...>; };

  template <template <class> class _Fn, class _Arg>
  concept __mcallable1_q = requires { typename __mcall1_q<_Fn, _Arg>; };

  template <template <class, class> class _Fn, class _First, class _Second>
  concept __mcallable2_q = requires { typename __mcall2_q<_Fn, _First, _Second>; };

  template <class... _Args>
  concept _Ok = (STDEXEC_IS_SAME(__ok_t<_Args>, __msuccess) && ...);

  //! The struct `__i` is the implementation of P2300's
  //! [_`META-APPLY`_](https://eel.is/c++draft/exec#util.cmplsig-5).
  //! > [Note [1](https://eel.is/c++draft/exec#util.cmplsig-note-1): 
  //! > The purpose of META-APPLY is to make it valid to use non-variadic
  //! > templates as Variant and Tuple arguments to gather-signatures. — end note]
  //! In addition to avoiding the dreaded "pack expanded into non-pack argument" error,
  //! it is part of the meta-error propagation mechanism. if any of the argument types
  //! are a specialization of `_ERROR_`, `__i` will short-circuit and return the error.
  //! `__minvoke` and `__minvoke_q` are implemented in terms of `__i`.
  template <bool _OK>
  struct __i;

  template <>
  struct __i<true> {
    template <template <class...> class _Fn, class... _Args>
    using __g = _Fn<_Args...>;

    template <class _Fn, class... _Args>
    using __f = _Fn::template __f<_Args...>;
  };

  template <>
  struct __i<false> {
    template <template <class...> class, class... _Args>
    using __g = __mfind_error<_Args...>;

    template <class... _Args>
    using __f = __mfind_error<_Args...>;
  };

#if STDEXEC_EDG()
  // Most compilers memoize alias template specializations, but
  // nvc++ does not. So we memoize the type computations by
  // indirecting through a class template specialization.
  template <template <class...> class _Fn, class... _Args>
  using __minvoke_q__ = __i<_Ok<_Args...>>::template __g<_Fn, _Args...>;

  template <template <class...> class _Fn, class... _Args>
  struct __minvoke_q_ { };

  template <template <class...> class _Fn, class... _Args>
    requires __typename<__minvoke_q__<_Fn, _Args...>>
  struct __minvoke_q_<_Fn, _Args...> {
    using __t = __minvoke_q__<_Fn, _Args...>;
  };

  template <template <class...> class _Fn, class... _Args>
  using __minvoke_q = __t<__minvoke_q_<_Fn, _Args...>>;

  template <class _Fn, class... _Args>
  using __minvoke__ = __i<_Ok<_Fn, _Args...>>::template __f<_Fn, _Args...>;

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
  using __minvoke_q = __i<_Ok<_Args...>>::template __g<_Fn, _Args...>;

  //! Metafunction invocation
  //! Given a metafunction, `_Fn`, and args.
  //! We expect `_Fn::__f` to be type alias template "implementing" the metafunction `_Fn`.
  template <class _Fn, class... _Args>
  using __minvoke = __i<_Ok<_Fn, _Args...>>::template __f<_Fn, _Args...>;

#endif

  template <template <class...> class _Fn>
  struct __qq {
    template <class... _Args>
    using __f = _Fn<_Args...>;
  };

  template <template <class> class _Fn>
  struct __q1 {
    template <class _Ty>
    using __f = _Fn<_Ty>;
  };

  template <template <class, class> class _Fn>
  struct __q2 {
    template <class _Ty, class _Uy>
    using __f = _Fn<_Ty, _Uy>;
  };

  //! This struct template is like
  //! [mpl::quote](https://www.boost.org/doc/libs/1_86_0/libs/mpl/doc/refmanual/quote.html).
  //! It turns an alias/class template into a metafunction that also propagates
  //! "meta-exceptions". All of the meta utilities recognize specializations of
  //! STDEXEC::_ERROR_ as an error type. Error types short-circuit the evaluation of the
  //! metafunction and are automatically propagated like an exception. Note: `__minvoke`
  //! and `__minvoke_q` also participate in this error propagation.
  //!
  //! This design lets us report type errors briefly at the library boundary, even if the
  //! actual error happens deep inside a meta-program.
  template <template <class...> class _Fn>
  struct __q {
    template <class... _Args>
    using __f = __i<_Ok<_Args...>>::template __g<_Fn, _Args...>;
  };

  template <template <class...> class _Fn>
  using __mtry_q = __q<_Fn>;

  template <class _Fn>
  struct __mtry : __mtry_q<_Fn::template __f> { };

  template <__merror _Error>
  struct __mtry<_Error> {
    template <class...>
    using __f = _Error;
  };

  template <template <class...> class _Fn, class... _Front>
  struct __mbind_front_q {
    template <class... _Args>
    using __f = __minvoke_q<_Fn, _Front..., _Args...>;
  };

  template <class _Fn, class... _Front>
  using __mbind_front = __mbind_front_q<_Fn::template __f, _Front...>;

  template <template <class...> class _Fn, class... _Back>
  struct __mbind_back_q {
    template <class... _Args>
    using __f = __minvoke_q<_Fn, _Args..., _Back...>;
  };

  template <class _Fn, class... _Back>
  using __mbind_back = __mbind_back_q<_Fn::template __f, _Back...>;

  template <template <class...> class _Tp, class... _Args>
  concept __minvocable_q = requires { typename __minvoke_q<_Tp, _Args...>; };

  template <class _Fn, class... _Args>
  concept __minvocable = __minvocable_q<_Fn::template __f, _Args...>;

  namespace __detail {
    template <class _Fn, class... _Args>
    struct __minvoke_force_ {
      using __t = __minvoke<_Fn, _Args...>;
    };
  } // namespace __detail

  template <class _Fn, class... _Args>
  using __minvoke_force = __t<__detail::__minvoke_force_<_Fn, _Args...>>;

  template <class _Fn, class... _Args>
  struct __mdefer { };

  template <class _Fn, class... _Args>
    requires __minvocable<_Fn, _Args...>
  struct __mdefer<_Fn, _Args...> {
    using __t = __minvoke<_Fn, _Args...>;
  };

  template <class _Fn, class... _Args>
  using __mmemoize = __t<__mdefer<_Fn, _Args...>>;

  template <template <class...> class _Fn, class... _Args>
  using __mmemoize_q = __mmemoize<__q<_Fn>, _Args...>;

#if STDEXEC_GCC()
  // GCC can not mangle builtins. __mangle_t introduces an
  // indirection that hides the builtin from the mangler.
  template <template <class...> class _Fn, class... _Args>
  using __mmangle_t = __mmemoize_q<_Fn, _Args...>;
#else
  template <template <class...> class _Fn, class... _Args>
  using __mmangle_t = _Fn<_Args...>;
#endif

  namespace __detail {
    template <bool>
    struct __if_ {
      template <class _Then, class _Else>
      using __f = _Then;
    };

    template <>
    struct __if_<false> {
      template <class _Then, class _Else>
      using __f = _Else;
    };

    template <class _Pred, class _Then, class _Else>
    using __if_t = __if_<bool(_Pred::value)>::template __f<_Then, _Else>;
  } // namespace __detail

  //! Metafunction selects `_Then` if the bool template is `true`, otherwise the second.
  //! This is similar to `std::conditional_t<Pred, Then, Else>` but instantiates fewer
  //! templates.
  template <class _Pred, class _Then, class _Else>
  using __if = __minvoke_q<__detail::__if_t, _Pred, _Then, _Else>;

  template <bool _Pred, class _Then, class _Else>
  using __if_c = __minvoke<__detail::__if_<_Pred>, _Then, _Else>;

  template <class _Pred, class _Then, class _Else, class... _Args>
  using __minvoke_if = __minvoke<__if<_Pred, _Then, _Else>, _Args...>;

  template <bool _Pred, class _Then, class _Else, class... _Args>
  using __minvoke_if_c = __minvoke<__if_c<_Pred, _Then, _Else>, _Args...>;

  template <class _Tp>
  struct __mconst {
    template <class...>
    using __f = _Tp;
  };

  template <template <class...> class _Try, class _Catch>
  struct __mtry_catch_q {
    template <class... _Args>
    using __f = __minvoke<__if_c<__minvocable_q<_Try, _Args...>, __q<_Try>, _Catch>, _Args...>;
  };

  template <class _Try, class _Catch>
  struct __mtry_catch {
    template <class... _Args>
    using __f = __minvoke_if_c<__minvocable<_Try, _Args...>, _Try, _Catch, _Args...>;
  };

  template <class _Fn, class _Default>
  using __mwith_default = __mtry_catch<_Fn, __mconst<_Default>>;

  template <template <class...> class _Fn, class _Default>
  using __mwith_default_q = __mtry_catch_q<_Fn, __mconst<_Default>>;

  template <class _Fn, class _Default, class... _Args>
  using __minvoke_or = __minvoke<__mwith_default<_Fn, _Default>, _Args...>;

  template <template <class...> class _Fn, class _Default, class... _Args>
  using __minvoke_or_q = __minvoke<__mwith_default_q<_Fn, _Default>, _Args...>;

  template <class _Fn, class _Continuation = __q<__mlist>>
  struct __mtransform {
    template <class... _Args>
    using __f = __minvoke<_Continuation, __minvoke<_Fn, _Args>...>;
  };

  template <bool>
  struct __mfold_right_ {
    template <class _Fn, class _State, class _Head, class... _Tail>
    using __f =
      __minvoke<__mfold_right_<sizeof...(_Tail) == 0>, _Fn, __minvoke<_Fn, _State, _Head>, _Tail...>;
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

  template <bool>
  struct __mfold_left_ {
    template <class _Fn, class _State, class _Head, class... _Tail>
    using __f =
      __minvoke<_Fn, __minvoke<__mfold_left_<sizeof...(_Tail) == 0>, _Fn, _State, _Tail...>, _Head>;
  };

  template <>
  struct __mfold_left_<true> { // empty pack
    template <class _Fn, class _State, class...>
    using __f = _State;
  };

  template <class _Init, class _Fn>
  struct __mfold_left {
    template <class... _Args>
    using __f = __minvoke<__mfold_left_<sizeof...(_Args) == 0>, _Fn, _Init, _Args...>;
  };

  // for:: [a] -> (a -> b) -> [b]
  template <class _Tp>
  struct __mfor;

  template <template <class...> class _Ap, class... _As>
  struct __mfor<_Ap<_As...>> {
    template <class _Fn>
    using __f = __minvoke<_Fn, _As...>;
  };

  template <class _Ret, class... _As>
  struct __mfor<_Ret(_As...)> {
    template <class _Fn>
    using __f = __minvoke<_Fn, _Ret, _As...>;
  };

  template <std::size_t... _Ns>
  struct __mfor<__indices<_Ns...>> {
    template <class _Fn>
    using __f = __minvoke<_Fn, __msize_t<_Ns>...>;
  };

  template <template <class _Np, _Np...> class _Cp, class _Np, _Np... _Ns>
  struct __mfor<_Cp<_Np, _Ns...>> {
    template <class _Fn>
    using __f = __minvoke<_Fn, std::integral_constant<_Np, _Ns>...>;
  };

  template <class _What, class... _With>
  struct __mfor<_ERROR_<_What, _With...>> {
    template <class _Fn>
    using __f = _ERROR_<_What, _With...>;
  };

  template <class _Fn, class _List>
  using __mapply = __mcall1<__mfor<_List>, _Fn>;

  template <template <class...> class _Fn, class _List>
  using __mapply_q = __mcall1<__mfor<_List>, __q<_Fn>>;

  template <class _Fn>
  struct __muncurry {
    template <class _List>
    using __f = __mapply<_Fn, _List>;
  };

  template <std::size_t _Ny, class _Ty, class _Continuation = __qq<__mlist>>
  using __mfill_c = __mapply<__mtransform<__mconst<_Ty>, _Continuation>, __make_indices<_Ny>>;

  template <class _Ny, class _Ty, class _Continuation = __qq<__mlist>>
  using __mfill = __mfill_c<_Ny::value, _Ty, _Continuation>;

  template <bool>
  struct __mconcat_ {
    template <
      class... _Ts,
      template <class...> class _Ap = __mlist,
      class... _As,
      template <class...> class _Bp = __mlist,
      class... _Bs,
      template <class...> class _Cp = __mlist,
      class... _Cs,
      template <class...> class _Dp = __mlist,
      class... _Ds,
      class... _Tail
    >
    static auto __f(
      __mlist<_Ts...> *,
      _Ap<_As...> *,
      _Bp<_Bs...> * = nullptr,
      _Cp<_Cs...> * = nullptr,
      _Dp<_Ds...> * = nullptr,
      _Tail *...__tail)
      -> __midentity<decltype(__mconcat_<(sizeof...(_Tail) == 0)>::__f(
        static_cast<__mlist<_Ts..., _As..., _Bs..., _Cs..., _Ds...> *>(nullptr),
        __tail...))>;
  };

  template <>
  struct __mconcat_<true> {
    template <class... _As>
    static auto __f(__mlist<_As...> *) -> __mlist<_As...>;
  };

  template <class _Continuation = __qq<__mlist>>
  struct __mconcat {
    template <class... _Args>
    using __f = __mapply<
      _Continuation,
      decltype(__mconcat_<(sizeof...(_Args) == 0)>::__f({}, static_cast<_Args *>(nullptr)...))
    >;
  };

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
    using __f = __msize_t<(bool(__minvoke<_Fn, _Ts>::value) + ... + 0)>;
  };

  template <class _Tp>
  struct __mcontains {
    template <class... _Args>
    using __f = __mbool<(__same_as<_Tp, _Args> || ...)>;
  };

  template <class _Continuation = __q<__mlist>>
  struct __mpush_back {
    template <class _List, class _Item>
    using __f = __mapply<__mbind_back<_Continuation, _Item>, _List>;
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

  template <template <class...> class _Second, template <class...> class _First>
  struct __mcompose_q {
    template <class... _Args>
    using __f = _Second<_First<_Args...>>;
  };

  template <class _Old, class _New, class _Continuation = __q<__mlist>>
  struct __mreplace {
    template <class... _Args>
    using __f = __minvoke<_Continuation, __if_c<__same_as<_Args, _Old>, _New, _Args>...>;
  };

  template <class _Old, class _Continuation = __q<__mlist>>
  struct __mremove {
    template <class... _Args>
    using __f = __minvoke<
      __mconcat<_Continuation>,
      __if_c<__same_as<_Args, _Old>, __mlist<>, __mlist<_Args>>...
    >;
  };

  template <class _Pred, class _Continuation = __q<__mlist>>
  struct __mremove_if {
    template <class... _Args>
    using __f = __minvoke<
      __mconcat<_Continuation>,
      __if<__minvoke<_Pred, _Args>, __mlist<>, __mlist<_Args>>...
    >;
  };

  template <class _Return>
  struct __qf {
    template <class... _Args>
    using __f = _Return(_Args...);
  };

  template <class _Ty, class...>
  using __mfront_ = _Ty;

  template <class... _As>
  using __mfront = __minvoke_q<__mfront_, _As...>;

  template <class... _As>
    requires(sizeof...(_As) == 1)
  using __msingle = __mfront<_As...>;

  template <class _Default>
  struct __msingle_or {
    template <class... _As>
      requires(sizeof...(_As) <= 1)
    using __f = __mfront<_As..., _Default>;
  };

  namespace __detail {
    template <class _Ty>
    extern __declfn_t<_Ty> __demangle_v;

    template <class _Ty>
    extern __declfn_t<_Ty &> __demangle_v<_Ty &>;

    template <class _Ty>
    extern __declfn_t<_Ty &&> __demangle_v<_Ty &&>;

    template <class _Ty>
    extern __declfn_t<_Ty const &> __demangle_v<_Ty const &>;

    template <class _Ty>
    using __demangle_t = decltype(__demangle_v<_Ty>());

    template <class _Sender>
    using __remangle_t = __copy_cvref_t<_Sender, typename __decay_t<_Sender>::__mangled_t>;
  } // namespace __detail

  // A utility for pretty-printing type names in diagnostics
  template <class _Ty>
  using __demangle_t = __copy_cvref_t<_Ty, __detail::__demangle_t<std::remove_cvref_t<_Ty>>>;

  template <class _Sender>
  using __remangle_t = __minvoke_or_q<__detail::__remangle_t, _Sender, _Sender>;

  namespace __detail {
    //////////////////////////////////////////////////////////////////////////////////////////
    // __get_pretty_name
    template <class>
    struct __xyzzy {
      struct __plugh { };
    };

    constexpr char __type_name_prefix[] = "__xyzzy<";
    constexpr char __type_name_suffix[] = ">::__plugh";

    [[nodiscard]]
    consteval std::string_view __find_pretty_name(std::string_view __sv) noexcept {
      const auto __beg_pos = __sv.find(__type_name_prefix);
      const auto __end_pos = __sv.rfind(__type_name_suffix);

      const auto __start = __beg_pos + sizeof(__type_name_prefix) - 1;
      const auto __len = __end_pos - __start;

      return __sv.substr(__start, __len);
    }

    template <class _Ty>
    [[nodiscard]]
    consteval std::string_view __get_pretty_name_helper() noexcept {
#if STDEXEC_EDG()
      return __detail::__find_pretty_name(std::string_view{STDEXEC_PRETTY_FUNCTION()});
#else
      return __detail::__find_pretty_name(std::source_location::current().function_name());
#endif
    }

    template <class _Ty>
    [[nodiscard]]
    consteval std::string_view __get_pretty_name() noexcept {
      return __detail::__get_pretty_name_helper<typename __xyzzy<_Ty>::__plugh>();
    }
  } // namespace __detail

  ////////////////////////////////////////////////////////////////////////////////////////////
  // __mnameof: get the pretty name of a type _Ty as a string_view at compile time
  template <class _Ty>
  inline constexpr std::string_view __mnameof = __detail::__get_pretty_name<__demangle_t<_Ty>>();

  static_assert(__mnameof<void> == "void");

  template <class _List1, class _List2>
  struct __mzip_with2_
    : __mzip_with2_<__mapply_q<__mlist, _List1>, __mapply_q<__mlist, _List2>> { };

  template <template <class...> class _Cp, class... _Cs, template <class...> class _Dp, class... _Ds>
  struct __mzip_with2_<_Cp<_Cs...>, _Dp<_Ds...>> {
    template <class _Fn, class _Continuation>
    using __f = __minvoke<_Continuation, __minvoke<_Fn, _Cs, _Ds>...>;
  };

  template <class _Fn, class _Continuation = __q<__mlist>>
  struct __mzip_with2 {
    template <class _Cp, class _Dp>
    using __f = __minvoke<__mzip_with2_<_Cp, _Dp>, _Fn, _Continuation>;
  };

  template <class _Ty, class _Uy>
  using __msame_as = __mbool<STDEXEC_IS_SAME(_Ty, _Uy)>;

  template <class _Ty, class _Uy>
  using __mconvertible_to = __mbool<STDEXEC_IS_CONVERTIBLE_TO(_Ty, _Uy)>;

  template <bool>
  struct __mfind_if_ {
    template <class _Fn, class _Continuation, class _Head, class... _Tail>
    using __f = __minvoke_if_c<
      __mcall1<_Fn, _Head>::value,
      __mbind_front<_Continuation, _Head>,
      __mbind_front<__mfind_if_<(sizeof...(_Tail) != 0)>, _Fn, _Continuation>,
      _Tail...
    >;
  };

  template <>
  struct __mfind_if_<false> {
    template <class _Fn, class _Continuation>
    using __f = __minvoke<_Continuation>;
  };

  template <class _Fn, class _Continuation = __q<__mlist>>
  struct __mfind_if {
    template <class... _Args>
    using __f = __minvoke<__mfind_if_<(sizeof...(_Args) != 0)>, _Fn, _Continuation, _Args...>;
  };

  template <class _Needle, class _Continuation = __q<__mlist>>
  struct __mfind {
    template <class... _Args>
    using __f = __mcall<__mfind_if<__mbind_front_q<__msame_as, _Needle>, _Continuation>, _Args...>;
  };

  template <class _Needle>
  struct __mfind_i {
    template <class... _Args>
    using __f =
      __msize_t<(sizeof...(_Args) - __minvoke<__mfind<_Needle, __msize>, _Args...>::value)>;
  };

  template <class... _Booleans>
  using __mand_t = __mbool<(_Booleans::value && ...)>;
  template <class... _Booleans>
  using __mand = __minvoke_q<__mand_t, _Booleans...>;

  template <class... _Booleans>
  using __mor_t = __mbool<(_Booleans::value || ...)>;
  template <class... _Booleans>
  using __mor = __minvoke_q<__mor_t, _Booleans...>;

  template <class _Boolean>
  using __mnot_t = __mbool<!_Boolean::value>;
  template <class _Boolean>
  using __mnot = __minvoke_q<__mnot_t, _Boolean>;

  template <class _Fn>
  struct __mall_of {
    template <class... _Args>
    using __f = __minvoke_q<__mand, __mcall1<_Fn, _Args>...>;
  };

  template <class _Fn>
  struct __mnone_of {
    template <class... _Args>
    using __f = __minvoke_q<__mand, __mnot<__mcall1<_Fn, _Args>>...>;
  };

  template <class _Fn>
  struct __many_of {
    template <class... _Args>
    using __f = __minvoke_q<__mor, __mcall1<_Fn, _Args>...>;
  };

#if !STDEXEC_NO_STD_PACK_INDEXING()
  STDEXEC_PRAGMA_PUSH()
  STDEXEC_PRAGMA_IGNORE_GNU("-Wc++26-extensions")

  template <bool>
  struct __m_at_ {
    template <class _Np, class... _Ts>
    using __f = _Ts...[_Np::value];
  };

  template <class _Np, class... _Ts>
  using __m_at = __minvoke<__m_at_<_Np::value == ~0ul>, _Np, _Ts...>;

  template <std::size_t _Np, class... _Ts>
  using __m_at_c = __minvoke<__m_at_<_Np == ~0ul>, __msize_t<_Np>, _Ts...>;

  STDEXEC_PRAGMA_POP()
#elif STDEXEC_HAS_BUILTIN(__type_pack_element)
  template <bool>
  struct __m_at_ {
    template <class _Np, class... _Ts>
    using __f = __type_pack_element<_Np::value, _Ts...>;
  };

  template <class _Np, class... _Ts>
  using __m_at = __minvoke<__m_at_<_Np::value == ~0ul>, _Np, _Ts...>;

  template <std::size_t _Np, class... _Ts>
  using __m_at_c = __minvoke<__m_at_<_Np == ~0ul>, __msize_t<_Np>, _Ts...>;
#else
  template <std::size_t>
  using __void_ptr = void *;

  template <class _Ty>
  using __mtype_ptr = __mtype<_Ty> *;

  template <class _Ty>
  struct __m_at_;

  template <std::size_t... _Is>
  struct __m_at_<__indices<_Is...>> {
    template <class _Up, class... _Us>
    static _Up __f_(__void_ptr<_Is>..., _Up *, _Us *...);
    template <class... _Ts>
    using __f = __t<decltype(__m_at_::__f_(__mtype_ptr<_Ts>()...))>;
  };

  template <std::size_t _Np, class... _Ts>
  using __m_at_c = __minvoke<__m_at_<__make_indices<_Np>>, _Ts...>;

  template <class _Np, class... _Ts>
  using __m_at = __m_at_c<_Np::value, _Ts...>;
#endif

  template <class _Set, class... _Ty>
  concept __mset_contains = (STDEXEC_IS_BASE_OF(__mtype<_Ty>, _Set) && ...);

  struct __mset_nil;

  namespace __set {
    template <class... _Ts>
    struct __inherit : __mtype<__mset_nil> {
      template <template <class...> class _Fn>
      using rebind = _Fn<_Ts...>;
    };

    template <class _Ty, class... _Ts>
    struct __inherit<_Ty, _Ts...>
      : __mtype<_Ty>
      , __inherit<_Ts...> {
      template <template <class...> class _Fn>
      using rebind = _Fn<_Ty, _Ts...>;
    };

    template <class... _Set>
    auto operator+(__inherit<_Set...> &) -> __inherit<_Set...>;

    template <class... _Set, class _Ty>
    auto operator%(__inherit<_Set...> &, __mtype<_Ty> &) -> __inherit<_Ty, _Set...> &;

    template <class... _Set, class _Ty>
      requires __mset_contains<__inherit<_Set...>, _Ty>
    auto operator%(__inherit<_Set...> &, __mtype<_Ty> &) -> __inherit<_Set...> &;

    template <class _ExpectedSet, class... _Ts>
    concept __mset_eq = (sizeof...(_Ts) == __mapply<__msize, _ExpectedSet>::value)
                     && __mset_contains<_ExpectedSet, _Ts...>;

    template <class _ExpectedSet>
    struct __eq {
      template <class... _Ts>
      using __f = __mbool<__mset_eq<_ExpectedSet, _Ts...>>;
    };
  } // namespace __set

  template <class... _Ts>
  using __mset = __set::__inherit<_Ts...>;

  template <class _Set, class... _Ts>
  using __mset_insert = decltype(+(__declval<_Set &>() % ... % __declval<__mtype<_Ts> &>()));

  template <class... _Ts>
  using __mmake_set = __mset_insert<__mset<>, _Ts...>;

  template <class _Set1, class _Set2>
  concept __mset_eq = __mapply<__set::__eq<_Set1>, _Set2>::value;

  template <class _Continuation = __q<__mlist>>
  struct __munique {
    template <class... _Ts>
    using __f = __mapply<_Continuation, __mmake_set<_Ts...>>;
  };
} // namespace STDEXEC
