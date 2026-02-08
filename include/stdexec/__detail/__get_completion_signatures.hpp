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

// include these after __execution_fwd.hpp
#include "__awaitable.hpp"
#include "__completion_signatures.hpp" // IWYU pragma: export
#include "__connect_awaitable.hpp"
#include "__diagnostics.hpp"
#include "__env.hpp"
#include "__meta.hpp"
#include "__tag_invoke.hpp"
#include "__tuple.hpp" // IWYU pragma: keep for __tuple

namespace STDEXEC {
  namespace __detail {
    template <class _Env>
    struct __promise : __connect_await::__with_await_transform<__promise<_Env>> {
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      auto get_env() const noexcept -> const _Env&;
    };

    // set_value_t(_Type) when _Type is not void, and set_value_t() when _Type is void
    template <class _Type>
    using __single_value_sig_t = __mcall1<__mremove<void, __qf<set_value_t>>, _Type>;
  } // namespace __detail

  /////////////////////////////////////////////////////////////////////////////
  // [exec.getcomplsigs]

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // get_completion_signatures
  STDEXEC_PRAGMA_PUSH()
  // warning C4913: user defined binary operator ',' exists but no overload could convert all operands,
  // default built-in binary operator ',' used
  STDEXEC_PRAGMA_IGNORE_MSVC(4913)

  struct _A_GET_COMPLETION_SIGNATURES_CUSTOMIZATION_RETURNED_A_TYPE_THAT_IS_NOT_A_COMPLETION_SIGNATURES_SPECIALIZATION;

  namespace __cmplsigs {
#define STDEXEC_GET_COMPLSIGS(...)                                                                 \
  STDEXEC_REMOVE_REFERENCE(                                                                        \
    STDEXEC_PP_FRONT(__VA_ARGS__))::template get_completion_signatures<__VA_ARGS__>()

#define STDEXEC_CHECKED_COMPLSIGS(_ARGS, ...)                                                      \
  STDEXEC::__cmplsigs::__checked_complsigs(                                                        \
    __VA_ARGS__, static_cast<__mlist<STDEXEC_PP_EXPAND _ARGS>*>(nullptr))

    template <class _Ty>
    concept __non_sender = !enable_sender<__decay_t<_Ty>>;

    template <__valid_completion_signatures _Completions>
    consteval auto __checked_complsigs(_Completions, void*) {
      return _Completions();
    }

    template <class _Completions, class _Sender, class... _Env>
      requires(!__valid_completion_signatures<_Completions>)
    consteval auto __checked_complsigs(_Completions, __mlist<_Sender, _Env...>*) {
      if constexpr (__merror<_Completions>) {
        return STDEXEC::__throw_compile_time_error(_Completions());
      } else if constexpr (STDEXEC_IS_BASE_OF(dependent_sender_error, _Completions)) {
        return _Completions();
      } else {
        return __throw_compile_time_error<
          _A_GET_COMPLETION_SIGNATURES_CUSTOMIZATION_RETURNED_A_TYPE_THAT_IS_NOT_A_COMPLETION_SIGNATURES_SPECIALIZATION,
          _WITH_COMPLETION_SIGNATURES_(_Completions),
          _WITH_PRETTY_SENDER_<_Sender>,
          __fn_t<_WITH_ENVIRONMENT_, _Env>...
        >();
      }
    }

    template <class _Sender, class... _Env>
    using __co_await_completions_t = //
      completion_signatures<
        __detail::__single_value_sig_t<__await_result_t<_Sender, __detail::__promise<_Env>...>>,
        set_error_t(std::exception_ptr),
        set_stopped_t()
      >;

    template <class _Sender, class... _Env>
    using __legacy_member_result_t = //
      decltype(__declval<_Sender>().get_completion_signatures(__declval<_Env>()...));

    template <class _Sender, class... _Env>
    using __legacy_static_member_result_t =      //
      decltype(STDEXEC_REMOVE_REFERENCE(_Sender) //
               ::static_get_completion_signatures(__declval<_Sender>(), __declval<_Env>()...));

    template <class _Sender>
    using __legacy_member_alias_t = STDEXEC_REMOVE_REFERENCE(_Sender)::completion_signatures;

    template <class _Sender, class... _Env>
    concept __with_legacy_member =
      requires(__declfn_t<_Sender&&> __sndr, __declfn_t<_Env&&>... __env) {
        __sndr().get_completion_signatures(__env()...);
      };

    template <class _Sender, class... _Env>
    concept __with_legacy_static_member =
      requires(__declfn_t<_Sender&&> __sndr, __declfn_t<_Env&&>... __env) {
        STDEXEC_REMOVE_REFERENCE(_Sender)
        ::static_get_completion_signatures(__sndr(), __env()...);
      };

    template <class _Sender, class... _Env>
    concept __with_consteval_static_member = //
      (__non_sender<_Env> && ...)            //
      && requires { STDEXEC_GET_COMPLSIGS(_Sender, _Env...); };

    // [WAR]: see nvbugs#5813160
    template <class _Sender>
    concept __with_non_dependent_consteval_static_member = //
      requires { STDEXEC_GET_COMPLSIGS(_Sender); };

    template <class _Sender, class... _Env>
    concept __with_legacy_tag_invoke =
      __tag_invocable<get_completion_signatures_t, _Sender, _Env...>;

    template <class _Sender, class... _Env>
    concept __with_legacy_non_dependent_tag_invoke =
      (sizeof...(_Env) == 0) && __tag_invocable<get_completion_signatures_t, _Sender, env<>>;

    template <class _Sender>
    concept __with_legacy_member_alias = requires {
      typename STDEXEC_REMOVE_REFERENCE(_Sender)::completion_signatures;
    };

    template <class _Sender, class... _Env>
    concept __with_co_await = __awaitable<_Sender, __detail::__promise<_Env>...>;

    template <class _Sender, class _Env>
    concept __with = __with_legacy_static_member<_Sender, _Env>            //
                  || __with_legacy_member<_Sender, _Env>                   //
                  || __with_legacy_member_alias<_Sender>                   //
                  || __with_consteval_static_member<_Sender, _Env>         //
                  || __with_non_dependent_consteval_static_member<_Sender> //
                  || __with_legacy_tag_invoke<_Sender, _Env>               //
                  || __with_legacy_non_dependent_tag_invoke<_Sender, _Env> //
                  || __with_co_await<_Sender, _Env>;
  } // namespace __cmplsigs

  template <class _Sender, class _Env>
  concept __has_get_completion_signatures =
    requires(__declfn_t<_Sender&&> __sndr, __declfn_t<_Env&&> __env) {
      { transform_sender(__sndr(), __env()) } -> __cmplsigs::__with<_Env>;
    };

  namespace __cmplsigs {
    template <class _Sender>
    [[nodiscard]]
    consteval auto __get_completion_signatures_helper() {
      if constexpr (__with_legacy_member_alias<_Sender>) {
        return STDEXEC_CHECKED_COMPLSIGS((_Sender), __legacy_member_alias_t<_Sender>());
      } else if constexpr (__with_consteval_static_member<_Sender>) {
        return STDEXEC_CHECKED_COMPLSIGS((_Sender), STDEXEC_GET_COMPLSIGS(_Sender));
      } else if constexpr (__with_legacy_static_member<_Sender>) {
        using __completions_t = __legacy_static_member_result_t<_Sender>;
        return STDEXEC_CHECKED_COMPLSIGS((_Sender), __completions_t());
      } else if constexpr (__with_legacy_member<_Sender>) {
        using __completions_t = __legacy_member_result_t<_Sender>;
        return STDEXEC_CHECKED_COMPLSIGS((_Sender), __completions_t());
      } else if constexpr (__with_legacy_tag_invoke<_Sender>) {
        using __completions_t = __tag_invoke_result_t<get_completion_signatures_t, _Sender>;
        return STDEXEC_CHECKED_COMPLSIGS((_Sender), __completions_t());
      } else if constexpr (__with_legacy_non_dependent_tag_invoke<_Sender>) {
        using __completions_t = __tag_invoke_result_t<get_completion_signatures_t, _Sender, env<>>;
        return STDEXEC_CHECKED_COMPLSIGS((_Sender), __completions_t());
      } else if constexpr (__with_co_await<_Sender>) {
        return __co_await_completions_t<_Sender>();
      } else {
        return STDEXEC::__dependent_sender<_Sender>();
      }
    }

    template <class _Sender, class _Env>
    [[nodiscard]]
    consteval auto __get_completion_signatures_helper() {
      if constexpr (__with_legacy_member_alias<_Sender>) {
        return STDEXEC_CHECKED_COMPLSIGS((_Sender, _Env), __legacy_member_alias_t<_Sender>());
      } else if constexpr (__with_consteval_static_member<_Sender, _Env>) {
        return STDEXEC_CHECKED_COMPLSIGS((_Sender, _Env), STDEXEC_GET_COMPLSIGS(_Sender, _Env));
      } else if constexpr (__with_non_dependent_consteval_static_member<_Sender>) {
        return STDEXEC_CHECKED_COMPLSIGS((_Sender, _Env), STDEXEC_GET_COMPLSIGS(_Sender));
      } else if constexpr (__with_legacy_static_member<_Sender, _Env>) {
        using __completions_t = __legacy_static_member_result_t<_Sender, _Env>;
        return STDEXEC_CHECKED_COMPLSIGS((_Sender, _Env), __completions_t());
      } else if constexpr (__with_legacy_member<_Sender, _Env>) {
        using __completions_t = __legacy_member_result_t<_Sender, _Env>;
        return STDEXEC_CHECKED_COMPLSIGS((_Sender, _Env), __completions_t());
      } else if constexpr (__with_legacy_tag_invoke<_Sender, _Env>) {
        using __completions_t = __tag_invoke_result_t<get_completion_signatures_t, _Sender, _Env>;
        return STDEXEC_CHECKED_COMPLSIGS((_Sender, _Env), __completions_t());
      } else if constexpr (__with_co_await<_Sender, _Env>) {
        return __co_await_completions_t<_Sender, _Env>();
      } else {
        return __unrecognized_sender_error_t<_Sender, _Env>();
      }
    }

    // For backwards compatibility
    struct get_completion_signatures_t {
      template <class _Sender>
      constexpr auto operator()(_Sender&&) const noexcept {
        return __cmplsigs::__get_completion_signatures_helper<_Sender>();
      }

      template <class _Sender, class _Env>
      constexpr auto operator()(_Sender&&, const _Env&) const noexcept {
        using __new_sndr_t = transform_sender_result_t<_Sender, _Env>;
        static_assert(!__merror<__new_sndr_t>);
        return __cmplsigs::__get_completion_signatures_helper<__new_sndr_t, _Env>();
      }
    };
  } // namespace __cmplsigs

  STDEXEC_PRAGMA_POP()

  template <class _Sender>
  consteval auto get_completion_signatures() {
    return __cmplsigs::__get_completion_signatures_helper<_Sender>();
  }

  template <class _Sender, class _Env>
    requires __has_get_completion_signatures<_Sender, _Env>
  consteval auto get_completion_signatures() {
    using __new_sndr_t = transform_sender_result_t<_Sender, _Env>;
    static_assert(!__merror<__new_sndr_t>);
    return __cmplsigs::__get_completion_signatures_helper<__new_sndr_t, _Env>();
  }

  // Legacy interface:
  template <class _Sender, class... _Env>
    requires(sizeof...(_Env) <= 1)
  constexpr auto get_completion_signatures(_Sender&&, const _Env&...) noexcept {
    return STDEXEC::get_completion_signatures<_Sender, _Env...>();
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  // An minimally constrained alias for the result of get_completion_signatures:
  template <class _Sender, class... _Env>
    requires enable_sender<__decay_t<_Sender>>
  using __completion_signatures_of_t =
    decltype(STDEXEC::get_completion_signatures<_Sender, _Env...>());

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  // __get_child_completion_signatures
  template <class _Parent, class _Child, class... _Env>
  [[nodiscard]]
  consteval auto __get_child_completion_signatures() {
    return STDEXEC::get_completion_signatures<
      __copy_cvref_t<_Parent, _Child>,
      __fwd_env_t<_Env>...
    >();
  }

#if STDEXEC_NO_STD_CONSTEXPR_EXCEPTIONS()
  template <class _Sender>
  concept __is_dependent_sender =
    __std::derived_from<__completion_signatures_of_t<_Sender>, dependent_sender_error>;
#else  // ^^^ no constexpr exceptions ^^^ / vvv constexpr exceptions vvv
  // When asked for its completions without an envitonment, a dependent sender
  // will throw an exception of a type derived from `dependent_sender_error`.
  template <class _Sender>
  [[nodiscard]]
  consteval bool __is_dependent_sender_helper() noexcept try {
    (void) STDEXEC::get_completion_signatures<_Sender>();
    return false; // didn't throw, not a dependent sender
  } catch (dependent_sender_error&) {
    return true;
  } catch (...) {
    return false; // different kind of exception was thrown; not a dependent sender
  }

  template <class _Sender>
  concept __is_dependent_sender = __mbool<__is_dependent_sender_helper<_Sender>()>::value;
#endif // ^^^ constexpr exceptions ^^^

  template <class _WantedTag, class _Sender, class _Env, class _Tuple, class _Variant>
  using __gather_completions_of_t = __gather_completions_t<
    _WantedTag,
    __completion_signatures_of_t<_Sender, _Env>,
    _Tuple,
    _Variant
  >;

  template <
    class _Sender,
    class _Env = env<>,
    class _Tuple = __qq<__decayed_std_tuple>,
    class _Variant = __qq<__std_variant>
  >
  using __value_types_of_t =
    __value_types_t<__completion_signatures_of_t<_Sender, _Env>, _Tuple, _Variant>;

  template <
    class _Sender,
    class _Env = env<>,
    template <class...> class _Tuple = __decayed_std_tuple,
    template <class...> class _Variant = __std_variant
  >
  using value_types_of_t =
    __value_types_t<__completion_signatures_of_t<_Sender, _Env>, __q<_Tuple>, __q<_Variant>>;

  template <
    class _Sender,
    class _Env = env<>,
    class _Variant = __qq<__std_variant>,
    class _Transform = __q1<__midentity>
  >
  using __error_types_of_t =
    __error_types_t<__completion_signatures_of_t<_Sender, _Env>, _Variant, _Transform>;

  template <class _Sender, class _Env = env<>, template <class...> class _Variant = __std_variant>
  using error_types_of_t =
    __error_types_t<__completion_signatures_of_t<_Sender, _Env>, __q<_Variant>>;

  template <class _Sender, class... _Env>
    requires __valid_completion_signatures<__completion_signatures_of_t<_Sender, _Env...>>
  inline constexpr bool sends_stopped =
    __sends_stopped<__completion_signatures_of_t<_Sender, _Env...>>;

  template <class _Tag, class _Sender, class... _Env>
  using __count_of =
    __msize_t<__detail::__count_of<_Tag, __completion_signatures_of_t<_Sender, _Env...>>>;
} // namespace STDEXEC
