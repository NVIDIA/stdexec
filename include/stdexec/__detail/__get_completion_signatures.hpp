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
#include "__diagnostics.hpp"
#include "__env.hpp"
#include "__meta.hpp"
#include "__tag_invoke.hpp"
#include "__tuple.hpp" // IWYU pragma: keep for __tuple

namespace STDEXEC {
  namespace __detail {
    template <class _Tp, class _Promise>
    concept __has_as_awaitable_member = requires(_Tp&& __t, _Promise& __promise) {
      static_cast<_Tp&&>(__t).as_awaitable(__promise);
    };

    // A partial duplicate of with_awaitable_senders to avoid circular type dependencies
    template <class _Promise>
    struct __with_await_transform {
      template <class _Ty>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      auto await_transform(_Ty&& __value) noexcept -> _Ty&& {
        return static_cast<_Ty&&>(__value);
      }

      template <class _Ty>
        requires __has_as_awaitable_member<_Ty, _Promise&>
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      auto await_transform(_Ty&& __value)
        noexcept(noexcept(__declval<_Ty>().as_awaitable(__declval<_Promise&>())))
          -> decltype(__declval<_Ty>().as_awaitable(__declval<_Promise&>())) {
        return static_cast<_Ty&&>(__value).as_awaitable(static_cast<_Promise&>(*this));
      }

      template <class _Ty>
        requires __has_as_awaitable_member<_Ty, _Promise&>
              || tag_invocable<as_awaitable_t, _Ty, _Promise&>
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      auto await_transform(_Ty&& __value)
        noexcept(nothrow_tag_invocable<as_awaitable_t, _Ty, _Promise&>)
          -> tag_invoke_result_t<as_awaitable_t, _Ty, _Promise&> {
        return tag_invoke(as_awaitable, static_cast<_Ty&&>(__value), static_cast<_Promise&>(*this));
      }
    };

    template <class _Env>
    struct __promise : __with_await_transform<__promise<_Env>> {
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      auto get_env() const noexcept -> const _Env&;
    };
  } // namespace __detail

  /////////////////////////////////////////////////////////////////////////////
  // [exec.getcomplsigs]
#if STDEXEC_NO_STD_CONSTEXPR_EXCEPTIONS()

  template <class... _What, class... _Values>
  [[nodiscard]]
  consteval auto __invalid_completion_signature(_Values...) -> __mexception<_What...> {
    return {};
  }

#else // ^^^ no constexpr exceptions ^^^ / vvv constexpr exceptions vvv

  // C++26, https://wg21.link/p3068
  template <class _What, class... _More, class... _Values>
  [[noreturn, nodiscard]]
  consteval auto __invalid_completion_signature([[maybe_unused]] _Values... __values)
    -> completion_signatures<> {
    if constexpr (__same_as<_What, dependent_sender_error>) {
      throw __mexception<dependent_sender_error, _More...>();
    } else if constexpr (sizeof...(_Values) == 1) {
      throw __sender_type_check_failure<_Values..., _What, _More...>(__values...);
    } else {
      throw __sender_type_check_failure<__tuple<_Values...>, _What, _More...>(__tuple{__values...});
    }
  }

#endif // ^^^ constexpr exceptions ^^^

  template <class... _What>
  [[nodiscard]]
  consteval auto __invalid_completion_signature(__mexception<_What...>) {
    return STDEXEC::__invalid_completion_signature<_What...>();
  }

  // Returns _Sender if sizeof...(_Env) == 0, otherwise it is the result of applying
  // transform_sender_result_t to _Sender with _Env.
  template <class _Sender, class... _Env>
  using __maybe_transform_sender_t =
    __mmemoize<__mfold_right<_Sender, __q<transform_sender_result_t>>, _Env...>;

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

#define STDEXEC_CHECKED_COMPLSIGS(_SENDER, _ENV, ...)                                              \
  (static_cast<void>(__VA_ARGS__),                                                                 \
   STDEXEC::__cmplsigs::__checked_complsigs<decltype(__VA_ARGS__)>(                                \
     static_cast<__types<_SENDER, _ENV...>*>(nullptr)))

    template <class _Ty>
    concept __non_sender = !enable_sender<__decay_t<_Ty>>;

    template <__valid_completion_signatures _Completions>
    consteval auto __checked_complsigs(void*) {
      return _Completions();
    }

    template <class _Completions, class _Sender, class... _Env>
      requires(!__valid_completion_signatures<_Completions>)
    consteval auto __checked_complsigs(__types<_Sender, _Env...>*) {
      if constexpr (__merror<_Completions>) {
        return STDEXEC::__invalid_completion_signature(_Completions());
      } else if constexpr (STDEXEC_IS_BASE_OF(dependent_sender_error, _Completions)) {
        return _Completions();
      } else {
        return __invalid_completion_signature<
          _A_GET_COMPLETION_SIGNATURES_CUSTOMIZATION_RETURNED_A_TYPE_THAT_IS_NOT_A_COMPLETION_SIGNATURES_SPECIALIZATION,
          _WITH_COMPLETION_SIGNATURES_(_Completions),
          _WITH_SENDER_<_Sender>,
          _WITH_ENVIRONMENT_<_Env>...
        >();
      }
    }

    template <class _Sender, __non_sender... _Env>
    using __member_result_t = //
      decltype(__declval<_Sender>().get_completion_signatures(__declval<_Env>()...));

    template <class _Sender, __non_sender... _Env>
    using __static_member_result_t =             //
      decltype(STDEXEC_REMOVE_REFERENCE(_Sender) //
               ::static_get_completion_signatures(__declval<_Sender>(), __declval<_Env>()...));

    template <class _Sender, class... _Env>
    concept __with_member = __mvalid<__member_result_t, _Sender, _Env...>;

    template <class _Sender>
    using __member_alias_t = STDEXEC_REMOVE_REFERENCE(_Sender)::completion_signatures;

    template <class _Sender, class... _Env>
    concept __with_static_member = __mvalid<__static_member_result_t, _Sender, _Env...>;

    template <class _Sender, class... _Env>
    concept __with_consteval_static_member = //
      (__non_sender<_Env> && ...)            //
      && requires { STDEXEC_GET_COMPLSIGS(_Sender, _Env...); };

    // [WAR]: see nvbugs#5813160
    template <class _Sender>
    concept __with_non_dependent_consteval_static_member = //
      requires { STDEXEC_GET_COMPLSIGS(_Sender); };

    template <class _Sender, class... _Env>
    concept __with_tag_invoke = tag_invocable<get_completion_signatures_t, _Sender, _Env...>;

    template <class _Sender, class... _Env>
    concept __with_legacy_tag_invoke = (sizeof...(_Env) == 0)
                                    && tag_invocable<get_completion_signatures_t, _Sender, env<>>;

    template <class _Sender>
    concept __with_member_alias = __mvalid<__member_alias_t, _Sender>;

    template <class _Sender, class... _Env>
    [[nodiscard]]
    consteval auto __get_completion_signatures_helper() {
      using namespace __cmplsigs;
      if constexpr (__with_static_member<_Sender, _Env...>) {
        using _Result = __static_member_result_t<_Sender, _Env...>;
        return STDEXEC_CHECKED_COMPLSIGS(_Sender, _Env, _Result());
      } else if constexpr (__with_member<_Sender, _Env...>) {
        return STDEXEC_CHECKED_COMPLSIGS(_Sender, _Env, __member_result_t<_Sender, _Env...>());
      } else if constexpr (__with_member_alias<_Sender>) {
        return STDEXEC_CHECKED_COMPLSIGS(_Sender, _Env, __member_alias_t<_Sender>());
      } else if constexpr (__with_consteval_static_member<_Sender, _Env...>) {
        return STDEXEC_CHECKED_COMPLSIGS(_Sender, _Env, STDEXEC_GET_COMPLSIGS(_Sender, _Env...));
      } else if constexpr (__with_non_dependent_consteval_static_member<_Sender>) {
        return STDEXEC_CHECKED_COMPLSIGS(_Sender, _Env, STDEXEC_GET_COMPLSIGS(_Sender));
      } else if constexpr (__with_tag_invoke<_Sender, _Env...>) {
        //__deprecated_tag_invoke_completion_signatures_warning(); // NOLINT(deprecated-declarations)
        using _Result = tag_invoke_result_t<get_completion_signatures_t, _Sender, _Env...>;
        return STDEXEC_CHECKED_COMPLSIGS(_Sender, _Env, _Result());
      } else if constexpr (__with_legacy_tag_invoke<_Sender, _Env...>) {
        // This branch is strictly for backwards compatibility
        using _Result = tag_invoke_result_t<get_completion_signatures_t, _Sender, env<>>;
        return STDEXEC_CHECKED_COMPLSIGS(_Sender, _Env, _Result());
      } else if constexpr (bool(__awaitable<_Sender, __detail::__promise<_Env>...>)) {
        // [WAR] The explicit cast to bool above is to work around a bug in nvc++ (nvbug#4707793)
        using _Result = __await_result_t<_Sender, __detail::__promise<_Env>...>;
        using _ValueSig = __minvoke<__mremove<void, __qf<set_value_t>>, _Result>;
        return completion_signatures<_ValueSig, set_error_t(std::exception_ptr), set_stopped_t()>();
      } else if constexpr (sizeof...(_Env) == 0) {
        return STDEXEC::__dependent_sender<_Sender>();
      } else if constexpr ((__is_debug_env<_Env> || ...)) {
        // This ought to cause a hard error that indicates where the problem is.
        using _Completions [[maybe_unused]] = decltype(STDEXEC_GET_COMPLSIGS(_Sender, _Env...));
        return __debug::__completion_signatures();
      } else {
        return __unrecognized_sender_error<_Sender, _Env...>();
      }
    }

    // For backwards compatibility
    struct get_completion_signatures_t {
      template <class _Sender, class... _Env>
      constexpr auto operator()(_Sender&&, const _Env&...) const noexcept {
        // _NewSender is _Sender if sizeof...(_Env) == 0, otherwise it is the result of applying
        // transform_sender_result_t to _Sender with _Env.
        using _NewSender = __maybe_transform_sender_t<_Sender, _Env...>;
        return __cmplsigs::__get_completion_signatures_helper<_NewSender, _Env...>();
      }
    };
  } // namespace __cmplsigs

  STDEXEC_PRAGMA_POP()

  template <class _Sender, class... _Env>
    requires(sizeof...(_Env) <= 1)
  [[nodiscard]]
  consteval auto get_completion_signatures() -> __well_formed_completions auto {
    using _NewSender = __maybe_transform_sender_t<_Sender, _Env...>;
    if constexpr (__merror<_NewSender>) {
      // Computing the type of the transformed sender returned an error type. Propagate it.
      return STDEXEC::__invalid_completion_signature(_NewSender());
    } else {
      return __cmplsigs::__get_completion_signatures_helper<_NewSender, _Env...>();
    }
  }

  // Legacy interface:
  template <class _Sender, class... _Env>
    requires(sizeof...(_Env) <= 1)
  constexpr auto get_completion_signatures(_Sender&&, const _Env&...) noexcept //
    -> __well_formed_completions auto {
    return STDEXEC::get_completion_signatures<_Sender, _Env...>();
  }

  // An minimally constrained alias for get_completion_signatures:
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
