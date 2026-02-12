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
#include "../functional.hpp"
#include "__completion_signatures.hpp"
#include "__completion_signatures_of.hpp"
#include "__concepts.hpp"
#include "__debug.hpp" // IWYU pragma: keep
#include "__get_completion_signatures.hpp"
#include "__meta.hpp"

#include <exception>

namespace STDEXEC {
  namespace __cmplsigs {
#if STDEXEC_NO_STD_CONSTEXPR_EXCEPTIONS()
    // Without constexpr exceptions, we cannot always produce a valid
    // completion_signatures type. We must permit get_completion_signatures to return an
    // error type because we can't throw it.
    template <class _Completions>
    concept __well_formed_completions_helper =
      __valid_completion_signatures<_Completions>
      || STDEXEC_IS_BASE_OF(STDEXEC::dependent_sender_error, _Completions)
      || __is_instance_of<_Completions, _ERROR_>;
#else
    // When we have constexpr exceptions, we can require that get_completion_signatures
    // always produces a valid completion_signatures type.
    template <class _Completions>
    concept __well_formed_completions_helper = __valid_completion_signatures<_Completions>;
#endif
  } // namespace __cmplsigs

  using __cmplsigs::get_completion_signatures_t;

  // The cast to bool is to hide the disjunction in __well_formed_completions_helper.
  template <class _Completions>
  concept __well_formed_completions = bool(
    __cmplsigs::__well_formed_completions_helper<_Completions>);

  namespace __cmplsigs {
    //////////////////////////////////////////////////////////////////////////////////////////////////
    template <template <class...> class _Tuple, class _Tag, class... _Args>
    constexpr auto __for_each_sig(_Tag (*)(_Args...)) -> _Tuple<_Tag, _Args...>;

    template <class _Sig, template <class...> class _Tuple>
    using __for_each_sig_t = decltype(__cmplsigs::__for_each_sig<_Tuple>(
      static_cast<_Sig*>(nullptr)));

    template <
      template <class...> class _Tuple,
      template <class...> class _Variant,
      class... _More,
      class... _What
    >
    constexpr auto __for_each_completion_signature_fn(_ERROR_<_What...>**) -> _ERROR_<_What...>;
    template <
      template <class...> class _Tuple,
      template <class...> class _Variant,
      class... _More,
      class... _Sigs
    >
    constexpr auto __for_each_completion_signature_fn(completion_signatures<_Sigs...>**)
      -> _Variant<__for_each_sig_t<_Sigs, _Tuple>..., _More...>;
  } // namespace __cmplsigs

  template <
    class _Sigs,
    template <class...> class _Tuple,
    template <class...> class _Variant,
    class... _More
  >
  using __for_each_completion_signature_t =
    decltype(__cmplsigs::__for_each_completion_signature_fn<_Tuple, _Variant, _More...>(
      static_cast<_Sigs**>(nullptr)));

  namespace __cmplsigs {
    ////////////////////////////////////////////////////////////////////////////////////////////////
    template <
      template <class...> class _SetVal,
      template <class...> class _SetErr,
      class _SetStp,
      class... _Values
    >
    constexpr auto __transform_sig(set_value_t (*)(_Values...)) -> _SetVal<_Values...>;

    template <
      template <class...> class _SetVal,
      template <class...> class _SetErr,
      class _SetStp,
      class _Error
    >
    constexpr auto __transform_sig(set_error_t (*)(_Error)) -> _SetErr<_Error>;

    template <template <class...> class _SetVal, template <class...> class _SetErr, class _SetStp>
    constexpr auto __transform_sig(set_stopped_t (*)()) -> _SetStp;

    template <
      class _Sig,
      template <class...> class _SetVal,
      template <class...> class _SetErr,
      class _SetStp
    >
    using __transform_sig_t = decltype(__cmplsigs::__transform_sig<_SetVal, _SetErr, _SetStp>(
      static_cast<_Sig*>(nullptr)));

    template <
      template <class...> class _SetVal,
      template <class...> class _SetErr,
      class _SetStp,
      template <class...> class _Variant,
      class... _More,
      class... _What
    >
    constexpr auto __transform_sigs_fn(_ERROR_<_What...>**) -> _ERROR_<_What...>;

    template <
      template <class...> class _SetVal,
      template <class...> class _SetErr,
      class _SetStp,
      template <class...> class _Variant,
      class... _More,
      class... _Sigs
    >
    constexpr auto __transform_sigs_fn(completion_signatures<_Sigs...>**)
      -> _Variant<__transform_sig_t<_Sigs, _SetVal, _SetErr, _SetStp>..., _More...>;
  } // namespace __cmplsigs

  template <
    class _Sigs,
    template <class...> class _SetVal,
    template <class...> class _SetErr,
    class _SetStp,
    template <class...> class _Variant,
    class... _More
  >
  using __transform_completion_signatures_t =
    decltype(__cmplsigs::__transform_sigs_fn<_SetVal, _SetErr, _SetStp, _Variant, _More...>(
      static_cast<_Sigs**>(nullptr)));

  namespace __cmplsigs {
    ////////////////////////////////////////////////////////////////////////////////////////////////
    template <class _WantedTag, bool = true>
    struct __gather_sigs_fn;

    template <>
    struct __gather_sigs_fn<set_value_t> {
      template <
        class _Sigs,
        template <class...> class _Then,
        template <class...> class _Else,
        template <class...> class _Variant,
        class... _More
      >
      using __f = __transform_completion_signatures_t<
        _Sigs,
        _Then,
        __mbind_front_q<_Else, set_error_t>::template __f,
        _Else<set_stopped_t>,
        _Variant,
        _More...
      >;
    };

    template <>
    struct __gather_sigs_fn<set_error_t> {
      template <
        class _Sigs,
        template <class...> class _Then,
        template <class...> class _Else,
        template <class...> class _Variant,
        class... _More
      >
      using __f = __transform_completion_signatures_t<
        _Sigs,
        __mbind_front_q<_Else, set_value_t>::template __f,
        _Then,
        _Else<set_stopped_t>,
        _Variant,
        _More...
      >;
    };

    template <>
    struct __gather_sigs_fn<set_stopped_t> {
      template <
        class _Sigs,
        template <class...> class _Then,
        template <class...> class _Else,
        template <class...> class _Variant,
        class... _More
      >
      using __f = __transform_completion_signatures_t<
        _Sigs,
        __mbind_front_q<_Else, set_value_t>::template __f,
        __mbind_front_q<_Else, set_error_t>::template __f,
        _Then<>,
        _Variant,
        _More...
      >;
    };

    template <class _WantedTag>
    struct __gather_sigs_fn<_WantedTag, false> {
      template <
        class _Error,
        template <class...> class,
        template <class...> class,
        template <class...> class,
        class...
      >
      using __f = _Error;
    };

    template <class... _Values>
    using __default_set_value = completion_signatures<set_value_t(_Values...)>;

    template <class... _Error>
    using __default_set_error = completion_signatures<set_error_t(_Error...)>;

    template <class _Tag, class... _Args>
    using __default_completion = completion_signatures<_Tag(_Args...)>;
  } // namespace __cmplsigs

  template <
    class _Sigs,
    class _WantedTag,
    template <class...> class _Then,
    template <class...> class _Else,
    template <class...> class _Variant,
    class... _More
  >
  using __gather_completion_signatures_t = __cmplsigs::__gather_sigs_fn<
    _WantedTag,
    __ok<_Sigs>
  >::template __f<_Sigs, _Then, _Else, _Variant, _More...>;

  /////////////////////////////////////////////////////////////////////////////
  // transform_completion_signatures
  // ==========================

  // `transform_completion_signatures` takes a sender, and environment, and a bunch of other
  // template arguments for munging the completion signatures of a sender in interesting ways.

  //  ```c++
  //  template <class... Args>
  //    using __default_set_value = completion_signatures<set_value_t(Args...)>;

  //  template <class Err>
  //    using __default_set_error = completion_signatures<set_error_t(Err)>;

  //  template <
  //    class Completions,
  //    class AdditionalSigs = completion_signatures<>,
  //    template <class...> class SetValue = __default_set_value,
  //    template <class> class SetError = __default_set_error,
  //    class SetStopped = completion_signatures<set_stopped_t()>>
  //  using transform_completion_signatures =
  //    completion_signatures< ... >;
  //  ```

  //  * `SetValue` : an alias template that accepts a set of value types and returns an instance of
  //    `completion_signatures`.

  //  * `SetError` : an alias template that accepts an error types and returns a an instance of
  //    `completion_signatures`.

  //  * `SetStopped` : an instantiation of `completion_signatures` with a list of completion
  //    signatures `Sigs...` to the added to the list if the sender can complete with a stopped
  //    signal.

  //  * `AdditionalSigs` : an instantiation of `completion_signatures` with a list of completion
  //    signatures `Sigs...` to the added to the list unconditionally.

  //  `transform_completion_signatures` does the following:

  //  * Let `VCs...` be a pack of the `completion_signatures` types in the `__typelist` named by
  //    `value_types_of_t<Sndr, Env, SetValue, __typelist>`, and let `Vs...` be the concatenation of
  //    the packs that are template arguments to each `completion_signature` in `VCs...`.

  //  * Let `ECs...` be a pack of the `completion_signatures` types in the `__typelist` named by
  //    `error_types_of_t<Sndr, Env, __errorlist>`, where `__errorlist` is an alias template such
  //    that `__errorlist<Ts...>` names `__typelist<SetError<Ts>...>`, and let `Es...` be the
  //    concatenation of the packs that are the template arguments to each `completion_signature` in
  //    `ECs...`.

  //  * Let `Ss...` be an empty pack if `sends_stopped<Sndr, Env>` is `false`; otherwise, a pack
  //    containing the template arguments of the `completion_signatures` instantiation named by
  //    `SetStopped`.

  //  * Let `MoreSigs...` be a pack of the template arguments of the `completion_signatures`
  //    instantiation named by `AdditionalSigs`.

  //  Then `transform_completion_signatures<Completions, AdditionalSigs, SetValue, SetError,
  //  SendsStopped>` names the type `completion_signatures< Sigs... >` where `Sigs...` is the unique
  //  set of types in `[Vs..., Es..., Ss..., MoreSigs...]`.

  //  If any of the above type computations are ill-formed, `transform_completion_signatures<Sndr,
  //  Env, AdditionalSigs, SetValue, SetError, SendsStopped>` is ill-formed.
  template <
    class _Sigs,
    class _MoreSigs = completion_signatures<>,
    template <class...> class _ValueTransform = __cmplsigs::__default_set_value,
    template <class...> class _ErrorTransform = __cmplsigs::__default_set_error,
    class _StoppedSigs = completion_signatures<set_stopped_t()>
  >
  using transform_completion_signatures = __transform_completion_signatures_t<
    _Sigs,
    _ValueTransform,
    _ErrorTransform,
    _StoppedSigs,
    __mtry_q<__concat_completion_signatures_t>::__f,
    _MoreSigs
  >;

  template <
    class _Sndr,
    class _Env = env<>,
    class _MoreSigs = completion_signatures<>,
    template <class...> class _ValueTransform = __cmplsigs::__default_set_value,
    template <class...> class _ErrorTransform = __cmplsigs::__default_set_error,
    class _StoppedSigs = completion_signatures<set_stopped_t()>
  >
  using transform_completion_signatures_of = transform_completion_signatures<
    completion_signatures_of_t<_Sndr, _Env>,
    _MoreSigs,
    _ValueTransform,
    _ErrorTransform,
    _StoppedSigs
  >;

  struct _IN_TRANSFORM_COMPLETION_SIGNATURES_;
  struct _A_TRANSFORM_FUNCTION_RETURNED_A_TYPE_THAT_IS_NOT_A_COMPLETION_SIGNATURES_SPECIALIZATION_;
  struct _COULD_NOT_CALL_THE_TRANSFORM_FUNCTION_WITH_THE_GIVEN_TEMPLATE_ARGUMENTS_;

  namespace __cmplsigs {
    template <class _Fn, class... _Args>
    using __transform_result_t = decltype(__declval<_Fn>().template operator()<_Args...>());

    template <class _SetTag, class... _Args, class _Fn>
    [[nodiscard]]
    consteval auto
      __transform_expr(const _Fn& __fn) -> __transform_result_t<const _Fn&, _SetTag, _Args...> {
      return __fn.template operator()<_SetTag, _Args...>();
    }

    template <class _Fn>
    [[nodiscard]]
    consteval auto __transform_expr(const _Fn& __fn) -> __call_result_t<const _Fn&> {
      return __fn();
    }

    template <class _Fn, class... _Args>
    using __transform_expr_t = decltype(__cmplsigs::__transform_expr<_Args...>(
      __declval<const _Fn&>()));

    // transform_completion_signatures:
    template <class... _Args, class _Fn>
    [[nodiscard]]
    consteval auto __apply_transform(const _Fn& __fn) {
      if constexpr (__minvocable_q<__transform_expr_t, _Fn, _Args...>) {
        using __completions_t = __transform_expr_t<_Fn, _Args...>;
        if constexpr (__well_formed_completions<__completions_t>) {
          return __cmplsigs::__transform_expr<_Args...>(__fn);
        } else {
          (void) __cmplsigs::__transform_expr<_Args...>(__fn); // potentially throwing
          return STDEXEC::__throw_compile_time_error<
            _IN_TRANSFORM_COMPLETION_SIGNATURES_,
            _A_TRANSFORM_FUNCTION_RETURNED_A_TYPE_THAT_IS_NOT_A_COMPLETION_SIGNATURES_SPECIALIZATION_,
            _WITH_FUNCTION_(_Fn),
            _WITH_ARGUMENTS_(_Args...)
          >();
        }
      } else {
        return STDEXEC::__throw_compile_time_error<
          _IN_TRANSFORM_COMPLETION_SIGNATURES_,
          _COULD_NOT_CALL_THE_TRANSFORM_FUNCTION_WITH_THE_GIVEN_TEMPLATE_ARGUMENTS_,
          _WITH_FUNCTION_(_Fn),
          _WITH_ARGUMENTS_(_Args...)
        >();
      }
    }

    template <class _ValueFn, class _ErrorFn, class _StoppedFn>
    struct __transform_one {
      template <class _SetTag, class... _Args>
      [[nodiscard]]
      consteval auto operator()(_SetTag (*)(_Args...)) const {
        if constexpr (_SetTag() == set_value) {
          return __cmplsigs::__apply_transform<_Args...>(__value_fn);
        } else if constexpr (_SetTag() == set_error) {
          return __cmplsigs::__apply_transform<_Args...>(__error_fn);
        } else {
          return __cmplsigs::__apply_transform<_Args...>(__stopped_fn);
        }
      }

      _ValueFn __value_fn;
      _ErrorFn __error_fn;
      _StoppedFn __stopped_fn;
    };

    template <class _TransformOne>
    struct __transform_all_fn {
      _TransformOne __tfx1;

      template <class... Sigs>
      [[nodiscard]]
      consteval auto operator()(Sigs*... sigs) const {
        return __concat_completion_signatures(__tfx1(sigs)...);
      }
    };

    template <class _TransformOne>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
      __transform_all_fn(_TransformOne) -> __transform_all_fn<_TransformOne>;
  } // namespace __cmplsigs

  template <class _SetTag>
  struct __keep_completion {
    template <class... _Ts>
    consteval auto operator()() const noexcept -> completion_signatures<_SetTag(_Ts...)> {
      return {};
    }
  };

  struct __ignore_completion {
    template <class...>
    consteval auto operator()() const noexcept -> completion_signatures<> {
      return {};
    }
  };

  template <class _SetTag, class _Fn, class... _AlgoTag>
  struct __transform_arguments {
    template <class... _Args>
    consteval auto operator()() const noexcept {
      if constexpr ((__minvocable<_Fn, _Args> && ...)) {
        return completion_signatures<_SetTag(__minvoke<_Fn, _Args>...)>();
      } else {
        return (__check_transform<_Args>(), ...);
      }
    }

   private:
    template <class _Arg>
    static consteval auto __check_transform() {
      if constexpr (__minvocable<_Fn, _Arg>) {
        return __msuccess();
      } else {
        return STDEXEC::__throw_compile_time_error<
          _WHERE_(_IN_ALGORITHM_, _AlgoTag)...,
          _FUNCTION_IS_NOT_CALLABLE_WITH_THE_GIVEN_ARGUMENTS_,
          _WITH_METAFUNCTION_(_Fn),
          _WITH_ARGUMENTS_(_Arg)
        >();
      }
    }
  };

  template <class _SetTag, class... _AlgoTag>
  struct __decay_arguments {
    template <class... _Args>
    consteval auto operator()() const noexcept {
      if constexpr (__decay_copyable<_Args...>) {
        return completion_signatures<_SetTag(__decay_t<_Args>...)>();
      } else {
        // NB: this uses an overloaded comma operator on the _ERROR_ type to find an error
        // in a pack of types.
        return (__check_decay<_Args>(), ...);
      }
    }

   private:
    template <class _Arg>
    static consteval auto __check_decay() {
      if constexpr (__decay_copyable<_Arg>) {
        return __msuccess();
      } else {
        return STDEXEC::__throw_compile_time_error<
          _WHERE_(_IN_ALGORITHM_, _AlgoTag)...,
          _TYPE_IS_NOT_DECAY_COPYABLE_,
          _WITH_TYPE_<_Arg>
        >();
      }
    }
  };

  template <
    class _Completions,
    class _ValueFn = __keep_completion<set_value_t>,
    class _ErrorFn = __keep_completion<set_error_t>,
    class _StoppedFn = __keep_completion<set_stopped_t>,
    class _ExtraSigs = completion_signatures<>
  >
  consteval auto __transform_completion_signatures(
    _Completions,
    _ValueFn __value_fn = {},
    _ErrorFn __error_fn = {},
    _StoppedFn __stopped_fn = {},
    _ExtraSigs = {}) {
    STDEXEC_COMPLSIGS_LET(__completions, _Completions{}) {
      STDEXEC_COMPLSIGS_LET(__extra_sigs, _ExtraSigs{}) {
        __cmplsigs::__transform_one<_ValueFn, _ErrorFn, _StoppedFn> __tfx1{
          __value_fn, __error_fn, __stopped_fn};
        return __concat_completion_signatures(
          __completions.__apply(__cmplsigs::__transform_all_fn(__tfx1)), __extra_sigs);
      }
    }
  }

  using __eptr_completion = completion_signatures<set_error_t(std::exception_ptr)>;

  template <class _NoExcept>
  using __eptr_completion_unless_t = __if<_NoExcept, completion_signatures<>, __eptr_completion>;

  template <bool _NoExcept>
  using __eptr_completion_unless = __eptr_completion_unless_t<__mbool<_NoExcept>>;

  template <
    class _Sender,
    class _Env = env<>,
    class _More = completion_signatures<>,
    class _SetValue = __qq<__cmplsigs::__default_set_value>,
    class _SetError = __qq<__cmplsigs::__default_set_error>,
    class _SetStopped = completion_signatures<set_stopped_t()>
  >
  using __try_make_completion_signatures = __transform_completion_signatures_t<
    __completion_signatures_of_t<_Sender, _Env>,
    _SetValue::template __f,
    _SetError::template __f,
    _SetStopped,
    __mtry_q<__concat_completion_signatures_t>::__f,
    _More
  >;

  // The following utilities are needed fairly often:
  template <class _Fun, class... _Args>
    requires __invocable<_Fun, _Args...>
  using __nothrow_invocable_t = __mbool<__nothrow_invocable<_Fun, _Args...>>;

  template <class _Catch, class _Tag, class _Fun, class _Sender, class... _Env>
  using __with_error_invoke_t = __if<
    __gather_completion_signatures_t<
      __completion_signatures_of_t<_Sender, _Env...>,
      _Tag,
      __mbind_front<__mtry_catch_q<__nothrow_invocable_t, _Catch>, _Fun>::template __f,
      __mconst<__mbool<true>>::__f,
      __mand
    >,
    completion_signatures<>,
    __eptr_completion
  >;

  template <class _Fun, class... _Args>
    requires __invocable<_Fun, _Args...>
  using __set_value_from_t = completion_signatures<
    __minvoke<__mremove<void, __qf<set_value_t>>, __invoke_result_t<_Fun, _Args...>>
  >;

  template <class _Completions>
  using __decay_copyable_results_t =
    __cmplsigs::__partitions_of_t<_Completions>::__decay_copyable::__all;

  template <class _Completions>
  using __nothrow_decay_copyable_results_t =
    __cmplsigs::__partitions_of_t<_Completions>::__nothrow_decay_copyable::__all;
} // namespace STDEXEC
