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
#include "__debug.hpp"  // IWYU pragma: keep
#include "__get_completion_signatures.hpp"
#include "__meta.hpp"

#include <exception>

namespace STDEXEC
{
  namespace __cmplsigs
  {
#if STDEXEC_NO_STDCPP_CONSTEXPR_EXCEPTIONS()
    // Without constexpr exceptions, we cannot always produce a valid
    // completion_signatures type. We must permit get_completion_signatures to return an
    // error type because we can't throw it.
    template <class _Completions>
    concept __well_formed_completions_helper = __valid_completion_signatures<_Completions>
                                            || STDEXEC_IS_BASE_OF(STDEXEC::dependent_sender_error,
                                                                  _Completions)
                                            || __is_instance_of<_Completions, _ERROR_>;
#else
    // When we have constexpr exceptions, we can require that get_completion_signatures
    // always produces a valid completion_signatures type.
    template <class _Completions>
    concept __well_formed_completions_helper = __valid_completion_signatures<_Completions>;
#endif
  }  // namespace __cmplsigs

  using __cmplsigs::get_completion_signatures_t;

  // The cast to bool is to hide the disjunction in __well_formed_completions_helper.
  template <class _Completions>
  concept __well_formed_completions = bool(
    __cmplsigs::__well_formed_completions_helper<_Completions>);

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __for_each_completion_signature_t
  namespace __cmplsigs
  {
    template <template <class...> class _Tuple, class _Tag, class... _Args>
    constexpr auto __for_each_sig(_Tag (*)(_Args...)) -> _Tuple<_Tag, _Args...>;

    template <class _Sig, template <class...> class _Tuple>
    using __for_each_sig_t = decltype(__cmplsigs::__for_each_sig<_Tuple>(
      static_cast<_Sig *>(nullptr)));

    template <template <class...> class _Tuple,
              template <class...> class _Variant,
              class... _More,
              class... _What>
    constexpr auto __for_each_sigs(__undefined<_ERROR_<_What...>> *) -> _ERROR_<_What...>;
    template <template <class...> class _Tuple,
              template <class...> class _Variant,
              class... _More,
              class... _Sigs>
    constexpr auto __for_each_sigs(__undefined<completion_signatures<_Sigs...>> *)
      -> _Variant<__for_each_sig_t<_Sigs, _Tuple>..., _More...>;
  }  // namespace __cmplsigs

  template <class _Sigs,
            template <class...> class _Tuple,
            template <class...> class _Variant,
            class... _More>
  using __for_each_completion_signature_t =
    decltype(__cmplsigs::__for_each_sigs<_Tuple, _Variant, _More...>(
      static_cast<__undefined<_Sigs> *>(nullptr)));

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __transform_reduce_completion_signatures_t
  // __transform_completion_signatures_t
  // __transform_completion_signatures_of_t
  namespace __cmplsigs
  {
    template <template <class...> class _SetVal,
              template <class...> class _SetErr,
              class _SetStp,
              class... _Values>
    constexpr auto __transform_sig(set_value_t (*)(_Values...)) -> _SetVal<_Values...>;

    template <template <class...> class _SetVal,
              template <class...> class _SetErr,
              class _SetStp,
              class _Error>
    constexpr auto __transform_sig(set_error_t (*)(_Error)) -> _SetErr<_Error>;

    template <template <class...> class _SetVal, template <class...> class _SetErr, class _SetStp>
    constexpr auto __transform_sig(set_stopped_t (*)()) -> _SetStp;

    template <class _Sig,
              template <class...> class _SetVal,
              template <class...> class _SetErr,
              class _SetStp>
    using __transform_sig_t = decltype(__cmplsigs::__transform_sig<_SetVal, _SetErr, _SetStp>(
      static_cast<_Sig *>(nullptr)));

    template <template <class...> class _SetVal,
              template <class...> class _SetErr,
              class _SetStp,
              template <class...> class _Variant,
              class... _More,
              class... _What>
    constexpr auto __transform_reduce_sigs(__undefined<_ERROR_<_What...>> *) -> _ERROR_<_What...>;

    template <template <class...> class _SetVal,
              template <class...> class _SetErr,
              class _SetStp,
              template <class...> class _Variant,
              class... _More,
              class... _Sigs>
    constexpr auto __transform_reduce_sigs(__undefined<completion_signatures<_Sigs...>> *)
      -> _Variant<__transform_sig_t<_Sigs, _SetVal, _SetErr, _SetStp>..., _More...>;

    template <class... _Values>
    using __default_set_value = completion_signatures<set_value_t(_Values...)>;

    template <class... _Error>
    using __default_set_error = completion_signatures<set_error_t(_Error...)>;

    template <class _Tag, class... _Args>
    using __default_completion = completion_signatures<_Tag(_Args...)>;
  }  // namespace __cmplsigs

  template <class _Sigs,
            template <class...> class _SetVal,
            template <class...> class _SetErr,
            class _SetStp,
            template <class...> class _Variant,
            class... _More>
  using __transform_reduce_completion_signatures_t =
    decltype(__cmplsigs::__transform_reduce_sigs<_SetVal, _SetErr, _SetStp, _Variant, _More...>(
      static_cast<__undefined<_Sigs> *>(nullptr)));

  template <class _Sigs,
            class _MoreSigs                           = completion_signatures<>,
            template <class...> class _ValueTransform = __cmplsigs::__default_set_value,
            template <class...> class _ErrorTransform = __cmplsigs::__default_set_error,
            class _StoppedSigs                        = completion_signatures<set_stopped_t()>>
  using __transform_completion_signatures_t =
    __transform_reduce_completion_signatures_t<_Sigs,
                                               _ValueTransform,
                                               _ErrorTransform,
                                               _StoppedSigs,
                                               __mtry_q<__concat_completion_signatures_t>::__f,
                                               _MoreSigs>;

  template <class _Sndr,
            class _Env                                = env<>,
            class _MoreSigs                           = completion_signatures<>,
            template <class...> class _ValueTransform = __cmplsigs::__default_set_value,
            template <class...> class _ErrorTransform = __cmplsigs::__default_set_error,
            class _StoppedSigs                        = completion_signatures<set_stopped_t()>>
  using __transform_completion_signatures_of_t =
    __transform_completion_signatures_t<completion_signatures_of_t<_Sndr, _Env>,
                                        _MoreSigs,
                                        _ValueTransform,
                                        _ErrorTransform,
                                        _StoppedSigs>;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __gather_completion_signatures_t
  namespace __cmplsigs
  {
    template <class _WantedTag, bool = true>
    struct __gather_sigs_fn;

    template <>
    struct __gather_sigs_fn<set_value_t>
    {
      template <class _Sigs,
                template <class...> class _Then,
                template <class...> class _Else,
                template <class...> class _Variant,
                class... _More>
      using __f = __transform_reduce_completion_signatures_t<
        _Sigs,
        _Then,
        __mbind_front_q<_Else, set_error_t>::template __f,
        _Else<set_stopped_t>,
        _Variant,
        _More...>;
    };

    template <>
    struct __gather_sigs_fn<set_error_t>
    {
      template <class _Sigs,
                template <class...> class _Then,
                template <class...> class _Else,
                template <class...> class _Variant,
                class... _More>
      using __f = __transform_reduce_completion_signatures_t<
        _Sigs,
        __mbind_front_q<_Else, set_value_t>::template __f,
        _Then,
        _Else<set_stopped_t>,
        _Variant,
        _More...>;
    };

    template <>
    struct __gather_sigs_fn<set_stopped_t>
    {
      template <class _Sigs,
                template <class...> class _Then,
                template <class...> class _Else,
                template <class...> class _Variant,
                class... _More>
      using __f = __transform_reduce_completion_signatures_t<
        _Sigs,
        __mbind_front_q<_Else, set_value_t>::template __f,
        __mbind_front_q<_Else, set_error_t>::template __f,
        _Then<>,
        _Variant,
        _More...>;
    };

    template <class _WantedTag>
    struct __gather_sigs_fn<_WantedTag, false>
    {
      template <class _Error,
                template <class...> class,
                template <class...> class,
                template <class...> class,
                class...>
      using __f = _Error;
    };
  }  // namespace __cmplsigs

  template <class _Sigs,
            class _WantedTag,
            template <class...> class _Then,
            template <class...> class _Else,
            template <class...> class _Variant,
            class... _More>
  using __gather_completion_signatures_t =                                //
    __cmplsigs::__gather_sigs_fn<_WantedTag, __ok<_Sigs>>::template __f<  //
      _Sigs,
      _Then,
      _Else,
      _Variant,
      _More...>;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // begin implementation of __transform_completion_signatures
  struct _IN_TRANSFORM_COMPLETION_SIGNATURES_;
  struct _A_TRANSFORM_FUNCTION_RETURNED_A_TYPE_THAT_IS_NOT_A_COMPLETION_SIGNATURES_SPECIALIZATION_;
  struct _COULD_NOT_CALL_THE_TRANSFORM_FUNCTION_WITH_THE_GIVEN_TEMPLATE_ARGUMENTS_;

  namespace __cmplsigs
  {
    template <class _Fn, class... _Args>
    using __transform_result_t = decltype(__declval<_Fn>().template operator()<_Args...>());

    template <class... _Args, class _Fn>
    [[nodiscard]]
    consteval auto
    __transform_expr(_Fn const &__fn, long) -> __transform_result_t<_Fn const &, _Args...>
    {
      return __fn.template operator()<_Args...>();
    }

    template <class _Fn>
    [[nodiscard]]
    consteval auto __transform_expr(_Fn const &__fn, int) -> __call_result_t<_Fn const &>
    {
      return __fn();
    }

    template <class _Fn, class... _Args>
    using __transform_expr_t =
      decltype(__cmplsigs::__transform_expr<_Args...>(__declval<_Fn const &>(), 0));

    template <class... _Args, class _Fn>
    [[nodiscard]]
    consteval auto __apply_transform(_Fn const &__fn)
    {
      if constexpr (__minvocable_q<__transform_expr_t, _Fn, _Args...>)
      {
        using __completions_t = __transform_expr_t<_Fn, _Args...>;
        if constexpr (__well_formed_completions<__completions_t>)
        {
          return __cmplsigs::__transform_expr<_Args...>(__fn, 0);
        }
        else
        {
          (void) __cmplsigs::__transform_expr<_Args...>(__fn, 0);  // potentially throwing
          return STDEXEC::__throw_compile_time_error<
            _IN_TRANSFORM_COMPLETION_SIGNATURES_,
            _A_TRANSFORM_FUNCTION_RETURNED_A_TYPE_THAT_IS_NOT_A_COMPLETION_SIGNATURES_SPECIALIZATION_,
            _WITH_FUNCTION_(_Fn),
            _WITH_ARGUMENTS_(_Args...)>();
        }
      }
      else
      {
        return STDEXEC::__throw_compile_time_error<
          _IN_TRANSFORM_COMPLETION_SIGNATURES_,
          _COULD_NOT_CALL_THE_TRANSFORM_FUNCTION_WITH_THE_GIVEN_TEMPLATE_ARGUMENTS_,
          _WITH_FUNCTION_(_Fn),
          _WITH_ARGUMENTS_(_Args...)>();
      }
    }

    template <class _ValueFn, class _ErrorFn, class _StoppedFn>
    struct __transform_one
    {
      template <class _SetTag, class... _Args>
      [[nodiscard]]
      consteval auto operator()(_SetTag (*)(_Args...)) const
      {
        if constexpr (_SetTag() == set_value)
        {
          return __cmplsigs::__apply_transform<_Args...>(__value_fn);
        }
        else if constexpr (_SetTag() == set_error)
        {
          return __cmplsigs::__apply_transform<_Args...>(__error_fn);
        }
        else
        {
          return __cmplsigs::__apply_transform<_Args...>(__stopped_fn);
        }
      }

      _ValueFn   __value_fn;
      _ErrorFn   __error_fn;
      _StoppedFn __stopped_fn;
    };

    template <class _ValueFn, class _ErrorFn, class _StoppedFn>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __transform_one(_ValueFn, _ErrorFn, _StoppedFn)
      -> __transform_one<_ValueFn, _ErrorFn, _StoppedFn>;

    template <class _TransformOne>
    struct __transform_all_fn
    {
      template <class... Sigs>
      [[nodiscard]]
      consteval auto operator()(Sigs *...sigs) const
      {
        return __concat_completion_signatures(__tfx1(sigs)...);
      }

      _TransformOne __tfx1;
    };

    template <class _TransformOne>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
      __transform_all_fn(_TransformOne) -> __transform_all_fn<_TransformOne>;
  }  // namespace __cmplsigs

  //////////////////////////////////////////////////////////////////////////////////////////////////
  //! Commonly needed transformations of completion signatures for use with
  //! \c __transform_completion_signatures.
  //!
  //! * __keep_completion
  //! * __ignore_completion
  //! * __transform_arguments
  //! * __decay_arguments
  template <class _SetTag>
  struct __keep_completion
  {
    template <class... _Ts>
    consteval auto operator()() const noexcept -> completion_signatures<_SetTag(_Ts...)>
    {
      return {};
    }
  };

  struct __ignore_completion
  {
    template <class...>
    consteval auto operator()() const noexcept -> completion_signatures<>
    {
      return {};
    }
  };

  template <class _SetTag, class _Fn, class... _AlgoTag>
  struct __transform_arguments
  {
    template <class... _Args>
    consteval auto operator()() const noexcept
    {
      if constexpr ((__minvocable<_Fn, _Args> && ...))
      {
        return completion_signatures<_SetTag(__minvoke<_Fn, _Args>...)>();
      }
      else
      {
        return (__check_transform<_Args>(), ...);
      }
    }

   private:
    template <class _Arg>
    static consteval auto __check_transform()
    {
      if constexpr (__minvocable<_Fn, _Arg>)
      {
        return __msuccess();
      }
      else
      {
        return STDEXEC::__throw_compile_time_error<
          _WHERE_(_IN_ALGORITHM_, _AlgoTag)...,
          _FUNCTION_IS_NOT_CALLABLE_WITH_THE_GIVEN_ARGUMENTS_,
          _WITH_METAFUNCTION_(_Fn),
          _WITH_ARGUMENTS_(_Arg)>();
      }
    }
  };

  template <class _SetTag, class... _AlgoTag>
  struct __decay_arguments
  {
    template <class... _Args>
    consteval auto operator()() const noexcept
    {
      if constexpr (__decay_copyable<_Args...>)
      {
        return completion_signatures<_SetTag(__decay_t<_Args>...)>();
      }
      else
      {
        // NB: this uses an overloaded comma operator on the _ERROR_ type to find an error
        // in a pack of types.
        return (__check_decay<_Args>(), ...);
      }
    }

   private:
    template <class _Arg>
    static consteval auto __check_decay()
    {
      if constexpr (__decay_copyable<_Arg>)
      {
        return __msuccess();
      }
      else
      {
        return STDEXEC::__throw_compile_time_error<_WHAT_(_TYPE_IS_NOT_DECAY_COPYABLE_),
                                                   _WHERE_(_IN_ALGORITHM_, _AlgoTag)...,
                                                   _WITH_TYPE_<_Arg>>();
      }
    }
  };

  //! \brief Transforms completion signatures using provided transformation functions.
  //!
  //! This consteval function transforms a set of completion signatures by applying
  //! custom transformation functions to value, error, and stopped completion cases.
  //! The result can be augmented with additional extra signatures.
  //!
  //! \tparam _Completions The input completion signatures to transform.
  //! \tparam _ValueFn     Function object that transforms set_value_t completions.
  //!                      Defaults to __keep_completion<set_value_t>.
  //! \tparam _ErrorFn     Function object that transforms set_error_t completions.
  //!                      Defaults to __keep_completion<set_error_t>.
  //! \tparam _StoppedFn   Function object that transforms set_stopped_t completions.
  //!                      Defaults to __keep_completion<set_stopped_t>.
  //! \tparam _ExtraSigs   Additional completion signatures to append to the result.
  //!                      Defaults to an empty completion_signatures.
  //!
  //! \param __completions The input completion signatures object.
  //! \param __value_fn    Value transformation function instance.
  //! \param __error_fn    Error transformation function instance.
  //! \param __stopped_fn  Stopped transformation function instance.
  //! \param __extra_sigs  Extra signatures to append to the result.
  //!
  //! \return A transformed completion_signatures object combining the transformed
  //!         input signatures with the extra signatures.
  //!
  //! \par Example
  //!
  //! The following example demonstrates how to use \c __transform_completion_signatures
  //! to compute the completion signatures of the \c then sender.
  //!
  //! \code{.cpp}
  //! namespace ex = STDEXEC;
  //!
  //! template <class Fn, class... Args>
  //! consteval auto _transform_values()
  //! {
  //!   if constexpr (!std::invocable<Fn, Args...>)
  //!   {
  //!     // If Fn cannot be invoked with the given arguments, produce a compile-time
  //!     // error.
  //!     return ex::__throw_compile_time_error<
  //!       _WHAT_(_FUNCTION_IS_NOT_CALLABLE_WITH_THE_GIVEN_ARGUMENTS_),
  //!       _WHERE_(_IN_ALGORITHM_, then_t),
  //!       _WITH_FUNCTION_(Fn),
  //!       _WITH_ARGUMENTS_(Args...)>();
  //!   }
  //!   else
  //!   {
  //!     // transform the value types of the child sender into the value types of the
  //!     // then sender by applying Fn to them.
  //!     using result_t         = std::invoke_result_t<Fn, Args...>;
  //!     constexpr bool is_void = std::is_void_v<result_t>;
  //!     constexpr bool nothrow = std::is_nothrow_invocable_v<Fn, Args...>;
  //!     if constexpr (is_void && nothrow)
  //!     {
  //!       return ex::completion_signatures<set_value_t()>();
  //!     }
  //!     else if constexpr (is_void && !nothrow)
  //!     {
  //!       return ex::completion_signatures<set_value_t(),
  //!                                        set_error_t(std::exception_ptr)>();
  //!     }
  //!     else if constexpr (!is_void && nothrow)
  //!     {
  //!       return ex::completion_signatures<set_value_t(result_t)>();
  //!     }
  //!     else /* !is_void && !nothrow */
  //!     {
  //!       return ex::completion_signatures<set_value_t(result_t),
  //!                                        set_error_t(std::exception_ptr)>();
  //!     }
  //!   }
  //! }
  //!
  //! template <class Child, class Fn>
  //! struct then_sender
  //! {
  //!   using sender_concept = ex::sender_t;
  //!
  //!   template <class Self, class... Env>
  //!   static consteval auto get_completion_signatures()
  //!   {
  //!     // Compute the completion signatures of the child sender, and then transform
  //!     // them into the completion signatures of the `then` sender.
  //!     auto child_completions = ex::__get_child_completion_signatures<Self, Child, Env...>();
  //!     auto value_fn = []<class... Args>() { return _transform_values<Fn, Args...>(); };
  //!
  //!     return ex::__transform_completion_signatures(child_completions, value_fn);
  //!   }
  //!
  //!   // ...
  //! };
  //! \endcode
  //!
  //! \note This function is evaluated at compile-time (consteval).
  template <class _Completions,
            class _ValueFn   = __keep_completion<set_value_t>,
            class _ErrorFn   = __keep_completion<set_error_t>,
            class _StoppedFn = __keep_completion<set_stopped_t>,
            class _ExtraSigs = completion_signatures<>>
  consteval auto __transform_completion_signatures(_Completions,
                                                   _ValueFn   __value_fn   = {},
                                                   _ErrorFn   __error_fn   = {},
                                                   _StoppedFn __stopped_fn = {},
                                                   _ExtraSigs              = {})
  {
    STDEXEC_COMPLSIGS_LET(__completions, _Completions{})
    {
      STDEXEC_COMPLSIGS_LET(__extra_sigs, _ExtraSigs{})
      {
        __cmplsigs::__transform_one __tfx1{__value_fn, __error_fn, __stopped_fn};
        return __concat_completion_signatures(__completions.__apply(
                                                __cmplsigs::__transform_all_fn(__tfx1)),
                                              __extra_sigs);
      }
    }
  }

  using __eptr_completion_t = completion_signatures<set_error_t(std::exception_ptr)>;

  template <class _NoExcept>
  using __eptr_completion_unless_t = __if<_NoExcept, completion_signatures<>, __eptr_completion_t>;

  // The following utilities are needed fairly often:
  template <class _Fun, class... _Args>
    requires __invocable<_Fun, _Args...>
  using __nothrow_invocable_t = __mbool<__nothrow_invocable<_Fun, _Args...>>;

  template <class _Catch, class _Tag, class _Fun, class _Sender, class... _Env>
  using __with_error_invoke_t =
    __if<__gather_completion_signatures_t<
           __completion_signatures_of_t<_Sender, _Env...>,
           _Tag,
           __mbind_front<__mtry_catch_q<__nothrow_invocable_t, _Catch>, _Fun>::template __f,
           __mconst<__mbool<true>>::__f,
           __mand>,
         completion_signatures<>,
         __eptr_completion_t>;

  template <class _Fun, class... _Args>
    requires __invocable<_Fun, _Args...>
  using __set_value_from_t =
    completion_signatures<__single_value_sig_t<__invoke_result_t<_Fun, _Args...>>>;

  template <class _Completions>
  using __decay_copyable_results_t =
    __cmplsigs::__partitions_of_t<_Completions>::__decay_copyable::__all;

  template <class _Completions>
  using __nothrow_decay_copyable_results_t =
    __cmplsigs::__partitions_of_t<_Completions>::__nothrow_decay_copyable::__all;

#define STDEXEC_TRANSFORM_COMPLETION_SIGNATURES_DEPRECATION_MESSAGE \
  "Please migrate to the exec::transform_completion_signatures API in <exec/completion_signatures.hpp>"

  // Deprecated interfaces:
  template <class _Sigs,
            class _MoreSigs                           = completion_signatures<>,
            template <class...> class _ValueTransform = __cmplsigs::__default_set_value,
            template <class...> class _ErrorTransform = __cmplsigs::__default_set_error,
            class _StoppedSigs                        = completion_signatures<set_stopped_t()>>
  using transform_completion_signatures
    [[deprecated(STDEXEC_TRANSFORM_COMPLETION_SIGNATURES_DEPRECATION_MESSAGE)]] =
      __transform_reduce_completion_signatures_t<_Sigs,
                                                 _ValueTransform,
                                                 _ErrorTransform,
                                                 _StoppedSigs,
                                                 __mtry_q<__concat_completion_signatures_t>::__f,
                                                 _MoreSigs>;

  template <class _Sndr,
            class _Env                                = env<>,
            class _MoreSigs                           = completion_signatures<>,
            template <class...> class _ValueTransform = __cmplsigs::__default_set_value,
            template <class...> class _ErrorTransform = __cmplsigs::__default_set_error,
            class _StoppedSigs                        = completion_signatures<set_stopped_t()>>
  using transform_completion_signatures_of
    [[deprecated(STDEXEC_TRANSFORM_COMPLETION_SIGNATURES_DEPRECATION_MESSAGE)]] =
      __transform_completion_signatures_t<completion_signatures_of_t<_Sndr, _Env>,
                                          _MoreSigs,
                                          _ValueTransform,
                                          _ErrorTransform,
                                          _StoppedSigs>;
}  // namespace STDEXEC
