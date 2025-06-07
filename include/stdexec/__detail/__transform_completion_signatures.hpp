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
#include "__completion_signatures.hpp"
#include "__concepts.hpp"
#include "__debug.hpp" // IWYU pragma: keep
#include "__senders_core.hpp"
#include "__meta.hpp"

#include <exception>
#include <tuple>
#include <variant>

namespace stdexec {
#if STDEXEC_ENABLE_EXTRA_TYPE_CHECKING()
  // __checked_completion_signatures is for catching logic bugs in a sender's metadata. If sender<S>
  // and sender_in<S, Ctx> are both true, then they had better report the same metadata. This
  // completion signatures wrapper enforces that at compile time.
  template <class _Sender, class... _Env>
  auto __checked_completion_signatures(_Sender &&__sndr, _Env &&...__env) noexcept {
    using __completions_t = __completion_signatures_of_t<_Sender, _Env...>;
    stdexec::__debug_sender(static_cast<_Sender &&>(__sndr), __env...);
    return __completions_t{};
  }

  template <class _Sender, class... _Env>
    requires sender_in<_Sender, _Env...>
  using completion_signatures_of_t =
    decltype(stdexec::__checked_completion_signatures(__declval<_Sender>(), __declval<_Env>()...));
#else
  template <class _Sender, class... _Env>
    requires sender_in<_Sender, _Env...>
  using completion_signatures_of_t = __completion_signatures_of_t<_Sender, _Env...>;
#endif

  struct __not_a_variant {
    __not_a_variant() = delete;
  };

  template <class... _Ts>
  using __std_variant = __minvoke_if_c<
    sizeof...(_Ts) != 0,
    __mtransform<__q1<__decay_t>, __munique<__qq<std::variant>>>,
    __mconst<__not_a_variant>,
    _Ts...
  >;

  template <class... _Ts>
  using __nullable_std_variant =
    __mcall<__munique<__mbind_front<__qq<std::variant>, std::monostate>>, __decay_t<_Ts>...>;

  template <class... _Ts>
  using __decayed_std_tuple = __meval<std::tuple, __decay_t<_Ts>...>;

  namespace __sigs {
    // The following code is used to normalize completion signatures. "Normalization" means that
    // that rvalue-references are stripped from the types in the completion signatures. For example,
    // the completion signature `set_value_t(int &&)` would be normalized to `set_value_t(int)`,
    // but `set_value_t(int)` and `set_value_t(int &)` would remain unchanged.
    template <class _Tag, class... _Args>
    auto __normalize_sig_impl(_Args &&...) -> _Tag (*)(_Args...);

    template <class _Tag, class... _Args>
    auto __normalize_sig(_Tag (*)(_Args...))
      -> decltype(__sigs::__normalize_sig_impl<_Tag>(__declval<_Args>()...));

    template <class... _Sigs>
    auto __repack_completions(_Sigs *...) -> completion_signatures<_Sigs...>;

    template <class... _Sigs>
    auto __normalize_completions(completion_signatures<_Sigs...> *)
      -> decltype(__sigs::__repack_completions(
        __sigs::__normalize_sig(static_cast<_Sigs *>(nullptr))...));

    template <class _Completions>
    using __normalize_completions_t = decltype(__sigs::__normalize_completions(
      static_cast<_Completions *>(nullptr)));
  } // namespace __sigs

  template <class... _SigPtrs>
  using __completion_signature_ptrs = decltype(__sigs::__repack_completions(
    static_cast<_SigPtrs>(nullptr)...));

  template <class... _Sigs>
  using __concat_completion_signatures =
    __mconcat<__qq<completion_signatures>>::__f<__mconcat<__qq<__mmake_set>>::__f<_Sigs...>>;

  namespace __sigs {
    //////////////////////////////////////////////////////////////////////////////////////////////////
    template <template <class...> class _Tuple, class _Tag, class... _Args>
    auto __for_each_sig(_Tag (*)(_Args...)) -> _Tuple<_Tag, _Args...>;

    template <class _Sig, template <class...> class _Tuple>
    using __for_each_sig_t = decltype(__sigs::__for_each_sig<_Tuple>(static_cast<_Sig *>(nullptr)));

    template <
      template <class...> class _Tuple,
      template <class...> class _Variant,
      class... _More,
      class _What,
      class... _With
    >
    auto
      __for_each_completion_signature_fn(_ERROR_<_What, _With...> **) -> _ERROR_<_What, _With...>;

    template <
      template <class...> class _Tuple,
      template <class...> class _Variant,
      class... _More,
      class... _Sigs
    >
    auto __for_each_completion_signature_fn(completion_signatures<_Sigs...> **)
      -> _Variant<__for_each_sig_t<_Sigs, _Tuple>..., _More...>;
  } // namespace __sigs

  template <
    class _Sigs,
    template <class...> class _Tuple,
    template <class...> class _Variant,
    class... _More
  >
  using __for_each_completion_signature =
    decltype(__sigs::__for_each_completion_signature_fn<_Tuple, _Variant, _More...>(
      static_cast<_Sigs **>(nullptr)));

  namespace __sigs {
    ////////////////////////////////////////////////////////////////////////////////////////////////
    template <
      template <class...> class _SetVal,
      template <class...> class _SetErr,
      class _SetStp,
      class... _Values
    >
    auto __transform_sig(set_value_t (*)(_Values...)) -> _SetVal<_Values...>;

    template <
      template <class...> class _SetVal,
      template <class...> class _SetErr,
      class _SetStp,
      class _Error
    >
    auto __transform_sig(set_error_t (*)(_Error)) -> _SetErr<_Error>;

    template <template <class...> class _SetVal, template <class...> class _SetErr, class _SetStp>
    auto __transform_sig(set_stopped_t (*)()) -> _SetStp;

    template <
      class _Sig,
      template <class...> class _SetVal,
      template <class...> class _SetErr,
      class _SetStp
    >
    using __transform_sig_t = decltype(__sigs::__transform_sig<_SetVal, _SetErr, _SetStp>(
      static_cast<_Sig *>(nullptr)));

    template <
      template <class...> class _SetVal,
      template <class...> class _SetErr,
      class _SetStp,
      template <class...> class _Variant,
      class... _More,
      class _What,
      class... _With
    >
    auto __transform_sigs_fn(_ERROR_<_What, _With...> **) -> _ERROR_<_What, _With...>;

    template <
      template <class...> class _SetVal,
      template <class...> class _SetErr,
      class _SetStp,
      template <class...> class _Variant,
      class... _More,
      class... _Sigs
    >
    auto __transform_sigs_fn(completion_signatures<_Sigs...> **)
      -> _Variant<__transform_sig_t<_Sigs, _SetVal, _SetErr, _SetStp>..., _More...>;
  } // namespace __sigs

  template <
    class _Sigs,
    template <class...> class _SetVal,
    template <class...> class _SetErr,
    class _SetStp,
    template <class...> class _Variant,
    class... _More
  >
  using __transform_completion_signatures =
    decltype(__sigs::__transform_sigs_fn<_SetVal, _SetErr, _SetStp, _Variant, _More...>(
      static_cast<_Sigs **>(nullptr)));

  namespace __sigs {
    ////////////////////////////////////////////////////////////////////////////////////////////////
    template <class _WantedTag>
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
      using __f = __transform_completion_signatures<
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
      using __f = __transform_completion_signatures<
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
      using __f = __transform_completion_signatures<
        _Sigs,
        __mbind_front_q<_Else, set_value_t>::template __f,
        __mbind_front_q<_Else, set_error_t>::template __f,
        _Then<>,
        _Variant,
        _More...
      >;
    };

    template <class... _Values>
    using __default_set_value = completion_signatures<set_value_t(_Values...)>;

    template <class... _Error>
    using __default_set_error = completion_signatures<set_error_t(_Error...)>;

    template <class _Tag, class... _Args>
    using __default_completion = completion_signatures<_Tag(_Args...)>;
  } // namespace __sigs

  template <
    class _Sigs,
    class _WantedTag,
    template <class...> class _Then,
    template <class...> class _Else,
    template <class...> class _Variant,
    class... _More
  >
  using __gather_completion_signatures = typename __sigs::__gather_sigs_fn<
    _WantedTag
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
    template <class...> class _ValueTransform = __sigs::__default_set_value,
    template <class...> class _ErrorTransform = __sigs::__default_set_error,
    class _StoppedSigs = completion_signatures<set_stopped_t()>
  >
  using transform_completion_signatures = __transform_completion_signatures<
    _Sigs,
    _ValueTransform,
    _ErrorTransform,
    _StoppedSigs,
    __mtry_q<__concat_completion_signatures>::__f,
    _MoreSigs
  >;

  template <
    class _Sndr,
    class _Env = env<>,
    class _MoreSigs = completion_signatures<>,
    template <class...> class _ValueTransform = __sigs::__default_set_value,
    template <class...> class _ErrorTransform = __sigs::__default_set_error,
    class _StoppedSigs = completion_signatures<set_stopped_t()>
  >
  using transform_completion_signatures_of = transform_completion_signatures<
    completion_signatures_of_t<_Sndr, _Env>,
    _MoreSigs,
    _ValueTransform,
    _ErrorTransform,
    _StoppedSigs
  >;

  using __eptr_completion = completion_signatures<set_error_t(std::exception_ptr)>;

  template <class _NoExcept>
  using __eptr_completion_if_t = __if<_NoExcept, completion_signatures<>, __eptr_completion>;

  template <bool _NoExcept>
  using __eptr_completion_if = __eptr_completion_if_t<__mbool<_NoExcept>>;

  template <
    class _Sender,
    class _Env = env<>,
    class _More = completion_signatures<>,
    class _SetValue = __qq<__sigs::__default_set_value>,
    class _SetError = __qq<__sigs::__default_set_error>,
    class _SetStopped = completion_signatures<set_stopped_t()>
  >
  using __try_make_completion_signatures = __transform_completion_signatures<
    __completion_signatures_of_t<_Sender, _Env>,
    _SetValue::template __f,
    _SetError::template __f,
    _SetStopped,
    __mtry_q<__concat_completion_signatures>::__f,
    _More
  >;

  template <class _SetTag, class _Completions, class _Tuple, class _Variant>
  using __gather_completions = __gather_completion_signatures<
    _Completions,
    _SetTag,
    __mcompose_q<__types, _Tuple::template __f>::template __f,
    __mconst<__types<>>::__f,
    __mconcat<_Variant>::template __f
  >;

  template <class _SetTag, class _Sender, class _Env, class _Tuple, class _Variant>
  using __gather_completions_of =
    __gather_completions<_SetTag, __completion_signatures_of_t<_Sender, _Env>, _Tuple, _Variant>;

  template <
    class _Sender,
    class _Env = env<>,
    class _Sigs = completion_signatures<>,
    template <class...> class _SetValue = __sigs::__default_set_value,
    template <class...> class _SetError = __sigs::__default_set_error,
    class _SetStopped = completion_signatures<set_stopped_t()>
  >
  using make_completion_signatures =
    transform_completion_signatures_of<_Sender, _Env, _Sigs, _SetValue, _SetError, _SetStopped>;

  template <
    class _Sigs,
    class _Tuple = __q<__decayed_std_tuple>,
    class _Variant = __q<__std_variant>
  >
  using __value_types_t = __gather_completions<set_value_t, _Sigs, _Tuple, _Variant>;

  template <
    class _Sender,
    class _Env = env<>,
    class _Tuple = __q<__decayed_std_tuple>,
    class _Variant = __q<__std_variant>
  >
  using __value_types_of_t =
    __value_types_t<__completion_signatures_of_t<_Sender, _Env>, _Tuple, _Variant>;

  template <class _Sigs, class _Variant = __q<__std_variant>>
  using __error_types_t = __gather_completions<set_error_t, _Sigs, __q<__midentity>, _Variant>;

  template <class _Sender, class _Env = env<>, class _Variant = __q<__std_variant>>
  using __error_types_of_t = __error_types_t<__completion_signatures_of_t<_Sender, _Env>, _Variant>;

  template <
    class _Sender,
    class _Env = env<>,
    template <class...> class _Tuple = __decayed_std_tuple,
    template <class...> class _Variant = __std_variant
  >
  using value_types_of_t = __value_types_of_t<_Sender, _Env, __q<_Tuple>, __q<_Variant>>;

  template <class _Sender, class _Env = env<>, template <class...> class _Variant = __std_variant>
  using error_types_of_t = __error_types_of_t<_Sender, _Env, __q<_Variant>>;

  template <class _Tag, class _Sender, class... _Env>
  using __count_of = __gather_completion_signatures<
    __completion_signatures_of_t<_Sender, _Env...>,
    _Tag,
    __mconst<__msize_t<1>>::__f,
    __mconst<__msize_t<0>>::__f,
    __mplus_t
  >;

  template <class _Tag, class _Sender, class... _Env>
    requires sender_in<_Sender, _Env...>
  inline constexpr bool __sends = __v<__gather_completion_signatures<
    __completion_signatures_of_t<_Sender, _Env...>,
    _Tag,
    __mconst<__mtrue>::__f,
    __mconst<__mfalse>::__f,
    __mor_t
  >>;

  template <class _Sender, class... _Env>
  concept sends_stopped = sender_in<_Sender, _Env...> && __sends<set_stopped_t, _Sender, _Env...>;

  template <class _Sender, class... _Env>
  using __single_sender_value_t =
    __value_types_t<__completion_signatures_of_t<_Sender, _Env...>, __q<__msingle>, __q<__msingle>>;

  template <class _Sender, class... _Env>
  concept __single_value_sender = sender_in<_Sender, _Env...>
                               && requires { typename __single_sender_value_t<_Sender, _Env...>; };

  template <class _Sender, class... _Env>
  using __single_value_variant_sender_t =
    __value_types_t<__completion_signatures_of_t<_Sender, _Env...>, __qq<__types>, __q<__msingle>>;

  template <class _Sender, class... _Env>
  concept __single_value_variant_sender = sender_in<_Sender, _Env...> && requires {
    typename __single_value_variant_sender_t<_Sender, _Env...>;
  };

  // The following utilities are needed fairly often:
  template <class _Fun, class... _Args>
    requires __invocable<_Fun, _Args...>
  using __nothrow_invocable_t = __mbool<__nothrow_invocable<_Fun, _Args...>>;

  template <class _Catch, class _Tag, class _Fun, class _Sender, class... _Env>
  using __with_error_invoke_t = __if<
    __gather_completion_signatures<
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
  using __set_value_invoke_t = completion_signatures<
    __minvoke<__mremove<void, __qf<set_value_t>>, __invoke_result_t<_Fun, _Args...>>
  >;

  template <class _Completions>
  using __decay_copyable_results_t =
    __for_each_completion_signature<_Completions, __decay_copyable_t, __mand_t>;

  template <class _Completions>
  using __nothrow_decay_copyable_results_t =
    __for_each_completion_signature<_Completions, __nothrow_decay_copyable_t, __mand_t>;
} // namespace stdexec
