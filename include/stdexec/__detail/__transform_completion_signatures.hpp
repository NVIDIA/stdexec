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

#include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "__completion_signatures.hpp"
#include "__debug.hpp"
#include "__senders_core.hpp"

#include <tuple>
#include <variant>

namespace stdexec {
#if STDEXEC_ENABLE_EXTRA_TYPE_CHECKING()
  // __checked_completion_signatures is for catching logic bugs in a sender's metadata. If sender<S>
  // and sender_in<S, Ctx> are both true, then they had better report the same metadata. This
  // completion signatures wrapper enforces that at compile time.
  template <class _Sender, class... _Env>
  auto __checked_completion_signatures(_Sender&& __sndr, _Env&&... __env) noexcept {
    using __completions_t = __completion_signatures_of_t<_Sender, _Env...>;
    stdexec::__debug_sender(static_cast<_Sender&&>(__sndr), __env...);
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
  using __variant = //
    __minvoke<
      __if_c<
        sizeof...(_Ts) != 0,
        __transform<__q<__decay_t>, __munique<__q<std::variant>>>,
        __mconst<__not_a_variant>>,
      _Ts...>;

  using __nullable_variant_t = __munique<__mbind_front_q<std::variant, std::monostate>>;

  template <class... _Ts>
  using __decayed_tuple = __meval<std::tuple, __decay_t<_Ts>...>;

  template <class _Tag, class _Tuple>
  struct __select_completions_for {
    template <same_as<_Tag> _Tag2, class... _Args>
    using __f = __minvoke<_Tag2, _Tuple, _Args...>;
  };

  template <class _Tuple>
  struct __invoke_completions {
    template <class _Tag, class... _Args>
    using __f = __minvoke<_Tag, _Tuple, _Args...>;
  };

  template <class _Tag, class _Tuple>
  using __select_completions_for_or = //
    __with_default<__select_completions_for<_Tag, _Tuple>, __>;

  template <class _Tag, class _Completions>
  using __only_gather_signal = //
    __compl_sigs::__maybe_for_all_sigs<
      _Completions,
      __select_completions_for_or<_Tag, __qf<_Tag>>,
      __remove<__, __q<completion_signatures>>>;

  template <class _Tag, class _Completions, class _Tuple, class _Variant>
  using __gather_signal = //
    __compl_sigs::__maybe_for_all_sigs<
      __only_gather_signal<_Tag, _Completions>,
      __invoke_completions<_Tuple>,
      _Variant>;

  template <class _Tag, class _Sender, class _Env, class _Tuple, class _Variant>
  using __gather_completions_for = //
    __meval<                       //
      __gather_signal,
      _Tag,
      __completion_signatures_of_t<_Sender, _Env>,
      _Tuple,
      _Variant>;

  template <                             //
    class _Sender,                       //
    class _Env = empty_env,              //
    class _Tuple = __q<__decayed_tuple>, //
    class _Variant = __q<__variant>>
  using __try_value_types_of_t = //
    __gather_completions_for<set_value_t, _Sender, _Env, _Tuple, _Variant>;

  template <                             //
    class _Sender,                       //
    class _Env = empty_env,              //
    class _Tuple = __q<__decayed_tuple>, //
    class _Variant = __q<__variant>>
    requires sender_in<_Sender, _Env>
  using __value_types_of_t = //
    __msuccess_or_t<__try_value_types_of_t<_Sender, _Env, _Tuple, _Variant>>;

  template <class _Sender, class _Env = empty_env, class _Variant = __q<__variant>>
  using __try_error_types_of_t =
    __gather_completions_for<set_error_t, _Sender, _Env, __q<__midentity>, _Variant>;

  template <class _Sender, class _Env = empty_env, class _Variant = __q<__variant>>
    requires sender_in<_Sender, _Env>
  using __error_types_of_t = __msuccess_or_t<__try_error_types_of_t<_Sender, _Env, _Variant>>;

  template <                                            //
    class _Sender,                                      //
    class _Env = empty_env,                             //
    template <class...> class _Tuple = __decayed_tuple, //
    template <class...> class _Variant = __variant>
    requires sender_in<_Sender, _Env>
  using value_types_of_t = __value_types_of_t<_Sender, _Env, __q<_Tuple>, __q<_Variant>>;

  template <class _Sender, class _Env = empty_env, template <class...> class _Variant = __variant>
    requires sender_in<_Sender, _Env>
  using error_types_of_t = __error_types_of_t<_Sender, _Env, __q<_Variant>>;

  template <class _Tag, class _Sender, class _Env = empty_env>
  using __try_count_of = //
    __compl_sigs::__maybe_for_all_sigs<
      __completion_signatures_of_t<_Sender, _Env>,
      __q<__mfront>,
      __mcount<_Tag>>;

  template <class _Tag, class _Sender, class _Env = empty_env>
    requires sender_in<_Sender, _Env>
  using __count_of = __msuccess_or_t<__try_count_of<_Tag, _Sender, _Env>>;

  template <class _Tag, class _Sender, class _Env = empty_env>
    requires __mvalid<__count_of, _Tag, _Sender, _Env>
  inline constexpr bool __sends = (__v<__count_of<_Tag, _Sender, _Env>> != 0);

  template <class _Sender, class _Env = empty_env>
    requires __mvalid<__count_of, set_stopped_t, _Sender, _Env>
  inline constexpr bool sends_stopped = __sends<set_stopped_t, _Sender, _Env>;

  template <class _Sender, class _Env = empty_env>
  using __single_sender_value_t =
    __value_types_of_t<_Sender, _Env, __msingle_or<void>, __q<__msingle>>;

  template <class _Sender, class _Env = empty_env>
  using __single_value_variant_sender_t = value_types_of_t<_Sender, _Env, __types, __msingle>;

  template <class _Sender, class _Env = empty_env>
  concept __single_typed_sender =
    sender_in<_Sender, _Env> && __mvalid<__single_sender_value_t, _Sender, _Env>;

  template <class _Sender, class _Env = empty_env>
  concept __single_value_variant_sender =
    sender_in<_Sender, _Env> && __mvalid<__single_value_variant_sender_t, _Sender, _Env>;

  template <class... Errs>
  using __nofail = __mbool<sizeof...(Errs) == 0>;

  template <class _Sender, class _Env = empty_env>
  concept __nofail_sender =
    sender_in<_Sender, _Env> && (__v<error_types_of_t<_Sender, _Env, __nofail>>);

  /////////////////////////////////////////////////////////////////////////////
  namespace __compl_sigs {
    template <class... _Args>
    using __default_set_value = completion_signatures<set_value_t(_Args...)>;

    template <class _Error>
    using __default_set_error = completion_signatures<set_error_t(_Error)>;

    template <__valid_completion_signatures... _Sigs>
    using __ensure_concat_ = __minvoke<__mconcat<__q<completion_signatures>>, _Sigs...>;

    template <class... _Sigs>
    using __ensure_concat = __mtry_eval<__ensure_concat_, _Sigs...>;

    template <class _Sender, class _Env, class _Sigs, class _SetVal, class _SetErr, class _SetStp>
    using __compl_sigs_impl = //
      __concat_completion_signatures_t<
        _Sigs,
        __mtry_eval<__try_value_types_of_t, _Sender, _Env, _SetVal, __q<__ensure_concat>>,
        __mtry_eval<__try_error_types_of_t, _Sender, _Env, __transform<_SetErr, __q<__ensure_concat>>>,
        __if<__try_count_of<set_stopped_t, _Sender, _Env>, _SetStp, completion_signatures<>>>;

    template <class _Sender, class _Env, class _Sigs, class _SetVal, class _SetErr, class _SetStp>
      requires __mvalid<__compl_sigs_impl, _Sender, _Env, _Sigs, _SetVal, _SetErr, _SetStp>
    extern __compl_sigs_impl<_Sender, _Env, _Sigs, _SetVal, _SetErr, _SetStp> __compl_sigs_v;

    template <class _Sender, class _Env, class _Sigs, class _SetVal, class _SetErr, class _SetStp>
    using __compl_sigs_t =
      decltype(__compl_sigs_v<_Sender, _Env, _Sigs, _SetVal, _SetErr, _SetStp>);

    template <                                                    //
      class _Sender,                                              //
      class _Env = empty_env,                                     //
      class _Sigs = completion_signatures<>,                      //
      class _SetValue = __q<__default_set_value>,                 //
      class _SetError = __q<__default_set_error>,                 //
      class _SetStopped = completion_signatures<set_stopped_t()>> //
    using __try_make_completion_signatures =                      //
      __meval<__compl_sigs_t, _Sender, _Env, _Sigs, _SetValue, _SetError, _SetStopped>;
  } // namespace __compl_sigs

  using __compl_sigs::__try_make_completion_signatures;

  /////////////////////////////////////////////////////////////////////////////
  // NOT TO SPEC
  //
  // make_completion_signatures
  // ==========================
  //
  // `make_completion_signatures` takes a sender, and environment, and a bunch
  // of other template arguments for munging the completion signatures of a
  // sender in interesting ways.
  //
  //  ```c++
  //  template <class... Args>
  //    using __default_set_value = completion_signatures<set_value_t(Args...)>;
  //
  //  template <class Err>
  //    using __default_set_error = completion_signatures<set_error_t(Err)>;
  //
  //  template <
  //    sender Sndr,
  //    class Env = empty_env,
  //    class AddlSigs = completion_signatures<>,
  //    template <class...> class SetValue = __default_set_value,
  //    template <class> class SetError = __default_set_error,
  //    class SetStopped = completion_signatures<set_stopped_t()>>
  //      requires sender_in<Sndr, Env>
  //  using make_completion_signatures =
  //    completion_signatures< ... >;
  //  ```
  //
  //  * `SetValue` : an alias template that accepts a set of value types and
  //    returns an instance of `completion_signatures`.
  //  * `SetError` : an alias template that accepts an error types and returns a
  //    an instance of `completion_signatures`.
  //  * `SetStopped` : an instantiation of `completion_signatures` with a list
  //    of completion signatures `Sigs...` to the added to the list if the
  //    sender can complete with a stopped signal.
  //  * `AddlSigs` : an instantiation of `completion_signatures` with a list of
  //    completion signatures `Sigs...` to the added to the list
  //    unconditionally.
  //
  //  `make_completion_signatures` does the following:
  //  * Let `VCs...` be a pack of the `completion_signatures` types in the
  //    `__typelist` named by `value_types_of_t<Sndr, Env, SetValue,
  //    __typelist>`, and let `Vs...` be the concatenation of the packs that are
  //    template arguments to each `completion_signature` in `VCs...`.
  //  * Let `ECs...` be a pack of the `completion_signatures` types in the
  //    `__typelist` named by `error_types_of_t<Sndr, Env, __errorlist>`, where
  //    `__errorlist` is an alias template such that `__errorlist<Ts...>` names
  //    `__typelist<SetError<Ts>...>`, and let `Es...` by the concatenation of
  //    the packs that are the template arguments to each `completion_signature`
  //    in `ECs...`.
  //  * Let `Ss...` be an empty pack if `sends_stopped<Sndr, Env>` is
  //    `false`; otherwise, a pack containing the template arguments of the
  //    `completion_signatures` instantiation named by `SetStopped`.
  //  * Let `MoreSigs...` be a pack of the template arguments of the
  //    `completion_signatures` instantiation named by `AddlSigs`.
  //
  //  Then `make_completion_signatures<Sndr, Env, AddlSigs, SetValue, SetError,
  //  SendsStopped>` names the type `completion_signatures< Sigs... >` where
  //  `Sigs...` is the unique set of types in `[Vs..., Es..., Ss...,
  //  MoreSigs...]`.
  //
  //  If any of the above type computations are ill-formed,
  //  `make_completion_signatures<Sndr, Env, AddlSigs, SetValue, SetError,
  //  SendsStopped>` is an alias for an empty struct
  template <                                                                 //
    class _Sender,                                                           //
    class _Env = empty_env,                                                  //
    __valid_completion_signatures _Sigs = completion_signatures<>,           //
    template <class...> class _SetValue = __compl_sigs::__default_set_value, //
    template <class> class _SetError = __compl_sigs::__default_set_error,    //
    __valid_completion_signatures _SetStopped = completion_signatures<set_stopped_t()>>
    requires sender_in<_Sender, _Env>
  using transform_completion_signatures_of = //
    __msuccess_or_t<                         //
      __try_make_completion_signatures<_Sender, _Env, _Sigs, __q<_SetValue>, __q<_SetError>, _SetStopped>>;

  template <                                                                 //
    class _Sender,                                                           //
    class _Env = empty_env,                                                  //
    __valid_completion_signatures _Sigs = completion_signatures<>,           //
    template <class...> class _SetValue = __compl_sigs::__default_set_value, //
    template <class> class _SetError = __compl_sigs::__default_set_error,    //
    __valid_completion_signatures _SetStopped = completion_signatures<set_stopped_t()>>
    requires sender_in<_Sender, _Env>
  using make_completion_signatures =
    transform_completion_signatures_of<_Sender, _Env, _Sigs, _SetValue, _SetError, _SetStopped>;

  // The following utilities are needed fairly often:
  using __with_exception_ptr = completion_signatures<set_error_t(std::exception_ptr)>;

  template <class _Fun, class... _Args>
    requires __invocable<_Fun, _Args...>
  using __nothrow_invocable_t = __mbool<__nothrow_invocable<_Fun, _Args...>>;

  template <class _Tag, class _Fun, class _Sender, class _Env, class _Catch>
  using __with_error_invoke_t = //
    __if<
      __gather_completions_for<
        _Tag,
        _Sender,
        _Env,
        __mbind_front<__mtry_catch_q<__nothrow_invocable_t, _Catch>, _Fun>,
        __q<__mand>>,
      completion_signatures<>,
      __with_exception_ptr>;

  template <class _Fun, class... _Args>
    requires __invocable<_Fun, _Args...>
  using __set_value_invoke_t = //
    completion_signatures<
      __minvoke<__remove<void, __qf<set_value_t>>, __invoke_result_t<_Fun, _Args...>>>;
} // namespace stdexec
