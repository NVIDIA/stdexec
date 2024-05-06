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
#include "__concepts.hpp"
#include "__coroutine.hpp"
#include "__debug.hpp"
#include "__diagnostics.hpp"
#include "__domain.hpp"
#include "__tag_invoke.hpp"
#include "__transform_sender.hpp"
#include "__utility.hpp"

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // completion_signatures
  namespace __compl_sigs {
    template <class... _Args>
    inline constexpr bool __is_compl_sig<set_value_t(_Args...)> = true;
    template <class _Error>
    inline constexpr bool __is_compl_sig<set_error_t(_Error)> = true;
    template <>
    inline constexpr bool __is_compl_sig<set_stopped_t()> = true;
  } // namespace __compl_sigs

  template <__completion_signature... _Sigs>
  struct completion_signatures { };

  namespace __compl_sigs {
    template <class _TaggedTuple, class _Tag, class... _Ts>
    auto __as_tagged_tuple_(_Tag (*)(_Ts...), _TaggedTuple*)
      -> __mconst<__minvoke<_TaggedTuple, _Tag, _Ts...>>;

    template <class _Sig, class _TaggedTuple>
    using __as_tagged_tuple = decltype(__compl_sigs::__as_tagged_tuple_(
      static_cast<_Sig*>(nullptr),
      static_cast<_TaggedTuple*>(nullptr)));

    template <class _TaggedTuple, class _Variant, class... _Sigs>
    auto __for_all_sigs_(completion_signatures<_Sigs...>*, _TaggedTuple*, _Variant*)
      -> __mconst<__minvoke<_Variant, __minvoke<__as_tagged_tuple<_Sigs, _TaggedTuple>>...>>;

    template <class _Completions, class _TaggedTuple, class _Variant>
    using __for_all_sigs =                      //
      __minvoke<                                //
        decltype(__compl_sigs::__for_all_sigs_( //
          static_cast<_Completions*>(nullptr),
          static_cast<_TaggedTuple*>(nullptr),
          static_cast<_Variant*>(nullptr)))>;

    template <class _Completions, class _TaggedTuple, class _Variant>
    using __maybe_for_all_sigs = __meval<__for_all_sigs, _Completions, _TaggedTuple, _Variant>;

    template <class...>
    struct __normalize_completions {
      void operator()() const;
    };

    template <class _Ry, class... _As, class... _Rest>
    struct __normalize_completions<_Ry(_As...), _Rest...> : __normalize_completions<_Rest...> {
      auto operator()(_Ry (*)(_As&&...)) const -> _Ry (*)(_As...);
      using __normalize_completions<_Rest...>::operator();
    };

    // MSVCBUG: https://developercommunity.visualstudio.com/t/ICE-in-stdexec-metaprogramming/10642778
    // Previously a lambda was used here, along with __result_of in __normalize_completions.
    // As a workaround for this compiler bug, the lambda was replaced by a function
    // and the use of __result_of was expanded inline.
    template <class... _As, class _Ry, class... _Bs>
    auto __merge_sigs(_Ry (*)(_Bs...)) -> _Ry (*)(__if_c<__same_as<_Bs&&, _Bs>, _As, _Bs>...);

    template <class _Ry, class... _As, class... _Rest>
      requires __callable<__normalize_completions<_Rest...>, _Ry (*)(_As&&...)>
    struct __normalize_completions<_Ry(_As...), _Rest...> {
      auto operator()(_Ry (*)(_As&&...)) const -> decltype(__merge_sigs<_As...>(
        __declval<__call_result_t<__normalize_completions<_Rest...>, _Ry (*)(_As&&...)>>()));

      template <class _Sig>
      auto operator()(_Sig*) const -> __call_result_t<__normalize_completions<_Rest...>, _Sig*>;
    };

    template <class _Fn, class... _As>
    using __norm_sig_t = _Fn (*)(_As&&...);

    template <class T>
    extern __undefined<T> __norm;

    template <class _Ry, class... _As>
    extern __norm_sig_t<_Ry, _As...> __norm<_Ry(_As...)>;

    template <class _Sig>
    using __norm_t = decltype(+__norm<_Sig>);

    inline constexpr auto __convert_to_completion_signatures =
      []<class... Sigs>(__types<Sigs*...>*) -> completion_signatures<Sigs...> {
      return {};
    };

    template <class... _Sigs>
    using __unique_completion_signatures = __result_of<
      __convert_to_completion_signatures,
      __minvoke<
        __transform<
          __mbind_front_q<__call_result_t, __normalize_completions<_Sigs...>>,
          __munique<__q<__types>>>,
        __norm_t<_Sigs>...>*>;
  } // namespace __compl_sigs

  template <class _Completions>
  concept __valid_completion_signatures = //
    __same_as<__ok_t<_Completions>, __msuccess>
    && __is_instance_of<_Completions, completion_signatures>;

  template <class _Completions>
  using __invalid_completion_signatures_t = //
    __mbool<!__valid_completion_signatures<_Completions>>;

  template <__mstring _Msg = "Expected an instance of template completion_signatures<>"_mstr>
  struct _INVALID_COMPLETION_SIGNATURES_TYPE_ {
    template <class... _Completions>
    using __f = //
      __mexception<
        _INVALID_COMPLETION_SIGNATURES_TYPE_<>,
        __minvoke<
          __mfind_if<
            __q<__invalid_completion_signatures_t>,
            __mcompose<__q<_WITH_TYPE_>, __q<__mfront>>>,
          _Completions...>>;
  };

  template <class... _Completions>
  using __concat_completion_signatures_impl_t = //
    __minvoke<
      __if_c<
        (__valid_completion_signatures<_Completions> && ...),
        __mconcat<__q<__compl_sigs::__unique_completion_signatures>>,
        _INVALID_COMPLETION_SIGNATURES_TYPE_<>>,
      _Completions...>;

  template <class... _Completions>
  struct __concat_completion_signatures_ {
    using __t = __meval<__concat_completion_signatures_impl_t, _Completions...>;
  };

  template <class... _Completions>
  using __concat_completion_signatures_t = __t<__concat_completion_signatures_<_Completions...>>;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.receivers]
  template <class _Receiver, class _Tag, class... _Args>
  auto __try_completion(_Tag (*)(_Args...))
    -> __mexception<_MISSING_COMPLETION_SIGNAL_<_Tag(_Args...)>, _WITH_RECEIVER_<_Receiver>>;

  template <class _Receiver, class _Tag, class... _Args>
    requires nothrow_tag_invocable<_Tag, _Receiver, _Args...>
  auto __try_completion(_Tag (*)(_Args...)) -> __msuccess;

  template <class _Receiver, class... _Sigs>
  auto __try_completions(completion_signatures<_Sigs...>*) //
    -> decltype((
      __msuccess(),
      ...,
      stdexec::__try_completion<_Receiver>(static_cast<_Sigs*>(nullptr))));

  template <class _Sender, class _Env>
  using __unrecognized_sender_error = //
    __mexception<_UNRECOGNIZED_SENDER_TYPE_<>, _WITH_SENDER_<_Sender>, _WITH_ENVIRONMENT_<_Env>>;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.get_completion_signatures]
  namespace __compl_sigs {
    template <class _Sender, class _Env>
    using __tfx_sender =
      transform_sender_result_t<__late_domain_of_t<_Sender, _Env>, _Sender, _Env>;

    template <class _Sender, class _Env>
    concept __with_tag_invoke = //
      tag_invocable<get_completion_signatures_t, __tfx_sender<_Sender, _Env>, _Env>;

    template <class _Sender, class _Env>
    using __member_alias_t = //
      typename __decay_t<__tfx_sender<_Sender, _Env>>::completion_signatures;

    template <class _Sender, class _Env = empty_env>
    concept __with_member_alias = __mvalid<__member_alias_t, _Sender, _Env>;

    struct get_completion_signatures_t {
      template <class _Sender, class _Env>
      static auto __impl() {
        static_assert(sizeof(_Sender), "Incomplete type used with get_completion_signatures");
        static_assert(sizeof(_Env), "Incomplete type used with get_completion_signatures");

        // Compute the type of the transformed sender:
        using _TfxSender = __tfx_sender<_Sender, _Env>;

        if constexpr (__merror<_TfxSender>) {
          // Computing the type of the transformed sender returned an error type. Propagate it.
          return static_cast<_TfxSender (*)()>(nullptr);
        } else if constexpr (__with_tag_invoke<_Sender, _Env>) {
          using _Result = tag_invoke_result_t<get_completion_signatures_t, _TfxSender, _Env>;
          return static_cast<_Result (*)()>(nullptr);
        } else if constexpr (__with_member_alias<_Sender, _Env>) {
          using _Result = __member_alias_t<_Sender, _Env>;
          return static_cast<_Result (*)()>(nullptr);
        } else if constexpr (__awaitable<_Sender, __env::__promise<_Env>>) {
          using _AwaitResult = __await_result_t<_Sender, __env::__promise<_Env>>;
          using _Result = completion_signatures<
            // set_value_t() or set_value_t(T)
            __minvoke<__remove<void, __qf<set_value_t>>, _AwaitResult>,
            set_error_t(std::exception_ptr),
            set_stopped_t()>;
          return static_cast<_Result (*)()>(nullptr);
        } else if constexpr (__is_debug_env<_Env>) {
          using __tag_invoke::tag_invoke;
          // This ought to cause a hard error that indicates where the problem is.
          using _Completions
            [[maybe_unused]] = tag_invoke_result_t<get_completion_signatures_t, _Sender, _Env>;
          return static_cast<__debug::__completion_signatures (*)()>(nullptr);
        } else {
          using _Result = __mexception<
            _UNRECOGNIZED_SENDER_TYPE_<>,
            _WITH_SENDER_<_Sender>,
            _WITH_ENVIRONMENT_<_Env>>;
          return static_cast<_Result (*)()>(nullptr);
        }
      }

      // NOT TO SPEC: if we're unable to compute the completion signatures,
      // return an error type instead of SFINAE.
      template <class _Sender, class _Env = empty_env>
      constexpr auto operator()(_Sender&&, _Env&& = {}) const noexcept //
        -> decltype(__impl<_Sender, _Env>()()) {
        return {};
      }
    };
  } // namespace __compl_sigs

  using __compl_sigs::get_completion_signatures_t;
  inline constexpr get_completion_signatures_t get_completion_signatures{};
} // namespace stdexec
