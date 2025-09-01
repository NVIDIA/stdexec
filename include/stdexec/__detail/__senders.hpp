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
#include "__completion_signatures.hpp"
#include "__concepts.hpp"
#include "__connect_awaitable.hpp"
#include "__debug.hpp"
#include "__env.hpp"
#include "__operation_states.hpp"
#include "__receivers.hpp"
#include "__senders_core.hpp"
#include "__tag_invoke.hpp"
#include "__transform_completion_signatures.hpp"
#include "__transform_sender.hpp"
#include "__type_traits.hpp"

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.get_completion_signatures]
  namespace __detail {
    struct __dependent_completions { };

    using dependent_completions = _ERROR_<__detail::__dependent_completions>;

    template <class _Completions>
    concept __well_formed_completions_helper = __valid_completion_signatures<_Completions>
                                            || __same_as<_Completions, dependent_completions>;

    template <class _Completions>
    inline constexpr bool __well_formed_completions_v =
      __well_formed_completions_helper<_Completions>;

    template <class _Completions>
    concept __well_formed_completions = __well_formed_completions_v<_Completions>;
  } // namespace __detail

  using __detail::dependent_completions;

  namespace __sigs {
    template <class _Sender, class _Env>
    using __tfx_sender =
      transform_sender_result_t<__late_domain_of_t<_Sender, _Env>, _Sender, _Env>;

    template <class _Sender, class... _Env>
    using __member_result_t = decltype(__declval<_Sender>()
                                         .get_completion_signatures(__declval<_Env>()...));

    template <class _Sender, class... _Env>
    using __static_member_result_t = decltype(STDEXEC_REMOVE_REFERENCE(
      _Sender)::get_completion_signatures(__declval<_Sender>(), __declval<_Env>()...));

    template <class _Sender, class... _Env>
    concept __with_member = __mvalid<__member_result_t, _Sender, _Env...>;

    template <class _Sender, class... _Env>
    concept __with_static_member = __mvalid<__static_member_result_t, _Sender, _Env...>;

    template <class _Sender, class... _Env>
    concept __with_tag_invoke = tag_invocable<get_completion_signatures_t, _Sender, _Env...>;

    template <class _Sender, class... _Env>
    concept __with_legacy_tag_invoke = (sizeof...(_Env) == 0)
                                    && tag_invocable<get_completion_signatures_t, _Sender, env<>>;

    template <class _Sender>
    using __member_alias_t = typename __decay_t<_Sender>::completion_signatures;

    template <class _Sender>
    concept __with_member_alias = __mvalid<__member_alias_t, _Sender>;

    struct get_completion_signatures_t {
      template <class _Sender, class... _Env>
        requires(sizeof...(_Env) <= 1)
      static auto __impl() {
        // Compute the type of the transformed sender:
        using __tfx_fn = __if_c<sizeof...(_Env) == 0, __mconst<_Sender>, __q<__tfx_sender>>;
        using _TfxSender = __minvoke<__tfx_fn, _Sender, _Env...>;

        if constexpr (__merror<_TfxSender>) {
          // Computing the type of the transformed sender returned an error type. Propagate it.
          return static_cast<_TfxSender (*)()>(nullptr);
        } else if constexpr (__with_member_alias<_TfxSender>) {
          using _Result = __member_alias_t<_TfxSender>;
          return static_cast<_Result (*)()>(nullptr);
        } else if constexpr (__with_static_member<_TfxSender, _Env...>) {
          using _Result = __static_member_result_t<_TfxSender, _Env...>;
          return static_cast<_Result (*)()>(nullptr);
        } else if constexpr (__with_member<_TfxSender, _Env...>) {
          using _Result = __member_result_t<_TfxSender, _Env...>;
          return static_cast<_Result (*)()>(nullptr);
        } else if constexpr (__with_tag_invoke<_TfxSender, _Env...>) {
          using _Result = tag_invoke_result_t<get_completion_signatures_t, _TfxSender, _Env...>;
          return static_cast<_Result (*)()>(nullptr);
        } else if constexpr (__with_legacy_tag_invoke<_TfxSender, _Env...>) {
          // This branch is strictly for backwards compatibility
          using _Result = tag_invoke_result_t<get_completion_signatures_t, _Sender, env<>>;
          return static_cast<_Result (*)()>(nullptr);
          // [WAR] The explicit cast to bool below is to work around a bug in nvc++ (nvbug#4707793)
        } else if constexpr (bool(__awaitable<_TfxSender, __env::__promise<_Env>...>)) {
          using _AwaitResult = __await_result_t<_TfxSender, __env::__promise<_Env>...>;
          using _Result = completion_signatures<
            // set_value_t() or set_value_t(T)
            __minvoke<__mremove<void, __qf<set_value_t>>, _AwaitResult>,
            set_error_t(std::exception_ptr),
            set_stopped_t()
          >;
          return static_cast<_Result (*)()>(nullptr);
        } else if constexpr (sizeof...(_Env) == 0) {
          // It's possible this is a dependent sender.
          return static_cast<dependent_completions (*)()>(nullptr);
        } else if constexpr ((__is_debug_env<_Env> || ...)) {
          using __tag_invoke::tag_invoke;
          // This ought to cause a hard error that indicates where the problem is.
          using _Completions
            [[maybe_unused]] = tag_invoke_result_t<get_completion_signatures_t, _Sender, _Env...>;
          return static_cast<__debug::__completion_signatures (*)()>(nullptr);
        } else {
          using _Result = __mexception<
            _UNRECOGNIZED_SENDER_TYPE_<>,
            _WITH_SENDER_<_Sender>,
            _WITH_ENVIRONMENT_<_Env>...
          >;
          return static_cast<_Result (*)()>(nullptr);
        }
      }

      // NOT TO SPEC: if we're unable to compute the completion signatures,
      // return an error type instead of SFINAE.
      template <class _Sender, class... _Env>
        requires(sizeof...(_Env) <= 1)
      constexpr auto
        operator()(_Sender&&, _Env&&...) const noexcept -> decltype(__impl<_Sender, _Env...>()()) {
        return {};
      }
    };
  } // namespace __sigs

  using __sigs::get_completion_signatures_t;
  inline constexpr get_completion_signatures_t get_completion_signatures{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.connect]
  namespace __connect {
    template <class _Sender, class _Receiver>
    using __tfx_sender = __mmemoize_q<
      transform_sender_result_t,
      __late_domain_of_t<_Sender, env_of_t<_Receiver>>,
      _Sender,
      env_of_t<_Receiver>
    >;

    template <class _Sender, class _Receiver>
    using __member_result_t = decltype(__declval<_Sender>().connect(__declval<_Receiver>()));

    template <class _Sender, class _Receiver>
    using __static_member_result_t = decltype(STDEXEC_REMOVE_REFERENCE(
      _Sender)::connect(__declval<_Sender>(), __declval<_Receiver>()));

    template <class _Sender, class _Receiver>
    concept __with_member = __mvalid<__member_result_t, _Sender, _Receiver>;

    template <class _Sender, class _Receiver>
    concept __with_static_member = __mvalid<__static_member_result_t, _Sender, _Receiver>;

    template <class _Sender, class _Receiver>
    concept __with_tag_invoke = tag_invocable<connect_t, _Sender, _Receiver>;

    template <class _Sender, class _Receiver>
    concept __with_co_await = __callable<__connect_awaitable_t, _Sender, _Receiver>;

    struct _NO_USABLE_CONNECT_CUSTOMIZATION_FOUND_ {
      void operator()() const noexcept = delete;
    };

    struct connect_t {
      template <class _Sender, class _Receiver>
        requires sender_in<_Sender, env_of_t<_Receiver>> && __receiver_from<_Receiver, _Sender>
      STDEXEC_ATTRIBUTE(always_inline)
      static constexpr auto __type_check_arguments() -> bool {
        if constexpr (sender_in<_Sender, env_of_t<_Receiver>>) {
          // Instantiate __debug_sender via completion_signatures_of_t to check that the actual
          // completions match the expected completions.
          using __checked_signatures
            [[maybe_unused]] = completion_signatures_of_t<_Sender, env_of_t<_Receiver>>;
        } else {
          __diagnose_sender_concept_failure<_Sender, env_of_t<_Receiver>>();
        }
        return true;
      }

      template <class _OpState>
      static constexpr void __check_operation_state() noexcept {
        static_assert(operation_state<_OpState>, STDEXEC_ERROR_CANNOT_CONNECT_SENDER_TO_RECEIVER);
      }

      template <class _Sender, class _Receiver>
      static constexpr auto __select_impl() noexcept {
        using _Domain = __late_domain_of_t<_Sender, env_of_t<_Receiver>>;
        using _TfxSender = __tfx_sender<_Sender, _Receiver>;
        constexpr bool _NothrowTfxSender =
          __nothrow_callable<transform_sender_t, _Domain, _Sender, env_of_t<_Receiver>>;

        static_assert(sender<_Sender>, "The first argument to stdexec::connect must be a sender");
        static_assert(
          receiver<_Receiver>, "The second argument to stdexec::connect must be a receiver");
#if STDEXEC_ENABLE_EXTRA_TYPE_CHECKING()
        static_assert(__type_check_arguments<_TfxSender, _Receiver>());
#endif

        if constexpr (__with_static_member<_TfxSender, _Receiver>) {
          using _Result = __static_member_result_t<_TfxSender, _Receiver>;
          __check_operation_state<_Result>();
          constexpr bool _Nothrow = _NothrowTfxSender
                                 && noexcept(
                                      __declval<_TfxSender>()
                                        .connect(__declval<_TfxSender>(), __declval<_Receiver>()));
          return static_cast<_Result (*)() noexcept(_Nothrow)>(nullptr);
        } else if constexpr (__with_member<_TfxSender, _Receiver>) {
          using _Result = __member_result_t<_TfxSender, _Receiver>;
          __check_operation_state<_Result>();
          constexpr bool _Nothrow = _NothrowTfxSender
                                 && noexcept(__declval<_TfxSender>()
                                               .connect(__declval<_Receiver>()));
          return static_cast<_Result (*)() noexcept(_Nothrow)>(nullptr);
        } else if constexpr (__with_tag_invoke<_TfxSender, _Receiver>) {
          using _Result = tag_invoke_result_t<connect_t, _TfxSender, _Receiver>;
          __check_operation_state<_Result>();
          constexpr bool _Nothrow = _NothrowTfxSender
                                 && nothrow_tag_invocable<connect_t, _TfxSender, _Receiver>;
          return static_cast<_Result (*)() noexcept(_Nothrow)>(nullptr);
        } else if constexpr (__with_co_await<_TfxSender, _Receiver>) {
          using _Result = __call_result_t<__connect_awaitable_t, _TfxSender, _Receiver>;
          return static_cast<_Result (*)()>(nullptr);
        } else if constexpr (__is_debug_env<env_of_t<_Receiver>>) {
          using _Result = __debug::__debug_operation;
          return static_cast<_Result (*)() noexcept(_NothrowTfxSender)>(nullptr);
        } else {
          return _NO_USABLE_CONNECT_CUSTOMIZATION_FOUND_();
        }
      }

      template <class _Sender, class _Receiver>
      using __select_impl_t = decltype(__select_impl<_Sender, _Receiver>());

      template <class _Sender, class _Receiver>
      auto operator()(_Sender&& __sndr, _Receiver&& __rcvr) const
        noexcept(__nothrow_callable<__select_impl_t<_Sender, _Receiver>>)
          -> __call_result_t<__select_impl_t<_Sender, _Receiver>> {

        static_assert(sender_in<_Sender, env_of_t<_Receiver>>);
        static_assert(__receiver_from<_Receiver, _Sender>);

        using _TfxSender = __tfx_sender<_Sender, _Receiver>;
        auto&& __env = get_env(__rcvr);
        auto __domain = __get_late_domain(__sndr, __env);

        if constexpr (__with_static_member<_TfxSender, _Receiver>) {
          auto&& __tfx_sndr = transform_sender(__domain, static_cast<_Sender&&>(__sndr), __env);
          return __tfx_sndr
            .connect(static_cast<_TfxSender&&>(__tfx_sndr), static_cast<_Receiver&&>(__rcvr));
        } else if constexpr (__with_member<_TfxSender, _Receiver>) { // NOLINT(bugprone-branch-clone)
          return transform_sender(__domain, static_cast<_Sender&&>(__sndr), __env)
            .connect(static_cast<_Receiver&&>(__rcvr));
        } else if constexpr (__with_tag_invoke<_TfxSender, _Receiver>) {
          return tag_invoke(
            connect_t(),
            transform_sender(__domain, static_cast<_Sender&&>(__sndr), __env),
            static_cast<_Receiver&&>(__rcvr));
        } else if constexpr (__with_co_await<_TfxSender, _Receiver>) {
          return __connect_awaitable(
            transform_sender(__domain, static_cast<_Sender&&>(__sndr), __env),
            static_cast<_Receiver&&>(__rcvr));
        } else {
          // This should generate an instantiation backtrace that contains useful
          // debugging information.
          return transform_sender(__domain, static_cast<_Sender&&>(__sndr), __env)
            .connect(static_cast<_Receiver&&>(__rcvr));
        }
      }

      static constexpr auto query(forwarding_query_t) noexcept -> bool {
        return false;
      }
    };
  } // namespace __connect

  using __connect::connect_t;
  inline constexpr __connect::connect_t connect{};

  /////////////////////////////////////////////////////////////////////////////
  // [exec.snd]
  template <class _Tag, class... _Args>
  auto __tag_of_sig_(_Tag (*)(_Args...)) -> _Tag;
  template <class _Sig>
  using __tag_of_sig_t = decltype(stdexec::__tag_of_sig_(static_cast<_Sig*>(nullptr)));

  template <class _Sender, class _SetSig, class _Env = env<>>
  concept sender_of = sender_in<_Sender, _Env>
                   && same_as<
                        __types<_SetSig>,
                        __gather_completions_of<
                          __tag_of_sig_t<_SetSig>,
                          _Sender,
                          _Env,
                          __mcompose_q<__types, __qf<__tag_of_sig_t<_SetSig>>::template __f>,
                          __mconcat<__qq<__types>>
                        >
                   >;

  template <class _Error>
    requires false
  using __nofail_t = _Error;

  template <class _Sender, class _Env = env<>>
  concept __nofail_sender = sender_in<_Sender, _Env> && requires {
    typename __gather_completion_signatures<
      __completion_signatures_of_t<_Sender, _Env>,
      set_error_t,
      __nofail_t,
      __sigs::__default_completion,
      __types
    >;
  };

  /////////////////////////////////////////////////////////////////////////////
  // early sender type-checking
  template <class _Sender>
  concept __well_formed_sender = __detail::__well_formed_completions<
    __minvoke<__with_default_q<__completion_signatures_of_t, dependent_completions>, _Sender>
  >;
} // namespace stdexec
