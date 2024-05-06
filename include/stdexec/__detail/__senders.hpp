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
#include "__awaitable.hpp"
#include "__completion_signatures.hpp"
#include "__concepts.hpp"
#include "__connect_awaitable.hpp"
#include "__debug.hpp"
#include "__env.hpp"
#include "__operation_states.hpp"
#include "__receivers.hpp"
#include "__senders_core.hpp"
#include "__transform_completion_signatures.hpp"
#include "__transform_sender.hpp"
#include "__type_traits.hpp"

namespace stdexec {
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

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.connect]
  namespace __connect {
    template <class _Sender, class _Receiver>
    using __tfx_sender = //
      transform_sender_result_t<
        __late_domain_of_t<_Sender, env_of_t<_Receiver&>>,
        _Sender,
        env_of_t<_Receiver&>>;

    template <class _Sender, class _Receiver>
    concept __connectable_with_tag_invoke_ =     //
      receiver<_Receiver>                        //
      && sender_in<_Sender, env_of_t<_Receiver>> //
      && __receiver_from<_Receiver, _Sender>     //
      && tag_invocable<connect_t, _Sender, _Receiver>;

    template <class _Sender, class _Receiver>
    concept __connectable_with_tag_invoke = //
      __connectable_with_tag_invoke_<__tfx_sender<_Sender, _Receiver>, _Receiver>;

    template <class _Sender, class _Receiver>
    concept __connectable_with_co_await = //
      __callable<__connect_awaitable_t, __tfx_sender<_Sender, _Receiver>, _Receiver>;

    struct connect_t {
      template <class _Sender, class _Env>
      static constexpr auto __check_signatures() -> bool {
        if constexpr (sender_in<_Sender, _Env>) {
          // Instantiate __debug_sender via completion_signatures_of_t to check that the actual
          // completions match the expected completions.
          //
          // Instantiate completion_signatures_of_t only if sender_in is true to workaround Clang
          // not implementing CWG#2369 yet (connect() does not have a constraint for _Sender
          // satisfying sender_in).
          using __checked_signatures [[maybe_unused]] = completion_signatures_of_t<_Sender, _Env>;
        }
        return true;
      }

      template <class _Sender, class _Receiver>
      static constexpr auto __select_impl() noexcept {
        using _Domain = __late_domain_of_t<_Sender, env_of_t<_Receiver&>>;
        constexpr bool _NothrowTfxSender =
          __nothrow_callable<get_env_t, _Receiver&>
          && __nothrow_callable<transform_sender_t, _Domain, _Sender, env_of_t<_Receiver&>>;
        using _TfxSender = __tfx_sender<_Sender, _Receiver&>;

#if STDEXEC_ENABLE_EXTRA_TYPE_CHECKING()
        static_assert(__check_signatures<_TfxSender, env_of_t<_Receiver>>());
#endif

        if constexpr (__connectable_with_tag_invoke<_Sender, _Receiver>) {
          using _Result = tag_invoke_result_t<connect_t, _TfxSender, _Receiver>;
          constexpr bool _Nothrow = //
            _NothrowTfxSender && nothrow_tag_invocable<connect_t, _TfxSender, _Receiver>;
          return static_cast<_Result (*)() noexcept(_Nothrow)>(nullptr);
        } else if constexpr (__connectable_with_co_await<_Sender, _Receiver>) {
          using _Result = __call_result_t<__connect_awaitable_t, _TfxSender, _Receiver>;
          return static_cast<_Result (*)()>(nullptr);
        } else {
          using _Result = __debug::__debug_operation;
          return static_cast<_Result (*)() noexcept(_NothrowTfxSender)>(nullptr);
        }
      }

      template <class _Sender, class _Receiver>
      using __select_impl_t = decltype(__select_impl<_Sender, _Receiver>());

      template <sender _Sender, receiver _Receiver>
        requires __connectable_with_tag_invoke<_Sender, _Receiver>
              || __connectable_with_co_await<_Sender, _Receiver>
              || __is_debug_env<env_of_t<_Receiver>>
      auto operator()(_Sender&& __sndr, _Receiver&& __rcvr) const
        noexcept(__nothrow_callable<__select_impl_t<_Sender, _Receiver>>)
          -> __call_result_t<__select_impl_t<_Sender, _Receiver>> {
        using _TfxSender = __tfx_sender<_Sender, _Receiver&>;
        auto&& __env = get_env(__rcvr);
        auto __domain = __get_late_domain(__sndr, __env);

        if constexpr (__connectable_with_tag_invoke<_Sender, _Receiver>) {
          static_assert(
            operation_state<tag_invoke_result_t<connect_t, _TfxSender, _Receiver>>,
            "stdexec::connect(sender, receiver) must return a type that "
            "satisfies the operation_state concept");
          return tag_invoke(
            connect_t(),
            transform_sender(__domain, static_cast<_Sender&&>(__sndr), __env),
            static_cast<_Receiver&&>(__rcvr));
        } else if constexpr (__connectable_with_co_await<_Sender, _Receiver>) {
          return __connect_awaitable( //
            transform_sender(__domain, static_cast<_Sender&&>(__sndr), __env),
            static_cast<_Receiver&&>(__rcvr));
        } else {
          // This should generate an instantiation backtrace that contains useful
          // debugging information.
          using __tag_invoke::tag_invoke;
          tag_invoke(
            *this,
            transform_sender(__domain, static_cast<_Sender&&>(__sndr), __env),
            static_cast<_Receiver&&>(__rcvr));
        }
      }

      constexpr STDEXEC_MEMFN_DECL(auto forwarding_query)(this connect_t) noexcept -> bool {
        return false;
      }
    };
  } // namespace __connect

  using __connect::connect_t;
  inline constexpr __connect::connect_t connect{};

  /////////////////////////////////////////////////////////////////////////////
  // [exec.snd]
  template <class _Sender, class _Receiver>
  concept sender_to =                          //
    receiver<_Receiver>                        //
    && sender_in<_Sender, env_of_t<_Receiver>> //
    && __receiver_from<_Receiver, _Sender>     //
    && requires(_Sender&& __sndr, _Receiver&& __rcvr) {
         connect(static_cast<_Sender&&>(__sndr), static_cast<_Receiver&&>(__rcvr));
       };

  template <class _Tag, class... _Args>
  auto __tag_of_sig_(_Tag (*)(_Args...)) -> _Tag;
  template <class _Sig>
  using __tag_of_sig_t = decltype(stdexec::__tag_of_sig_(static_cast<_Sig*>(nullptr)));

  template <class _Sender, class _SetSig, class _Env = empty_env>
  concept sender_of =        //
    sender_in<_Sender, _Env> //
    && same_as<
      __types<_SetSig>,
      __gather_completions_for<
        __tag_of_sig_t<_SetSig>,
        _Sender,
        _Env,
        __qf<__tag_of_sig_t<_SetSig>>,
        __q<__types>>>;
} // namespace stdexec
