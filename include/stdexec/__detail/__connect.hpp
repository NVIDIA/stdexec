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
#include "__completion_signatures_of.hpp"
#include "__connect_awaitable.hpp"
#include "__debug.hpp"
#include "__meta.hpp"
#include "__operation_states.hpp"
#include "__tag_invoke.hpp"
#include "__type_traits.hpp"

namespace STDEXEC {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.connect]
  namespace __connect {
    template <class _Sender, class _Receiver>
    using __tfx_sender = __mmemoize_q<transform_sender_result_t, _Sender, env_of_t<_Receiver>>;

    template <class _Sender, class _Receiver>
    using __member_result_t = decltype(__declval<_Sender>().connect(__declval<_Receiver>()));

    template <class _Sender, class _Receiver>
    using __static_member_result_t =             //
      decltype(STDEXEC_REMOVE_REFERENCE(_Sender) //
               ::static_connect(__declval<_Sender>(), __declval<_Receiver>()));

    template <class _Sender, class _Receiver>
    concept __with_member = __mvalid<__member_result_t, _Sender, _Receiver>;

    template <class _Sender, class _Receiver>
    concept __with_static_member = __mvalid<__static_member_result_t, _Sender, _Receiver>;

    template <class _Sender, class _Receiver>
    concept __with_legacy_tag_invoke = tag_invocable<connect_t, _Sender, _Receiver>;

    template <class _Sender, class _Receiver>
    concept __with_co_await = __callable<__connect_awaitable_t, _Sender, _Receiver>;

    template <class _Sender, class _Receiver>
    struct _NO_USABLE_CONNECT_CUSTOMIZATION_FOUND_ {
      void operator()() const noexcept = delete;
    };

    /////////////////////////////////////////////////////////////////////////////
    // connect_t
    struct connect_t {
      template <class _Sender, class _Receiver>
      STDEXEC_ATTRIBUTE(always_inline)
      static constexpr auto __type_check_arguments() noexcept -> bool {
        if constexpr (sender_in<_Sender, env_of_t<_Receiver>>) {
          // Instantiate __debug_sender via completion_signatures_of_t to check that the actual
          // completions match the expected completions.
          using __checked_signatures
            [[maybe_unused]] = completion_signatures_of_t<_Sender, env_of_t<_Receiver>>;
        } else {
          __diagnose_sender_concept_failure<__demangle_t<_Sender>, env_of_t<_Receiver>>();
        }
        return true;
      }

      template <class _OpState>
      static constexpr void __check_operation_state() noexcept {
        static_assert(operation_state<_OpState>, STDEXEC_ERROR_CANNOT_CONNECT_SENDER_TO_RECEIVER);
      }

      template <class _Sender, class _Receiver>
      static consteval auto __get_declfn() noexcept {
        static_assert(sender<_Sender>, "The first argument to STDEXEC::connect must be a sender");
        if constexpr (!receiver<_Receiver>) {
          static_assert(
            __nothrow_move_constructible<__decay_t<_Receiver>>,
            "Receivers must be nothrow move constructible");
          static_assert(
            receiver<_Receiver>, "The second argument to STDEXEC::connect must be a receiver");
        }

        static_assert(sender_in<_Sender, env_of_t<_Receiver>>);
        static_assert(__receiver_from<_Receiver, _Sender>);

        using _TfxSender = __tfx_sender<_Sender, _Receiver>;
        constexpr bool _NothrowTfxSender =
          __nothrow_callable<transform_sender_t, _Sender, env_of_t<_Receiver>>;

#if STDEXEC_ENABLE_EXTRA_TYPE_CHECKING()
        static_assert(__type_check_arguments<_TfxSender, _Receiver>());
#endif

        if constexpr (__with_static_member<_TfxSender, _Receiver>) {
          using _Result = __static_member_result_t<_TfxSender, _Receiver>;
          __check_operation_state<_Result>();
          constexpr bool _Nothrow = _NothrowTfxSender
                                 && noexcept(STDEXEC_REMOVE_REFERENCE(_TfxSender)::static_connect(
                                   __declval<_TfxSender>(), __declval<_Receiver>()));
          return __declfn<_Result, _Nothrow>();
        } else if constexpr (__with_member<_TfxSender, _Receiver>) {
          using _Result = __member_result_t<_TfxSender, _Receiver>;
          __check_operation_state<_Result>();
          constexpr bool _Nothrow = _NothrowTfxSender
                                 && noexcept(__declval<_TfxSender>()
                                               .connect(__declval<_Receiver>()));
          return __declfn<_Result, _Nothrow>();
        } else if constexpr (__with_co_await<_TfxSender, _Receiver>) {
          using _Result = __call_result_t<__connect_awaitable_t, _TfxSender, _Receiver>;
          return __declfn<_Result, false>();
        } else if constexpr (__is_debug_env<env_of_t<_Receiver>>) {
          using _Result = __debug::__debug_operation;
          return __declfn<_Result, _NothrowTfxSender>();
        } else {
          return _NO_USABLE_CONNECT_CUSTOMIZATION_FOUND_<
            _WITH_SENDER_<__demangle_t<_TfxSender>>,
            _WITH_RECEIVER_<_Receiver>
          >();
        }
      }

      template <class _Sender, class _Receiver, auto _DeclFn = __get_declfn<_Sender, _Receiver>()>
        requires __callable<__mtypeof<_DeclFn>>
      constexpr auto operator()(_Sender&& __sndr, _Receiver&& __rcvr) const
        noexcept(noexcept(_DeclFn())) -> decltype(_DeclFn()) {

        using _TfxSender = __tfx_sender<_Sender, _Receiver>;
        auto&& __env = get_env(__rcvr);

        if constexpr (__with_static_member<_TfxSender, _Receiver>) {
          auto&& __tfx_sndr = transform_sender(static_cast<_Sender&&>(__sndr), __env);
          return STDEXEC_REMOVE_REFERENCE(_TfxSender)::static_connect(
            static_cast<_TfxSender&&>(__tfx_sndr), static_cast<_Receiver&&>(__rcvr));
        } else if constexpr (__with_member<_TfxSender, _Receiver>) { // NOLINT(bugprone-branch-clone)
          auto&& __tfx_sndr = transform_sender(static_cast<_Sender&&>(__sndr), __env);
          return static_cast<_TfxSender&&>(__tfx_sndr).connect(static_cast<_Receiver&&>(__rcvr));
        } else if constexpr (__with_co_await<_TfxSender, _Receiver>) {
          auto&& __tfx_sndr = transform_sender(static_cast<_Sender&&>(__sndr), __env);
          return __connect_awaitable(
            static_cast<_TfxSender&&>(__tfx_sndr), static_cast<_Receiver&&>(__rcvr));
        } else {
          // This should generate an instantiation backtrace that contains useful
          // debugging information.
          auto&& __tfx_sndr = transform_sender(static_cast<_Sender&&>(__sndr), __env);
          return static_cast<_TfxSender&&>(__tfx_sndr).connect(static_cast<_Receiver&&>(__rcvr));
        }
      }

      template <class _Sender, class _Receiver, auto _DeclFn = __get_declfn<_Sender, _Receiver>()>
        requires __callable<__mtypeof<_DeclFn>>
              || __with_legacy_tag_invoke<__tfx_sender<_Sender, _Receiver>, _Receiver>
      [[deprecated("the use of tag_invoke for connect is deprecated")]]
      constexpr auto operator()(_Sender&& __sndr, _Receiver&& __rcvr) const noexcept(
        __nothrow_callable<transform_sender_t, _Sender, env_of_t<_Receiver>>
        && nothrow_tag_invocable<connect_t, __tfx_sender<_Sender, _Receiver>, _Receiver>)
        -> tag_invoke_result_t<connect_t, __tfx_sender<_Sender, _Receiver>, _Receiver> {
        using _TfxSender = __tfx_sender<_Sender, _Receiver>;
        using _Result = tag_invoke_result_t<connect_t, _TfxSender, _Receiver>;
        __check_operation_state<_Result>();
        auto&& __env = get_env(__rcvr);
        auto&& __tfx_sndr = transform_sender(static_cast<_Sender&&>(__sndr), __env);
        return tag_invoke(
          connect_t(), static_cast<_TfxSender&&>(__tfx_sndr), static_cast<_Receiver&&>(__rcvr));
      }

      static constexpr auto query(forwarding_query_t) noexcept -> bool {
        return false;
      }
    };
  } // namespace __connect

  using __connect::connect_t;
  inline constexpr __connect::connect_t connect{};

  template <class _Sender, class _Receiver>
  concept __nothrow_connectable = sender_to<_Sender, _Receiver>
                               && __nothrow_callable<connect_t, _Sender, _Receiver>;

} // namespace STDEXEC
