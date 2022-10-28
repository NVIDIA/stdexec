/*
 * Copyright (c) 2022 NVIDIA Corporation
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

// Internal header, do not include directly

#ifdef STDEXEC_DISABLE_STD_DEPRECATIONS
#define STDEXEC_STD_DEPRECATED
#else
#define STDEXEC_STD_DEPRECATED [[deprecated("Please access this entity in the ::stdexec:: namespace. Define STDEXEC_DISABLE_STD_DEPRECATIONS to silence this warning.")]]
#endif

namespace std {
  //////////////////////////////////////////////////////////////////////////////
  // <functional>
  STDEXEC_STD_DEPRECATED
  inline constexpr stdexec::tag_invoke_t tag_invoke{};

  template <class _Tag, class... _Ts>
    using tag_invoke_result STDEXEC_STD_DEPRECATED = stdexec::tag_invoke_result<_Tag, _Ts...>;

  template <class _Tag, class... _Ts>
    using tag_invoke_result_t STDEXEC_STD_DEPRECATED = stdexec::tag_invoke_result_t<_Tag, _Ts...>;

  template <class _Tag, class... _Ts>
    concept tag_invocable /*STDEXEC_STD_DEPRECATED*/ = stdexec::tag_invocable<_Tag, _Ts...>;

  template <class _Tag, class... _Ts>
    concept nothrow_tag_invocable /*STDEXEC_STD_DEPRECATED*/ = stdexec::nothrow_tag_invocable<_Tag, _Ts...>;

  template <auto& _Tag>
    using tag_t STDEXEC_STD_DEPRECATED = stdexec::tag_t<_Tag>;

  //////////////////////////////////////////////////////////////////////////////
  // <stop_token>
  template <class _Token>
    concept stoppable_token /*STDEXEC_STD_DEPRECATED*/ = stdexec::stoppable_token<_Token>;

  template <class _Token, typename _Callback, typename _Initializer = _Callback>
    concept stoppable_token_for /*STDEXEC_STD_DEPRECATED*/ = stdexec::stoppable_token_for<_Token, _Callback, _Initializer>;

  template <class _Token>
    concept unstoppable_token /*STDEXEC_STD_DEPRECATED*/ = stdexec::unstoppable_token<_Token>;
 
  using never_stop_token STDEXEC_STD_DEPRECATED = stdexec::never_stop_token;
  using in_place_stop_token STDEXEC_STD_DEPRECATED = stdexec::in_place_stop_token;
  using in_place_stop_source STDEXEC_STD_DEPRECATED = stdexec::in_place_stop_source;

  template <class _Callback>
    using in_place_stop_callback STDEXEC_STD_DEPRECATED = stdexec::in_place_stop_callback<_Callback>;

  //////////////////////////////////////////////////////////////////////////////
  // <execution>
  namespace execution {
    // [exec.queries], general queries
    using get_scheduler_t STDEXEC_STD_DEPRECATED = stdexec::get_scheduler_t;
    using get_delegatee_scheduler_t STDEXEC_STD_DEPRECATED = stdexec::get_delegatee_scheduler_t;
    using get_allocator_t STDEXEC_STD_DEPRECATED = stdexec::get_allocator_t;
    using get_stop_token_t STDEXEC_STD_DEPRECATED = stdexec::get_stop_token_t;
    STDEXEC_STD_DEPRECATED
    inline constexpr stdexec::get_scheduler_t get_scheduler{};
    STDEXEC_STD_DEPRECATED
    inline constexpr stdexec::get_delegatee_scheduler_t get_delegatee_scheduler{};
    STDEXEC_STD_DEPRECATED
    inline constexpr stdexec::get_allocator_t get_allocator{};
    STDEXEC_STD_DEPRECATED
    inline constexpr stdexec::get_stop_token_t get_stop_token{};

    template <class _StopTokenProvider>
      using stop_token_of_t STDEXEC_STD_DEPRECATED = stdexec::stop_token_of_t<_StopTokenProvider>;

    // [exec.env], execution environments
    using no_env STDEXEC_STD_DEPRECATED = stdexec::no_env;
    using get_env_t STDEXEC_STD_DEPRECATED = stdexec::get_env_t;
    //using forwarding_env_query_t STDEXEC_STD_DEPRECATED = stdexec::forwarding_env_query_t; // BUGBUG
    STDEXEC_STD_DEPRECATED
    inline constexpr stdexec::get_env_t get_env{};
    //inline constexpr stdexec::forwarding_env_query_t forwarding_env_query{}; // BUGBUG

    template <class _EnvProvider>
      using env_of_t STDEXEC_STD_DEPRECATED = stdexec::env_of_t<_EnvProvider>;

    // [exec.sched], schedulers
    template <class _Scheduler>
      concept scheduler /*STDEXEC_STD_DEPRECATED*/ = stdexec::scheduler<_Scheduler>;

    // [exec.sched_queries], scheduler queries
    using forward_progress_guarantee STDEXEC_STD_DEPRECATED = stdexec::forward_progress_guarantee;
    using forwarding_scheduler_query_t STDEXEC_STD_DEPRECATED = stdexec::forwarding_scheduler_query_t;
    using get_forward_progress_guarantee_t STDEXEC_STD_DEPRECATED = stdexec::get_forward_progress_guarantee_t;
    STDEXEC_STD_DEPRECATED
    inline constexpr stdexec::forwarding_scheduler_query_t forwarding_scheduler_query{};
    STDEXEC_STD_DEPRECATED
    inline constexpr stdexec::get_forward_progress_guarantee_t get_forward_progress_guarantee{};

    // [exec.recv], receivers
    template <class _Receiver>
      concept receiver /*STDEXEC_STD_DEPRECATED*/ = stdexec::receiver<_Receiver>;

    template <class _Receiver, class _Completions>
      concept receiver_of /*STDEXEC_STD_DEPRECATED*/ = stdexec::receiver_of<_Receiver, _Completions>;

    using set_value_t STDEXEC_STD_DEPRECATED = stdexec::set_value_t;
    using set_error_t STDEXEC_STD_DEPRECATED = stdexec::set_error_t;
    using set_stopped_t STDEXEC_STD_DEPRECATED = stdexec::set_stopped_t;
    STDEXEC_STD_DEPRECATED
    inline constexpr stdexec::set_value_t set_value{};
    STDEXEC_STD_DEPRECATED
    inline constexpr stdexec::set_error_t set_error{};
    STDEXEC_STD_DEPRECATED
    inline constexpr stdexec::set_stopped_t set_stopped{};

    // [exec.recv_queries], receiver queries
    // using stdexec::forwarding_receiver_query_t; // BUGBUG
    // using stdexec::forwarding_receiver_query; // BUGBUG

    // [exec.op_state], operation states
    template <class _OpState>
      concept operation_state /*STDEXEC_STD_DEPRECATED*/ = stdexec::operation_state<_OpState>;

    using start_t STDEXEC_STD_DEPRECATED = stdexec::start_t;
    STDEXEC_STD_DEPRECATED
    inline constexpr stdexec::start_t start{};

    // [exec.snd], senders
    template <class _Sender, class _Env = stdexec::no_env>
      concept sender /*STDEXEC_STD_DEPRECATED*/ = stdexec::sender<_Sender, _Env>;

    template <class _Sender, class _Receiver>
      concept sender_to /*STDEXEC_STD_DEPRECATED*/ = stdexec::sender_to<_Sender, _Receiver>;

    template<class _Sender, class _SetSig, class _Env = stdexec::no_env>
      concept sender_of /*STDEXEC_STD_DEPRECATED*/ = stdexec::sender_of<_Sender, _SetSig, _Env>;

    // [exec.sndtraits], completion signatures
    using get_completion_signatures_t STDEXEC_STD_DEPRECATED = stdexec::get_completion_signatures_t;
    STDEXEC_STD_DEPRECATED
    inline constexpr stdexec::get_completion_signatures_t get_completion_signatures{};

    template<class _Sender, class _Env = stdexec::no_env>
      using completion_signatures_of_t STDEXEC_STD_DEPRECATED = stdexec::completion_signatures_of_t<_Sender, _Env>;

    template <class _Env>
      using dependent_completion_signatures STDEXEC_STD_DEPRECATED = stdexec::dependent_completion_signatures<_Env>;

    template <class _Sender,
              class _Env = stdexec::no_env,
              template <class...> class _Tuple = stdexec::__decayed_tuple,
              template <class...> class _Variant = stdexec::__variant>
      using value_types_of_t STDEXEC_STD_DEPRECATED = stdexec::value_types_of_t<_Sender, _Env, _Tuple, _Variant>;

    template <class _Sender,
              class _Env = stdexec::no_env,
              template <class...> class _Variant = stdexec::__variant>
      using error_types_of_t STDEXEC_STD_DEPRECATED = stdexec::error_types_of_t<_Sender, _Env, _Variant>;

    template <class _Sender, class _Env = stdexec::no_env>
      STDEXEC_STD_DEPRECATED
      inline constexpr bool sends_stopped = stdexec::sends_stopped<_Sender, _Env>;

    // [exec.connect], the connect sender algorithm
    using connect_t STDEXEC_STD_DEPRECATED = stdexec::connect_t;
    STDEXEC_STD_DEPRECATED
    inline constexpr stdexec::connect_t connect{};

    template <class _Sender, class _Receiver>
      using connect_result_t STDEXEC_STD_DEPRECATED = stdexec::connect_result_t<_Sender, _Receiver>;

    // [exec.snd_queries], sender queries
    using forwarding_sender_query_t STDEXEC_STD_DEPRECATED = stdexec::forwarding_sender_query_t;
    template <class _Tag>
      using get_completion_scheduler_t STDEXEC_STD_DEPRECATED = stdexec::get_completion_scheduler_t<_Tag>;
    STDEXEC_STD_DEPRECATED
    inline constexpr stdexec::forwarding_sender_query_t forwarding_sender_query{};

    template <class _Tag>
      STDEXEC_STD_DEPRECATED
      inline constexpr stdexec::get_completion_scheduler_t<_Tag> get_completion_scheduler{};

    // [exec.factories], sender factories
    using schedule_t STDEXEC_STD_DEPRECATED = stdexec::schedule_t;
    using transfer_just_t STDEXEC_STD_DEPRECATED = stdexec::transfer_just_t;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto just = stdexec::just;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto just_error = stdexec::just_error;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto just_stopped = stdexec::just_stopped;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto schedule = stdexec::schedule;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto transfer_just = stdexec::transfer_just;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto read = stdexec::read;

    template <class _Scheduler>
      using schedule_result_t STDEXEC_STD_DEPRECATED = stdexec::schedule_result_t<_Scheduler>;

    // [exec.adapt], sender adaptors
    template <class _Closure>
      using sender_adaptor_closure STDEXEC_STD_DEPRECATED = stdexec::sender_adaptor_closure<_Closure>;

    using on_t STDEXEC_STD_DEPRECATED = stdexec::on_t;
    using transfer_t STDEXEC_STD_DEPRECATED = stdexec::transfer_t;
    using schedule_from_t STDEXEC_STD_DEPRECATED = stdexec::schedule_from_t;
    using then_t STDEXEC_STD_DEPRECATED = stdexec::then_t;
    using upon_error_t STDEXEC_STD_DEPRECATED = stdexec::upon_error_t;
    using upon_stopped_t STDEXEC_STD_DEPRECATED = stdexec::upon_stopped_t;
    using let_value_t STDEXEC_STD_DEPRECATED = stdexec::let_value_t;
    using let_error_t STDEXEC_STD_DEPRECATED = stdexec::let_error_t;
    using let_stopped_t STDEXEC_STD_DEPRECATED = stdexec::let_stopped_t;
    using bulk_t STDEXEC_STD_DEPRECATED = stdexec::bulk_t;
    using split_t STDEXEC_STD_DEPRECATED = stdexec::split_t;
    using when_all_t STDEXEC_STD_DEPRECATED = stdexec::when_all_t;
    using when_all_with_variant_t STDEXEC_STD_DEPRECATED = stdexec::when_all_with_variant_t;
    using transfer_when_all_t STDEXEC_STD_DEPRECATED = stdexec::transfer_when_all_t;
    using transfer_when_all_with_variant_t STDEXEC_STD_DEPRECATED = stdexec::transfer_when_all_with_variant_t;
    using into_variant_t STDEXEC_STD_DEPRECATED = stdexec::into_variant_t;
    using stopped_as_optional_t STDEXEC_STD_DEPRECATED = stdexec::stopped_as_optional_t;
    using stopped_as_error_t STDEXEC_STD_DEPRECATED = stdexec::stopped_as_error_t;
    using ensure_started_t STDEXEC_STD_DEPRECATED = stdexec::ensure_started_t;

    STDEXEC_STD_DEPRECATED
    inline constexpr auto on = stdexec::on;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto transfer = stdexec::transfer;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto schedule_from = stdexec::schedule_from;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto then = stdexec::then;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto upon_error = stdexec::upon_error;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto upon_stopped = stdexec::upon_stopped;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto let_value = stdexec::let_value;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto let_error = stdexec::let_error;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto let_stopped = stdexec::let_stopped;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto bulk = stdexec::bulk;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto split = stdexec::split;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto when_all = stdexec::when_all;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto when_all_with_variant = stdexec::when_all_with_variant;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto transfer_when_all = stdexec::transfer_when_all;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto transfer_when_all_with_variant = stdexec::transfer_when_all_with_variant;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto into_variant = stdexec::into_variant;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto stopped_as_optional = stdexec::stopped_as_optional;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto stopped_as_error = stdexec::stopped_as_error;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto ensure_started = stdexec::ensure_started;

    // [exec.consumers], sender consumers
    using start_detached_t STDEXEC_STD_DEPRECATED = stdexec::start_detached_t;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto start_detached = stdexec::start_detached;

    // [exec.utils], sender and receiver utilities
    // [exec.utils.rcvr_adptr]
    template <class _Derived, class _Base = stdexec::__adaptors::__not_a_receiver>
      using receiver_adaptor STDEXEC_STD_DEPRECATED = stdexec::receiver_adaptor<_Derived, _Base>;

    // [exec.utils.cmplsigs]
    template <class... _Sigs>
      using completion_signatures STDEXEC_STD_DEPRECATED = stdexec::completion_signatures<_Sigs...>;

    // [exec.utils.mkcmplsigs]
    template<
      class _Sender,
      class _Env = stdexec::no_env,
      class _Sigs = stdexec::completion_signatures<>,
      template <class...> class _SetValue = stdexec::__compl_sigs::__default_set_value,
      template <class> class _SetError = stdexec::__compl_sigs::__default_set_error,
      class _SetStopped = stdexec::completion_signatures<stdexec::set_stopped_t()>>
    using make_completion_signatures STDEXEC_STD_DEPRECATED =
      stdexec::make_completion_signatures<_Sender, _Env, _Sigs, _SetValue, _SetError, _SetStopped>;

    // [exec.ctx], execution contexts
    using run_loop STDEXEC_STD_DEPRECATED = stdexec::run_loop;

    // [exec.execute], execute
    using execute_t STDEXEC_STD_DEPRECATED = stdexec::execute_t;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto execute = stdexec::execute;

    #if !_STD_NO_COROUTINES_
    // [exec.as_awaitable]
    using as_awaitable_t STDEXEC_STD_DEPRECATED = stdexec::as_awaitable_t;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto as_awaitable = stdexec::as_awaitable;

    // [exec.with_awaitable_senders]
    template <class _Promise>
      using with_awaitable_senders STDEXEC_STD_DEPRECATED = stdexec::with_awaitable_senders<_Promise>;
    #endif // !_STD_NO_COROUTINES_
  } // namespace execution

  namespace this_thread {
    using execute_may_block_caller_t STDEXEC_STD_DEPRECATED = stdexec::execute_may_block_caller_t;
    using sync_wait_t STDEXEC_STD_DEPRECATED = stdexec::sync_wait_t;
    using sync_wait_with_variant_t STDEXEC_STD_DEPRECATED = stdexec::sync_wait_with_variant_t;

    STDEXEC_STD_DEPRECATED
    inline constexpr auto execute_may_block_caller = stdexec::execute_may_block_caller;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto sync_wait = stdexec::sync_wait;
    STDEXEC_STD_DEPRECATED
    inline constexpr auto sync_wait_with_variant = stdexec::sync_wait_with_variant;
  } // namespace this_thread
} // namespace std
