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

namespace std {
  //////////////////////////////////////////////////////////////////////////////
  // <functional>
  inline constexpr stdexec::tag_invoke_t tag_invoke{};

  template <class _Tag, class... _Ts>
    using tag_invoke_result = stdexec::tag_invoke_result<_Tag, _Ts...>;

  template <class _Tag, class... _Ts>
    using tag_invoke_result_t = stdexec::tag_invoke_result_t<_Tag, _Ts...>;

  template <class _Tag, class... _Ts>
    concept tag_invocable = stdexec::tag_invocable<_Tag, _Ts...>;

  template <class _Tag, class... _Ts>
    concept nothrow_tag_invocable = stdexec::nothrow_tag_invocable<_Tag, _Ts...>;

  template <auto& _Tag>
    using tag_t = stdexec::tag_t<_Tag>;

  //////////////////////////////////////////////////////////////////////////////
  // <stop_token>
  template <class _Token>
    concept stoppable_token = stdexec::stoppable_token<_Token>;

  template <class _Token, typename _Callback, typename _Initializer = _Callback>
    concept stoppable_token_for = stdexec::stoppable_token_for<_Token, _Callback, _Initializer>;

  template <class _Token>
    concept unstoppable_token = stdexec::unstoppable_token<_Token>;
 
  using never_stop_token = stdexec::never_stop_token;
  using in_place_stop_token = stdexec::in_place_stop_token;
  using in_place_stop_source = stdexec::in_place_stop_source;

  template <class _Callback>
    using in_place_stop_callback = stdexec::in_place_stop_callback<_Callback>;

  //////////////////////////////////////////////////////////////////////////////
  // <execution>
  namespace execution {
    // [exec.queries], general queries
    using get_scheduler_t = stdexec::get_scheduler_t;
    using get_delegatee_scheduler_t = stdexec::get_delegatee_scheduler_t;
    using get_allocator_t = stdexec::get_allocator_t;
    using get_stop_token_t = stdexec::get_stop_token_t;
    inline constexpr stdexec::get_scheduler_t get_scheduler{};
    inline constexpr stdexec::get_delegatee_scheduler_t get_delegatee_scheduler{};
    inline constexpr stdexec::get_allocator_t get_allocator{};
    inline constexpr stdexec::get_stop_token_t get_stop_token{};

    template <class _StopTokenProvider>
      using stop_token_of_t = stdexec::stop_token_of_t<_StopTokenProvider>;

    // [exec.env], execution environments
    using no_env = stdexec::no_env;
    using get_env_t = stdexec::get_env_t;
    //using forwarding_env_query_t = stdexec::forwarding_env_query_t; // BUGBUG
    inline constexpr stdexec::get_env_t get_env{};
    //inline constexpr stdexec::forwarding_env_query_t forwarding_env_query{}; // BUGBUG

    template <class _EnvProvider>
      using env_of_t = stdexec::env_of_t<_EnvProvider>;

    // [exec.sched], schedulers
    template <class _Scheduler>
      concept scheduler = stdexec::scheduler<_Scheduler>;

    // [exec.sched_queries], scheduler queries
    using forward_progress_guarantee = stdexec::forward_progress_guarantee;
    using forwarding_scheduler_query_t = stdexec::forwarding_scheduler_query_t;
    using get_forward_progress_guarantee_t = stdexec::get_forward_progress_guarantee_t;
    inline constexpr stdexec::forwarding_scheduler_query_t forwarding_scheduler_query{};
    inline constexpr stdexec::get_forward_progress_guarantee_t get_forward_progress_guarantee{};

    // [exec.recv], receivers
    template <class _Receiver>
      concept receiver = stdexec::receiver<_Receiver>;

    template <class _Receiver, class _Completions>
      concept receiver_of = stdexec::receiver_of<_Receiver, _Completions>;

    using set_value_t = stdexec::set_value_t;
    using set_error_t = stdexec::set_error_t;
    using set_stopped_t = stdexec::set_stopped_t;
    inline constexpr stdexec::set_value_t set_value{};
    inline constexpr stdexec::set_error_t set_error{};
    inline constexpr stdexec::set_stopped_t set_stopped{};

    // [exec.recv_queries], receiver queries
    // using stdexec::forwarding_receiver_query_t; // BUGBUG
    // using stdexec::forwarding_receiver_query; // BUGBUG

    // [exec.op_state], operation states
    template <class _OpState>
      concept operation_state = stdexec::operation_state<_OpState>;

    using start_t = stdexec::start_t;
    inline constexpr stdexec::start_t start{};

    // [exec.snd], senders
    template <class _Sender, class _Env = no_env>
      concept sender = stdexec::sender<_Sender, _Env>;

    template <class _Sender, class _Receiver>
      concept sender_to = stdexec::sender_to<_Sender, _Receiver>;

    template<class _Sender, class _SetSig, class _Env = no_env>
      concept sender_of = stdexec::sender_of<_Sender, _SetSig, _Env>;

    // [exec.sndtraits], completion signatures
    using get_completion_signatures_t = stdexec::get_completion_signatures_t;
    inline constexpr stdexec::get_completion_signatures_t get_completion_signatures{};

    template<class _Sender, class _Env = no_env>
      using completion_signatures_of_t = stdexec::completion_signatures_of_t<_Sender, _Env>;

    template <class _Env>
      using dependent_completion_signatures = stdexec::dependent_completion_signatures<_Env>;

    template <class _Sender,
              class _Env = no_env,
              template <class...> class _Tuple = stdexec::__decayed_tuple,
              template <class...> class _Variant = stdexec::__variant>
      using value_types_of_t = stdexec::value_types_of_t<_Sender, _Env, _Tuple, _Variant>;

    template <class _Sender,
              class _Env = no_env,
              template <class...> class _Variant = stdexec::__variant>
      using error_types_of_t = stdexec::error_types_of_t<_Sender, _Env, _Variant>;

    template <class _Sender, class _Env = no_env>
      inline constexpr bool sends_stopped = stdexec::sends_stopped<_Sender, _Env>;

    // [exec.connect], the connect sender algorithm
    using connect_t = stdexec::connect_t;
    inline constexpr stdexec::connect_t connect{};

    template <class _Sender, class _Receiver>
      using connect_result_t = stdexec::connect_result_t<_Sender, _Receiver>;

    // [exec.snd_queries], sender queries
    using forwarding_sender_query_t = stdexec::forwarding_sender_query_t;
    template <class _Tag>
      using get_completion_scheduler_t = stdexec::get_completion_scheduler_t<_Tag>;
    inline constexpr stdexec::forwarding_sender_query_t forwarding_sender_query{};

    template <class _Tag>
      inline constexpr stdexec::get_completion_scheduler_t<_Tag> get_completion_scheduler{};

    // [exec.factories], sender factories
    using schedule_t = stdexec::schedule_t;
    using transfer_just_t = stdexec::transfer_just_t;
    inline constexpr auto just = stdexec::just;
    inline constexpr auto just_error = stdexec::just_error;
    inline constexpr auto just_stopped = stdexec::just_stopped;
    inline constexpr auto schedule = stdexec::schedule;
    inline constexpr auto transfer_just = stdexec::transfer_just;
    inline constexpr auto read = stdexec::read;

    template <class _Scheduler>
      using schedule_result_t = stdexec::schedule_result_t<_Scheduler>;

    // [exec.adapt], sender adaptors
    template <class _Closure>
      using sender_adaptor_closure = stdexec::sender_adaptor_closure<_Closure>;

    using on_t = stdexec::on_t;
    using transfer_t = stdexec::transfer_t;
    using schedule_from_t = stdexec::schedule_from_t;
    using then_t = stdexec::then_t;
    using upon_error_t = stdexec::upon_error_t;
    using upon_stopped_t = stdexec::upon_stopped_t;
    using let_value_t = stdexec::let_value_t;
    using let_error_t = stdexec::let_error_t;
    using let_stopped_t = stdexec::let_stopped_t;
    using bulk_t = stdexec::bulk_t;
    using split_t = stdexec::split_t;
    using when_all_t = stdexec::when_all_t;
    using when_all_with_variant_t = stdexec::when_all_with_variant_t;
    using transfer_when_all_t = stdexec::transfer_when_all_t;
    using transfer_when_all_with_variant_t = stdexec::transfer_when_all_with_variant_t;
    using into_variant_t = stdexec::into_variant_t;
    using stopped_as_optional_t = stdexec::stopped_as_optional_t;
    using stopped_as_error_t = stdexec::stopped_as_error_t;
    using ensure_started_t = stdexec::ensure_started_t;

    inline constexpr auto on = stdexec::on;
    inline constexpr auto transfer = stdexec::transfer;
    inline constexpr auto schedule_from = stdexec::schedule_from;
    inline constexpr auto then = stdexec::then;
    inline constexpr auto upon_error = stdexec::upon_error;
    inline constexpr auto upon_stopped = stdexec::upon_stopped;
    inline constexpr auto let_value = stdexec::let_value;
    inline constexpr auto let_error = stdexec::let_error;
    inline constexpr auto let_stopped = stdexec::let_stopped;
    inline constexpr auto bulk = stdexec::bulk;
    inline constexpr auto split = stdexec::split;
    inline constexpr auto when_all = stdexec::when_all;
    inline constexpr auto when_all_with_variant = stdexec::when_all_with_variant;
    inline constexpr auto transfer_when_all = stdexec::transfer_when_all;
    inline constexpr auto transfer_when_all_with_variant = stdexec::transfer_when_all_with_variant;
    inline constexpr auto into_variant = stdexec::into_variant;
    inline constexpr auto stopped_as_optional = stdexec::stopped_as_optional;
    inline constexpr auto stopped_as_error = stdexec::stopped_as_error;
    inline constexpr auto ensure_started = stdexec::ensure_started;

    // [exec.consumers], sender consumers
    using start_detached_t = stdexec::start_detached_t;
    inline constexpr auto start_detached = stdexec::start_detached;

    // [exec.utils], sender and receiver utilities
    // [exec.utils.rcvr_adptr]
    template <class _Derived, class _Base = stdexec::__adaptors::__not_a_receiver>
      using receiver_adaptor = stdexec::receiver_adaptor<_Derived, _Base>;

    // [exec.utils.cmplsigs]
    template <class... _Sigs>
      using completion_signatures = stdexec::completion_signatures<_Sigs...>;

    // [exec.utils.mkcmplsigs]
    template<
      class _Sender,
      class _Env = no_env,
      class _Sigs = completion_signatures<>,
      template <class...> class _SetValue = stdexec::__compl_sigs::__default_set_value,
      template <class> class _SetError = stdexec::__compl_sigs::__default_set_error,
      class _SetStopped = completion_signatures<set_stopped_t()>>
    using make_completion_signatures =
      stdexec::make_completion_signatures<_Sender, _Env, _Sigs, _SetValue, _SetError, _SetStopped>;

    // [exec.ctx], execution contexts
    using run_loop = stdexec::run_loop;

    // [exec.execute], execute
    using execute_t = stdexec::execute_t;
    inline constexpr auto execute = stdexec::execute;

    #if !_STD_NO_COROUTINES_
    // [exec.as_awaitable]
    using as_awaitable_t = stdexec::as_awaitable_t;
    inline constexpr auto as_awaitable = stdexec::as_awaitable;

    // [exec.with_awaitable_senders]
    template <class _Promise>
      using with_awaitable_senders = stdexec::with_awaitable_senders<_Promise>;
    #endif // !_STD_NO_COROUTINES_
  } // namespace execution

  namespace this_thread {
    using execute_may_block_caller_t = stdexec::execute_may_block_caller_t;
    using sync_wait_t = stdexec::sync_wait_t;
    using sync_wait_with_variant_t = stdexec::sync_wait_with_variant_t;

    inline constexpr auto execute_may_block_caller = stdexec::execute_may_block_caller;
    inline constexpr auto sync_wait = stdexec::sync_wait;
    inline constexpr auto sync_wait_with_variant = stdexec::sync_wait_with_variant;
  } // namespace this_thread
} // namespace std
