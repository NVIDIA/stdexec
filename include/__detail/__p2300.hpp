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
  // <functional>
  using _P2300::tag_invoke;
  using _P2300::tag_invoke_result;
  using _P2300::tag_invoke_result_t;
  using _P2300::tag_invocable;
  using _P2300::nothrow_tag_invocable;
  using _P2300::tag_t;

  // <stop_token>
  using _P2300::stoppable_token;
  using _P2300::stoppable_token_for;
  using _P2300::unstoppable_token;
  using _P2300::never_stop_token;
  using _P2300::in_place_stop_token;
  using _P2300::in_place_stop_source;
  using _P2300::in_place_stop_callback;

  namespace execution {
    // [exec.queries], general queries
    using _P2300::execution::get_scheduler_t;
    using _P2300::execution::get_delegatee_scheduler_t;
    using _P2300::execution::get_allocator_t;
    using _P2300::execution::get_stop_token_t;
    using _P2300::execution::get_scheduler;
    using _P2300::execution::get_delegatee_scheduler;
    using _P2300::execution::get_allocator;
    using _P2300::execution::get_stop_token;

    using _P2300::execution::stop_token_of_t;

    // [exec.env], execution environments
    using _P2300::execution::no_env;
    using _P2300::execution::get_env_t;
    //using _P2300::execution::forwarding_env_query_t; // BUGBUG
    using _P2300::execution::get_env;
    //using _P2300::execution::forwarding_env_query; // BUGBUG

    using _P2300::execution::env_of_t;

    // [exec.sched], schedulers
    using _P2300::execution::scheduler;

    // [exec.sched_queries], scheduler queries
    using _P2300::execution::forward_progress_guarantee;
    using _P2300::execution::forwarding_scheduler_query_t;
    using _P2300::execution::get_forward_progress_guarantee_t;
    using _P2300::execution::forwarding_scheduler_query;
    using _P2300::execution::get_forward_progress_guarantee;

    // [exec.recv], receivers
    using _P2300::execution::receiver;

    using _P2300::execution::receiver_of;

    using _P2300::execution::set_value_t;
    using _P2300::execution::set_error_t;
    using _P2300::execution::set_stopped_t;
    using _P2300::execution::set_value;
    using _P2300::execution::set_error;
    using _P2300::execution::set_stopped;

    // [exec.recv_queries], receiver queries
    // using _P2300::execution::forwarding_receiver_query_t; // BUGBUG
    // using _P2300::execution::forwarding_receiver_query; // BUGBUG

    // [exec.op_state], operation states
    using _P2300::execution::operation_state;

    using _P2300::execution::start_t;
    using _P2300::execution::start;

    // [exec.snd], senders
    using _P2300::execution::sender;
    using _P2300::execution::sender_to;
    using _P2300::execution::sender_of;

    // [exec.sndtraits], completion signatures
    using _P2300::execution::get_completion_signatures_t;
    using _P2300::execution::get_completion_signatures;
    using _P2300::execution::completion_signatures_of_t;

    using _P2300::execution::dependent_completion_signatures;

    using _P2300::execution::value_types_of_t;
    using _P2300::execution::error_types_of_t;
    using _P2300::execution::sends_stopped;

    // [exec.connect], the connect sender algorithm
    using _P2300::execution::connect_t;
    using _P2300::execution::connect;

    using _P2300::execution::connect_result_t;

    // [exec.snd_queries], sender queries
    using _P2300::execution::forwarding_sender_query_t;
    using _P2300::execution::get_completion_scheduler_t;
    using _P2300::execution::forwarding_sender_query;

    using _P2300::execution::get_completion_scheduler;

    // [exec.factories], sender factories
    using _P2300::execution::just;
    using _P2300::execution::just_error;
    using _P2300::execution::just_stopped;
    using _P2300::execution::schedule_t;
    using _P2300::execution::transfer_just_t;
    using _P2300::execution::schedule;
    using _P2300::execution::transfer_just;
    using _P2300::execution::read;

    using _P2300::execution::schedule_result_t;

    // [exec.adapt], sender adaptors
    using _P2300::execution::sender_adaptor_closure;

    using _P2300::execution::on_t;
    using _P2300::execution::transfer_t;
    using _P2300::execution::schedule_from_t;
    using _P2300::execution::then_t;
    using _P2300::execution::upon_error_t;
    using _P2300::execution::upon_stopped_t;
    using _P2300::execution::let_value_t;
    using _P2300::execution::let_error_t;
    using _P2300::execution::let_stopped_t;
    using _P2300::execution::bulk_t;
    using _P2300::execution::split_t;
    using _P2300::execution::when_all_t;
    using _P2300::execution::when_all_with_variant_t;
    using _P2300::execution::transfer_when_all_t;
    using _P2300::execution::transfer_when_all_with_variant_t;
    using _P2300::execution::into_variant_t;
    using _P2300::execution::stopped_as_optional_t;
    using _P2300::execution::stopped_as_error_t;
    using _P2300::execution::ensure_started_t;

    using _P2300::execution::on;
    using _P2300::execution::transfer;
    using _P2300::execution::schedule_from;

    using _P2300::execution::then;
    using _P2300::execution::upon_error;
    using _P2300::execution::upon_stopped;

    using _P2300::execution::let_value;
    using _P2300::execution::let_error;
    using _P2300::execution::let_stopped;

    using _P2300::execution::bulk;

    using _P2300::execution::split;
    using _P2300::execution::when_all;
    using _P2300::execution::when_all_with_variant;
    using _P2300::execution::transfer_when_all;
    using _P2300::execution::transfer_when_all_with_variant;

    using _P2300::execution::into_variant;

    using _P2300::execution::stopped_as_optional;
    using _P2300::execution::stopped_as_error;

    using _P2300::execution::ensure_started;

    // [exec.consumers], sender consumers
    using _P2300::execution::start_detached_t;
    using _P2300::execution::start_detached;

    // [exec.utils], sender and receiver utilities
    // [exec.utils.rcvr_adptr]
    using _P2300::execution::receiver_adaptor;

    // [exec.utils.cmplsigs]
    using _P2300::execution::completion_signatures;

    // [exec.utils.mkcmplsigs]
    using _P2300::execution::make_completion_signatures;

    // [exec.ctx], execution contexts
    using _P2300::execution::run_loop;

    // [exec.execute], execute
    using _P2300::execution::execute_t;
    using _P2300::execution::execute;

    #if !_STD_NO_COROUTINES_
    // [exec.as_awaitable]
    using _P2300::execution::as_awaitable_t;
    using _P2300::execution::as_awaitable;

    // [exec.with_awaitable_senders]
    struct with_awaitable_senders;
    #endif // !_STD_NO_COROUTINES_
  } // namespace execution

  namespace this_thread {
    using _P2300::this_thread::execute_may_block_caller_t;
    using _P2300::this_thread::execute_may_block_caller;
    using _P2300::this_thread::sync_wait_t;
    using _P2300::this_thread::sync_wait_with_variant_t;
    using _P2300::this_thread::sync_wait;
    using _P2300::this_thread::sync_wait_with_variant;
  } // namespace this_thread
} // namespace std
