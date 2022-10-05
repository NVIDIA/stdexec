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
  using stdexec::tag_invoke;
  using stdexec::tag_invoke_result;
  using stdexec::tag_invoke_result_t;
  using stdexec::tag_invocable;
  using stdexec::nothrow_tag_invocable;
  using stdexec::tag_t;

  // <stop_token>
  using stdexec::stoppable_token;
  using stdexec::stoppable_token_for;
  using stdexec::unstoppable_token;
  using stdexec::never_stop_token;
  using stdexec::in_place_stop_token;
  using stdexec::in_place_stop_source;
  using stdexec::in_place_stop_callback;

  namespace execution {
    // [exec.queries], general queries
    using stdexec::get_scheduler_t;
    using stdexec::get_delegatee_scheduler_t;
    using stdexec::get_allocator_t;
    using stdexec::get_stop_token_t;
    using stdexec::get_scheduler;
    using stdexec::get_delegatee_scheduler;
    using stdexec::get_allocator;
    using stdexec::get_stop_token;

    using stdexec::stop_token_of_t;

    // [exec.env], execution environments
    using stdexec::no_env;
    using stdexec::get_env_t;
    //using stdexec::forwarding_env_query_t; // BUGBUG
    using stdexec::get_env;
    //using stdexec::forwarding_env_query; // BUGBUG

    using stdexec::env_of_t;

    // [exec.sched], schedulers
    using stdexec::scheduler;

    // [exec.sched_queries], scheduler queries
    using stdexec::forward_progress_guarantee;
    using stdexec::forwarding_scheduler_query_t;
    using stdexec::get_forward_progress_guarantee_t;
    using stdexec::forwarding_scheduler_query;
    using stdexec::get_forward_progress_guarantee;

    // [exec.recv], receivers
    using stdexec::receiver;

    using stdexec::receiver_of;

    using stdexec::set_value_t;
    using stdexec::set_error_t;
    using stdexec::set_stopped_t;
    using stdexec::set_value;
    using stdexec::set_error;
    using stdexec::set_stopped;

    // [exec.recv_queries], receiver queries
    // using stdexec::forwarding_receiver_query_t; // BUGBUG
    // using stdexec::forwarding_receiver_query; // BUGBUG

    // [exec.op_state], operation states
    using stdexec::operation_state;

    using stdexec::start_t;
    using stdexec::start;

    // [exec.snd], senders
    using stdexec::sender;
    using stdexec::sender_to;
    using stdexec::sender_of;

    // [exec.sndtraits], completion signatures
    using stdexec::get_completion_signatures_t;
    using stdexec::get_completion_signatures;
    using stdexec::completion_signatures_of_t;

    using stdexec::dependent_completion_signatures;

    using stdexec::value_types_of_t;
    using stdexec::error_types_of_t;
    using stdexec::sends_stopped;

    // [exec.connect], the connect sender algorithm
    using stdexec::connect_t;
    using stdexec::connect;

    using stdexec::connect_result_t;

    // [exec.snd_queries], sender queries
    using stdexec::forwarding_sender_query_t;
    using stdexec::get_completion_scheduler_t;
    using stdexec::forwarding_sender_query;

    using stdexec::get_completion_scheduler;

    // [exec.factories], sender factories
    using stdexec::just;
    using stdexec::just_error;
    using stdexec::just_stopped;
    using stdexec::schedule_t;
    using stdexec::transfer_just_t;
    using stdexec::schedule;
    using stdexec::transfer_just;
    using stdexec::read;

    using stdexec::schedule_result_t;

    // [exec.adapt], sender adaptors
    using stdexec::sender_adaptor_closure;

    using stdexec::on_t;
    using stdexec::transfer_t;
    using stdexec::schedule_from_t;
    using stdexec::then_t;
    using stdexec::upon_error_t;
    using stdexec::upon_stopped_t;
    using stdexec::let_value_t;
    using stdexec::let_error_t;
    using stdexec::let_stopped_t;
    using stdexec::bulk_t;
    using stdexec::split_t;
    using stdexec::when_all_t;
    using stdexec::when_all_with_variant_t;
    using stdexec::transfer_when_all_t;
    using stdexec::transfer_when_all_with_variant_t;
    using stdexec::into_variant_t;
    using stdexec::stopped_as_optional_t;
    using stdexec::stopped_as_error_t;
    using stdexec::ensure_started_t;

    using stdexec::on;
    using stdexec::transfer;
    using stdexec::schedule_from;

    using stdexec::then;
    using stdexec::upon_error;
    using stdexec::upon_stopped;

    using stdexec::let_value;
    using stdexec::let_error;
    using stdexec::let_stopped;

    using stdexec::bulk;

    using stdexec::split;
    using stdexec::when_all;
    using stdexec::when_all_with_variant;
    using stdexec::transfer_when_all;
    using stdexec::transfer_when_all_with_variant;

    using stdexec::into_variant;

    using stdexec::stopped_as_optional;
    using stdexec::stopped_as_error;

    using stdexec::ensure_started;

    // [exec.consumers], sender consumers
    using stdexec::start_detached_t;
    using stdexec::start_detached;

    // [exec.utils], sender and receiver utilities
    // [exec.utils.rcvr_adptr]
    using stdexec::receiver_adaptor;

    // [exec.utils.cmplsigs]
    using stdexec::completion_signatures;

    // [exec.utils.mkcmplsigs]
    using stdexec::make_completion_signatures;

    // [exec.ctx], execution contexts
    using stdexec::run_loop;

    // [exec.execute], execute
    using stdexec::execute_t;
    using stdexec::execute;

    #if !_STD_NO_COROUTINES_
    // [exec.as_awaitable]
    using stdexec::as_awaitable_t;
    using stdexec::as_awaitable;

    // [exec.with_awaitable_senders]
    struct with_awaitable_senders;
    #endif // !_STD_NO_COROUTINES_
  } // namespace execution

  namespace this_thread {
    using stdexec::execute_may_block_caller_t;
    using stdexec::execute_may_block_caller;
    using stdexec::sync_wait_t;
    using stdexec::sync_wait_with_variant_t;
    using stdexec::sync_wait;
    using stdexec::sync_wait_with_variant;
  } // namespace this_thread
} // namespace std
