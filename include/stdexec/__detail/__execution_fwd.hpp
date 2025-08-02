/*
 * Copyright (c) 2023 NVIDIA Corporation
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

#include "__config.hpp" // IWYU pragma: export
#include "__meta.hpp"
#include "__concepts.hpp"
#include "__type_traits.hpp"

// IWYU pragma: always_keep

namespace stdexec {
  struct __none_such;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct default_domain;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __rcvrs {
    struct set_value_t;
    struct set_error_t;
    struct set_stopped_t;
  } // namespace __rcvrs

  using __rcvrs::set_value_t;
  using __rcvrs::set_error_t;
  using __rcvrs::set_stopped_t;
  extern const set_value_t set_value;
  extern const set_error_t set_error;
  extern const set_stopped_t set_stopped;

  template <class _Tag>
  concept __completion_tag = __one_of<_Tag, set_value_t, set_error_t, set_stopped_t>;

  struct receiver_t;

  template <class _Sender>
  extern const bool enable_receiver;

  namespace __env {
    template <class _Query, class _Value>
    struct prop;

    template <class _Query, auto _Value>
    struct cprop;

    template <class... _Envs>
    struct env;
  } // namespace __env

  using __env::prop;
  using __env::cprop;
  using __env::env;
  using empty_env [[deprecated("stdexec::empty_env is now spelled stdexec::env<>")]] = env<>;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __get_env {
    struct get_env_t;
  } // namespace __get_env

  using __get_env::get_env_t;
  extern const get_env_t get_env;

  template <class _EnvProvider>
  using env_of_t = __call_result_t<get_env_t, _EnvProvider>;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  enum class forward_progress_guarantee {
    concurrent,
    parallel,
    weakly_parallel
  };

  namespace __queries {
    struct forwarding_query_t;
    struct execute_may_block_caller_t;
    struct get_forward_progress_guarantee_t;
    struct get_scheduler_t;
    struct get_delegation_scheduler_t;
    struct get_allocator_t;
    struct get_stop_token_t;
    template <__completion_tag _CPO>
    struct get_completion_scheduler_t;
    struct get_domain_t;
  } // namespace __queries

  using __queries::forwarding_query_t;
  using __queries::execute_may_block_caller_t;
  using __queries::get_forward_progress_guarantee_t;
  using __queries::get_allocator_t;
  using __queries::get_scheduler_t;
  using __queries::get_delegation_scheduler_t;
  using __queries::get_stop_token_t;
  using __queries::get_completion_scheduler_t;
  using __queries::get_domain_t;

  extern const forwarding_query_t forwarding_query;
  extern const execute_may_block_caller_t execute_may_block_caller;
  extern const get_forward_progress_guarantee_t get_forward_progress_guarantee;
  extern const get_scheduler_t get_scheduler;
  extern const get_delegation_scheduler_t get_delegation_scheduler;
  extern const get_allocator_t get_allocator;
  extern const get_stop_token_t get_stop_token;
  template <__completion_tag _CPO>
  extern const get_completion_scheduler_t<_CPO> get_completion_scheduler;
  extern const get_domain_t get_domain;

  struct never_stop_token;
  class inplace_stop_source;
  class inplace_stop_token;
  template <class _Fn>
  class inplace_stop_callback;

  template <class _Tp>
  using stop_token_of_t = __decay_t<__call_result_t<get_stop_token_t, _Tp>>;

  template <class _Sender, class _CPO>
  concept __has_completion_scheduler =
    __callable<get_completion_scheduler_t<_CPO>, env_of_t<const _Sender&>>;

  template <class _Sender, class _CPO>
  using __completion_scheduler_for =
    __call_result_t<get_completion_scheduler_t<_CPO>, env_of_t<const _Sender&>>;

  template <class _Env>
  using __domain_of_t = __decay_t<__call_result_t<get_domain_t, _Env>>;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __sigs {
    template <class _Sig>
    inline constexpr bool __is_compl_sig = false;

    struct get_completion_signatures_t;
  } // namespace __sigs

  template <class _Sig>
  concept __completion_signature = __sigs::__is_compl_sig<_Sig>;

  template <class... _Sigs>
  struct completion_signatures;

  using __sigs::get_completion_signatures_t;
  extern const get_completion_signatures_t get_completion_signatures;

  template <class _Sender, class... _Env>
  using __completion_signatures_of_t =
    __call_result_t<get_completion_signatures_t, _Sender, _Env...>;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __connect {
    struct connect_t;
  } // namespace __connect

  using __connect::connect_t;
  extern const connect_t connect;

  template <class _Sender, class _Receiver>
  concept __nothrow_connectable = __nothrow_callable<connect_t, _Sender, _Receiver>;

  struct sender_t;

  template <class _Sender>
  extern const bool enable_sender;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct operation_state_t;

  namespace __start {
    struct start_t;
  } // namespace __start

  using __start::start_t;
  extern const start_t start;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __sched {
    struct schedule_t;
  } // namespace __sched

  using __sched::schedule_t;
  extern const schedule_t schedule;

  struct scheduler_t;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __as_awaitable {
    struct as_awaitable_t;
  } // namespace __as_awaitable

  using __as_awaitable::as_awaitable_t;
  extern const as_awaitable_t as_awaitable;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct transform_sender_t;
  extern const transform_sender_t transform_sender;

  template <class _Domain, class _Sender, class... _Env>
  using transform_sender_result_t = __call_result_t<transform_sender_t, _Domain, _Sender, _Env...>;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __starts_on_ns {
    struct starts_on_t;
  } // namespace __starts_on_ns

  using __starts_on_ns::starts_on_t;
  extern const starts_on_t starts_on;

  using start_on_t [[deprecated("start_on_t has been renamed starts_on_t")]] = starts_on_t;
  [[deprecated("start_on has been renamed starts_on")]]
  extern const starts_on_t start_on;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __schfr {
    struct schedule_from_t;
  } // namespace __schfr

  using __schfr::schedule_from_t;
  extern const schedule_from_t schedule_from;

  namespace __continues_on {
    struct continues_on_t;
  } // namespace __continues_on

  using __continues_on::continues_on_t;
  extern const continues_on_t continues_on;

  using transfer_t [[deprecated("transfer_t has been renamed continues_on_t")]] = continues_on_t;
  [[deprecated("transfer has been renamed continues_on")]]
  extern const continues_on_t transfer;

  using continue_t [[deprecated("continue_on_t has been renamed continues_on_t")]] = continues_on_t;
  [[deprecated("continue_on has been renamed continues_on")]]
  extern const continues_on_t continue_on;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __transfer_just {
    struct transfer_just_t;
  } // namespace __transfer_just

  using __transfer_just::transfer_just_t;
  extern const transfer_just_t transfer_just;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __bulk {
    struct bulk_t;
    struct bulk_chunked_t;
    struct bulk_unchunked_t;
  } // namespace __bulk

  using __bulk::bulk_t;
  using __bulk::bulk_chunked_t;
  using __bulk::bulk_unchunked_t;
  extern const bulk_t bulk;
  extern const bulk_chunked_t bulk_chunked;
  extern const bulk_unchunked_t bulk_unchunked;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __split {
    struct split_t;
    struct __split_t;
  } // namespace __split

  using __split::split_t;
  extern const split_t split;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __ensure_started {
    struct ensure_started_t;
    struct __ensure_started_t;
  } // namespace __ensure_started

  using __ensure_started::ensure_started_t;
  extern const ensure_started_t ensure_started;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __on {
    struct on_t;
  } // namespace __on

  using __on::on_t;
  extern const on_t on;

  namespace __detail {
    struct __sexpr_apply_t;
  } // namespace __detail

  using __detail::__sexpr_apply_t;
  extern const __sexpr_apply_t __sexpr_apply;
} // namespace stdexec

template <class...>
[[deprecated]]
void __print() {
}

template <class...>
struct __undef;
