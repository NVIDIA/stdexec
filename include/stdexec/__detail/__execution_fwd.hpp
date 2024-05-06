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

#include "__config.hpp"
#include "__meta.hpp"
#include "__concepts.hpp"
#include "__type_traits.hpp"

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

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __env {
    struct get_env_t;

    struct empty_env {
      using __t = empty_env;
      using __id = empty_env;
    };
  } // namespace __env

  using __env::empty_env;
  using __env::get_env_t;
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
    struct __has_algorithm_customizations_t;
    struct get_scheduler_t;
    struct get_delegatee_scheduler_t;
    struct get_allocator_t;
    struct get_stop_token_t;
    template <__completion_tag _CPO>
    struct get_completion_scheduler_t;
  } // namespace __queries

  using __queries::forwarding_query_t;
  using __queries::execute_may_block_caller_t;
  using __queries::__has_algorithm_customizations_t;
  using __queries::get_forward_progress_guarantee_t;
  using __queries::get_allocator_t;
  using __queries::get_scheduler_t;
  using __queries::get_delegatee_scheduler_t;
  using __queries::get_stop_token_t;
  using __queries::get_completion_scheduler_t;

  extern const forwarding_query_t forwarding_query;
  extern const execute_may_block_caller_t execute_may_block_caller;
  extern const __has_algorithm_customizations_t __has_algorithm_customizations;
  extern const get_forward_progress_guarantee_t get_forward_progress_guarantee;
  extern const get_scheduler_t get_scheduler;
  extern const get_delegatee_scheduler_t get_delegatee_scheduler;
  extern const get_allocator_t get_allocator;
  extern const get_stop_token_t get_stop_token;
  template <__completion_tag _CPO>
  extern const get_completion_scheduler_t<_CPO> get_completion_scheduler;

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

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __compl_sigs {
    template <class _Sig>
    inline constexpr bool __is_compl_sig = false;

    struct get_completion_signatures_t;
  } // namespace __compl_sigs

  template <class _Sig>
  concept __completion_signature = __compl_sigs::__is_compl_sig<_Sig>;

  template <__completion_signature... _Sigs>
  struct completion_signatures;

  using __compl_sigs::get_completion_signatures_t;
  extern const get_completion_signatures_t get_completion_signatures;

  template <class _Sender, class... _Env>
  using __completion_signatures_of_t = //
    __call_result_t<get_completion_signatures_t, _Sender, _Env...>;

  namespace __detail {
    template <class _Tag>
    struct __make_sexpr_t;
  } // namespace __detail

  template <class _Tag>
  extern const __detail::__make_sexpr_t<_Tag> __make_sexpr;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __connect {
    struct connect_t;
  } // namespace __connect

  using __connect::connect_t;
  extern const connect_t connect;

  template <class _Sender, class _Receiver>
  using connect_result_t = __call_result_t<connect_t, _Sender, _Receiver>;

  template <class _Sender, class _Receiver>
  concept __nothrow_connectable = __nothrow_callable<connect_t, _Sender, _Receiver>;

  struct sender_t;

  template <class _Sender>
  extern const bool enable_sender;

  //////////////////////////////////////////////////////////////////////////////////////////////////
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

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __as_awaitable {
    struct as_awaitable_t;
  } // namespace __as_awaitable

  using __as_awaitable::as_awaitable_t;
  extern const as_awaitable_t as_awaitable;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __start_on {
    struct start_on_t;
  } // namespace __start_on

  using __start_on::start_on_t;
  extern const start_on_t start_on;

  using on_t = start_on_t;
  extern const on_t on;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __continue_on {
    struct continue_on_t;
  } // namespace __continue_on

  using __continue_on::continue_on_t;
  extern const continue_on_t continue_on;

  using transfer_t = continue_on_t;
  extern const transfer_t transfer;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __transfer_just {
    struct transfer_just_t;
  } // namespace __transfer_just

  using __transfer_just::transfer_just_t;
  extern const transfer_just_t transfer_just;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __bulk {
    struct bulk_t;
  } // namespace __bulk

  using __bulk::bulk_t;
  extern const bulk_t bulk;

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
  namespace __on_v2 {
    struct on_t;
  } // namespace __on_v2

  namespace v2 {
    using __on_v2::on_t;
  } // namespace v2

  namespace __detail {
    struct __sexpr_apply_t;
  } // namespace __detail

  using __detail::__sexpr_apply_t;
  extern const __sexpr_apply_t __sexpr_apply;
} // namespace stdexec

template <class...>
[[deprecated]]
void print() {
}

template <class>
struct undef;
