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

namespace stdexec {
  struct __none_such;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct default_domain;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __receivers {
    struct set_value_t;
    struct set_error_t;
    struct set_stopped_t;
  }

  using __receivers::set_value_t;
  using __receivers::set_error_t;
  using __receivers::set_stopped_t;
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
  }

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

  template <class _Tp>
  using stop_token_of_t = __decay_t<__call_result_t<get_stop_token_t, _Tp>>;

  template <class _Sender, class _CPO>
  concept __has_completion_scheduler =
    __callable<get_completion_scheduler_t<_CPO>, env_of_t<const _Sender&>>;

  template <class _Sender, class _CPO>
  using __completion_scheduler_for =
    __call_result_t<get_completion_scheduler_t<_CPO>, env_of_t<const _Sender&>>;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __get_completion_signatures {
    struct get_completion_signatures_t;
  }

  using __get_completion_signatures::get_completion_signatures_t;
  extern const get_completion_signatures_t get_completion_signatures;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __connect {
    struct connect_t;
  }

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
  }

  using __start::start_t;
  extern const start_t start;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __schedule {
    struct schedule_t;
  }

  using __schedule::schedule_t;
  extern const schedule_t schedule;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __as_awaitable {
    struct as_awaitable_t;
  }

  using __as_awaitable::as_awaitable_t;
  extern const as_awaitable_t as_awaitable;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __transfer {
    struct transfer_t;
  }

  using __transfer::transfer_t;
  extern const transfer_t transfer;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __transfer_just {
    struct transfer_just_t;
  }

  using __transfer_just::transfer_just_t;
  extern const transfer_just_t transfer_just;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __bulk {
    struct bulk_t;
  }

  using __bulk::bulk_t;
  extern const bulk_t bulk;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  namespace __on_v2 {
    struct on_t;
  }

  namespace v2 {
    using __on_v2::on_t;
  }

  namespace __detail {
    struct __sexpr_apply_t;
  }

  using __detail::__sexpr_apply_t;
  extern const __sexpr_apply_t __sexpr_apply;
}

template <class...>
[[deprecated]] void print() {}

template <class>
struct undef;
