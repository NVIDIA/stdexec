/*
 * Copyright (c) 2023 Lee Howes
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

#include "stdexec/execution.hpp"
// For the default implementation, test will override
#include "exec/static_thread_pool.hpp"

struct __exec_system_scheduler_impl;
struct __exec_system_sender_impl;

// Low-level APIs
// Phase 2 will move these to weak symbols and allow replacement in tests
// Default implementation based on static_thread_pool
struct __exec_system_context_impl {
  exec::static_thread_pool pool;

  __exec_system_scheduler_impl get_scheduler();
};

struct __exec_system_scheduler_impl {
  __exec_system_context_impl* ctx_;

  __exec_system_sender_impl schedule() const;

  stdexec::forward_progress_guarantee get_forward_progress_guarantee() const {
    return stdexec::forward_progress_guarantee::parallel;
  }
};

struct __exec_system_sender_impl {

};



// Phase 1 implementation, single single implementation
static __exec_system_context_impl* __get_exec_system_context_impl() {
  static __exec_system_context_impl impl_;

  return &impl_;
}

inline __exec_system_scheduler_impl __exec_system_context_impl::get_scheduler() {
  return __exec_system_scheduler_impl{this};
}

__exec_system_sender_impl __exec_system_scheduler_impl::schedule() const {
  // TODO
  return __exec_system_sender_impl{};
}




namespace exec {



  class system_scheduler;
  class system_sender;

  class system_context {
  public:
    system_context() {
      impl_ = __get_exec_system_context_impl();
      // TODO error handling
    }

    system_scheduler get_scheduler();

  private:
    __exec_system_context_impl* impl_ = nullptr;

  };

  class system_scheduler {
  private:
    friend system_sender tag_invoke(
      stdexec::schedule_t, const system_scheduler&) noexcept;

    friend stdexec::forward_progress_guarantee tag_invoke(
      stdexec::get_forward_progress_guarantee_t,
      const system_scheduler&) noexcept;

    // Pointer that we ref count?
    system_scheduler(__exec_system_scheduler_impl impl) : impl_(impl) {}

    __exec_system_scheduler_impl impl_;
    friend class system_context;
  };

  class system_sender {
  public:
    system_sender(__exec_system_sender_impl impl) : impl_{impl} {}

  private:

    __exec_system_sender_impl impl_;
  };


  inline system_scheduler system_context::get_scheduler() {
    return system_scheduler{impl_->get_scheduler()};
  }

  system_sender tag_invoke(
      stdexec::schedule_t, const system_scheduler& sched) noexcept {
    return system_sender(sched.impl_.schedule());
  }

  stdexec::forward_progress_guarantee tag_invoke(
      stdexec::get_forward_progress_guarantee_t,
      const system_scheduler& sched) noexcept {
    return sched.impl_.get_forward_progress_guarantee();
  }
} // namespace exec
