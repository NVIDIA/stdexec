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
// Phase 2 will move to pointers and ref counting ala COM
// Phase 3 will move these to weak symbols and allow replacement in tests
// Default implementation based on static_thread_pool
struct __exec_system_context_impl {
  exec::static_thread_pool pool_;

  __exec_system_scheduler_impl get_scheduler();
};

struct __exec_system_scheduler_impl {
  __exec_system_context_impl* ctx_;
  decltype(ctx_->pool_.get_scheduler()) pool_scheduler_;

  __exec_system_sender_impl schedule() const;

  stdexec::forward_progress_guarantee get_forward_progress_guarantee() const {
    return stdexec::forward_progress_guarantee::parallel;
  }

  bool equals(const __exec_system_scheduler_impl* rhs) const {
    return rhs == this;
  }
};

struct __exec_system_sender_impl {
  decltype(stdexec::schedule(std::declval<__exec_system_scheduler_impl>().pool_scheduler_)) pool_sender_;
};



// Phase 1 implementation, single single implementation
static __exec_system_context_impl* __get_exec_system_context_impl() {
  static __exec_system_context_impl impl_;

  return &impl_;
}

inline __exec_system_scheduler_impl __exec_system_context_impl::get_scheduler() {
  return __exec_system_scheduler_impl{this, pool_.get_scheduler()};
}

__exec_system_sender_impl __exec_system_scheduler_impl::schedule() const {
  // TODO: Can schedule on thread pool, but not absolutely necessary. Doing in start() for now.
  return __exec_system_sender_impl{stdexec::schedule(pool_scheduler_)};
}




namespace exec {
  namespace __system_scheduler {

  } // namespace system_scheduler


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
  public:
    // Pointer that we ref count?
    system_scheduler(__exec_system_scheduler_impl impl) : impl_(impl) {}

    bool operator==(const system_scheduler& rhs) const noexcept {
      impl_.equals(&(rhs.impl_));
    }

  private:
    friend system_sender tag_invoke(
      stdexec::schedule_t, const system_scheduler&) noexcept;

    friend stdexec::forward_progress_guarantee tag_invoke(
      stdexec::get_forward_progress_guarantee_t,
      const system_scheduler&) noexcept;



    __exec_system_scheduler_impl impl_;
    friend class system_context;
  };

  class system_sender {
  public:
    using is_sender = void;
    using completion_signatures =
      stdexec::completion_signatures< stdexec::set_value_t(), stdexec::set_stopped_t() >;

    system_sender(__exec_system_scheduler_impl scheduler_impl, __exec_system_sender_impl impl) :
        scheduler_impl_{scheduler_impl}, impl_{impl} {}

  private:
    template <class R_>
    struct __op {
      using R = stdexec::__t<R_>;
      decltype(stdexec::connect(std::declval<__exec_system_sender_impl>().pool_sender_, std::declval<R>())) impl_;

      friend void tag_invoke(stdexec::start_t, __op& op) noexcept {
        stdexec::start(op.impl_);
      }

      // TODO: Type-erase operation state to remove coupling with pool_sender
    };

    template <class R>
    friend auto tag_invoke(stdexec::connect_t, system_sender snd, R&& rec) //
      noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<R>, R>)
        -> __op<stdexec::__x<std::remove_cvref_t<R>>> {
      return {stdexec::connect(snd.impl_.pool_sender_, (R&&) rec)};
    }

    struct __env {
      friend system_scheduler
        tag_invoke(stdexec::get_completion_scheduler_t<stdexec::set_value_t>, const __env& self) //
        noexcept {
        return {self.impl_};
      }

      friend system_scheduler
        tag_invoke(stdexec::get_completion_scheduler_t<stdexec::set_stopped_t>, const __env& self) //
        noexcept {
        return {self.impl_};
      }

      __exec_system_scheduler_impl impl_;
    };

    friend __env tag_invoke(stdexec::get_env_t, const system_sender& snd) noexcept {
      // TODO: Ref add
      return {snd.scheduler_impl_};
    }

    /*
    friend system_bulk_sender tag_invoke(
      std::execution::bulk_t,
      const system_scheduler&,
      Sh&& sh,
      F&& f) noexcept;
    */

      // TODO: Do we need both? Should we get scheduler from sender or do we need sender at all?
    __exec_system_scheduler_impl scheduler_impl_;
    __exec_system_sender_impl impl_;
  };


  inline system_scheduler system_context::get_scheduler() {
    return system_scheduler{impl_->get_scheduler()};
  }

  system_sender tag_invoke(
      stdexec::schedule_t, const system_scheduler& sched) noexcept {
    return system_sender(sched.impl_, sched.impl_.schedule());
  }

  stdexec::forward_progress_guarantee tag_invoke(
      stdexec::get_forward_progress_guarantee_t,
      const system_scheduler& sched) noexcept {
    return sched.impl_.get_forward_progress_guarantee();
  }

/*
  friend system_bulk_sender tag_invoke(
      std::execution::bulk_t,
      const system_scheduler& sched,
      Sh&& sh,
      F&& f) noexcept {
    return system_bulk_sender(impl.bulk_schedule(TBD…));
  }
*/
} // namespace exec
