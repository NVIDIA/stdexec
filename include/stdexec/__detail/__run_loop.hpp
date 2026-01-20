/*
 * Copyright (c) 2025 NVIDIA Corporation
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
#include "__atomic.hpp"
#include "__atomic_intrusive_queue.hpp"
#include "__completion_signatures.hpp"
#include "__domain.hpp"
#include "__env.hpp"
#include "__receivers.hpp"
#include "__schedulers.hpp"
#include "__stop_token.hpp"

#include <cstddef>

namespace STDEXEC {
  /////////////////////////////////////////////////////////////////////////////
  // run_loop
  class __run_loop_base : __immovable {
   public:
    __run_loop_base() = default;

    ~__run_loop_base() noexcept {
      STDEXEC_ASSERT(__task_count_.load(__std::memory_order_acquire) == 0);
    }

    STDEXEC_ATTRIBUTE(host, device)
    void run() noexcept {
      // execute work items until the __finishing_ flag is set:
      while (!__finishing_.load(__std::memory_order_acquire)) {
        __queue_.wait_for_item();
        __execute_all();
      }
      // drain the queue, taking care to execute any tasks that get added while
      // executing the remaining tasks (also wait for other tasks that might still be in flight):
      while (__execute_all() || __task_count_.load(__std::memory_order_acquire) > 0)
        ;
    }

    STDEXEC_ATTRIBUTE(host, device)
    void finish() noexcept {
      // Increment our task count to avoid lifetime issues. This is preventing
      // a use-after-free issue if finish is called from a different thread.
      // We increment the task counter by two to prevent the run loop from
      // exiting before we schedule the noop task.
      __task_count_.fetch_add(2, __std::memory_order_release);
      if (!__finishing_.exchange(true, __std::memory_order_acq_rel)) {
        // push an empty work item to the queue to wake up the consuming thread
        // and let it finish.
        // The count will be decremented once the tasks executes.
        __queue_.push(&__noop_task);
        // If the task got pushed, simply subtract one again, the other decrement
        // happens when the noop task got executed.
        __task_count_.fetch_sub(1, __std::memory_order_release);
        return;
      }
      // We are done finishing. Decrement the count by two, which signals final completion.
      __task_count_.fetch_sub(2, __std::memory_order_release);
    }

    struct __task : __immovable {
      using __execute_fn_t = void(__task*) noexcept;

      constexpr __task() = default;
      STDEXEC_ATTRIBUTE(host, device)
      constexpr explicit __task(__execute_fn_t* __execute_fn) noexcept
        : __execute_fn_(__execute_fn) {
      }

      STDEXEC_ATTRIBUTE(host, device)
      constexpr void __execute() noexcept {
        (*__execute_fn_)(this);
      }

      __execute_fn_t* __execute_fn_ = nullptr;
      __task* __next_ = nullptr;
    };

    template <class _Rcvr>
    struct __opstate_t : __task {
      __std::atomic<std::size_t>* __task_count_;
      __atomic_intrusive_queue<&__task::__next_>* __queue_;
      _Rcvr __rcvr_;

      STDEXEC_ATTRIBUTE(host, device)
      static constexpr void __execute_impl(__task* __p) noexcept {
        static_assert(noexcept(get_stop_token(__declval<env_of_t<_Rcvr>>()).stop_requested()));
        auto& __rcvr = static_cast<__opstate_t*>(__p)->__rcvr_;

        // NOLINTNEXTLINE(bugprone-branch-clone)
        if constexpr (unstoppable_token<stop_token_of_t<env_of_t<_Rcvr>>>) {
          set_value(static_cast<_Rcvr&&>(__rcvr));
        } else if (get_stop_token(get_env(__rcvr)).stop_requested()) {
          set_stopped(static_cast<_Rcvr&&>(__rcvr));
        } else {
          set_value(static_cast<_Rcvr&&>(__rcvr));
        }
      }

      STDEXEC_ATTRIBUTE(host, device)
      constexpr explicit __opstate_t(
        __std::atomic<std::size_t>* __task_count,
        __atomic_intrusive_queue<&__task::__next_>* __queue,
        _Rcvr __rcvr)
        : __task{&__execute_impl}
        , __task_count_(__task_count)
        , __queue_{__queue}
        , __rcvr_{static_cast<_Rcvr&&>(__rcvr)} {
      }

      STDEXEC_ATTRIBUTE(host, device)
      constexpr void start() noexcept {
        __task_count_->fetch_add(1, __std::memory_order_release);
        __queue_->push(this);
      }
    };

    // Returns true if any tasks were executed.
    STDEXEC_ATTRIBUTE(host, device)
    constexpr bool __execute_all() noexcept {
      // Dequeue all tasks at once. This returns an __intrusive_queue.
      auto __queue = __queue_.pop_all();

      // Execute all the tasks in the queue.
      auto __it = __queue.begin();
      if (__it == __queue.end()) {
        return false; // No tasks to execute.
      }

      std::size_t __task_count = 0;

      do {
        // Take care to increment the iterator before executing the task,
        // because __execute() may invalidate the current node.
        auto __prev = __it++;
        (*__prev)->__execute();
        ++__task_count;
      } while (__it != __queue.end());

      __queue.clear();
      __task_count_.fetch_sub(__task_count, __std::memory_order_release);
      return true;
    }

    STDEXEC_ATTRIBUTE(host, device) static constexpr void __noop_(__task*) noexcept {
    }

    __std::atomic<std::size_t> __task_count_{0};
    __std::atomic<bool> __finishing_{false};
    __atomic_intrusive_queue<&__task::__next_> __queue_{};
    __task __noop_task{&__noop_};
  };

  template <class _Env>
  struct basic_run_loop : __run_loop_base {
   private:
    struct __attrs_t {
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto query(get_completion_scheduler_t<set_value_t>) const noexcept;
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto query(get_completion_scheduler_t<set_stopped_t>) const noexcept;

      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto query(get_completion_domain_t<set_value_t>) const noexcept;
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto query(get_completion_domain_t<set_stopped_t>) const noexcept;

      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto query(get_completion_behavior_t<set_value_t>) const noexcept {
        return completion_behavior::asynchronous;
      }
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto query(get_completion_behavior_t<set_stopped_t>) const noexcept {
        return completion_behavior::asynchronous;
      }

      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto query(execute_may_block_caller_t) const noexcept {
        return false;
      }

      basic_run_loop* __loop_;
    };

   public:
    STDEXEC_ATTRIBUTE(host, device)
    constexpr explicit basic_run_loop(_Env __env) noexcept
      : __env_{static_cast<_Env&&>(__env)} {
    }

    class scheduler : __attrs_t {
     private:
      friend basic_run_loop;

      STDEXEC_ATTRIBUTE(host, device)
      constexpr explicit scheduler(basic_run_loop* __loop) noexcept
        : __attrs_t{__loop} {
      }

     public:
      using scheduler_concept = scheduler_t;

      struct __sndr_t {
        using sender_concept = sender_t;

        template <class _Rcvr>
        STDEXEC_ATTRIBUTE(nodiscard, host, device)
        constexpr auto connect(_Rcvr __rcvr) const noexcept -> __opstate_t<_Rcvr> {
          return __opstate_t<_Rcvr>{
            &__loop_->__task_count_, &__loop_->__queue_, static_cast<_Rcvr&&>(__rcvr)};
        }

        template <class, class...>
        STDEXEC_ATTRIBUTE(nodiscard, host, device)
        static consteval auto get_completion_signatures() noexcept {
          return completion_signatures<set_value_t(), set_stopped_t()>{};
        }

        STDEXEC_ATTRIBUTE(nodiscard, host, device)
        constexpr auto get_env() const noexcept -> __attrs_t {
          return __attrs_t{__loop_};
        }

       private:
        friend scheduler;
        STDEXEC_ATTRIBUTE(host, device)
        constexpr explicit __sndr_t(basic_run_loop* __loop) noexcept
          : __loop_(__loop) {
        }

        basic_run_loop* __loop_;
      };

      using __scheduler
        [[deprecated("run_loop::__scheduler has been renamed run_loop::scheduler")]] = scheduler;

      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto schedule() const noexcept -> __sndr_t {
        return __sndr_t{this->__loop_};
      }

      using __attrs_t::query;

      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto
        query(get_forward_progress_guarantee_t) const noexcept -> forward_progress_guarantee {
        return forward_progress_guarantee::parallel;
      }

      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      friend constexpr bool operator==(const scheduler& __a, const scheduler& __b) noexcept {
        return __a.__loop_ == __b.__loop_;
      }

      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      friend constexpr bool operator!=(const scheduler& __a, const scheduler& __b) noexcept {
        return __a.__loop_ != __b.__loop_;
      }
    };

    STDEXEC_ATTRIBUTE(nodiscard, host, device)
    constexpr auto get_scheduler() noexcept -> scheduler {
      return scheduler{this};
    }

    STDEXEC_ATTRIBUTE(nodiscard, host, device)
    constexpr auto get_env() const noexcept -> const _Env& {
      return __env_;
    }

   private:
    _Env __env_;
  };

  // A run_loop with an empty environment. This is a struct instead of a type alias to give
  // it a simpler type name that is easier to read in diagnostics.
  struct run_loop : basic_run_loop<env<>> {
    constexpr run_loop() noexcept
      : basic_run_loop<env<>>{env{}} {
    }
  };

  template <class _Env>
  STDEXEC_ATTRIBUTE(host, device)
  constexpr auto basic_run_loop<_Env>::__attrs_t::query(
    get_completion_scheduler_t<set_value_t>) const noexcept {
    if constexpr (__callable<get_scheduler_t, _Env&>) {
      return STDEXEC::get_scheduler(__loop_->__env_);
    } else {
      return scheduler{__loop_};
    }
  }

  template <class _Env>
  STDEXEC_ATTRIBUTE(host, device)
  constexpr auto basic_run_loop<_Env>::__attrs_t::query(
    get_completion_scheduler_t<set_stopped_t>) const noexcept {
    return query(get_completion_scheduler<set_value_t>);
  }

  template <class _Env>
  STDEXEC_ATTRIBUTE(host, device)
  constexpr auto basic_run_loop<_Env>::__attrs_t::query(
    get_completion_domain_t<set_value_t>) const noexcept {
    if constexpr (__callable<get_domain_t, _Env&>) {
      return __call_result_t<get_domain_t, _Env&>();
    } else {
      return default_domain{};
    }
  }

  template <class _Env>
  STDEXEC_ATTRIBUTE(host, device)
  constexpr auto basic_run_loop<_Env>::__attrs_t::query(
    get_completion_domain_t<set_stopped_t>) const noexcept {
    return query(get_completion_domain<set_value_t>);
  }

} // namespace STDEXEC
