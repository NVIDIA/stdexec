/*
 * Copyright (c) 2024 Maikel Nadolski
 * Copyright (c) 2024 NVIDIA Corporation
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

#include "./timed_scheduler.hpp"
#include "./__detail/intrusive_heap.hpp"

#include "../stdexec/__detail/__intrusive_mpsc_queue.hpp"
#include "../stdexec/__detail/__spin_loop_pause.hpp"

#include <bit>

namespace exec {
  namespace detail {
    struct timed_thread_operation_base {
      enum class command_type {
        schedule,
        stop
      };

      timed_thread_operation_base(
        void (*set_value)(timed_thread_operation_base*) noexcept,
        command_type command = command_type::schedule) noexcept
        : command_{command}
        , set_value_{set_value} {
      }

      std::atomic<void*> next_{nullptr};
      command_type command_;
      void (*set_value_)(timed_thread_operation_base*) noexcept;
    };

    struct timed_thread_schedule_operation_base : timed_thread_operation_base {
      timed_thread_schedule_operation_base(
        std::chrono::steady_clock::time_point time_point,
        void (*set_stopped)(timed_thread_operation_base*) noexcept,
        void (*set_value)(timed_thread_operation_base*) noexcept) noexcept
        : timed_thread_operation_base{set_value, command_type::schedule}
        , time_point_{time_point}
        , set_stopped_{set_stopped} {
      }

      std::chrono::steady_clock::time_point time_point_;
      timed_thread_schedule_operation_base* prev_ = nullptr;
      timed_thread_schedule_operation_base* left_ = nullptr;
      timed_thread_schedule_operation_base* right_ = nullptr;
      void (*set_stopped_)(timed_thread_operation_base*) noexcept;
    };

    struct timed_thread_stop_operation : timed_thread_operation_base {
      timed_thread_stop_operation(
        void (*set_value)(timed_thread_operation_base*) noexcept,
        timed_thread_schedule_operation_base* target) noexcept
        : timed_thread_operation_base{set_value, command_type::stop}
        , target_{target} {
      }

      timed_thread_schedule_operation_base* target_;
    };
  } // namespace detail

  class timed_thread_scheduler;

  template <class Rcvr>
  struct timed_thread_schedule_at_op {
    class __t;
  };

  class timed_thread_context {
   private:
    static constexpr std::ptrdiff_t context_closed = std::numeric_limits<std::ptrdiff_t>::min() / 2;
   public:
    timed_thread_context() noexcept
      : run_thread_(&timed_thread_context::run, this) {
    }

    ~timed_thread_context() {
      request_stop();
      run_thread_.join();
    }

    timed_thread_scheduler get_scheduler() noexcept;

   private:
    template <class Rcvr>
    friend struct timed_thread_schedule_at_op;

    using command_type = detail::timed_thread_operation_base;
    using task_type = detail::timed_thread_schedule_operation_base;
    using stop_type = detail::timed_thread_stop_operation;
    using time_point = std::chrono::steady_clock::time_point;

    void run() {
      while (true) {
        while (command_type* op = command_queue_.pop_front()) {
          if (op->command_ == command_type::command_type::schedule) {
            heap_.insert(static_cast<task_type*>(op));
          } else {
            STDEXEC_ASSERT(op->command_ == command_type::command_type::stop);
            stop_type* stop_op = static_cast<stop_type*>(op);
            if (heap_.erase(stop_op->target_)) {
              stop_op->target_->set_stopped_(stop_op->target_);
            }
            stop_op->set_value_(stop_op);
          }
        }
        time_point now = std::chrono::steady_clock::now();
        task_type* op = heap_.front();
        while (op && op->time_point_ <= now) {
          heap_.pop_front();
          op->set_value_(op);
          op = heap_.front();
        }
        time_point deadline = op ? op->time_point_ : now + std::chrono::seconds(2);
        std::unique_lock lock{ready_mutex_};
        cv_.wait_until(lock, deadline, [this] { return ready_ || stop_requested_; });
        bool stop_requested = stop_requested_;
        ready_ = false;
        lock.unlock();
        if (stop_requested) {
          std::ptrdiff_t expected = 0;
          while (!n_submissions_in_flight_.compare_exchange_weak(
            expected, context_closed, std::memory_order_relaxed)) {
            stdexec::__spin_loop_pause();
            expected = 0;
          }
          op = heap_.front();
          while (op) {
            heap_.pop_front();
            op->set_stopped_(op);
            op = heap_.front();
          }
          break;
        }
      }
    }

    void schedule(command_type* op) {
      std::ptrdiff_t n = n_submissions_in_flight_.fetch_add(1, std::memory_order_relaxed);
      if (n < 0) {
        if (op->command_ == command_type::command_type::schedule) {
          static_cast<task_type*>(op)->set_stopped_(op);
        } else {
          STDEXEC_ASSERT(op->command_ == command_type::command_type::stop);
          static_cast<stop_type*>(op)->set_value_(op);
        }
        n_submissions_in_flight_.compare_exchange_strong(
          n, context_closed, std::memory_order_relaxed);
        return;
      }
      if (command_queue_.push_back(op)) {
        std::scoped_lock lock{ready_mutex_};
        ready_ = true;
        cv_.notify_one();
      }
      n_submissions_in_flight_.fetch_sub(1, std::memory_order_relaxed);
    }

    void request_stop() {
      std::scoped_lock lock{ready_mutex_};
      stop_requested_ = true;
      cv_.notify_one();
    }

    stdexec::__intrusive_mpsc_queue<&command_type::next_> command_queue_;
    intrusive_heap<&task_type::time_point_, &task_type::prev_, &task_type::left_, &task_type::right_>
      heap_;
    std::atomic<std::ptrdiff_t> n_submissions_in_flight_{0};
    std::mutex ready_mutex_;
    bool ready_{false};
    bool stop_requested_{false};
    std::condition_variable cv_;
    std::thread run_thread_;
  };

  template <class Receiver>
  class timed_thread_schedule_at_op<Receiver>::__t : detail::timed_thread_schedule_operation_base {
   public:
    __t(
      timed_thread_context& context,
      std::chrono::steady_clock::time_point time_point,
      Receiver receiver) noexcept
      : detail::timed_thread_schedule_operation_base{
        time_point,
        [](detail::timed_thread_operation_base* op) noexcept {
          auto* self = static_cast<__t*>(op);
          int counter = self->ref_count_.fetch_sub(1, std::memory_order_relaxed);
          if (counter == 1) {
            self->stop_callback_.reset();
            stdexec::set_stopped(std::move(self->receiver_));
          }
        },
        [](detail::timed_thread_operation_base* op) noexcept {
          auto* self = static_cast<__t*>(op);
          int counter = self->ref_count_.fetch_sub(1, std::memory_order_relaxed);
          if (counter == 1) {
            self->stop_callback_.reset();
            stdexec::set_value(std::move(self->receiver_));
          }
        }}
      , context_{context}
      , receiver_{std::move(receiver)}
      , stop_op_{
          [](detail::timed_thread_operation_base* op) noexcept {
            auto* stop = static_cast<detail::timed_thread_stop_operation*>(op);
            auto* self = static_cast<__t*>(stop->target_);
            int counter = self->ref_count_.fetch_sub(1, std::memory_order_relaxed);
            if (counter == 1) {
              self->stop_callback_.reset();
              stdexec::set_stopped(std::move(self->receiver_));
            }
          },
          this} {
    }

    friend void tag_invoke(stdexec::start_t, __t& self) noexcept {
      self.stop_callback_.emplace(
        stdexec::get_stop_token(stdexec::get_env(self.receiver_)), on_stopped_t{self});
      int expected = 0;
      if (self.ref_count_.compare_exchange_strong(expected, 1, std::memory_order_relaxed)) {
        self.schedule_this();
      } else {
        self.stop_callback_.reset();
        stdexec::set_stopped(std::move(self.receiver_));
      }
    }

   private:
    void schedule_this() noexcept {
      context_.schedule(this);
    }

    struct on_stopped_t {
      __t& self_;

      void operator()() const noexcept {
        self_.request_stop();
      }
    };

    using callback_type = typename stdexec::stop_token_of_t<
      stdexec::env_of_t<Receiver>>::template callback_type<on_stopped_t>;

    void request_stop() noexcept {
      if (ref_count_.fetch_add(1, std::memory_order_relaxed) == 1) {
        context_.schedule(&stop_op_);
      }
    }

    timed_thread_context& context_;
    Receiver receiver_;
    detail::timed_thread_stop_operation stop_op_;
    std::optional<callback_type> stop_callback_;
    std::atomic<int> ref_count_{0};
  };

  class timed_thread_scheduler {
   public:
    using time_point = std::chrono::steady_clock::time_point;
    using duration = std::chrono::steady_clock::duration;

    class schedule_at {
     public:
      using sender_concept = stdexec::sender_t;
      using completion_signatures =
        stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_stopped_t()>;

      schedule_at(
        timed_thread_context& context,
        std::chrono::steady_clock::time_point time_point) noexcept
        : context_{&context}
        , time_point_{time_point} {
      }

      friend auto tag_invoke(
        stdexec::get_env_t,
        const schedule_at& self) noexcept -> stdexec::__env::
        __with<timed_thread_scheduler, stdexec::get_completion_scheduler_t<stdexec::set_value_t>> {
        return stdexec::__env::__with(
          timed_thread_scheduler{*self.context_},
          stdexec::get_completion_scheduler<stdexec::set_value_t>);
      }

      template <class Receiver>
      friend auto
        tag_invoke(stdexec::connect_t, const schedule_at& self, Receiver receiver) noexcept ->
        typename timed_thread_schedule_at_op<Receiver>::__t {
        return {*self.context_, self.time_point_, std::move(receiver)};
      }

     private:
      timed_thread_scheduler get_scheduler() const noexcept;

      timed_thread_context* context_;
      std::chrono::steady_clock::time_point time_point_;
    };

    explicit timed_thread_scheduler(timed_thread_context& context) noexcept
      : context_{&context} {
    }

    friend auto tag_invoke(now_t, const timed_thread_scheduler&) noexcept -> time_point {
      return std::chrono::steady_clock::now();
    }

    friend auto
      tag_invoke(schedule_at_t, const timed_thread_scheduler& self, time_point tp) noexcept
      -> schedule_at {
      return schedule_at{*self.context_, tp};
    }

    friend auto tag_invoke(stdexec::schedule_t, const timed_thread_scheduler& self) noexcept
      -> schedule_at {
      return exec::schedule_at(self, time_point());
    }

    friend auto operator==(
      const timed_thread_scheduler& sched1,
      const timed_thread_scheduler& sched2) noexcept -> bool = default;

   private:
    timed_thread_context* context_;
  };

  inline timed_thread_scheduler timed_thread_context::get_scheduler() noexcept {
    return timed_thread_scheduler{*this};
  }
} // namespace exec