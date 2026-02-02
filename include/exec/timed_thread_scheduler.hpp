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

#include "__detail/intrusive_heap.hpp"
#include "timed_scheduler.hpp" // IWYU pragma: keep for schedule_at and schedule_after

#include "../stdexec/__detail/__atomic.hpp"
#include "../stdexec/__detail/__intrusive_mpsc_queue.hpp"
#include "../stdexec/__detail/__receivers.hpp"
#include "../stdexec/__detail/__schedulers.hpp"
#include "../stdexec/__detail/__spin_loop_pause.hpp"

#include <chrono>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <optional>
#include <utility>

namespace exec {
  class timed_thread_scheduler;

  namespace _time_thrd_sched {
    struct timed_thread_operation_base {
      enum class command_type {
        schedule,
        stop
      };

      constexpr timed_thread_operation_base(
        void (*set_value)(timed_thread_operation_base*) noexcept,
        command_type command = command_type::schedule) noexcept
        : command_{command}
        , set_value_{set_value} {
      }

      STDEXEC::__std::atomic<void*> next_{nullptr};
      command_type command_;
      void (*set_value_)(timed_thread_operation_base*) noexcept;
    };

    template <class Tp>
    struct when_type {
      constexpr when_type() = default;

      constexpr explicit when_type(Tp tp, std::size_t n = 0) noexcept
        : time_point{std::move(tp)}
        , counter{n} {
      }

      Tp time_point{};
      std::size_t counter{};

      friend constexpr auto operator<(const when_type& lhs, const when_type& rhs) noexcept -> bool {
        return lhs.time_point < rhs.time_point
            || (!(rhs.time_point < lhs.time_point) && lhs.counter < rhs.counter);
      }
    };

    struct timed_thread_schedule_operation_base : timed_thread_operation_base {
      using time_point = std::chrono::steady_clock::time_point;

      constexpr explicit timed_thread_schedule_operation_base(
        time_point tp,
        void (*set_stopped)(timed_thread_operation_base*) noexcept,
        void (*set_value)(timed_thread_operation_base*) noexcept) noexcept
        : timed_thread_operation_base{set_value, command_type::schedule}
        , time_point_{tp}
        , set_stopped_{set_stopped} {
      }

      time_point time_point_;
      // we increase the when counter to ensure that the heap is stable
      // when two operations have the same time_point
      // We do so only when the operation is started, not when it is constructed
      when_type<time_point> when_{};
      timed_thread_schedule_operation_base* prev_ = nullptr;
      timed_thread_schedule_operation_base* left_ = nullptr;
      timed_thread_schedule_operation_base* right_ = nullptr;
      void (*set_stopped_)(timed_thread_operation_base*) noexcept;
    };

    struct timed_thread_stop_operation : timed_thread_operation_base {
      constexpr timed_thread_stop_operation(
        void (*set_value)(timed_thread_operation_base*) noexcept,
        timed_thread_schedule_operation_base* target) noexcept
        : timed_thread_operation_base{set_value, command_type::stop}
        , target_{target} {
      }

      timed_thread_schedule_operation_base* target_;
    };

    template <class Rcvr>
    class timed_thread_schedule_at_op;
  } // namespace _time_thrd_sched

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

    constexpr auto get_scheduler() noexcept -> timed_thread_scheduler;

   private:
    template <class Rcvr>
    friend class _time_thrd_sched::timed_thread_schedule_at_op;

    using command_type = _time_thrd_sched::timed_thread_operation_base;
    using task_type = _time_thrd_sched::timed_thread_schedule_operation_base;
    using stop_type = _time_thrd_sched::timed_thread_stop_operation;
    using time_point = std::chrono::steady_clock::time_point;

    void run() {
      while (true) {
        while (command_type* op = command_queue_.pop_front()) {
          if (op->command_ == command_type::command_type::schedule) {
            auto* task = static_cast<task_type*>(op);
            task->when_ = _time_thrd_sched::when_type{task->time_point_, submission_counter_++};
            heap_.insert(task);
          } else {
            STDEXEC_ASSERT(op->command_ == command_type::command_type::stop);
            auto* stop_op = static_cast<stop_type*>(op);
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
            expected, context_closed, STDEXEC::__std::memory_order_relaxed)) {
            STDEXEC::__spin_loop_pause();
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
      std::ptrdiff_t n = n_submissions_in_flight_
                           .fetch_add(1, STDEXEC::__std::memory_order_relaxed);
      if (n < 0) {
        if (op->command_ == command_type::command_type::schedule) {
          static_cast<task_type*>(op)->set_stopped_(op);
        } else {
          STDEXEC_ASSERT(op->command_ == command_type::command_type::stop);
          static_cast<stop_type*>(op)->set_value_(op);
        }
        n_submissions_in_flight_
          .compare_exchange_strong(n, context_closed, STDEXEC::__std::memory_order_relaxed);
        return;
      }
      if (command_queue_.push_back(op)) {
        std::scoped_lock lock{ready_mutex_};
        ready_ = true;
        cv_.notify_one();
      }
      n_submissions_in_flight_.fetch_sub(1, STDEXEC::__std::memory_order_relaxed);
    }

    void request_stop() {
      std::scoped_lock lock{ready_mutex_};
      stop_requested_ = true;
      cv_.notify_one();
    }

    STDEXEC::__intrusive_mpsc_queue<&command_type::next_> command_queue_;
    intrusive_heap<
      task_type,
      _time_thrd_sched::when_type<time_point>,
      &task_type::when_,
      &task_type::prev_,
      &task_type::left_,
      &task_type::right_
    >
      heap_;
    STDEXEC::__std::atomic<std::ptrdiff_t> n_submissions_in_flight_{0};
    std::mutex ready_mutex_;
    bool ready_{false};
    bool stop_requested_{false};
    std::condition_variable cv_;
    std::thread run_thread_;
    std::size_t submission_counter_{1};
  };

  namespace _time_thrd_sched {
    template <class Receiver>
    class timed_thread_schedule_at_op : _time_thrd_sched::timed_thread_schedule_operation_base {
     public:
      constexpr explicit timed_thread_schedule_at_op(
        timed_thread_context& context,
        std::chrono::steady_clock::time_point time_point,
        Receiver receiver) noexcept
        : _time_thrd_sched::timed_thread_schedule_operation_base{
            time_point,
            [](_time_thrd_sched::timed_thread_operation_base* op) noexcept {
              auto* self = static_cast<timed_thread_schedule_at_op*>(op);
              int counter = self->ref_count_.fetch_sub(1, STDEXEC::__std::memory_order_relaxed);
              if (counter == 1) {
                self->stop_callback_.reset();
                STDEXEC::set_stopped(std::move(self->rcvr_));
              }
            },
            [](_time_thrd_sched::timed_thread_operation_base* op) noexcept {
              auto* self = static_cast<timed_thread_schedule_at_op*>(op);
              int counter = self->ref_count_.fetch_sub(1, STDEXEC::__std::memory_order_relaxed);
              if (counter == 1) {
                self->stop_callback_.reset();
                STDEXEC::set_value(std::move(self->rcvr_));
              }
            }}
        , context_{context}
        , rcvr_{std::move(receiver)}
        , stop_op_{
            [](_time_thrd_sched::timed_thread_operation_base* op) noexcept {
              auto* stop = static_cast<_time_thrd_sched::timed_thread_stop_operation*>(op);
              auto* self = static_cast<timed_thread_schedule_at_op*>(stop->target_);
              int counter = self->ref_count_.fetch_sub(1, STDEXEC::__std::memory_order_relaxed);
              if (counter == 1) {
                self->stop_callback_.reset();
                STDEXEC::set_stopped(std::move(self->rcvr_));
              }
            },
            this} {
      }

      constexpr void start() & noexcept {
        stop_callback_
          .emplace(STDEXEC::get_stop_token(STDEXEC::get_env(rcvr_)), on_stopped_t{*this});
        int expected = 0;
        if (ref_count_.compare_exchange_strong(expected, 1, STDEXEC::__std::memory_order_relaxed)) {
          schedule_this();
        } else {
          stop_callback_.reset();
          STDEXEC::set_stopped(std::move(rcvr_));
        }
      }

     private:
      constexpr void schedule_this() noexcept {
        context_.schedule(this);
      }

      struct on_stopped_t {
        timed_thread_schedule_at_op& self_;

        constexpr void operator()() const noexcept {
          self_.request_stop();
        }
      };

      using callback_type =
        STDEXEC::stop_token_of_t<STDEXEC::env_of_t<Receiver>>::template callback_type<on_stopped_t>;

      constexpr void request_stop() noexcept {
        if (ref_count_.fetch_add(1, STDEXEC::__std::memory_order_relaxed) == 1) {
          context_.schedule(&stop_op_);
        }
      }

      timed_thread_context& context_;
      Receiver rcvr_;
      _time_thrd_sched::timed_thread_stop_operation stop_op_;
      std::optional<callback_type> stop_callback_;
      STDEXEC::__std::atomic<int> ref_count_{0};
    };
  } // namespace _time_thrd_sched

  class timed_thread_scheduler {
   public:
    using time_point = std::chrono::steady_clock::time_point;
    using duration = std::chrono::steady_clock::duration;

    class schedule_at_sender {
     public:
      using sender_concept = STDEXEC::sender_t;
      using completion_signatures =
        STDEXEC::completion_signatures<STDEXEC::set_value_t(), STDEXEC::set_stopped_t()>;

      struct attrs {
        [[nodiscard]]
        constexpr auto
          query(STDEXEC::get_completion_scheduler_t<STDEXEC::set_value_t>) const noexcept
          -> timed_thread_scheduler {
          return timed_thread_scheduler{*context_};
        }

        timed_thread_context* context_;
      };

      constexpr explicit schedule_at_sender(
        timed_thread_context& context,
        std::chrono::steady_clock::time_point time_point) noexcept
        : context_{&context}
        , time_point_{time_point} {
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> attrs {
        return attrs{context_};
      }

      template <class Receiver>
      constexpr auto connect(Receiver receiver) const & noexcept
        -> _time_thrd_sched::timed_thread_schedule_at_op<Receiver> {
        return _time_thrd_sched::timed_thread_schedule_at_op<Receiver>{
          *context_, time_point_, std::move(receiver)};
      }

     private:
      [[nodiscard]]
      constexpr auto get_scheduler() const noexcept -> timed_thread_scheduler;

      timed_thread_context* context_;
      std::chrono::steady_clock::time_point time_point_;
    };

    constexpr explicit timed_thread_scheduler(timed_thread_context& context) noexcept
      : context_{&context} {
    }

    [[nodiscard]]
    static auto now() noexcept -> time_point {
      return std::chrono::steady_clock::now();
    }

    [[nodiscard]]
    constexpr auto schedule_at(time_point tp) const noexcept -> schedule_at_sender {
      return schedule_at_sender{*context_, tp};
    }

    [[nodiscard]]
    constexpr auto schedule() const noexcept -> schedule_at_sender {
      return schedule_at(time_point());
    }

    constexpr auto operator==(const timed_thread_scheduler&) const noexcept -> bool = default;

   private:
    timed_thread_context* context_;
  };

  inline constexpr auto timed_thread_context::get_scheduler() noexcept -> timed_thread_scheduler {
    return timed_thread_scheduler{*this};
  }
} // namespace exec
