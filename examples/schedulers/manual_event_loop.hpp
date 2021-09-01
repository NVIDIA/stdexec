/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Copyright (c) NVIDIA
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

#include <stop_token.hpp>
#include <execution.hpp>

#include <condition_variable>
#include <mutex>
#include <type_traits>

namespace example {
  struct manual_event_loop;

  namespace detail {
    struct task_base {
      void execute() noexcept {
        this->execute_(this);
      }

      task_base* next_;
      void(* execute_)(task_base*) noexcept;
      manual_event_loop* const loop_;
    };

    template <class Op>
    void _execute(task_base* t) noexcept {
      auto& self = *static_cast<Op*>(t);
      using token_t = decltype(std::execution::get_stop_token(self.receiver_));
      if constexpr (std::unstoppable_token<token_t>) {
        std::execution::set_value(std::move(self.receiver_));
      } else {
        if (std::execution::get_stop_token(self.receiver_).stop_requested()) {
          std::execution::set_done(std::move(self.receiver_));
        } else {
          std::execution::set_value(std::move(self.receiver_));
        }
      }
    }

    template <class Receiver_>
    struct operation : task_base {
      friend void tag_invoke(std::execution::start_t, operation& op) noexcept {
        op.start_();
      }

      void start_() noexcept;
      [[no_unique_address]] std::__t<Receiver_> receiver_;
    };
  }

  struct manual_event_loop {
    template <class>
    friend struct detail::operation;
  public:
    struct scheduler {
      struct schedule_task {
        template <
            template <class...> class Tuple,
            template <class...> class Variant>
        using value_types = Variant<Tuple<>>;

        template <template <class...> class Variant>
        using error_types = Variant<>;

        static constexpr bool sends_done = true;

        template <class Receiver>
        using _op_t =
            detail::operation<std::__id_t<std::remove_cvref_t<Receiver>>>;

        template <std::execution::receiver_of Receiver>
        friend _op_t<Receiver> tag_invoke(
            std::execution::connect_t, schedule_task self, Receiver&& receiver) {
          return {{nullptr, &detail::_execute<_op_t<Receiver>>, self.loop_},
                  (Receiver &&) receiver};
        }

        manual_event_loop* const loop_;
      };

      friend schedule_task tag_invoke(std::execution::schedule_t, scheduler self) noexcept {
        return {self.loop_};
      }

      bool operator==(const scheduler&) const noexcept = default;

      manual_event_loop* loop_;
    };

    scheduler get_scheduler() {
      return scheduler{this};
    }

    void run();

    void finish();

  private:
    void enqueue(detail::task_base* task);

    std::mutex mutex_;
    std::condition_variable cv_;
    detail::task_base* head_ = nullptr;
    detail::task_base* tail_ = nullptr;
    bool stop_ = false;
  };

  template <class Receiver>
  inline void detail::operation<Receiver>::start_() noexcept {
    loop_->enqueue(this);
  }

  inline void manual_event_loop::run() {
    std::unique_lock lock{mutex_};
    while (true) {
      while (head_ == nullptr) {
        if (stop_) return;
        cv_.wait(lock);
      }
      auto* task = head_;
      head_ = task->next_;
      if (head_ == nullptr) {
        tail_ = nullptr;
      }
      lock.unlock();
      task->execute();
      lock.lock();
    }
  }

  inline void manual_event_loop::finish() {
    std::unique_lock lock{mutex_};
    stop_ = true;
    cv_.notify_all();
  }

  inline void manual_event_loop::enqueue(detail::task_base* task) {
    std::unique_lock lock{mutex_};
    if (head_ == nullptr) {
      head_ = task;
    } else {
      tail_->next_ = task;
    }
    tail_ = task;
    task->next_ = nullptr;
    cv_.notify_one();
  }
} // namespace example
