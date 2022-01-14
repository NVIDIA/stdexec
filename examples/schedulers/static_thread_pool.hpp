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

#include <execution.hpp>
#include "../detail/intrusive_queue.hpp"

#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace example {
  struct task_base {
    task_base* next;
    void (*__execute)(task_base*) noexcept;
  };

  template <typename ReceiverID>
    class operation;

  class static_thread_pool {
    template <typename ReceiverId>
      friend class operation;
   public:
    static_thread_pool();
    static_thread_pool(std::uint32_t threadCount);
    ~static_thread_pool();

    struct scheduler {
      bool operator==(const scheduler&) const = default;

     private:
      template <typename ReceiverId>
        friend class operation;

      using set_value_t = std::decay_t<decltype(std::execution::set_value)>;
      using set_error_t = std::decay_t<decltype(std::execution::set_error)>;
      using set_stopped_t = std::decay_t<decltype(std::execution::set_stopped)>;
      using traits = std::execution::completion_signatures<
          set_value_t(),
          set_error_t(std::exception_ptr),
          set_stopped_t()>;

      class sender : public traits {
        template <typename Receiver>
        operation<std::__x<std::decay_t<Receiver>>>
        make_operation_(Receiver&& r) const {
          return operation<std::__x<std::decay_t<Receiver>>>{pool_, (Receiver &&) r};
        }

        template <std::execution::receiver_of Receiver>
        friend operation<std::__x<std::decay_t<Receiver>>>
        tag_invoke(std::execution::connect_t, sender s, Receiver&& r) {
          return s.make_operation_((Receiver &&) r);
        }

        template <class CPO>
        friend static_thread_pool::scheduler
        tag_invoke(std::execution::get_completion_scheduler_t<CPO>, sender s) noexcept {
          return static_thread_pool::scheduler{s.pool_};
        }

        friend struct static_thread_pool::scheduler;

        explicit sender(static_thread_pool& pool) noexcept
          : pool_(pool) {}

        static_thread_pool& pool_;
      };

      sender make_sender_() const {
        return sender{*pool_};
      }

      friend sender
      tag_invoke(std::execution::schedule_t, const scheduler& s) noexcept {
        return s.make_sender_();
      }

      friend std::execution::forward_progress_guarantee tag_invoke(
          std::execution::get_forward_progress_guarantee_t,
          const static_thread_pool&) noexcept {
        return std::execution::forward_progress_guarantee::parallel;
      }

      friend class static_thread_pool;
      explicit scheduler(static_thread_pool& pool) noexcept
        : pool_(&pool) {}

      static_thread_pool* pool_;
    };

    scheduler get_scheduler() noexcept { return scheduler{*this}; }

    void request_stop() noexcept;

   private:
    class thread_state {
     public:
      task_base* try_pop();
      task_base* pop();
      bool try_push(task_base* task);
      void push(task_base* task);
      void request_stop();

     private:
      std::mutex mut_;
      std::condition_variable cv_;
      intrusive_queue<&task_base::next> queue_;
      bool stopRequested_ = false;
    };

    void run(std::uint32_t index) noexcept;
    void join() noexcept;

    void enqueue(task_base* task) noexcept;

    std::uint32_t threadCount_;
    std::vector<std::thread> threads_;
    std::vector<thread_state> threadStates_;
    std::atomic<std::uint32_t> nextThread_;
  };

  template <typename ReceiverId>
    class operation : task_base {
      using Receiver = typename ReceiverId::type;
      friend static_thread_pool::scheduler::sender;

      static_thread_pool& pool_;
      Receiver receiver_;

      explicit operation(static_thread_pool& pool, Receiver&& r)
        : pool_(pool)
        , receiver_((Receiver &&) r) {
        this->__execute = [](task_base* t) noexcept {
          auto& op = *static_cast<operation*>(t);
          auto stoken =
            std::execution::get_stop_token(
              std::execution::get_env(op.receiver_));
          if (stoken.stop_requested()) {
            std::execution::set_stopped((Receiver &&) op.receiver_);
          } else {
            std::execution::set_value((Receiver &&) op.receiver_);
          }
        };
      }

      void enqueue_(task_base* op) const {
        pool_.enqueue(op);
      }

      friend void tag_invoke(std::execution::start_t, operation& op) noexcept {
        op.enqueue_(&op);
      }
    };

  inline static_thread_pool::static_thread_pool()
    : static_thread_pool(std::thread::hardware_concurrency()) {}

  inline static_thread_pool::static_thread_pool(std::uint32_t threadCount)
    : threadCount_(threadCount)
    , threadStates_(threadCount)
    , nextThread_(0) {
    assert(threadCount > 0);

    threads_.reserve(threadCount);

    try {
      for (std::uint32_t i = 0; i < threadCount; ++i) {
        threads_.emplace_back([this, i] { run(i); });
      }
    } catch (...) {
      request_stop();
      join();
      throw;
    }
  }

  inline static_thread_pool::~static_thread_pool() {
    request_stop();
    join();
  }

  inline void static_thread_pool::request_stop() noexcept {
    for (auto& state : threadStates_) {
      state.request_stop();
    }
  }

  inline void static_thread_pool::run(std::uint32_t index) noexcept {
    while (true) {
      task_base* task = nullptr;
      for (std::uint32_t i = 0; i < threadCount_; ++i) {
        auto queueIndex = (index + i) < threadCount_
            ? (index + i)
            : (index + i - threadCount_);
        auto& state = threadStates_[queueIndex];
        task = state.try_pop();
        if (task != nullptr) {
          break;
        }
      }

      if (task == nullptr) {
        task = threadStates_[index].pop();
        if (task == nullptr) {
          // request_stop() was called.
          return;
        }
      }

      task->__execute(task);
    }
  }

  inline void static_thread_pool::join() noexcept {
    for (auto& t : threads_) {
      t.join();
    }
    threads_.clear();
  }

  inline void static_thread_pool::enqueue(task_base* task) noexcept {
    const std::uint32_t threadCount = static_cast<std::uint32_t>(threads_.size());
    const std::uint32_t startIndex =
        nextThread_.fetch_add(1, std::memory_order_relaxed) % threadCount;

    // First try to enqueue to one of the threads without blocking.
    for (std::uint32_t i = 0; i < threadCount; ++i) {
      const auto index = (startIndex + i) < threadCount
          ? (startIndex + i)
          : (startIndex + i - threadCount);
      if (threadStates_[index].try_push(task)) {
        return;
      }
    }

    // Otherwise, do a blocking enqueue on the selected thread.
    threadStates_[startIndex].push(task);
  }

  inline task_base* static_thread_pool::thread_state::try_pop() {
    std::unique_lock lk{mut_, std::try_to_lock};
    if (!lk || queue_.empty()) {
      return nullptr;
    }
    return queue_.pop_front();
  }

  inline task_base* static_thread_pool::thread_state::pop() {
    std::unique_lock lk{mut_};
    while (queue_.empty()) {
      if (stopRequested_) {
        return nullptr;
      }
      cv_.wait(lk);
    }
    return queue_.pop_front();
  }

  inline bool static_thread_pool::thread_state::try_push(task_base* task) {
    std::unique_lock lk{mut_, std::try_to_lock};
    if (!lk) {
      return false;
    }
    const bool wasEmpty = queue_.empty();
    queue_.push_back(task);
    if (wasEmpty) {
      cv_.notify_one();
    }
    return true;
  }

  inline void static_thread_pool::thread_state::push(task_base* task) {
    std::lock_guard lk{mut_};
    const bool wasEmpty = queue_.empty();
    queue_.push_back(task);
    if (wasEmpty) {
      cv_.notify_one();
    }
  }

  inline void static_thread_pool::thread_state::request_stop() {
    std::lock_guard lk{mut_};
    stopRequested_ = true;
    cv_.notify_one();
  }
} // namespace example
