/*
 * Copyright (c) 2026 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdexec_relacy.hpp>

#include <atomic>

struct task_node
{
  task_node* next_ = nullptr;
};

struct remote_queue
{
  remote_queue*           next_ = nullptr;
  std::atomic<task_node*> head_{nullptr};
};

struct static_thread_pool_remote_poll : rl::test_suite<static_thread_pool_remote_poll, 3>
{
  static constexpr int running  = 0;
  static constexpr int sleeping = 1;
  static constexpr int notified = 2;

  enum class remote_poll_mode
  {
    speculative,
    before_sleep
  };

  struct poll_result
  {
    bool any    = false;
    bool second = false;
  };

  std::atomic<int>           state_{running};
  std::atomic<remote_queue*> remote_head_{nullptr};
  std::atomic<bool>          first_notification_published_{false};
  remote_queue               first_queue_{};
  remote_queue               second_queue_{};
  task_node                  first_task_{};
  task_node                  first_extra_task_{};
  task_node                  second_task_{};
  bool                       worker_would_sleep_ = false;

  void before()
  {
    state_.store(running, std::memory_order_relaxed);
    remote_head_.store(nullptr, std::memory_order_relaxed);
    first_notification_published_.store(false, std::memory_order_relaxed);
    first_queue_.next_  = nullptr;
    second_queue_.next_ = nullptr;
    first_queue_.head_.store(nullptr, std::memory_order_relaxed);
    second_queue_.head_.store(nullptr, std::memory_order_relaxed);
    first_task_.next_       = nullptr;
    first_extra_task_.next_ = nullptr;
    second_task_.next_      = nullptr;
    worker_would_sleep_     = false;
  }

  void publish_remote_queue(remote_queue& queue)
  {
    auto* old_head = remote_head_.load(std::memory_order_acquire);
    do
    {
      queue.next_ = old_head;
    }
    while (!remote_head_.compare_exchange_weak(old_head,
                                               &queue,
                                               std::memory_order_acq_rel,
                                               std::memory_order_acquire));
  }

  auto push(remote_queue& queue, task_node& task) -> bool
  {
    auto* old_head = queue.head_.load(std::memory_order_relaxed);
    do
    {
      task.next_ = old_head;
    }
    while (!queue.head_.compare_exchange_weak(old_head,
                                              &task,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire));
    return old_head == nullptr;
  }

  void notify()
  {
    state_.exchange(notified, std::memory_order_release);
  }

  void enqueue(remote_queue& queue, task_node& task)
  {
    bool const was_empty = push(queue, task);
    if (was_empty)
    {
      notify();
    }
  }

  auto drain(remote_queue& queue) -> task_node*
  {
    auto* old_head = queue.head_.load(std::memory_order_relaxed);
    while (!queue.head_.compare_exchange_weak(old_head,
                                              nullptr,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire))
    {
    }
    return old_head;
  }

  auto poll_remote(remote_poll_mode mode) -> poll_result
  {
    poll_result result{};
    auto*       queue = remote_head_.load(std::memory_order_acquire);
    while (queue != nullptr)
    {
      if (mode == remote_poll_mode::before_sleep
          || queue->head_.load(std::memory_order_relaxed) != nullptr)
      {
        for (auto* task = drain(*queue); task != nullptr; task = task->next_)
        {
          result.any    = true;
          result.second = result.second || task == &second_task_;
        }
      }
      queue = queue->next_;
    }
    return result;
  }

  void worker_poll()
  {
    auto result = poll_remote(remote_poll_mode::speculative);
    if (result.second)
    {
      return;
    }
    if (result.any)
    {
      result = poll_remote(remote_poll_mode::speculative);
      if (result.second)
      {
        return;
      }
    }

    int expected = running;
    if (!state_.compare_exchange_weak(expected,
                                      sleeping,
                                      std::memory_order_relaxed,
                                      std::memory_order_relaxed))
    {
      state_.exchange(running, std::memory_order_acquire);
      result = poll_remote(remote_poll_mode::speculative);
      if (result.second)
      {
        return;
      }
      if (result.any)
      {
        result = poll_remote(remote_poll_mode::speculative);
        if (result.second)
        {
          return;
        }
      }

      expected = running;
      if (!state_.compare_exchange_weak(expected,
                                        sleeping,
                                        std::memory_order_relaxed,
                                        std::memory_order_relaxed))
      {
        return;
      }
    }

    result = poll_remote(remote_poll_mode::before_sleep);
    if (result.any)
    {
      int expected_sleeping = sleeping;
      state_.compare_exchange_strong(expected_sleeping,
                                     running,
                                     std::memory_order_relaxed,
                                     std::memory_order_relaxed);
      return;
    }

    worker_would_sleep_ = true;
  }

  void thread(unsigned thread_id)
  {
    if (thread_id == 0)
    {
      publish_remote_queue(first_queue_);
      enqueue(first_queue_, first_task_);
      enqueue(first_queue_, first_extra_task_);
      first_notification_published_.store(true, std::memory_order_release);
    }
    else if (thread_id == 1)
    {
      while (!first_notification_published_.load(std::memory_order_acquire))
      {
      }
      publish_remote_queue(second_queue_);
      enqueue(second_queue_, second_task_);
    }
    else
    {
      worker_poll();
    }
  }

  void after()
  {
    if (worker_would_sleep_)
    {
      RL_ASSERT(state_.load(std::memory_order_acquire) != sleeping);
    }
  }
};

auto main() -> int
{
  rl::test_params p;
  p.iteration_count       = 50000;
  p.execution_depth_limit = 10000;
  p.search_type           = rl::random_scheduler_type;
  return rl::simulate<static_thread_pool_remote_poll>(p) ? 0 : 1;
}
