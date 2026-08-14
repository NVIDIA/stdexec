/*
 * Copyright (c) 2026 NVIDIA Corporation
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

#include <stdexec_relacy.hpp>

#include <atomic>

struct remote_queue_node
{
  remote_queue_node* next_ = nullptr;
};

struct static_thread_pool_remote_poll : rl::test_suite<static_thread_pool_remote_poll, 3>
{
  static constexpr int running  = 0;
  static constexpr int sleeping = 1;
  static constexpr int notified = 2;

  std::atomic<int>                state_{running};
  std::atomic<remote_queue_node*> head_{nullptr};
  std::atomic<bool>               first_notification_published_{false};
  remote_queue_node               first_{};
  remote_queue_node               second_{};
  bool                            worker_would_sleep_ = false;

  void before()
  {
    state_.store(running, std::memory_order_relaxed);
    head_.store(nullptr, std::memory_order_relaxed);
    first_notification_published_.store(false, std::memory_order_relaxed);
    first_.next_        = nullptr;
    second_.next_       = nullptr;
    worker_would_sleep_ = false;
  }

  void publish(remote_queue_node& node)
  {
    auto* old_head = head_.load(std::memory_order_relaxed);
    do
    {
      node.next_ = old_head;
    }
    while (!head_.compare_exchange_weak(old_head, &node, std::memory_order_acq_rel));
  }

  void notify()
  {
    state_.exchange(notified, std::memory_order_release);
  }

  auto sees_second_node() -> bool
  {
    for (auto* node = head_.load(std::memory_order_acquire); node != nullptr; node = node->next_)
    {
      if (node == &second_)
      {
        return true;
      }
    }
    return false;
  }

  void worker_poll()
  {
    if (sees_second_node())
    {
      return;
    }

    int expected = running;
    if (!state_.compare_exchange_weak(expected,
                                      sleeping,
                                      std::memory_order_relaxed,
                                      std::memory_order_relaxed))
    {
      // This acquire RMW must consume a preceding notification or leave a
      // later notification visible to the next sleep transition.
      state_.exchange(running, std::memory_order_acquire);
      if (sees_second_node())
      {
        return;
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

    if (sees_second_node())
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
      publish(first_);
      notify();
      first_notification_published_.store(true, std::memory_order_release);
    }
    else if (thread_id == 1)
    {
      while (!first_notification_published_.load(std::memory_order_acquire))
      {
      }
      publish(second_);
      notify();
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
