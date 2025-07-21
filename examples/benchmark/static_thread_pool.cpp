/*
 * Copyright (c) 2023 Maikel Nadolski
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
#include "./common.hpp"
#include <exec/static_thread_pool.hpp>

struct RunThread {
  void operator()(
    exec::static_thread_pool& pool,
    std::size_t total_scheds,
    std::size_t tid,
    std::barrier<>& barrier,
#ifndef STDEXEC_NO_MONOTONIC_BUFFER_RESOURCE
    std::span<char> buffer,
#endif
    std::atomic<bool>& stop,
    exec::numa_policy numa) {
    int numa_node = numa.thread_index_to_node(tid);
    numa.bind_to_node(numa_node);
    exec::nodemask mask{};
    mask.set(static_cast<std::size_t>(numa_node));
    auto scheduler = pool.get_constrained_scheduler(&mask);
    std::mutex mut;
    std::condition_variable cv;
    while (true) {
      barrier.arrive_and_wait();
      if (stop.load()) {
        break;
      }
#ifndef STDEXEC_NO_MONOTONIC_BUFFER_RESOURCE
      pmr::monotonic_buffer_resource resource{
        buffer.data(), buffer.size(), pmr::null_memory_resource()};
      pmr::polymorphic_allocator<char> alloc(&resource);
      auto [start, end] = exec::_pool_::even_share(total_scheds, tid, pool.available_parallelism());
      std::size_t scheds = end - start;
      std::atomic<std::size_t> counter{scheds};
      auto env = stdexec::prop{stdexec::get_allocator, alloc};
      while (scheds) {
        stdexec::start_detached(
          stdexec::schedule(scheduler) | stdexec::then([&] {
            auto prev = counter.fetch_sub(1);
            if (prev == 1) {
              std::lock_guard lock{mut};
              cv.notify_one();
            }
          }),
          env);
        --scheds;
      }
#else
      auto [start, end] = exec::_pool_::even_share(total_scheds, tid, pool.available_parallelism());
      std::size_t scheds = end - start;
      std::atomic<std::size_t> counter{scheds};
      while (scheds) {
        stdexec::start_detached(stdexec::schedule(scheduler) | stdexec::then([&] {
                                  auto prev = counter.fetch_sub(1);
                                  if (prev == 1) {
                                    std::lock_guard lock{mut};
                                    cv.notify_one();
                                  }
                                }));
        --scheds;
      }
#endif
      std::unique_lock lock{mut};
      cv.wait(lock, [&] { return counter.load() == 0; });
      lock.unlock();
      barrier.arrive_and_wait();
    }
  }
};

struct my_numa_distribution : public exec::default_numa_policy {
  [[nodiscard]]
  auto thread_index_to_node(std::size_t index) const noexcept -> int {
    return exec::default_numa_policy::thread_index_to_node(2 * index);
  }
};

auto main(int argc, char** argv) -> int {
  my_numa_distribution numa{};
  my_main<exec::static_thread_pool, RunThread>(argc, argv, numa);
}