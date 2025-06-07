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

#include <execpools/tbb/tbb_thread_pool.hpp>
#include <tbb/task_group.h>

struct RunThread {
  void operator()(
    execpools::tbb_thread_pool& pool,
    std::size_t total_scheds,
    std::size_t tid,
    std::barrier<>& barrier,
#ifndef STDEXEC_NO_MONOTONIC_BUFFER_RESOURCE
    [[maybe_unused]] std::span<char> buffer,
#endif
    std::atomic<bool>& stop,
    exec::numa_policy numa) {
    int numa_node = numa.thread_index_to_node(tid);
    numa.bind_to_node(numa_node);
    auto scheduler = pool.get_scheduler();
    while (true) {
      barrier.arrive_and_wait();
      if (stop.load()) {
        break;
      }
      auto [start, end] = exec::_pool_::even_share(total_scheds, tid, pool.available_parallelism());
      std::size_t scheds = end - start;
      tbb::task_group tg{};
      stdexec::sync_wait(stdexec::schedule(scheduler) | stdexec::then([&] {
                           for (std::size_t i = 0; i < scheds; ++i) {
                             tg.run([&] {
                               // empty
                             });
                           }
                         }));
      tg.wait();
      barrier.arrive_and_wait();
    }
  }
};

auto main(int argc, char** argv) -> int {
  my_main<execpools::tbb_thread_pool, RunThread>(argc, argv);
}
