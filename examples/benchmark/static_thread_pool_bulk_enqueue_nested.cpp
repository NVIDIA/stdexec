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

#if STDEXEC_HAS_STD_RANGES()
#  include <exec/sequence/ignore_all_values.hpp>
#  include <ranges>

struct RunThread {
  void operator()(
    exec::static_thread_pool& pool,
    std::size_t total_scheds,
    std::size_t tid,
    std::barrier<>& barrier,
#  ifndef STDEXEC_NO_MONOTONIC_BUFFER_RESOURCE
    std::span<char> buffer,
#  endif
    std::atomic<bool>& stop,
    exec::numa_policy numa) {
    int numa_node = numa.thread_index_to_node(tid);
    numa.bind_to_node(numa_node);
    auto scheduler = pool.get_scheduler_on_thread(tid);
    while (true) {
      barrier.arrive_and_wait();
      if (stop.load()) {
        break;
      }
#  ifndef STDEXEC_NO_MONOTONIC_BUFFER_RESOURCE
      pmr::monotonic_buffer_resource rsrc{buffer.data(), buffer.size()};
      pmr::polymorphic_allocator<char> alloc{&rsrc};
      auto env = stdexec::prop{stdexec::get_allocator, alloc};
      auto [start, end] = exec::_pool_::even_share(total_scheds, tid, pool.available_parallelism());
      auto iterate = exec::iterate(std::views::iota(start, end)) | exec::ignore_all_values()
                   | stdexec::write_env(env);
#  else
      auto [start, end] = exec::_pool_::even_share(total_scheds, tid, pool.available_parallelism());
      auto iterate = exec::iterate(std::views::iota(start, end)) | exec::ignore_all_values();
#  endif
      stdexec::sync_wait(stdexec::starts_on(scheduler, iterate));
      barrier.arrive_and_wait();
    }
  }
};

auto main(int argc, char** argv) -> int {
  my_main<exec::static_thread_pool, RunThread>(argc, argv);
}
#else
int main() {
}
#endif