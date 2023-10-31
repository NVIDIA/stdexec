#include "./common.hpp"

#include <tbbexec/tbb_thread_pool.hpp>
#include <tbb/task_group.h>

struct RunThread {
  void operator()(
    tbbexec::tbb_thread_pool& pool,
    std::size_t total_scheds,
    std::size_t tid,
    std::barrier<>& barrier,
#ifndef STDEXEC_NO_MONOTONIC_BUFFER_RESOURCE
    [[maybe_unused]] std::span<char> buffer,
#endif
    std::atomic<bool>& stop) {
    auto scheduler = pool.get_scheduler();
    std::mutex mut;
    std::condition_variable cv;
    while (true) {
      barrier.arrive_and_wait();
      if (stop.load()) {
        break;
      }
      auto [start, end] = exec::even_share(total_scheds, tid, pool.available_parallelism());
      std::size_t scheds = end - start;
      tbb::task_group tg{};
      stdexec::sync_wait(       //
        stdexec::schedule(scheduler) //
        | stdexec::then([&] {
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

int main(int argc, char** argv)
{
  my_main<tbbexec::tbb_thread_pool, RunThread>(argc, argv);
}