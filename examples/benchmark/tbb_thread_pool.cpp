#include "./common.hpp"
#include <tbbexec/tbb_thread_pool.hpp>

struct RunThread {
  void operator()(
    tbbexec::tbb_thread_pool& pool,
    std::size_t total_scheds,
    std::size_t tid,
    std::barrier<>& barrier,
#ifndef STDEXEC_NO_MONOTONIC_BUFFER_RESOURCE
    std::span<char> buffer,
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
#ifndef STDEXEC_NO_MONOTONIC_BUFFER_RESOURCE
      pmr::monotonic_buffer_resource resource{
        buffer.data(), buffer.size(), pmr::null_memory_resource()};
      pmr::polymorphic_allocator<char> alloc(&resource);
      auto [start, end] = exec::even_share(total_scheds, tid, pool.available_parallelism() - 1);
      std::size_t scheds = end - start;
      std::atomic<std::size_t> counter{scheds};
      auto env = exec::make_env(exec::with(stdexec::get_allocator, alloc));
      while (scheds) {
        stdexec::start_detached(       //
          stdexec::schedule(scheduler) //
            | stdexec::then([&] {
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
      auto [start, end] = exec::even_share(total_scheds, tid, pool.available_parallelism() - 1);
      std::size_t scheds = end - start;
      std::atomic<std::size_t> counter{scheds};
      while (scheds) {
        stdexec::start_detached(       //
          stdexec::schedule(scheduler) //
            | stdexec::then([&] {
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

int main(int argc, char** argv)
{
  my_main<tbbexec::tbb_thread_pool, RunThread>(argc, argv);
}