#include "./common.hpp"
#include <exec/static_thread_pool.hpp>

#if STDEXEC_HAS_STD_RANGES()
#include <ranges>

struct RunThread {
  void operator()(
    exec::static_thread_pool& pool,
    std::size_t total_scheds,
    std::size_t tid,
    std::barrier<>& barrier,
#ifndef STDEXEC_NO_MONOTONIC_BUFFER_RESOURCE
    std::span<char> buffer,
#endif
    std::atomic<bool>& stop) {
    auto scheduler = pool.get_scheduler();
    while (true) {
      barrier.arrive_and_wait();
      if (stop.load()) {
        break;
      }
#ifndef STDEXEC_NO_MONOTONIC_BUFFER_RESOURCE
      pmr::monotonic_buffer_resource rsrc{buffer.data(), buffer.size()};  
      pmr::polymorphic_allocator<char> alloc{&rsrc};
      auto env = exec::make_env(exec::with(stdexec::get_allocator, alloc));
      auto [start, end] = exec::even_share(total_scheds, tid, pool.available_parallelism());
      auto iterate = exec::schedule_all(pool, std::views::iota(start, end)) 
                   | exec::ignore_all_values()
                   | exec::write(env);
#else
      auto [start, end] = exec::even_share(total_scheds, tid, pool.available_parallelism());
      auto iterate = exec::schedule_all(pool, std::views::iota(start, end)) 
                   | exec::ignore_all_values();
#endif
      stdexec::sync_wait(stdexec::on(scheduler, iterate));
      barrier.arrive_and_wait();
    }
  }
};

int main(int argc, char** argv) {
  my_main<exec::static_thread_pool, RunThread>(argc, argv);
}
#else
int main() {}
#endif