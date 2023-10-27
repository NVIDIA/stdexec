#include <exec/static_thread_pool.hpp>
#include <exec/env.hpp>

#include <iostream>
#include <iomanip>
#include <memory_resource>
#include <barrier>
#include <ranges>
#include <span>

struct barrier_receiver {
  using is_receiver = void;

  std::atomic<std::size_t>* counter;
  std::mutex* mut;
  std::condition_variable* cv;

  template <
    stdexec::__one_of<stdexec::set_value_t, stdexec::set_stopped_t, stdexec::set_error_t> Tag,
    class... E>
  friend void tag_invoke(Tag, barrier_receiver rcvr, E&&...) noexcept {
    if (rcvr.counter->fetch_sub(1, std::memory_order_relaxed) == 1) {
      {
        std::lock_guard lock{*rcvr.mut};
      }
      rcvr.cv->notify_one();
    }
  }

  template <stdexec::same_as<stdexec::get_env_t> GetEnv>
  friend stdexec::empty_env tag_invoke(GetEnv, barrier_receiver) noexcept {
    return stdexec::empty_env{};
  }
};

struct RunThread {
  void operator()(
    exec::static_thread_pool& pool,
    std::size_t total_scheds,
    std::size_t tid,
    std::barrier<>& barrier,
    std::span<char> buffer,
    std::atomic<bool>& stop) {
    auto scheduler = pool.get_scheduler();
    using schedule_sender = decltype(stdexec::schedule(scheduler));
    using op_t = stdexec::connect_result_t<schedule_sender, barrier_receiver>;
    while (true) {
      barrier.arrive_and_wait();
      if (stop.load()) {
        break;
      }
      std::pmr::monotonic_buffer_resource resource{
        buffer.data(), buffer.size(), std::pmr::null_memory_resource()};
      std::pmr::polymorphic_allocator<std::optional<op_t>> alloc(&resource);
      auto [start, end] = exec::even_share(total_scheds, tid, pool.available_parallelism());
      std::size_t scheds = end - start;
      std::pmr::vector<std::optional<op_t>> ops(scheds, alloc);
      std::mutex mut;
      std::condition_variable cv;
      std::atomic<std::size_t> counter{scheds};
      for (std::size_t i = 0; i < scheds; ++i) {
        ops[i].emplace(stdexec::__conv{[&] {
          return stdexec::connect(
            stdexec::schedule(scheduler), barrier_receiver{&counter, &mut, &cv});
        }});
      }
      auto derefs = std::views::all(ops)
                  | std::views::transform([](auto& op) -> decltype(auto) { return *op; });
      pool.bulk_enqueue(*pool.get_remote_queue(), derefs.begin(), derefs.end());
      std::unique_lock lock{mut};
      cv.wait(lock, [&] { return counter.load(std::memory_order_relaxed) == 0; });
      barrier.arrive_and_wait();
    }
  }
};

struct statistics {
  std::chrono::milliseconds total_time_ms;
  std::size_t ops_per_sec;
};

statistics compute_perf(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end, std::size_t total_scheds) {
  auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  auto dur_ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur);
  auto dur_dbl = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  std::size_t ops_per_sec = static_cast<std::size_t>(total_scheds / dur_dbl.count());
  return {dur_ms, ops_per_sec};
}

int main(int argc, char** argv) {
  int nthreads = std::thread::hardware_concurrency();
  if (argc > 1) {
    nthreads = std::atoi(argv[1]);
  }
  std::size_t total_scheds = 100'000'000;
  std::vector<std::unique_ptr<char[]>> buffers(nthreads);
  std::vector<std::optional<std::pmr::monotonic_buffer_resource>> resource(nthreads);
  exec::static_thread_pool pool(nthreads);
  std::barrier<> barrier(nthreads + 1);
  std::vector<std::thread> threads;
  std::atomic<bool> stop{false};
  std::size_t buffer_size = 1000 << 20;
  for (auto& buf: buffers) {
    buf = std::make_unique_for_overwrite<char[]>(buffer_size);
  }
  for (std::size_t i = 0; i < pool.available_parallelism(); ++i) {
    threads.emplace_back(
      RunThread{},
      std::ref(pool),
      total_scheds,
      i,
      std::ref(barrier),
      std::span<char>{buffers[i].get(), buffer_size},
      std::ref(stop));
  }
  std::size_t nRuns = 100;
  std::vector<std::chrono::steady_clock::time_point> starts(nRuns);
  std::vector<std::chrono::steady_clock::time_point> ends(nRuns);
  for (std::size_t i = 0; i < 2; ++i) {
    barrier.arrive_and_wait();
    starts[i] = std::chrono::steady_clock::now();
    barrier.arrive_and_wait();
    ends[i] = std::chrono::steady_clock::now();
    auto [dur_ms, ops_per_sec] = compute_perf(starts[i], ends[i], total_scheds);
    std::cout << dur_ms.count() << " " << std::setprecision(3) << ((double)ops_per_sec) << "\n";
  }
  stop = true;
  barrier.arrive_and_wait();
  for (auto& thread: threads) {
    thread.join();
  }
}