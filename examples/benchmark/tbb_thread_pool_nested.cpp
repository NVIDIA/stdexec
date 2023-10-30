#include <tbbexec/tbb_thread_pool.hpp>
#include <tbb/task_group.h>
#include <exec/env.hpp>

#include <iostream>
#include <iomanip>
#include <barrier>
#include <span>
#include <cmath>

#if __has_include(<memory_resource>)
#include <memory_resource>
namespace pmr = std::pmr;
#elif __has_include(<experimental/memory_resource>)
#include <experimental/memory_resource>
namespace pmr = std::experimental::pmr;
#endif

struct RunThread {
  void operator()(
    tbbexec::tbb_thread_pool& pool,
    std::size_t total_scheds,
    std::size_t tid,
    std::barrier<>& barrier,
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

struct statistics {
  std::chrono::milliseconds total_time_ms;
  double ops_per_sec;
};

statistics compute_perf(
  std::chrono::steady_clock::time_point start,
  std::chrono::steady_clock::time_point end,
  std::size_t total_scheds) {
  auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  auto dur_ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur);
  auto dur_dbl = std::chrono::duration_cast<std::chrono::duration<double>>(dur);
  double ops_per_sec = total_scheds / dur_dbl.count();
  return {dur_ms, ops_per_sec};
}

struct statistics_all {
  std::chrono::milliseconds total_time_ms;
  double ops_per_sec;
  double average;
  double max;
  double min;
  double stddev;
};

statistics_all compute_perf(
  std::span<const std::chrono::steady_clock::time_point> start,
  std::span<const std::chrono::steady_clock::time_point> end,
  std::size_t i0,
  std::size_t i,
  std::size_t total_scheds) {
  double average = 0.0;
  double max = 0.0;
  double min = std::numeric_limits<double>::max();
  for (std::size_t j = i0; j <= i; ++j) {
    auto stats = compute_perf(start[j], end[j], total_scheds);
    average += stats.ops_per_sec / (i + 1 - i0);
    max = std::max(max, stats.ops_per_sec);
    min = std::min(min, stats.ops_per_sec);
  }
  // compute variant
  double variance = 0.0;
  for (std::size_t j = i0; j <= i; ++j) {
    auto stats = compute_perf(start[j], end[j], total_scheds);
    variance += (stats.ops_per_sec - average) * (stats.ops_per_sec - average);
  }
  variance /= (i + 1 - i0);
  double stddev = std::sqrt(variance);
  auto stats = compute_perf(start[i], end[i], total_scheds);
  statistics_all all{stats.total_time_ms, stats.ops_per_sec, average, max, min, stddev};
  return all;
}

int main(int argc, char** argv) {
  int nthreads = std::thread::hardware_concurrency();
  if (argc > 1) {
    nthreads = std::atoi(argv[1]);
  }
  std::size_t total_scheds = 10'000'000;
  tbbexec::tbb_thread_pool pool(nthreads);
  std::barrier<> barrier(nthreads + 1);
  std::vector<std::thread> threads;
  std::atomic<bool> stop{false};
  for (std::size_t i = 0; i < pool.available_parallelism(); ++i) {
    threads.emplace_back(
      RunThread{},
      std::ref(pool),
      total_scheds,
      i,
      std::ref(barrier),
      std::ref(stop));
  }
  std::size_t nRuns = 100;
  std::size_t warmup = 1;
  std::vector<std::chrono::steady_clock::time_point> starts(nRuns);
  std::vector<std::chrono::steady_clock::time_point> ends(nRuns);
  for (std::size_t i = 0; i < nRuns; ++i) {
    barrier.arrive_and_wait();
    starts[i] = std::chrono::steady_clock::now();
    barrier.arrive_and_wait();
    ends[i] = std::chrono::steady_clock::now();
    if (i < warmup) {
      std::cout << "warmup: skip results\n";
    } else {
      auto [dur_ms, ops_per_sec, avg, max, min, stddev] = compute_perf(starts, ends, warmup, i, total_scheds);
      auto percent = stddev / ops_per_sec * 100;
      std::cout << i + 1 << " " << dur_ms.count() << "ms, throughput: " << std::setprecision(3)
              << ops_per_sec << ", average: " << avg << ", max: " << max << ", min: " << min
              << ", stddev: " << stddev << " (" << percent << "%)\n";
    }
  }
  stop = true;
  barrier.arrive_and_wait();
  for (auto& thread: threads) {
    thread.join();
  }
  auto [dur_ms, ops_per_sec, avg, max, min, stddev] = compute_perf(starts, ends, warmup, nRuns - 1, total_scheds);
  std::cout << avg << " | " << max << " | " << min << " | " << stddev << "\n";
}