#include <exec/static_thread_pool.hpp>
#include <iostream>
#include <iomanip>

std::size_t even_share(std::size_t n, std::size_t m, std::size_t id) {
  std::size_t q = n / m;
  std::size_t r = n % m;
  if (id < r) {
    return q + std::min(r, std::size_t(1));
  }
  return q;
}

struct RunThread {
  void operator()(exec::static_thread_pool& pool, std::size_t total_scheds, std::size_t tid) {
    std::size_t scheds = even_share(total_scheds, pool.available_parallelism(), tid);
    auto scheduler = pool.get_scheduler();
    while (scheds) {
      stdexec::start_detached(stdexec::schedule(scheduler));
      --scheds;
    }
  }
};

int main(int argc, char** argv) {
  int nthreads = std::thread::hardware_concurrency();
  if (argc > 1) {
    nthreads = std::atoi(argv[1]);
  }
  auto start = std::chrono::steady_clock::now();
  std::size_t total_scheds = 100'000'000;
  {
    exec::static_thread_pool pool(nthreads);
    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < pool.available_parallelism(); ++i) {
      threads.emplace_back(RunThread{}, std::ref(pool), total_scheds, i);
    }
    for (auto& thread: threads) {
      thread.join();
    }
    pool.request_stop();
  }
  auto end = std::chrono::steady_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  auto dur_dbl = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  std::cout << "Total time: " << dur.count() << "ns, " << dur.count() / 1'000'000 << "ms\n";
  std::cout << "Total scheds: " << total_scheds << "\n";
  std::cout << "Total threads: " << nthreads << "\n";
  std::cout
    << "OP/s: "
    << std::setprecision(3) << total_scheds / dur_dbl.count()
    << "\n";
}