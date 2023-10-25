#include <tbbexec/tbb_thread_pool.hpp>
#include <iostream>
#include <iomanip>
#include <barrier>

std::size_t even_share(std::size_t n, std::size_t m, std::size_t id) {
  std::size_t q = n / m;
  std::size_t r = n % m;
  if (id < r) {
    return q + std::min(r, std::size_t(1));
  }
  return q;
}

struct RunThread {
  void operator()(tbbexec::tbb_thread_pool& pool, std::size_t total_scheds, std::size_t tid, std::barrier<>& barrier) {
    barrier.arrive_and_wait();
    std::size_t scheds = even_share(total_scheds, pool.available_parallelism(), tid);
    auto scheduler = pool.get_scheduler();
    while (scheds) {
      stdexec::start_detached(stdexec::schedule(scheduler));
      --scheds;
    }
  }
};

thread_local char* alloc_watermark = nullptr;
thread_local std::size_t alloc_watermark_size = 0;

void* operator new(std::size_t count) {
  if (alloc_watermark == nullptr) {
    alloc_watermark_size = 1000 << 20;
    alloc_watermark = static_cast<char*>(std::malloc(alloc_watermark_size));
    if (alloc_watermark == nullptr) {
      throw std::bad_alloc();
    }
  }
  if (alloc_watermark_size < count) {
    throw std::bad_alloc();
  }
  char* ptr = std::exchange(alloc_watermark, alloc_watermark + count);
  alloc_watermark_size -= count;
  return ptr;
}

void* operator new[](std::size_t count) {
  if (alloc_watermark == nullptr) {
    alloc_watermark_size = 1000 << 20;
    alloc_watermark = static_cast<char*>(std::malloc(alloc_watermark_size));
    if (alloc_watermark == nullptr) {
      throw std::bad_alloc();
    }
  }
  if (alloc_watermark_size < count) {
    throw std::bad_alloc();
  }
  char* ptr = std::exchange(alloc_watermark, alloc_watermark + count);
  alloc_watermark_size -= count;
  return ptr;
}

void operator delete(void* ptr) noexcept {
}

void operator delete(void* ptr, std::size_t size) noexcept {
}

void operator delete[](void* ptr) noexcept {
}

void operator delete[](void* ptr, std::size_t size) noexcept {
}

int main(int argc, char** argv) {
  int nthreads = std::thread::hardware_concurrency();
  if (argc > 1) {
    nthreads = std::atoi(argv[1]);
  }
  std::chrono::steady_clock::time_point start{};
  std::size_t total_scheds = 100'000'000;
  {
    tbbexec::tbb_thread_pool pool(nthreads);
    std::barrier<> barrier(nthreads + 1);
    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < pool.available_parallelism(); ++i) {
      threads.emplace_back(RunThread{}, std::ref(pool), total_scheds, i, std::ref(barrier));
    }
    barrier.arrive_and_wait();
    start = std::chrono::steady_clock::now();
    for (auto& thread: threads) {
      thread.join();
    }
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