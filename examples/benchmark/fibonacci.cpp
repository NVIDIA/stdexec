/*
 * Copyright (c) 2023 Intel Corporation
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

#include <cstdlib>
#include <iostream>

#include <exec/static_thread_pool.hpp>
#include <execpools/tbb/tbb_thread_pool.hpp>

#include <exec/any_sender_of.hpp>
#include <stdexec/execution.hpp>

auto serial_fib(long n) -> long {
  return n < 2 ? n : serial_fib(n - 1) + serial_fib(n - 2);
}

template <class... Ts>
using any_sender_of =
  exec::any_receiver_ref<stdexec::completion_signatures<Ts...>>::template any_sender<>;

using fib_sender = any_sender_of<stdexec::set_value_t(long)>;

template <typename Scheduler>
struct fib_s {
  using sender_concept = stdexec::sender_t;
  using completion_signatures = stdexec::completion_signatures<stdexec::set_value_t(long)>;

  long cutoff;
  long n;
  Scheduler sched;

  template <class Receiver>
  struct operation {
    Receiver rcvr_;
    long cutoff;
    long n;
    Scheduler sched;

    void start() & noexcept {
      if (n < cutoff) {
        stdexec::set_value(static_cast<Receiver&&>(rcvr_), serial_fib(n));
      } else {
        auto mkchild = [&](long n) {
          return stdexec::starts_on(sched, fib_sender(fib_s{cutoff, n, sched}));
        };

        stdexec::start_detached(
          stdexec::when_all(mkchild(n - 1), mkchild(n - 2))
          | stdexec::then([rcvr = static_cast<Receiver&&>(rcvr_)](long a, long b) mutable {
              stdexec::set_value(static_cast<Receiver&&>(rcvr), a + b);
            }));
      }
    }
  };

  template <stdexec::receiver_of<completion_signatures> Receiver>
  [[nodiscard]]
  auto connect(Receiver rcvr) const -> operation<Receiver> {
    return {static_cast<Receiver&&>(rcvr), cutoff, n, sched};
  }
};

template <class Scheduler>
fib_s(long cutoff, long n, Scheduler sched) -> fib_s<Scheduler>;

template <typename duration, typename F>
auto measure(F&& f) {
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  f();
  return std::chrono::duration_cast<duration>(std::chrono::steady_clock::now() - start).count();
}

auto main(int argc, char** argv) -> int {
  if (argc < 5) {
    std::cerr << "Usage: example.benchmark.fibonacci cutoff n nruns {tbb|static}" << std::endl;
    return -1;
  }

  // skip 'warmup' iterations for performance measurements
  static constexpr size_t warmup = 1;

  long cutoff = std::strtol(argv[1], nullptr, 10);
  long n = std::strtol(argv[2], nullptr, 10);
  std::size_t nruns = std::strtoul(argv[3], nullptr, 10);

  if (nruns <= warmup) {
    std::cerr << "nruns should be >= " << warmup << std::endl;
    return -1;
  }

  std::variant<execpools::tbb_thread_pool, exec::static_thread_pool> pool;

  if (argv[4] == std::string_view("tbb")) {
    pool.emplace<execpools::tbb_thread_pool>(static_cast<int>(std::thread::hardware_concurrency()));
  } else {
    pool.emplace<exec::static_thread_pool>(
      std::thread::hardware_concurrency(), exec::bwos_params{}, exec::get_numa_policy());
  }

  std::vector<unsigned long> times;
  long result;
  for (unsigned long i = 0; i < nruns; ++i) {
    auto snd = std::visit(
      [&](auto&& pool) { return fib_sender(fib_s{cutoff, n, pool.get_scheduler()}); }, pool);

    auto time = measure<std::chrono::milliseconds>(
      [&] { std::tie(result) = stdexec::sync_wait(std::move(snd)).value(); });
    times.push_back(static_cast<unsigned int>(time));
  }

  std::cout << "Avg time: "
            << (std::accumulate(times.begin() + warmup, times.end(), 0u) / (times.size() - warmup))
            << "ms. Result: " << result << std::endl;
}
