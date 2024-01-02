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

#include <iostream>

#include <tbbexec/tbb_thread_pool.hpp>
#include <exec/static_thread_pool.hpp>

#include <exec/any_sender_of.hpp>
#include <stdexec/execution.hpp>

long serial_fib(int n) {
  return n < 2 ? n : serial_fib(n - 1) + serial_fib(n - 2);
}

template <class... Ts>
using any_sender_of =
  typename exec::any_receiver_ref<stdexec::completion_signatures<Ts...>>::template any_sender<>;

using fib_sender = any_sender_of<
  stdexec::set_value_t(long),
  stdexec::set_error_t(std::exception_ptr),
  stdexec::set_stopped_t()>;

template <typename Scheduler>
struct fib_s {
  using sender_concept = stdexec::sender_t;
  using completion_signatures = stdexec::completion_signatures<
    stdexec::set_value_t(long),
    stdexec::set_error_t(std::exception_ptr),
    stdexec::set_stopped_t()>;

  long cutoff;
  long n;
  Scheduler sched;

  template <class Receiver>
  struct operation {
    Receiver rcvr_;
    long cutoff;
    long n;
    Scheduler sched;

    friend void tag_invoke(stdexec::start_t, operation& self) noexcept {
      if (self.n < self.cutoff) {
        stdexec::set_value((Receiver &&) self.rcvr_, serial_fib(self.n));
      } else {
        fib_s<Scheduler> child1{self.cutoff, self.n - 1, self.sched};
        fib_s<Scheduler> child2{self.cutoff, self.n - 2, self.sched};

        stdexec::start_detached(stdexec::on(
          self.sched,
          stdexec::when_all(fib_sender(child1), fib_sender(child2))
            | stdexec::then([rcvr = (Receiver &&) self.rcvr_](long a, long b) {
                stdexec::set_value((Receiver &&) rcvr, a + b);
              })));
      }
    }
  };

  template <stdexec::receiver_of<completion_signatures> Receiver>
  friend operation<Receiver> tag_invoke(stdexec::connect_t, fib_s self, Receiver rcvr) {
    return {(Receiver &&) rcvr, self.cutoff, self.n, self.sched};
  }
};

template <typename duration, typename F>
auto measure(F&& f) {
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  f();
  return std::chrono::duration_cast<duration>(std::chrono::steady_clock::now() - start).count();
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "Usage: example.benchmark.fibonacci cutoff n nruns {tbb|static}" << std::endl;
    return -1;
  }

  // skip 'warmup' iterations for performance measurements
  static constexpr size_t warmup = 1;

  int cutoff = std::atoi(argv[1]);
  int n = std::atoi(argv[2]);
  int nruns = std::atoi(argv[3]);

  if (nruns <= warmup) {
    std::cerr << "nruns should be >= " << warmup << std::endl;
    return -1;
  }

  std::variant<tbbexec::tbb_thread_pool, exec::static_thread_pool> pool;

  if (argv[4] == std::string_view("tbb")) {
    pool.emplace<tbbexec::tbb_thread_pool>(std::thread::hardware_concurrency());
  } else {
    pool.emplace<exec::static_thread_pool>(
      std::thread::hardware_concurrency(), exec::bwos_params{}, exec::get_numa_policy());
  }

  std::vector<unsigned long> times;
  long result;
  for (unsigned long i = 0; i < nruns; ++i) {
    auto snd = std::visit(
      [&](auto&& pool) {
        return fib_sender(fib_s{cutoff, n, pool.get_scheduler()});
      },
      pool);

    auto time = measure<std::chrono::milliseconds>([&] {
      std::tie(result) = stdexec::sync_wait(std::move(snd)).value();
    });
    times.push_back(time);
  }

  std::cout << "Avg time: "
            << (std::accumulate(times.begin() + warmup, times.end(), 0) / times.size())
            << "ms. Result: " << result << std::endl;
}
