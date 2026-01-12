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

#include <linux/version.h>

// Some kernel versions have <linux/io_uring.h> but don't support or don't
// allow user access to some of the necessary system calls.
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 4, 0) && __has_include(<linux/io_uring.h>)

#  include "exec/finally.hpp"
#  include "exec/linux/io_uring_context.hpp"
#  include "exec/scope.hpp"
#  include "exec/single_thread_context.hpp"
#  include "exec/when_any.hpp"

#  include "catch2/catch.hpp"

using namespace STDEXEC;
using namespace exec;
using namespace std::chrono_literals;

namespace {

  // clang-12 does not know jthread yet
  class jthread {
    std::thread thread_;

   public:
    template <typename Callable, typename... Args>
    explicit jthread(Callable&& callable, Args&&... args)
      : thread_(std::forward<Callable>(callable), std::forward<Args>(args)...) {
    }

    jthread(jthread&& other) noexcept
      : thread_(std::move(other.thread_)) {
    }

    auto operator=(jthread&& other) noexcept -> jthread& {
      thread_ = std::move(other.thread_);
      return *this;
    }

    ~jthread() {
      if (thread_.joinable()) {
        thread_.join();
      }
    }

    [[nodiscard]]
    auto get_id() const noexcept -> std::thread::id {
      return thread_.get_id();
    }
  };

  TEST_CASE("io_uring_context - unused context", "[types][io_uring][schedulers]") {
    io_uring_context context;
    CHECK(context.is_running() == false);
  }

  TEST_CASE("io_uring_context Satisfy concepts", "[types][io_uring][schedulers]") {
    STATIC_REQUIRE(timed_scheduler<io_uring_scheduler>);
    STATIC_REQUIRE_FALSE(std::is_move_assignable_v<io_uring_context>);
  }

  TEST_CASE("io_uring_context Schedule runs in io thread", "[types][io_uring][schedulers]") {
    io_uring_context context;
    io_uring_scheduler scheduler = context.get_scheduler();
    jthread io_thread{[&] { context.run_until_stopped(); }};
    {
      scope_guard guard{[&]() noexcept { context.request_stop(); }};
      bool is_called = false;
      sync_wait(schedule(scheduler) | then([&] {
                  CHECK(io_thread.get_id() == std::this_thread::get_id());
                  is_called = true;
                }));
      CHECK(is_called);

      is_called = false;
      sync_wait(schedule_after(scheduler, 1ms) | then([&] {
                  CHECK(io_thread.get_id() == std::this_thread::get_id());
                  is_called = true;
                }));
      CHECK(is_called);
    }
  }

  TEST_CASE(
    "io_uring_context Call io_uring::run_until_empty with start_detached",
    "[types][io_uring][schedulers]") {
    io_uring_context context;
    io_uring_scheduler scheduler = context.get_scheduler();
    bool is_called = false;
    start_detached(schedule(scheduler) | then([&] {
                     CHECK(context.is_running());
                     is_called = true;
                   }));
    context.run_until_empty();
    CHECK(is_called);
    CHECK(!context.is_running());
    CHECK(!context.stop_requested());
  }

  TEST_CASE(
    "io_uring_context Call now(io_uring) is running clock",
    "[types][io_uring][schedulers]") {
    io_uring_context context;
    io_uring_scheduler scheduler = context.get_scheduler();
    auto start = now(scheduler);
    std::this_thread::sleep_for(10ms);
    CHECK(start + 10ms <= now(scheduler));
  }

  TEST_CASE(
    "io_uring_context Call io_uring::run_until_empty with sync_wait",
    "[types][io_uring][schedulers]") {
    io_uring_context context;
    io_uring_scheduler scheduler = context.get_scheduler();
    auto just_run = just() | then([&] { context.run_until_empty(); });
    bool is_called = false;
    sync_wait(when_all(
      schedule_after(scheduler, 500us) | then([&] {
        CHECK(context.is_running());
        is_called = true;
      }),
      just_run));
    CHECK(is_called);
    CHECK(!context.is_running());
    CHECK(!context.stop_requested());
  }

  TEST_CASE(
    "io_uring_context Call io_uring::run with sync_wait and when_any",
    "[types][io_uring][schedulers]") {
    io_uring_context context;
    io_uring_scheduler scheduler = context.get_scheduler();
    bool is_called = false;
    sync_wait(when_any(
      schedule_after(scheduler, 500us) | then([&] {
        CHECK(context.is_running());
        is_called = true;
      }),
      context.run()));
    CHECK(is_called);
    CHECK(!context.is_running());
    CHECK(context.stop_requested());
  }

  TEST_CASE(
    "io_uring_context Call io_uring::run with sync_wait and when_all",
    "[types][io_uring][schedulers]") {
    io_uring_context context;
    io_uring_scheduler scheduler = context.get_scheduler();
    bool is_called = false;
    sync_wait(when_all(
      schedule_after(scheduler, 500us) | then([&] {
        CHECK(context.is_running());
        is_called = true;
      }),
      context.run(until::empty)));
    CHECK(is_called);
    CHECK(!context.is_running());
    CHECK(!context.stop_requested());
  }

  TEST_CASE(
    "io_uring_context Explicitly stop the io_uring_context",
    "[types][io_uring][schedulers]") {
    io_uring_context context;
    io_uring_scheduler scheduler = context.get_scheduler();
    {
      bool is_called = false;
      start_detached(schedule(scheduler) | then([&] {
                       CHECK(context.is_running());
                       is_called = true;
                     }));
      context.run_until_empty();
      CHECK(is_called);
      CHECK(!context.is_running());
      CHECK(!context.stop_requested());
    }
    context.request_stop();
    CHECK(context.stop_requested());
    context.run_until_stopped();
    CHECK(context.stop_requested());
    bool is_stopped = false;
    sync_wait(schedule(scheduler) | then([&] { CHECK(false); }) | STDEXEC::upon_stopped([&] {
                is_stopped = true;
              }));
    CHECK(is_stopped);
  }

  TEST_CASE(
    "io_uring_context Thread-safe to schedule from multiple threads",
    "[types][io_uring][schedulers]") {
    io_uring_context context;
    io_uring_scheduler scheduler = context.get_scheduler();
    jthread io_thread{[&] { context.run_until_stopped(); }};
    auto fn = [&] {
      CHECK(io_thread.get_id() == std::this_thread::get_id());
    };
    {
      scope_guard guard{[&]() noexcept { context.request_stop(); }};
      jthread thread1{[&] {
        for (int i = 0; i < 10; ++i) {
          sync_wait(when_all(
            schedule(scheduler) | then(fn),
            schedule(scheduler) | then(fn),
            schedule(scheduler) | then(fn),
            schedule(scheduler) | then(fn),
            schedule(scheduler) | then(fn),
            schedule(scheduler) | then(fn),
            schedule(scheduler) | then(fn),
            schedule(scheduler) | then(fn),
            schedule(scheduler) | then(fn),
            schedule(scheduler) | then(fn)));
        }
      }};
      jthread thread2{[&] {
        for (int i = 0; i < 10; ++i) {
          auto tp = std::chrono::steady_clock::now() + 500us;
          sync_wait(when_all(
            schedule_at(scheduler, tp) | then(fn),
            schedule_at(scheduler, tp) | then(fn),
            schedule_at(scheduler, tp) | then(fn),
            schedule_at(scheduler, tp) | then(fn),
            schedule_at(scheduler, tp) | then(fn),
            schedule_at(scheduler, tp) | then(fn),
            schedule_at(scheduler, tp) | then(fn),
            schedule_at(scheduler, tp) | then(fn),
            schedule_at(scheduler, tp) | then(fn),
            schedule_at(scheduler, tp) | then(fn)));
        }
      }};
      jthread thread3{[&] {
        for (int i = 0; i < 10; ++i) {
          sync_wait(when_all(
            schedule_after(scheduler, 500us) | then(fn),
            schedule_after(scheduler, 500us) | then(fn),
            schedule_after(scheduler, 500us) | then(fn),
            schedule_after(scheduler, 500us) | then(fn),
            schedule_after(scheduler, 500us) | then(fn),
            schedule_after(scheduler, 500us) | then(fn),
            schedule_after(scheduler, 500us) | then(fn),
            schedule_after(scheduler, 500us) | then(fn),
            schedule_after(scheduler, 500us) | then(fn),
            schedule_after(scheduler, 500us) | then(fn)));
        }
      }};
    }
  }

  TEST_CASE("io_uring_context Stop io_uring_context", "[types][io_uring][schedulers]") {
    io_uring_context context;
    io_uring_scheduler scheduler = context.get_scheduler();
    jthread io_thread{[&] { context.run_until_stopped(); }};
    {
      single_thread_context ctx1{};
      auto sch1 = ctx1.get_scheduler();
      bool is_called = false;
      sync_wait(finally(
        schedule(scheduler) | then([&]() noexcept { is_called = true; }),
        schedule(sch1) | then([&]() noexcept { context.request_stop(); })));
      CHECK(is_called);
    }
    bool is_called = false;
    sync_wait(schedule(scheduler) | then([&] { is_called = true; }));
    CHECK_FALSE(is_called);
  }

  TEST_CASE("io_uring_context schedule_after 0s", "[types][io_uring][schedulers]") {
    io_uring_context context;
    io_uring_scheduler scheduler = context.get_scheduler();
    jthread io_thread{[&] { context.run_until_stopped(); }};
    {
      scope_guard guard{[&]() noexcept { context.request_stop(); }};
      bool is_called = false;
      sync_wait(when_any(
        schedule_after(scheduler, 0s) | then([&] {
          CHECK(io_thread.get_id() == std::this_thread::get_id());
          is_called = true;
        }),
        schedule_after(scheduler, 5ms)));
      CHECK(is_called);
    }
  }

  TEST_CASE("io_uring_context schedule_after -1s", "[types][io_uring][schedulers]") {
    io_uring_context context;
    io_uring_scheduler scheduler = context.get_scheduler();
    jthread io_thread{[&] { context.run_until_stopped(); }};
    {
      scope_guard guard{[&]() noexcept { context.request_stop(); }};
      bool is_called_1 = false;
      bool is_called_2 = false;
      auto start = std::chrono::steady_clock::now();
      auto timeout = 100ms;
      sync_wait(when_any(
        schedule_after(scheduler, -1s) | then([&] {
          CHECK(io_thread.get_id() == std::this_thread::get_id());
          is_called_1 = true;
        }),
        schedule_after(scheduler, timeout) | then([&] { is_called_2 = true; })));
      auto end = std::chrono::steady_clock::now();
      std::chrono::nanoseconds diff = end - start;
      CHECK(diff.count() < std::chrono::duration_cast<std::chrono::nanoseconds>(timeout).count());
      CHECK(is_called_1 == true);
      CHECK(is_called_2 == false);
    }
  }

  TEST_CASE("io_uring_context - reuse context after being used", "[types][io_uring][schedulers]") {
    io_uring_context context;
    io_uring_scheduler scheduler = context.get_scheduler();
    CHECK(sync_wait(exec::when_any(schedule(scheduler), context.run())));
    CHECK(!sync_wait(exec::when_any(schedule(scheduler), context.run())));
    context.reset();
    CHECK(sync_wait(exec::when_any(schedule(scheduler), context.run())));
    CHECK(!sync_wait(exec::when_any(schedule(scheduler), context.run())));
  }
} // namespace

#endif
