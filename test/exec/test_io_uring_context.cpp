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

#if __has_include(<linux/io_uring.h>)
#include "exec/linux/io_uring_context.hpp"
#include "exec/scope.hpp"
#include "exec/single_thread_context.hpp"
#include "exec/finally.hpp"
#include "exec/when_any.hpp"

#include "catch2/catch.hpp"

using namespace stdexec;
using namespace exec;
using namespace std::chrono_literals;

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

  jthread& operator=(jthread&& other) noexcept {
    thread_ = std::move(other.thread_);
    return *this;
  }

  ~jthread() {
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  std::thread::id get_id() const noexcept {
    return thread_.get_id();
  }
};

TEST_CASE("Satisfy concepts", "[types][io_uring][schedulers]") {
  STATIC_REQUIRE(timed_scheduler<io_uring_scheduler>);
  STATIC_REQUIRE_FALSE(std::is_move_assignable_v<io_uring_context>);
}

TEST_CASE("Schedule runs in io thread", "[types][io_uring][schedulers]") {
  io_uring_context context;
  io_uring_scheduler scheduler = context.get_scheduler();
  jthread io_thread{[&] {
    context.run();
  }};
  {
    scope_guard guard{[&]() noexcept {
      context.request_stop();
    }};
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
  "Call io_uring::sync_wait runs and returns but does not stop",
  "[types][io_uring][schedulers]") {
  io_uring_context context;
  io_uring_scheduler scheduler = context.get_scheduler();
  {
    bool is_called = false;
    sync_wait(context, schedule(scheduler) | then([&] {
                        CHECK(context.is_running());
                        is_called = true;
                      }));
    CHECK(is_called);
    CHECK(!context.is_running());
    CHECK(!context.stop_requested());
  }
  {
    bool is_called = false;
    sync_wait(context, schedule_after(scheduler, 500us) | then([&] {
                        CHECK(context.is_running());
                        is_called = true;
                      }));
    CHECK(is_called);
    CHECK(!context.is_running());
    CHECK(!context.stop_requested());
  }
  // Destructor stops the wakeup operation and blocks the thread until everything is done
}

TEST_CASE("Explicitly stop the io_uring_context", "[types][io_uring][schedulers]") {
  io_uring_context context;
  io_uring_scheduler scheduler = context.get_scheduler();
  {
    bool is_called = false;
    sync_wait(context, schedule(scheduler) | then([&] {
                         CHECK(context.is_running());
                         is_called = true;
                       }));
    CHECK(is_called);
    CHECK(!context.is_running());
    CHECK(!context.stop_requested());
  }
  context.request_stop();
  CHECK(context.stop_requested());
  context.run();
  CHECK(context.stop_requested());
  bool is_stopped = false;
  sync_wait(schedule(scheduler) | then([&] { CHECK(false); }) | stdexec::upon_stopped([&] {
                       is_stopped = true;
                     }));
  CHECK(is_stopped);
}

TEST_CASE("Sync wait returns a value", "[types][io_uring][schedulers]") {
  io_uring_context context;
  io_uring_scheduler scheduler = context.get_scheduler();
  {
    auto [value] = sync_wait(context, schedule(scheduler) | then([] { return 42; })).value();
    CHECK(value == 42);
  }
  {
    auto [value] =
      sync_wait(context, schedule_after(scheduler, 500us) | then([] { return 42; })).value();
    CHECK(value == 42);
  }
}

TEST_CASE("sync wait of a sender in another scheduler", "[types][io_uring][schedulers]") {
  io_uring_context context;
  exec::single_thread_context single_thread_context;
  auto scheduler = single_thread_context.get_scheduler();
  {
    bool is_called = false;
    sync_wait(context, schedule(scheduler) | then([&] {
                         CHECK(std::this_thread::get_id() == single_thread_context.get_thread_id());
                         CHECK(context.is_running());
                         is_called = true;
                       }));
    CHECK(is_called);
    CHECK(!context.is_running());
  }
}

TEST_CASE("Thread-safe to schedule from multiple threads", "[types][io_uring][schedulers]") {
  io_uring_context context;
  io_uring_scheduler scheduler = context.get_scheduler();
  jthread io_thread{[&] {
    context.run();
  }};
  {
    scope_guard guard{[&]() noexcept {
      context.request_stop();
    }};
    jthread thread1{[&] {
      for (int i = 0; i < 10; ++i) {
        sync_wait(when_all(
          schedule(scheduler)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule(scheduler)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule(scheduler)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule(scheduler)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule(scheduler)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule(scheduler)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule(scheduler)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule(scheduler)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule(scheduler)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule(scheduler)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); })));
      }
    }};
    jthread thread2{[&] {
      for (int i = 0; i < 10; ++i) {
        auto tp = std::chrono::steady_clock::now() + 500us;
        sync_wait(when_all(
          schedule_at(scheduler, tp)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule_at(scheduler, tp)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule_at(scheduler, tp)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule_at(scheduler, tp)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule_at(scheduler, tp)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule_at(scheduler, tp)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule_at(scheduler, tp)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule_at(scheduler, tp)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule_at(scheduler, tp)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule_at(scheduler, tp)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); })));
      }
    }};
    jthread thread3{[&] {
      for (int i = 0; i < 10; ++i) {
        sync_wait(when_all(
          schedule_after(scheduler, 500us)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule_after(scheduler, 500us)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule_after(scheduler, 500us)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule_after(scheduler, 500us)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule_after(scheduler, 500us)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule_after(scheduler, 500us)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule_after(scheduler, 500us)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule_after(scheduler, 500us)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule_after(scheduler, 500us)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); }),
          schedule_after(scheduler, 500us)
            | then([&] { CHECK(io_thread.get_id() == std::this_thread::get_id()); })));
      }
    }};
  }
}

TEST_CASE("Stop io_uring_context", "[types][io_uring][schedulers]") {
  io_uring_context context;
  io_uring_scheduler scheduler = context.get_scheduler();
  jthread io_thread{[&] {
    context.run();
  }};
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

TEST_CASE("After 0s", "[types][io_uring][schedulers]") {
  io_uring_context context;
  io_uring_scheduler scheduler = context.get_scheduler();
  jthread io_thread{[&] {
    context.run();
  }};
  {
    scope_guard guard{[&]() noexcept {
      context.request_stop();
    }};
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

TEST_CASE("After -1s", "[types][io_uring][schedulers]") {
  io_uring_context context;
  io_uring_scheduler scheduler = context.get_scheduler();
  jthread io_thread{[&] {
    context.run();
  }};
  {
    scope_guard guard{[&]() noexcept {
      context.request_stop();
    }};
    bool is_called = false;
    sync_wait(when_any(
      schedule_after(scheduler, -1ns) | then([&] {
        CHECK(io_thread.get_id() == std::this_thread::get_id());
        is_called = true;
      }),
      schedule_after(scheduler, 5ms)));
    CHECK(is_called);
  }
}

#endif