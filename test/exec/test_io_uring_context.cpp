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

#include "catch2/catch.hpp"

using namespace stdexec;
using namespace exec;
using namespace std::chrono_literals;

TEST_CASE("Satisfy concepts", "[types][io_uring_context][schedulers]") {
  STATIC_REQUIRE(timed_scheduler<io_uring_scheduler>);
  STATIC_REQUIRE_FALSE(std::is_move_assignable_v<io_uring_context>);
}

TEST_CASE("Schedule runs in io thread", "[types][io_uring_context][schedulers]") {
  io_uring_context context;
  io_uring_scheduler scheduler = context.get_scheduler();
  std::jthread io_thread{[&] {
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
  "Thread-safe to schedule from multiple threads",
  "[types][io_uring_context][schedulers]") {
  io_uring_context context;
  io_uring_scheduler scheduler = context.get_scheduler();
  std::jthread io_thread{[&] {
    context.run();
  }};
  auto call_100_times = [&] {
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
  };
  auto call_100_times_after = [&] {
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
  };
  auto call_100_times_at = [&] {
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
  };
  {
    scope_guard guard{[&]() noexcept {
      context.request_stop();
    }};
    std::jthread thread1{call_100_times};
    std::jthread thread2{call_100_times_after};
    std::jthread thread3{call_100_times_at};
    thread1.join();
    thread2.join();
    thread3.join();
  }
}

TEST_CASE("Stop io_uring_context", "[types][io_uring_context]") {
  io_uring_context context;
  io_uring_scheduler scheduler = context.get_scheduler();
  std::jthread io_thread{[&] {
    context.run();
  }};
  {
    single_thread_context ctx1{};
    auto sch1 = ctx1.get_scheduler();
    bool is_called = false;
    sync_wait(finally(
      schedule(scheduler) | then([&] { is_called = true; }),
      schedule(sch1) | then([&] { context.request_stop(); })));
    CHECK(is_called);
  }
  bool is_called = false;
  sync_wait(schedule(scheduler) | then([&] { is_called = true; }));
  CHECK_FALSE(is_called);
}

#endif