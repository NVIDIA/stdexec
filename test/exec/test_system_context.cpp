/*
 * Copyright (c) 2023 Lee Howes
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

#include <thread>

#define STDEXEC_SYSTEM_CONTEXT_HEADER_ONLY 1

#include <stdexec/execution.hpp>

#include <exec/async_scope.hpp>
#include <exec/inline_scheduler.hpp>
#include <exec/static_thread_pool.hpp>
#include <exec/system_context.hpp>

#include <catch2/catch.hpp>
#include <test_common/receivers.hpp>

namespace ex = stdexec;
namespace scr = exec::system_context_replaceability;

TEST_CASE("system_context can return a scheduler", "[types][system_scheduler]") {
  auto sched = exec::get_parallel_scheduler();
  STATIC_REQUIRE(ex::scheduler<decltype(sched)>);
}

TEST_CASE("system scheduler is not default constructible", "[types][system_scheduler]") {
  auto sched = exec::get_parallel_scheduler();
  using sched_t = decltype(sched);
  STATIC_REQUIRE(!std::is_default_constructible_v<sched_t>);
  STATIC_REQUIRE(std::is_destructible_v<sched_t>);
}

TEST_CASE("system scheduler is copyable and movable", "[types][system_scheduler]") {
  auto sched = exec::get_parallel_scheduler();
  using sched_t = decltype(sched);
  STATIC_REQUIRE(std::is_copy_constructible_v<sched_t>);
  STATIC_REQUIRE(std::is_move_constructible_v<sched_t>);
}

TEST_CASE("a copied scheduler is equal to the original", "[types][system_scheduler]") {
  auto sched1 = exec::get_parallel_scheduler();
  auto sched2 = sched1;
  REQUIRE(sched1 == sched2);
}

TEST_CASE(
  "two schedulers obtained from get_parallel_scheduler() are equal",
  "[types][system_scheduler]") {
  auto sched1 = exec::get_parallel_scheduler();
  auto sched2 = exec::get_parallel_scheduler();
  REQUIRE(sched1 == sched2);
}

TEST_CASE("system scheduler can produce a sender", "[types][system_scheduler]") {
  auto snd = ex::schedule(exec::get_parallel_scheduler());
  using sender_t = decltype(snd);

  STATIC_REQUIRE(ex::sender<sender_t>);
  STATIC_REQUIRE(ex::sender_of<sender_t, ex::set_value_t()>);
  STATIC_REQUIRE(ex::sender_of<sender_t, ex::set_stopped_t()>);
}

TEST_CASE("trivial schedule task on system context", "[types][system_scheduler]") {
  exec::parallel_scheduler sched = exec::get_parallel_scheduler();

  ex::sync_wait(ex::schedule(sched));
}

TEST_CASE("simple schedule task on system context", "[types][system_scheduler]") {
  std::thread::id this_id = std::this_thread::get_id();
  std::thread::id pool_id{};
  exec::parallel_scheduler sched = exec::get_parallel_scheduler();

  auto snd = ex::then(ex::schedule(sched), [&] { pool_id = std::this_thread::get_id(); });

  ex::sync_wait(std::move(snd));

  REQUIRE(pool_id != std::thread::id{});
  REQUIRE(this_id != pool_id);
  (void) snd;
}

TEST_CASE("simple schedule forward progress guarantee", "[types][system_scheduler]") {
  exec::parallel_scheduler sched = exec::get_parallel_scheduler();
  REQUIRE(ex::get_forward_progress_guarantee(sched) == ex::forward_progress_guarantee::parallel);
}

TEST_CASE("get_completion_scheduler", "[types][system_scheduler]") {
  exec::parallel_scheduler sched = exec::get_parallel_scheduler();
  REQUIRE(ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(ex::schedule(sched))) == sched);
}

TEST_CASE("simple chain task on system context", "[types][system_scheduler]") {
  std::thread::id this_id = std::this_thread::get_id();
  std::thread::id pool_id{};
  std::thread::id pool_id2{};
  exec::parallel_scheduler sched = exec::get_parallel_scheduler();

  auto snd = ex::then(ex::schedule(sched), [&] { pool_id = std::this_thread::get_id(); });
  auto snd2 = ex::then(std::move(snd), [&] { pool_id2 = std::this_thread::get_id(); });

  ex::sync_wait(std::move(snd2));

  REQUIRE(pool_id != std::thread::id{});
  REQUIRE(this_id != pool_id);
  REQUIRE(pool_id == pool_id2);
  (void) snd;
  (void) snd2;
}

TEST_CASE("checks stop_token before starting the work", "[types][system_scheduler]") {
  exec::parallel_scheduler sched = exec::get_parallel_scheduler();

  exec::async_scope scope;
  scope.request_stop();
  REQUIRE(scope.get_stop_source().stop_requested());

  bool called = false;
  auto snd = ex::then(ex::schedule(sched), [&called] { called = true; });

  // Start the sender in a stopped scope
  scope.spawn(std::move(snd));

  // Wait for everything to be completed.
  ex::sync_wait(scope.on_empty());

  // Assert.
  REQUIRE_FALSE(called);
}

TEST_CASE("simple bulk task on system context", "[types][system_scheduler]") {
  std::thread::id this_id = std::this_thread::get_id();
  constexpr size_t num_tasks = 16;
  std::thread::id pool_ids[num_tasks];
  exec::parallel_scheduler sched = exec::get_parallel_scheduler();

  auto bulk_snd = ex::bulk(ex::schedule(sched), ex::par, num_tasks, [&](size_t id) {
    pool_ids[id] = std::this_thread::get_id();
  });

  ex::sync_wait(std::move(bulk_snd));

  for (auto pool_id: pool_ids) {
    REQUIRE(pool_id != std::thread::id{});
    REQUIRE(this_id != pool_id);
  }
}

TEST_CASE("simple bulk chaining on system context", "[types][system_scheduler]") {
  std::thread::id this_id = std::this_thread::get_id();
  constexpr size_t num_tasks = 16;
  std::thread::id pool_id{};
  std::thread::id propagated_pool_ids[num_tasks];
  std::thread::id pool_ids[num_tasks];
  exec::parallel_scheduler sched = exec::get_parallel_scheduler();

  auto snd = ex::then(ex::schedule(sched), [&] {
    pool_id = std::this_thread::get_id();
    return pool_id;
  });

  auto bulk_snd = ex::bulk(
    std::move(snd), ex::par, num_tasks, [&](size_t id, std::thread::id propagated_pool_id) {
      propagated_pool_ids[id] = propagated_pool_id;
      pool_ids[id] = std::this_thread::get_id();
    });

  std::optional<std::tuple<std::thread::id>> res = ex::sync_wait(std::move(bulk_snd));

  // Assert: first `schedule` is run on a different thread than the current thread.
  REQUIRE(pool_id != std::thread::id{});
  REQUIRE(this_id != pool_id);
  // Assert: bulk items are run and they propagate the received value.
  for (size_t i = 0; i < num_tasks; ++i) {
    REQUIRE(pool_ids[i] != std::thread::id{});
    REQUIRE(propagated_pool_ids[i] == pool_id);
    REQUIRE(this_id != pool_ids[i]);
  }
  // Assert: the result of the bulk operation is the same as the result of the first `schedule`.
  CHECK(res.has_value());
  CHECK(std::get<0>(res.value()) == pool_id);
}

TEST_CASE("simple bulk_chunked task on system context", "[types][system_scheduler]") {
  std::thread::id this_id = std::this_thread::get_id();
  constexpr size_t num_tasks = 16;
  std::thread::id pool_ids[num_tasks];
  exec::parallel_scheduler sched = exec::get_parallel_scheduler();

  auto bulk_snd = ex::bulk_chunked(
    ex::schedule(sched), ex::par, num_tasks, [&](unsigned long b, unsigned long e) {
      for (unsigned long id = b; id < e; ++id)
        pool_ids[id] = std::this_thread::get_id();
    });

  ex::sync_wait(std::move(bulk_snd));

  for (auto pool_id: pool_ids) {
    REQUIRE(pool_id != std::thread::id{});
    REQUIRE(this_id != pool_id);
  }
}

TEST_CASE("simple bulk_unchunked task on system context", "[types][system_scheduler]") {
  std::thread::id this_id = std::this_thread::get_id();
  constexpr size_t num_tasks = 16;
  std::thread::id pool_ids[num_tasks];
  exec::parallel_scheduler sched = exec::get_parallel_scheduler();

  auto bulk_snd =
    ex::bulk_unchunked(ex::schedule(sched), ex::par, num_tasks, [&](unsigned long id) {
      pool_ids[id] = std::this_thread::get_id();
    });

  ex::sync_wait(std::move(bulk_snd));

  for (auto pool_id: pool_ids) {
    REQUIRE(pool_id != std::thread::id{});
    REQUIRE(this_id != pool_id);
  }
}

TEST_CASE(
  "bulk_unchunked with seq will run everything on one thread",
  "[types][system_scheduler]") {
  std::thread::id this_id = std::this_thread::get_id();
  constexpr size_t num_tasks = 16;
  std::thread::id pool_ids[num_tasks];
  exec::parallel_scheduler sched = exec::get_parallel_scheduler();

  auto bulk_snd = ex::bulk_unchunked(ex::schedule(sched), ex::seq, num_tasks, [&](size_t id) {
    pool_ids[id] = std::this_thread::get_id();
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  });

  ex::sync_wait(std::move(bulk_snd));

  for (auto pool_id: pool_ids) {
    REQUIRE(pool_id != std::thread::id{});
    REQUIRE(this_id != pool_id);
    REQUIRE(pool_id == pool_ids[0]); // All should be the same
  }
}

TEST_CASE("bulk_chunked on parallel_scheduler performs chunking", "[types][system_scheduler]") {
  std::atomic<bool> has_chunking = false;

  exec::parallel_scheduler sched = exec::get_parallel_scheduler();
  auto bulk_snd = ex::bulk_chunked(ex::schedule(sched), ex::par, 10'000, [&](int b, int e) {
    if (e - b > 1) {
      has_chunking = true;
    }
  });
  ex::sync_wait(std::move(bulk_snd));

  REQUIRE(has_chunking.load());
}

TEST_CASE(
  "bulk_chunked on parallel_scheduler covers the entire range",
  "[types][system_scheduler]") {
  constexpr size_t num_tasks = 200;
  bool covered[num_tasks];

  exec::parallel_scheduler sched = exec::get_parallel_scheduler();
  auto bulk_snd =
    ex::bulk_chunked(ex::schedule(sched), ex::par, num_tasks, [&](size_t b, size_t e) {
      for (auto i = b; i < e; ++i) {
        covered[i] = true;
      }
    });
  ex::sync_wait(std::move(bulk_snd));

  for (size_t i = 0; i < num_tasks; ++i) {
    REQUIRE(covered[i]);
  }
}

TEST_CASE(
  "bulk_chunked with seq on parallel_scheduler doesn't do chunking",
  "[types][system_scheduler]") {
  constexpr size_t num_tasks = 200;
  std::atomic<int> execution_count = 0;

  exec::parallel_scheduler sched = exec::get_parallel_scheduler();
  auto bulk_snd =
    ex::bulk_chunked(ex::schedule(sched), ex::seq, num_tasks, [&](size_t b, size_t e) {
      REQUIRE(b == 0);
      REQUIRE(e == num_tasks);
      execution_count++;
    });
  ex::sync_wait(std::move(bulk_snd));

  REQUIRE(execution_count.load() == 1);
}

struct my_parallel_scheduler_backend_impl
  : exec::__system_context_default_impl::__parallel_scheduler_backend_impl {
  using base_t = exec::__system_context_default_impl::__parallel_scheduler_backend_impl;

  my_parallel_scheduler_backend_impl() = default;

  [[nodiscard]]
  auto num_schedules() const -> int {
    return count_schedules_;
  }

  void schedule(std::span<std::byte> __s, scr::receiver& __r) noexcept override {
    count_schedules_++;
    base_t::schedule(__s, __r);
  }


 private:
  int count_schedules_ = 0;
};

struct my_inline_scheduler_backend_impl : scr::parallel_scheduler_backend {
  void schedule(std::span<std::byte>, scr::receiver& r) noexcept override {
    r.set_value();
  }

  void
    schedule_bulk_chunked(uint32_t count, std::span<std::byte>, scr::bulk_item_receiver& r) noexcept
    override {
    r.execute(0, count);
    r.set_value();
  }

  void schedule_bulk_unchunked(
    uint32_t count,
    std::span<std::byte>,
    scr::bulk_item_receiver& r) noexcept override {
    for (uint32_t i = 0; i < count; ++i)
      r.execute(i, i + 1);
    r.set_value();
  }
};

TEST_CASE(
  "can change the implementation of system context at runtime",
  "[types][system_scheduler]") {
  static auto my_scheduler_backend = std::make_shared<my_parallel_scheduler_backend_impl>();
  auto old_factory = scr::set_parallel_scheduler_backend(
    []() -> std::shared_ptr<scr::parallel_scheduler_backend> { return my_scheduler_backend; });

  std::thread::id this_id = std::this_thread::get_id();
  std::thread::id pool_id{};
  exec::parallel_scheduler sched = exec::get_parallel_scheduler();

  auto snd = ex::then(ex::schedule(sched), [&] { pool_id = std::this_thread::get_id(); });

  REQUIRE(my_scheduler_backend->num_schedules() == 0);
  ex::sync_wait(std::move(snd));
  REQUIRE(my_scheduler_backend->num_schedules() == 1);

  REQUIRE(pool_id != std::thread::id{});
  REQUIRE(this_id != pool_id);

  (void) scr::set_parallel_scheduler_backend(old_factory);
}

TEST_CASE(
  "can change the implementation of system context at runtime, with an inline scheduler",
  "[types][system_scheduler]") {
  auto old_factory = scr::set_parallel_scheduler_backend(
    []() -> std::shared_ptr<scr::parallel_scheduler_backend> {
      return std::make_shared<my_inline_scheduler_backend_impl>();
    });

  std::thread::id this_id = std::this_thread::get_id();
  std::thread::id pool_id{};
  exec::parallel_scheduler sched = exec::get_parallel_scheduler();

  auto snd = ex::then(ex::schedule(sched), [&] { pool_id = std::this_thread::get_id(); });

  ex::sync_wait(std::move(snd));

  REQUIRE(this_id == pool_id);

  (void) scr::set_parallel_scheduler_backend(old_factory);
}

TEST_CASE("empty environment always returns nullopt for any query", "[types][system_scheduler]") {
  struct my_receiver : scr::receiver {
    auto __query_env(__uuid, void*) noexcept -> bool override {
      return false;
    }

    void set_value() noexcept override {
    }

    void set_error(std::exception_ptr) noexcept override {
    }

    void set_stopped() noexcept override {
    }
  };

  my_receiver rcvr{};

  REQUIRE(rcvr.try_query<stdexec::inplace_stop_token>() == std::nullopt);
  REQUIRE(rcvr.try_query<int>() == std::nullopt);
  REQUIRE(rcvr.try_query<std::allocator<int>>() == std::nullopt);
}

TEST_CASE("environment with a stop token can expose its stop token", "[types][system_scheduler]") {
  struct my_receiver : scr::receiver {
    auto __query_env(__uuid uuid, void* dest) noexcept -> bool override {
      if (
        uuid
        == scr::__runtime_property_helper<stdexec::inplace_stop_token>::__property_identifier) {
        *static_cast<stdexec::inplace_stop_token*>(dest) = ss.get_token();
        return true;
      }
      return false;
    }

    void set_value() noexcept override {
    }

    void set_error(std::exception_ptr) noexcept override {
    }

    void set_stopped() noexcept override {
    }

    stdexec::inplace_stop_source ss;
  };

  my_receiver rcvr{};

  auto o1 = rcvr.try_query<stdexec::inplace_stop_token>();
  REQUIRE(o1.has_value());
  REQUIRE(o1.value().stop_requested() == false);
  REQUIRE(o1.value() == rcvr.ss.get_token());

  rcvr.ss.request_stop();
  REQUIRE(o1.value().stop_requested() == true);

  REQUIRE(rcvr.try_query<int>() == std::nullopt);
  REQUIRE(rcvr.try_query<std::allocator<int>>() == std::nullopt);
}
