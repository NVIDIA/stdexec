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
#include <iostream>
#include <chrono>

#define STDEXEC_SYSTEM_CONTEXT_HEADER_ONLY 1

#include <stdexec/execution.hpp>

#include <exec/async_scope.hpp>
#include <exec/inline_scheduler.hpp>
#include <exec/static_thread_pool.hpp>
#include <exec/system_context.hpp>

#include <catch2/catch.hpp>
#include <test_common/receivers.hpp>


namespace ex = stdexec;

TEST_CASE("system_context has default ctor and dtor", "[types][system_scheduler]") {
  STATIC_REQUIRE(std::is_default_constructible_v<exec::system_context>);
  STATIC_REQUIRE(std::is_destructible_v<exec::system_context>);
}

TEST_CASE("system_context is not copyable nor movable", "[types][system_scheduler]") {
  STATIC_REQUIRE_FALSE(std::is_copy_constructible_v<exec::system_context>);
  STATIC_REQUIRE_FALSE(std::is_move_constructible_v<exec::system_context>);
}

TEST_CASE("system_context can return a scheduler", "[types][system_scheduler]") {
  auto sched = exec::system_context{}.get_scheduler();
  STATIC_REQUIRE(ex::scheduler<decltype(sched)>);
}

TEST_CASE("can query max concurrency from system_context", "[types][system_scheduler]") {
  exec::system_context ctx;
  size_t max_concurrency = ctx.max_concurrency();
  REQUIRE(max_concurrency >= 1);
}

TEST_CASE("system scheduler is not default constructible", "[types][system_scheduler]") {
  auto sched = exec::system_context{}.get_scheduler();
  using sched_t = decltype(sched);
  STATIC_REQUIRE(!std::is_default_constructible_v<sched_t>);
  STATIC_REQUIRE(std::is_destructible_v<sched_t>);
}

TEST_CASE("system scheduler is copyable and movable", "[types][system_scheduler]") {
  auto sched = exec::system_context{}.get_scheduler();
  using sched_t = decltype(sched);
  STATIC_REQUIRE(std::is_copy_constructible_v<sched_t>);
  STATIC_REQUIRE(std::is_move_constructible_v<sched_t>);
}

TEST_CASE("a copied scheduler is equal to the original", "[types][system_scheduler]") {
  exec::system_context ctx;
  auto sched1 = ctx.get_scheduler();
  auto sched2 = sched1;
  REQUIRE(sched1 == sched2);
}

TEST_CASE(
  "two schedulers obtained from the same system_context are equal",
  "[types][system_scheduler]") {
  exec::system_context ctx;
  auto sched1 = ctx.get_scheduler();
  auto sched2 = ctx.get_scheduler();
  REQUIRE(sched1 == sched2);
}

TEST_CASE(
  "compare two schedulers obtained from different system_context objects",
  "[types][system_scheduler]") {
  exec::system_context ctx1;
  auto sched1 = ctx1.get_scheduler();
  exec::system_context ctx2;
  auto sched2 = ctx2.get_scheduler();
  // TODO: clarify the result of this in the paper
  REQUIRE(sched1 == sched2);
}

TEST_CASE("system scheduler can produce a sender", "[types][system_scheduler]") {
  auto snd = ex::schedule(exec::system_context{}.get_scheduler());
  using sender_t = decltype(snd);

  STATIC_REQUIRE(ex::sender<sender_t>);
  STATIC_REQUIRE(ex::sender_of<sender_t, ex::set_value_t()>);
  STATIC_REQUIRE(ex::sender_of<sender_t, ex::set_stopped_t()>);
}

TEST_CASE("trivial schedule task on system context", "[types][system_scheduler]") {
  exec::system_context ctx;
  exec::system_scheduler sched = ctx.get_scheduler();

  ex::sync_wait(ex::schedule(sched));
}

TEST_CASE("simple schedule task on system context", "[types][system_scheduler]") {
  std::thread::id this_id = std::this_thread::get_id();
  std::thread::id pool_id{};
  exec::system_context ctx;
  exec::system_scheduler sched = ctx.get_scheduler();

  auto snd = ex::then(ex::schedule(sched), [&] { pool_id = std::this_thread::get_id(); });

  ex::sync_wait(std::move(snd));

  REQUIRE(pool_id != std::thread::id{});
  REQUIRE(this_id != pool_id);
  (void) snd;
}

TEST_CASE("simple schedule forward progress guarantee", "[types][system_scheduler]") {
  exec::system_context ctx;
  exec::system_scheduler sched = ctx.get_scheduler();
  REQUIRE(ex::get_forward_progress_guarantee(sched) == ex::forward_progress_guarantee::parallel);
}

TEST_CASE("get_completion_scheduler", "[types][system_scheduler]") {
  exec::system_context ctx;
  exec::system_scheduler sched = ctx.get_scheduler();
  REQUIRE(ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(ex::schedule(sched))) == sched);
  REQUIRE(
    ex::get_completion_scheduler<ex::set_stopped_t>(ex::get_env(ex::schedule(sched))) == sched);
}

TEST_CASE("simple chain task on system context", "[types][system_scheduler]") {
  std::thread::id this_id = std::this_thread::get_id();
  std::thread::id pool_id{};
  std::thread::id pool_id2{};
  exec::system_context ctx;
  exec::system_scheduler sched = ctx.get_scheduler();

  auto snd = ex::then(ex::schedule(sched), [&] { pool_id = std::this_thread::get_id(); });
  auto snd2 = ex::then(std::move(snd), [&] { pool_id2 = std::this_thread::get_id(); });

  ex::sync_wait(std::move(snd2));

  REQUIRE(pool_id != std::thread::id{});
  REQUIRE(this_id != pool_id);
  REQUIRE(pool_id == pool_id2);
  (void) snd;
  (void) snd2;
}

// TODO: fix this test. This also makes tsan and asan unhappy.
// TEST_CASE("checks stop_token before starting the work", "[types][system_scheduler]") {
//   exec::system_context ctx;
//   exec::system_scheduler sched = ctx.get_scheduler();

//   exec::async_scope scope;
//   scope.request_stop();

//   bool called = false;
//   auto snd = ex::then(ex::schedule(sched), [&called] { called = true; });

//   // Start the sender in a stopped scope
//   scope.spawn(std::move(snd));

//   // Wait for everything to be completed.
//   ex::sync_wait(scope.on_empty());

//   // Assert.
//   // TODO: called should be false
//   REQUIRE(called);
// }

TEST_CASE("simple bulk task on system context", "[types][system_scheduler]") {
  std::thread::id this_id = std::this_thread::get_id();
  constexpr size_t num_tasks = 16;
  std::thread::id pool_ids[num_tasks];
  exec::system_context ctx;
  exec::system_scheduler sched = ctx.get_scheduler();

  auto bulk_snd = ex::bulk(ex::schedule(sched), num_tasks, [&](unsigned long id) {
    pool_ids[id] = std::this_thread::get_id();
  });

  ex::sync_wait(std::move(bulk_snd));

  for (size_t i = 0; i < num_tasks; ++i) {
    REQUIRE(pool_ids[i] != std::thread::id{});
    REQUIRE(this_id != pool_ids[i]);
  }
  (void) bulk_snd;
}

TEST_CASE("simple bulk chaining on system context", "[types][system_scheduler]") {
  std::thread::id this_id = std::this_thread::get_id();
  constexpr size_t num_tasks = 16;
  std::thread::id pool_id{};
  std::thread::id propagated_pool_ids[num_tasks];
  std::thread::id pool_ids[num_tasks];
  exec::system_context ctx;
  exec::system_scheduler sched = ctx.get_scheduler();

  auto snd = ex::then(ex::schedule(sched), [&] {
    pool_id = std::this_thread::get_id();
    return pool_id;
  });

  auto bulk_snd =
    ex::bulk(std::move(snd), num_tasks, [&](unsigned long id, std::thread::id propagated_pool_id) {
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

struct my_system_scheduler_impl : __exec_system_scheduler_interface {
  my_system_scheduler_impl()
    : base_{pool_} {
    __forward_progress_guarantee = base_.__forward_progress_guarantee;
    __schedule_operation_size = base_.__schedule_operation_size;
    __schedule_operation_alignment = base_.__schedule_operation_alignment;
    __destruct_schedule_operation = base_.__destruct_schedule_operation;
    __bulk_schedule_operation_size = base_.__bulk_schedule_operation_size;
    __bulk_schedule_operation_alignment = base_.__bulk_schedule_operation_alignment;
    __bulk_schedule = base_.__bulk_schedule;
    __destruct_bulk_schedule_operation = base_.__destruct_bulk_schedule_operation;

    __schedule = __schedule_impl; // have our own schedule implementation
  }

  int num_schedules() const {
    return count_schedules_;
  }

 private:
  exec::static_thread_pool pool_;
  exec::__system_context_default_impl::__system_scheduler_impl base_;
  int count_schedules_ = 0;

  static void* __schedule_impl(
    __exec_system_scheduler_interface* self_arg,
    void* preallocated,
    uint32_t psize,
    __exec_system_context_completion_callback_t callback,
    void* data) noexcept {
    auto self = static_cast<my_system_scheduler_impl*>(self_arg);
    // increment our counter.
    self->count_schedules_++;
    // delegate to the base implementation.
    return self->base_.__schedule(&self->base_, preallocated, psize, callback, data);
  }
};

struct my_system_context_impl : __exec_system_context_interface {
  my_system_context_impl() {
    __version = 202402;
    __get_scheduler = __get_scheduler_impl;
  }

  int num_schedules() const {
    return scheduler_.num_schedules();
  }

 private:
  my_system_scheduler_impl scheduler_{};

  static __exec_system_scheduler_interface*
    __get_scheduler_impl(__exec_system_context_interface* __self) noexcept {
    return &static_cast<my_system_context_impl*>(__self)->scheduler_;
  }
};

TEST_CASE("can change the implementation of system context", "[types][system_scheduler]") {
  // Not to spec.
  my_system_context_impl ctx_impl;
  __set_exec_system_context_impl(&ctx_impl);

  std::thread::id this_id = std::this_thread::get_id();
  std::thread::id pool_id{};
  exec::system_context ctx;
  exec::system_scheduler sched = ctx.get_scheduler();

  auto snd = ex::then(ex::schedule(sched), [&] { pool_id = std::this_thread::get_id(); });

  REQUIRE(ctx_impl.num_schedules() == 0);
  ex::sync_wait(std::move(snd));
  REQUIRE(ctx_impl.num_schedules() == 1);

  REQUIRE(pool_id != std::thread::id{});
  REQUIRE(this_id != pool_id);
}
