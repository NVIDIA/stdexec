/*
 * Copyright (c) 2024 Lucian Radu Teodorescu
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

#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>
#include <exec/system_context.hpp>
#include <exec/static_thread_pool.hpp>
#include <exec/__detail/__system_context_default_impl.hpp>

namespace ex = stdexec;

namespace {

  static int count_schedules = 0;

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

   private:
    exec::static_thread_pool pool_;
    exec::__system_context_default_impl::__system_scheduler_impl base_;

    static void* __schedule_impl(
      __exec_system_scheduler_interface* self_arg,
      void* preallocated,
      uint32_t psize,
      __exec_system_context_completion_callback_t callback,
      void* data) noexcept {
      printf("Using my_system_scheduler_impl::__schedule_impl\n");
      auto self = static_cast<my_system_scheduler_impl*>(self_arg);
      // increment our counter.
      count_schedules++;
      // delegate to the base implementation.
      return self->base_.__schedule(&self->base_, preallocated, psize, callback, data);
    }
  };

  struct my_system_context_impl : __exec_system_context_interface {
    my_system_context_impl() {
      __version = 202402;
      __get_scheduler = __get_scheduler_impl;
    }

   private:
    my_system_scheduler_impl scheduler_{};

    static __exec_system_scheduler_interface* __get_scheduler_impl(
      __exec_system_context_interface* __self) noexcept {
      return &static_cast<my_system_context_impl*>(__self)->scheduler_;
    }
  };
} // namespace

// Should replace the function defined in __system_context_default_impl.hpp
extern "C" __EXEC_WEAK_ATTRIBUTE __exec_system_context_interface* __get_exec_system_context_impl() {
  printf("Using my_system_context_impl\n");
  static my_system_context_impl instance;
  return &instance;
}

TEST_CASE(
  "Check that we are using a replaced system context (with weak linking)",
  "[system_scheduler][replaceability]") {
  std::thread::id this_id = std::this_thread::get_id();
  std::thread::id pool_id{};
  exec::system_context ctx;
  exec::system_scheduler sched = ctx.get_scheduler();

  auto snd = ex::then(ex::schedule(sched), [&] { pool_id = std::this_thread::get_id(); });

  REQUIRE(count_schedules == 0);
  ex::sync_wait(std::move(snd));
  REQUIRE(count_schedules == 1);

  REQUIRE(pool_id != std::thread::id{});
  REQUIRE(this_id != pool_id);
}
