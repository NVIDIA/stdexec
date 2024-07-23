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

  struct my_system_scheduler_impl_base {
    exec::static_thread_pool pool_;
  };

  struct my_system_scheduler_impl
    : my_system_scheduler_impl_base
    , exec::__detail::__system_scheduler_impl {
    my_system_scheduler_impl()
      : exec::__detail::__system_scheduler_impl{pool_}
      , parent_schedule_impl{std::exchange(schedule_fn, my_schedule_impl)} {
    }

   private:
    using schedule_fn_t = exec::system_operation_state*(
      exec::system_scheduler_interface*,
      void*,
      uint32_t,
      exec::system_context_completion_callback,
      void*);

    schedule_fn_t* parent_schedule_impl;

    static exec::system_operation_state* my_schedule_impl(
      exec::system_scheduler_interface* self_arg,
      void* preallocated,
      uint32_t psize,
      exec::system_context_completion_callback callback,
      void* data) {
      auto self = static_cast<my_system_scheduler_impl*>(self_arg);
      printf("Using my_system_scheduler_impl::my_schedule_impl\n");
      // increment our counter.
      count_schedules++;
      // delegate to the base implementation.
      return self->parent_schedule_impl(self, preallocated, psize, callback, data);
    }
  };

  struct my_system_context_impl : exec::system_context_base {
    my_system_context_impl() noexcept
      : exec::system_context_base(this) {
    }

    exec::system_scheduler_interface* get_scheduler() noexcept {
      return &scheduler_;
    }

   private:
    my_system_scheduler_impl scheduler_{};
  };
} // namespace

TEST_CASE(
  "Check that we are using a replaced system context (at runtime)",
  "[system_scheduler][replaceability]") {
  std::thread::id this_id = std::this_thread::get_id();
  std::thread::id pool_id{};

  exec::set_new_system_context_handler([]() -> exec::system_context_interface* {
    return new my_system_context_impl;
  });

  exec::system_context ctx;
  exec::system_scheduler sched = ctx.get_scheduler();

  auto snd = ex::then(ex::schedule(sched), [&] { pool_id = std::this_thread::get_id(); });

  REQUIRE(count_schedules == 0);
  ex::sync_wait(std::move(snd));
  REQUIRE(count_schedules == 1);

  REQUIRE(pool_id != std::thread::id{});
  REQUIRE(this_id != pool_id);
}
