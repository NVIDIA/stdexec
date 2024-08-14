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
#include <exec/__detail/__system_context_default_impl.hpp>

namespace ex = stdexec;

namespace {
  using namespace stdexec::tags;
  using exec::system_context_replaceability::receiver;
  using exec::system_context_replaceability::bulk_item_receiver;
  using exec::system_context_replaceability::storage;
  using exec::system_context_replaceability::system_scheduler;
  using exec::system_context_replaceability::__system_context_replaceability;

  static int count_schedules = 0;

  struct my_system_scheduler_impl : exec::__system_context_default_impl::__system_scheduler_impl {
    using base_t = exec::__system_context_default_impl::__system_scheduler_impl;

    my_system_scheduler_impl() = default;

    void schedule(
      exec::system_context_replaceability::storage __s,
      exec::system_context_replaceability::receiver* __r) noexcept override {
      count_schedules++;
      base_t::schedule(__s, __r);
    }
  };

  void* my_query_system_context_interface(std::type_index id) noexcept {
    if (id == typeid(exec::system_context_replaceability::system_scheduler)) {
      static my_system_scheduler_impl instance;
      return &instance;
    }
    return nullptr;
  }

} // namespace

// Should replace the function defined in __system_context_default_impl.hpp
extern STDEXEC_ATTRIBUTE((weak))
  void*
  __query_system_context_interface(std::type_index id) noexcept {
  return my_query_system_context_interface(id);
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
