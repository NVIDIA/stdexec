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
namespace scr = exec::system_context_replaceability;

namespace {

  static int count_schedules = 0;

  struct my_parallel_scheduler_backend_impl
    : exec::__system_context_default_impl::__parallel_scheduler_backend_impl {
    using base_t = exec::__system_context_default_impl::__parallel_scheduler_backend_impl;

    my_parallel_scheduler_backend_impl() = default;

    void schedule(std::span<std::byte> __s, scr::receiver& __r) noexcept override {
      count_schedules++;
      base_t::schedule(__s, __r);
    }
  };

} // namespace

namespace exec::system_context_replaceability {
  // Should replace the function defined in __system_context_default_impl.hpp
  auto query_parallel_scheduler_backend()
    -> std::shared_ptr<exec::system_context_replaceability::parallel_scheduler_backend> {
    return std::make_shared<my_parallel_scheduler_backend_impl>();
  }
} // namespace exec::system_context_replaceability

TEST_CASE(
  "Check that we are using a replaced system context (with weak linking)",
  "[system_scheduler][replaceability]") {
  std::thread::id this_id = std::this_thread::get_id();
  std::thread::id pool_id{};
  exec::parallel_scheduler sched = exec::get_parallel_scheduler();

  auto snd = ex::then(ex::schedule(sched), [&] { pool_id = std::this_thread::get_id(); });

  REQUIRE(count_schedules == 0);
  ex::sync_wait(std::move(snd));
  REQUIRE(count_schedules == 1);

  REQUIRE(pool_id != std::thread::id{});
  REQUIRE(this_id != pool_id);
}
