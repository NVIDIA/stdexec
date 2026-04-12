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

#include <stdexec/__detail/__parallel_scheduler.hpp>
#include <stdexec/__detail/__parallel_scheduler_default_impl.hpp>
#include <stdexec/execution.hpp>

#include <catch2/catch_all.hpp>

#if defined(STDEXEC_PARALLEL_SCHEDULER_HEADER_ONLY)
#  error This should be testing replacement of the system context with weak linking.
#endif

namespace ex  = STDEXEC;
namespace psr = ex::parallel_scheduler_replacement;

namespace
{

  static int count_schedules = 0;

  struct my_parallel_scheduler_backend_impl
    : ex::__parallel_scheduler_default_impl::__parallel_scheduler_backend_impl
  {
    using base_t = ex::__parallel_scheduler_default_impl::__parallel_scheduler_backend_impl;

    my_parallel_scheduler_backend_impl() = default;

    void schedule(psr::receiver_proxy& __r, std::span<std::byte> __s) noexcept override
    {
      count_schedules++;
      base_t::schedule(__r, __s);
    }
  };

}  // namespace

namespace STDEXEC::parallel_scheduler_replacement
{
  // Should replace the function defined in __parallel_scheduler_default_impl_entry.hpp
  auto query_parallel_scheduler_backend()
    -> std::shared_ptr<STDEXEC::parallel_scheduler_replacement::parallel_scheduler_backend>
  {
    return std::make_shared<my_parallel_scheduler_backend_impl>();
  }
}  // namespace STDEXEC::parallel_scheduler_replacement

TEST_CASE("Check that we are using a replaced system context (with weak linking)",
          "[scheduler][parallel_scheduler][replaceability]")
{
  std::thread::id             this_id = std::this_thread::get_id();
  std::thread::id             pool_id{};
  STDEXEC::parallel_scheduler sched = STDEXEC::get_parallel_scheduler();

  auto snd = ex::then(ex::schedule(sched), [&] { pool_id = std::this_thread::get_id(); });

  REQUIRE(count_schedules == 0);
  ex::sync_wait(std::move(snd));
  REQUIRE(count_schedules == 1);

  REQUIRE(pool_id != std::thread::id{});
  REQUIRE(this_id != pool_id);
}
