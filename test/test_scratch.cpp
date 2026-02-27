/*
 * Copyright (c) 2022 Lucian Radu Teodorescu
 * Copyright (c) 2022 NVIDIA Corporation
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
#include <test_common/receivers.hpp>
#include <test_common/schedulers.hpp>

namespace ex = STDEXEC;

namespace
{
  auto test_task_awaits_inline_sndr_without_stack_overflow() -> ex::task<int>
  {
    int result = 42;  //co_await nested();
    for (int i = 0; i < 1'000'000; ++i)
    {
      result += co_await ex::just(42);
    }
    co_return result;
  }

  TEST_CASE("test task can await a just_int sender without stack overflow", "[types][task]")
  {
    auto t   = test_task_awaits_inline_sndr_without_stack_overflow();
    auto [i] = ex::sync_wait(std::move(t)).value();
    CHECK(i == 42'000'042);
  }

  // template <ex::scheduler Sched = inline_scheduler>
  // inline auto _with_scheduler(Sched sched = {})
  // {
  //   return ex::write_env(ex::prop{ex::get_scheduler, std::move(sched)});
  // }

  // TEST_CASE("a scratch test case for minimal repro of a bug", "[scratch]")
  // {
  //   bool called{false};
  //   auto closure = ex::then(
  //     [&](int i) -> int
  //     {
  //       called = true;
  //       return i;
  //     });

  //   int               recv_value{0};
  //   impulse_scheduler sched1;
  //   impulse_scheduler sched2;
  //   impulse_scheduler sched3;
  //   auto              snd = ex::on(sched1, ex::just(19)) | ex::on(sched2, std::move(closure))
  //            | _with_scheduler(sched3);
  //   auto op = ex::connect(std::move(snd), expect_value_receiver_ex{recv_value});
  //   ex::start(op);
  // }
}  // namespace
