/*
 * Copyright (c) MAikel Nadolski
 * Copyright (c) 2021-2023 NVIDIA Corporation
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

#include <stdexec/coroutine.hpp>

#include <exec/single_thread_context.hpp>
#include <exec/task.hpp>

#include <catch2/catch_all.hpp>

#if !STDEXEC_NO_STDCPP_COROUTINES()

namespace ex = STDEXEC;

auto test_stickiness(auto scheduler1, auto id1) -> exec::task<void>
{
  co_await exec::reschedule_coroutine_on(scheduler1);
  CHECK(std::this_thread::get_id() == id1);
}

TEST_CASE("stress test for stickiness and coroutine rescheduling",
          "[types][task][reschedule_coroutine_on]")
{
  [[maybe_unused]]
  int                         i = GENERATE(repeat(10000, values({1})));
  exec::single_thread_context context1;
  ex::scheduler auto          scheduler1 = context1.get_scheduler();

  auto main_id = std::this_thread::get_id();
  auto id1 = context1.get_thread_id();
  auto t   = test_stickiness(scheduler1, id1);
  ex::sync_wait(
    std::move(t)
    | ex::then([=] { CHECK(std::this_thread::get_id() == main_id); }));
}

#else

TEST_CASE("dummy test", "[types][task]")
{
  CHECK(true);
}

#endif
