/*
 * Copyright (c) 2022 Lucian Radu Teodorescu
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
#include <catch2/catch_all.hpp>

#include <stdexec/execution.hpp>

#include <test_common/receivers.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/senders.hpp>
#include <test_common/type_helpers.hpp>

namespace ex = STDEXEC;

namespace
{

  // TODO: implement upon_stopped
  TEST_CASE("upon_stopped returns a sender", "[adaptors][upon_stopped]")
  {
    auto snd = ex::upon_stopped(ex::just_stopped(), []() {});
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("upon_stopped with environment returns a sender", "[adaptors][upon_stopped]")
  {
    auto snd = ex::upon_stopped(ex::just_stopped(), []() {});
    static_assert(ex::sender_in<decltype(snd), ex::env<>>);
    (void) snd;
  }

  TEST_CASE("upon_stopped simple example", "[adaptors][upon_stopped]")
  {
    bool called{false};
    auto snd = ex::upon_stopped(ex::just_stopped(),
                                [&]()
                                {
                                  called = true;
                                  return 0;
                                });
    auto op  = ex::connect(std::move(snd), expect_value_receiver{0});
    ex::start(op);
    // The receiver checks that it's called
    // we also check that the function was invoked
    CHECK(called);
  }

  TEST_CASE("upon_stopped reports the completion scheduler of the child channel that feeds each "
            "channel",
            "[adaptors][upon_stopped]")
  {
    ex::run_loop loop1, loop2, loop3;
    auto         sch1 = loop1.get_scheduler();
    auto         sch2 = loop2.get_scheduler();
    auto         sch3 = loop3.get_scheduler();
    auto         sndr = make_per_channel_scheduler_sender(sch1, sch2, sch3);

    SECTION("with a function that does not throw")
    {
      auto snd = sndr | ex::upon_stopped([]() noexcept { return 0; });
      STATIC_REQUIRE(!has_completion_scheduler<ex::set_value_t, decltype(snd)>);
      CHECK(ex::get_completion_scheduler<ex::set_error_t>(ex::get_env(snd)) == sch2);
      STATIC_REQUIRE(!has_completion_scheduler<ex::set_stopped_t, decltype(snd)>);
    }

    SECTION("with a function that may throw")
    {
      auto snd = sndr | ex::upon_stopped([]() { return 0; });
      STATIC_REQUIRE(!has_completion_scheduler<ex::set_value_t, decltype(snd)>);
      STATIC_REQUIRE(!has_completion_scheduler<ex::set_error_t, decltype(snd)>);
      STATIC_REQUIRE(!has_completion_scheduler<ex::set_stopped_t, decltype(snd)>);
    }

    SECTION("with a child that only completes with stopped")
    {
      ex::prop env{ex::get_start_scheduler, sch1};
      auto     snd = ex::just_stopped() | ex::upon_stopped([]() noexcept { return 1; });
      CHECK(ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(snd), env) == sch1);
    }
  }
}  // namespace
