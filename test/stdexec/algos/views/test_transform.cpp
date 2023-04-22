/*
 * Copyright (c) 2023 NVIDIA Corporation
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
#include <stdexec/ranges.hpp>

#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/senders.hpp>
#include <test_common/type_helpers.hpp>

#include <span>

namespace ex = stdexec;

#ifdef __cpp_lib_ranges

constexpr int input[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

struct times_two {
  constexpr int operator()(const int value) const noexcept {
    return 2 * value;
  }
};

TEST_CASE("views::transform returns a sender", "[adaptors][views][transform]") {
  auto snd = ex::views::transform(ex::just(std::span{input}), times_two{});
  static_assert(ex::sender<decltype(snd)>);
  (void) snd;
}

TEST_CASE("views::transform can be piped", "[adaptors][views][transform]") {
  ex::sender auto snd = ex::just(std::span{input}) | ex::views::transform(times_two{});
  (void) snd;
}

TEST_CASE("views::transform simple example", "[adaptors][views][transform]") {
  auto snd = ex::views::transform(ex::just(std::span{input}), times_two{});

  auto res = stdexec::sync_wait(std::move(snd));
  CHECK(res.has_value());

  auto expected = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18};
  CHECK(std::ranges::equal(std::get<0>(res.value()), expected));
}

TEST_CASE("views::transform is lazy and does not throw", "[adaptors][views][transform]") {
  auto snd = ex::views::transform(ex::just(std::span{input}), [](int x) -> int {
    throw std::logic_error{"err"};
    return x + 5;
  });

  auto res = ex::sync_wait(std::move(snd));
  CHECK(res.has_value());
  try {
    auto expected = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18};
    std::ranges::equal(std::get<0>(res.value()), expected);
    CHECK(false);
  } catch (const std::logic_error&) {
  }
}

TEST_CASE("views::transform can be used with just_error", "[adaptors][views][transform]") {
  ex::sender auto snd = ex::just_error(std::string{"err"}) //
                      | ex::views::transform(times_two{});
  auto op = ex::connect(std::move(snd), expect_error_receiver{std::string{"err"}});
  ex::start(op);
}

TEST_CASE("views::transform can be used with just_stopped", "[adaptors][views][transform]") {
  ex::sender auto snd = ex::just_stopped() | ex::views::transform(times_two{});
  auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
  ex::start(op);
}

TEST_CASE("views::transform advertises completion schedulers", "[adaptors][views][transform]") {
  inline_scheduler sched{};

  SECTION("for value channel") {
    ex::sender auto snd = ex::schedule(sched) | ex::views::transform(times_two{});
    REQUIRE(ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(snd)) == sched);
  }
  SECTION("for stop channel") {
    ex::sender auto snd = ex::just_stopped() | ex::transfer(sched)
                        | ex::views::transform(times_two{});
    REQUIRE(ex::get_completion_scheduler<ex::set_stopped_t>(ex::get_env(snd)) == sched);
  }
}

TEST_CASE("views::transform forwards env", "[adaptors][views][transform]") {
  SECTION("returns env by value") {
    auto snd = just_with_env<value_env, int>{value_env{100}, {0}}
             | ex::views::transform(times_two{});
    static_assert(std::same_as<decltype(ex::get_env(snd)), value_env>);
    CHECK(ex::get_env(snd).value == 100);
  }

  SECTION("returns env by reference") {
    auto snd = just_with_env<const value_env&, int>{value_env{100}, {0}}
             | ex::views::transform(times_two{});
    static_assert(std::same_as<decltype(ex::get_env(snd)), const value_env&>);
    CHECK(ex::get_env(snd).value == 100);
  }
}

#endif // __cpp_lib_ranges
