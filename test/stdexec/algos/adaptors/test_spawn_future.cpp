/*
 * Copyright (c) 2025 Ian Petersen
 * Copyright (c) 2025 NVIDIA Corporation
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
#include <test_common/scope_tokens.hpp>
#include <exec/static_thread_pool.hpp>

#include <atomic>
#include <string>
#include <thread>

namespace ex = stdexec;

namespace {
  TEST_CASE("future completion signature calculation works", "[adaptors][spawn_future]") {
    {
      using expected = ex::completion_signatures<ex::set_stopped_t()>;
      using actual = ex::__spawn_future::__future_completions_t<>;

      STATIC_REQUIRE(actual{} == expected{});
    }

    {
      using expected = ex::completion_signatures<ex::set_stopped_t(), ex::set_value_t()>;
      using actual = ex::__spawn_future::__future_completions_t<ex::set_value_t()>;

      STATIC_REQUIRE(actual{} == expected{});
    }

    {
      using expected = ex::completion_signatures<ex::set_stopped_t(), ex::set_value_t(std::string)>;
      using actual = ex::__spawn_future::__future_completions_t<ex::set_value_t(std::string)>;

      STATIC_REQUIRE(actual{} == expected{});
    }

    {
      using expected = ex::completion_signatures<
        ex::set_stopped_t(),
        ex::set_error_t(std::exception_ptr),
        ex::set_value_t(std::string)
      >;
      using actual =
        ex::__spawn_future::__future_completions_t<ex::set_value_t(const std::string&)>;

      STATIC_REQUIRE(actual{} == expected{});
    }
  }

  TEST_CASE("spawn_future(just(...)) is equivalent to just(...)", "[adaptors][spawn_future]") {
    constexpr auto checkEquivalence = [](const ex::sender auto& sender) {
      REQUIRE(ex::sync_wait(sender) == ex::sync_wait(ex::spawn_future(sender, null_token{})));
    };

    checkEquivalence(ex::just());
    checkEquivalence(ex::just(42));
    checkEquivalence(ex::just(42, std::string{"hello, world!"}));
  }

  TEST_CASE("deferred futures work", "[adaptors][spawn_future]") {
    exec::static_thread_pool pool;

    std::atomic<bool> waiting{false};
    std::atomic<bool> go{false};

    auto future = ex::spawn_future(
      ex::starts_on(pool.get_scheduler(), ex::just() | ex::let_value([&]() noexcept {
                                            waiting = true; // signal we've started running
                                            waiting.notify_one();
                                            go.wait(false);
                                            return ex::just(42, std::string{"hello, world!"});
                                          })),
      null_token{});

    // wait for the signal that the spawned work has started running
    waiting.wait(false);

    std::atomic<bool> firstBranchStarted = false;
    std::atomic<bool> futureCompleted = false;

    auto mainThreadId = std::this_thread::get_id();

    ex::sync_wait(
      ex::when_all(
        ex::just(std::move(future)) | ex::let_value([&](auto& future) noexcept {
          firstBranchStarted = true;
          return std::move(future);
        }) | ex::then([&](int i, std::string str) noexcept {
          futureCompleted = true;
          CHECK(i == 42);
          CHECK(str == "hello, world!");

          // we should be running on the pool thread
          CHECK(std::this_thread::get_id() != mainThreadId);
        }),
        ex::just() | ex::then([&]() noexcept {
          CHECK(std::this_thread::get_id() == mainThreadId);

          CHECK(firstBranchStarted);
          CHECK(!futureCompleted);
          // release the spawned work
          go = true;
          go.notify_one();
        })));

    CHECK(futureCompleted);
  }
} // namespace
