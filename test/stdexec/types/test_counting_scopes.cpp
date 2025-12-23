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
#include <test_common/receivers.hpp>
#include <exec/static_thread_pool.hpp>

#include <atomic>
#include <concepts>
#include <optional>
#include <type_traits>

namespace ex = STDEXEC;

#define SCOPE_TEST_CASE(name)                                                                      \
  TEMPLATE_TEST_CASE(                                                                              \
    name,                                                                                          \
    "[types][simple_counting_scope][counting_scope]",                                              \
    ex::simple_counting_scope,                                                                     \
    ex::counting_scope)

namespace {
  SCOPE_TEST_CASE("the token type satisfies scope_token") {
    STATIC_REQUIRE(ex::scope_token<typename TestType::token>);
  }

  SCOPE_TEST_CASE("the association type satisfies scope_association") {
    using assoc_t = decltype(std::declval<TestType&>().get_token().try_associate());

    STATIC_REQUIRE(ex::scope_association<assoc_t>);
  }

  SCOPE_TEST_CASE(
    "the scope type has a static size_t named max_associations with a positive value") {
    STATIC_REQUIRE(std::same_as<const std::size_t, decltype(TestType::max_associations)>);
    STATIC_REQUIRE(TestType::max_associations > 0);
  }

  SCOPE_TEST_CASE("the scope type is default-constructible and immovable") {
    STATIC_REQUIRE(std::is_default_constructible_v<TestType>);
    STATIC_REQUIRE(!std::movable<TestType>);
  }

  SCOPE_TEST_CASE("the join-sender is a sender") {
    TestType scope;

    STATIC_REQUIRE(ex::sender<decltype(scope.join())>);

    ex::run_loop loop;
    auto env = ex::prop(ex::get_scheduler, loop.get_scheduler());

    STATIC_REQUIRE(ex::sender_in<decltype(scope.join()), decltype(env)>);
    // the join-sender requires a scheduler in its receiver's environment
    STATIC_REQUIRE(!ex::sender_in<decltype(scope.join()), ex::env<>>);
  }

  SCOPE_TEST_CASE("unused scopes are safe to destroy") {
    TestType scope;
    (void) scope;
  }

  SCOPE_TEST_CASE("unused-and-closed scopes are safe to destroy") {
    TestType scope;
    scope.close();
  }

  SCOPE_TEST_CASE("joined scopes are safe to destroy") {
    TestType scope;
    wait_for_value(ex::associate(ex::just(), scope.get_token()));
    wait_for_value(scope.join());
  }

  SCOPE_TEST_CASE("deferred join-senders work") {
    TestType scope;

    // grab a reference on the scope
    auto assoc = scope.get_token().try_associate();
    REQUIRE(assoc);

    bool joinStarted = false;
    bool joinFinished = false;

    wait_for_value(
      ex::when_all(
        ex::just() | ex::let_value([&]() noexcept {
          // note that we've started evaluating this branch of the when_all
          joinStarted = true;
          return scope.join();
        }) | ex::then([&]() noexcept {
          // note that join() has completed
          joinFinished = true;
        }),
        ex::just() | ex::then([&]() noexcept {
          // the whole point is to confirm that join() will suspend if
          // started with outstanding operation
          REQUIRE(joinStarted);
          REQUIRE(!joinFinished);
          // trigger a disassociation
          assoc = decltype(assoc){};
          // resuming the suspended join operation involves scheduling the
          // resumption on the environment's scheduler, which is a
          // truly-async process on a run_loop so joinFinished should not
          // have been updated yet
          REQUIRE(!joinFinished);
        })));

    REQUIRE(joinStarted);
    REQUIRE(joinFinished);
  }

  SCOPE_TEST_CASE("closed scopes refuse new associations") {
    TestType scope;

    bool firstSenderRan = false;
    bool secondSenderRan = false;

    ex::sync_wait(
      ex::associate(ex::just(&firstSenderRan), scope.get_token())
      | ex::then([](auto* flag) noexcept { *flag = true; }));

    // this is a proxy for checking that the scope now needs joining
    REQUIRE(firstSenderRan);

    scope.close();

    std::optional result = ex::sync_wait(
      ex::associate(ex::just(&secondSenderRan), scope.get_token())
      | ex::then([](auto* flag) noexcept { *flag = true; }));

    REQUIRE(!secondSenderRan);
    // the associate-sender should complete with set_stopped
    REQUIRE(!result.has_value());

    ex::sync_wait(scope.join());
  }

  SCOPE_TEST_CASE("spawn lots of work looking for TSAN errors") {
    exec::static_thread_pool pool;
    auto sched = pool.get_scheduler();

    TestType scope;

    static constexpr int width = 256;
    static constexpr int depth = 1'024;

    STATIC_REQUIRE(std::size_t(width) * std::size_t(depth) < TestType::max_associations);

    std::atomic<int> opCount{0};
    std::atomic<bool> failed{false};

    auto spawnWork = [&](auto& spawnWork) noexcept -> void {
      ex::spawn(
        ex::starts_on(sched, ex::just() | ex::then([&]() noexcept {
                               auto newOpCount = 1
                                               + opCount.fetch_add(1, std::memory_order_relaxed);
                               if (newOpCount >= depth) {
                                 return;
                               }

                               for (int i = 0; i < width; ++i) {
                                 spawnWork(spawnWork);
                               }
                             }))
          | ex::upon_error([&failed](auto&& error) noexcept {
              WARN("Spawned work failed: " << error);
              failed = true;
            }),
        scope.get_token());
    };

    for (int i = 0; i < width; ++i) {
      spawnWork(spawnWork);
    }

    ex::sync_wait(scope.join());

    CHECK(opCount == (depth * width));
    REQUIRE(!failed);
  }

  SCOPE_TEST_CASE("spawn lots of concurrent join-senders looking for UB") {
    exec::static_thread_pool pool;
    auto sched = pool.get_scheduler();

    TestType outerScope;
    TestType innerScope;

    // block the inner scope from joining
    auto assoc = innerScope.get_token().try_associate();

    REQUIRE(assoc);

    std::atomic<bool> failed{false};

    for (int i = 0; i < 100; ++i) {
      ex::spawn(
        ex::starts_on(sched, innerScope.join()) | ex::upon_error([&](auto error) noexcept {
          if constexpr (std::same_as<std::exception_ptr, decltype(error)>) {
            try {
              std::rethrow_exception(std::move(error));
            } catch (const std::exception& e) {
              WARN("failed to spawn: " << e.what());
            } catch (...) {
              WARN("failed to spawn with unknown exception");
            }
          } else {
            WARN("failed to spawn: " << error);
          }

          failed = true;
        }),
        outerScope.get_token());
    }

    // unblock all the spawned join operations
    assoc = decltype(assoc){};

    ex::sync_wait(outerScope.join());

    REQUIRE(!failed);
  }

  TEST_CASE("counting_scope::request_stop signals associated senders", "[types][counting_scope]") {
    ex::counting_scope scope;
    scope.request_stop();

    wait_for_value(
      ex::associate(ex::read_env(ex::get_stop_token), scope.get_token())
        | ex::then([](auto token) noexcept { return token.stop_requested(); }),
      true);

    ex::sync_wait(scope.join());
  }
} // namespace
