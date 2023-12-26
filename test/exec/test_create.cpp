/*
 * Copyright (c) 2021-2022 NVIDIA Corporation
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
#include <exec/async_scope.hpp>
#include <exec/create.hpp>
#include <exec/static_thread_pool.hpp>

#include <optional>

using namespace std;
namespace ex = stdexec;

namespace {
  struct immovable {
    immovable() = default;
    immovable(immovable&&) = delete;
  };

  struct create_test_fixture {
    exec::static_thread_pool pool_{2};
    exec::async_scope scope_;

    ~create_test_fixture() {
      stdexec::sync_wait(scope_.on_empty());
    }

    void anIntAPI(int a, int b, void* context, void (*completed)(void* context, int result)) {
      // Execute some work asynchronously on some other thread. When its
      // work is finished, pass the result to the callback.
      scope_.spawn(ex::on(pool_.get_scheduler(), ex::then(ex::just(), [=]() noexcept {
                            auto result = a + b;
                            completed(context, result);
                          })));
    }

    void aVoidAPI(void* context, void (*completed)(void* context)) {
      // Execute some work asynchronously on some other thread. When its
      // work is finished, pass the result to the callback.
      scope_.spawn(ex::on(
        pool_.get_scheduler(), ex::then(ex::just(), [=]() noexcept { completed(context); })));
    }
  };

  TEST_CASE_METHOD(
    create_test_fixture,
    "wrap an async API that computes a result",
    "[detail][create]") {
    auto snd = [this](int a, int b) {
      return exec::create<ex::set_value_t(int)>([a, b, this]<class Context>(Context& ctx) noexcept {
        anIntAPI(a, b, &ctx, [](void* pv, int result) {
          ex::set_value(std::move(static_cast<Context*>(pv)->receiver), (int) result);
        });
      });
    }(1, 2);

    REQUIRE_NOTHROW([&] {
      auto [res] = stdexec::sync_wait(std::move(snd)).value();
      CHECK(res == 3);
    }());
  }

  TEST_CASE_METHOD(
    create_test_fixture,
    "wrap an async API that doesn't compute a result",
    "[detail][create]") {
    bool called = false;
    auto snd = [&called, this]() {
      return exec::create<ex::set_value_t()>(
        [this]<class Context>(Context& ctx) noexcept {
          aVoidAPI(&ctx, [](void* pv) {
            Context& ctx = *static_cast<Context*>(pv);
            *std::get<0>(ctx.args) = true;
            ex::set_value(std::move(ctx.receiver));
          });
        },
        &called);
    }();

    std::optional<std::tuple<>> res = stdexec::sync_wait(std::move(snd));
    CHECK(res.has_value());
    CHECK(called);
  }
} // anonymous namespace
