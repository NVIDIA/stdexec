/*
 * Copyright (c) NVIDIA
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
#include <execution.hpp>
#include <async_scope.hpp>
#include <examples/schedulers/static_thread_pool.hpp>

#include <optional>

using namespace std;
namespace exec = _P2300::execution;

namespace {
  struct immovable {
    immovable() = default;
    immovable(immovable&&) = delete;
  };

  struct create_test_fixture {
    example::static_thread_pool pool_{2};
    _P2519::execution::async_scope scope_;

    ~create_test_fixture() {
      _P2300::this_thread::sync_wait(scope_.empty());
    }

    void anIntAPI(int a, int b, void* context, void (*completed)(void* context, int result)) {
      // Execute some work asynchronously on some other thread. When its
      // work is finished, pass the result to the callback.
      scope_.spawn(
        exec::on(
          pool_.get_scheduler(),
          exec::then(exec::just(), [=]() noexcept {
            auto result = a + b;
            completed(context, result);
          })
        )
      );
    }

    void aVoidAPI(void* context, void (*completed)(void* context)) {
      // Execute some work asynchronously on some other thread. When its
      // work is finished, pass the result to the callback.
      scope_.spawn(
        exec::on(
          pool_.get_scheduler(),
          exec::then(exec::just(), [=]() noexcept {
            completed(context);
          })
        )
      );
    }
  };
} // anonymous namespace

TEST_CASE_METHOD(create_test_fixture, "wrap an async API that computes a result", "[detail][create]") {
  auto snd = [this](int a, int b) {
    return _PXXXX::execution::create<exec::set_value_t(int)>(
      [a, b, this]<class Context>(Context& ctx) noexcept {
        anIntAPI(a, b, &ctx, [](void* pv, int result) {
          exec::set_value(std::move(static_cast<Context*>(pv)->receiver), result);
        });
      }
    );
  }(1, 2);

  REQUIRE_NOTHROW([&] {
    auto [res] = _P2300::this_thread::sync_wait(std::move(snd)).value();
    CHECK(res == 3);
  }());
}

TEST_CASE_METHOD(create_test_fixture, "wrap an async API that doesn't compute a result", "[detail][create]") {
  bool called = false;
  auto snd = [&called, this]() {
    return _PXXXX::execution::create<exec::set_value_t()>(
      [this]<class Context>(Context& ctx) noexcept {
        aVoidAPI(&ctx, [](void* pv) {
          Context& ctx = *static_cast<Context*>(pv);
          *std::get<0>(ctx.args) = true;
          exec::set_value(std::move(ctx.receiver));
        });
      },
      &called
    );
  }();

  std::optional<std::tuple<>> res = _P2300::this_thread::sync_wait(std::move(snd));
  CHECK(res.has_value());
  CHECK(called);
}
