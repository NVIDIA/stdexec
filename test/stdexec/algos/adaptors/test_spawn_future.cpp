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
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>
#include <test_common/scope_helpers.hpp>
#include <test_common/scope_tokens.hpp>

#include <array>
#include <atomic>
#include <optional>
#include <string>
#include <thread>

namespace ex = STDEXEC;

namespace {
  TEST_CASE("future completion signature calculation works", "[adaptors][spawn_future]") {
    {
      using expected = ex::completion_signatures<ex::set_stopped_t()>;
      using actual = ex::__spawn_future::__future_completions_t<ex::env<>>;

      STATIC_REQUIRE(actual{} == expected{});
    }

    {
      using expected = ex::completion_signatures<ex::set_stopped_t(), ex::set_value_t()>;
      using actual = ex::__spawn_future::__future_completions_t<ex::env<>, ex::set_value_t()>;

      STATIC_REQUIRE(actual{} == expected{});
    }

    {
      using expected = ex::completion_signatures<ex::set_stopped_t(), ex::set_value_t(std::string)>;
      using actual =
        ex::__spawn_future::__future_completions_t<ex::env<>, ex::set_value_t(std::string)>;

      STATIC_REQUIRE(actual{} == expected{});
    }

    {
      using expected = ex::completion_signatures<
        ex::set_stopped_t(),
        ex::set_error_t(std::exception_ptr),
        ex::set_value_t(std::string)
      >;
      using actual =
        ex::__spawn_future::__future_completions_t<ex::env<>, ex::set_value_t(const std::string&)>;

      STATIC_REQUIRE(actual{} == expected{});
    }
  }

  TEST_CASE("spawn_future(just(...)) is equivalent to just(...)", "[adaptors][spawn_future]") {
    constexpr auto checkEquivalence = [](const ex::sender auto& sender) {
      CHECK(ex::sync_wait(sender) == ex::sync_wait(ex::spawn_future(sender, null_token{})));
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

  TEST_CASE("spawn_future doesn't leak", "[adaptors][spawn_future]") {
    counting_resource rsc;
    std::pmr::polymorphic_allocator<> alloc(&rsc);

    REQUIRE(rsc.allocated() == 0);

    auto future = ex::spawn_future(
      ex::read_env(ex::get_allocator) | ex::then([&](auto& envAlloc) noexcept {
        CHECK(alloc == envAlloc);
        return rsc.allocated();
      }),
      null_token{},
      ex::prop(ex::get_allocator, alloc));

    auto allocated = rsc.allocated();

    CHECK(allocated > 0);

    CHECK(ex::sync_wait(std::move(future)) == std::tuple{allocated});

    CHECK(rsc.allocated() == 0);
  }

  TEST_CASE(
    "spawn_future reads an allocator from the sender's environment",
    "[adaptors][spawn_future]") {
    counting_resource rsc;
    std::pmr::polymorphic_allocator<> alloc(&rsc);

    scope_with_alloc scope{alloc};

    REQUIRE(rsc.allocated() == 0);

    auto future = ex::spawn_future(
      ex::read_env(ex::get_allocator) | ex::then([&](auto&& envAlloc) noexcept {
        // we should've pulled the scope's allocator into our environment
        CHECK(alloc == envAlloc);

        return rsc.allocated();
      }),
      scope.get_token());

    // we should've allocated some memory for the operation
    auto allocated = rsc.allocated();
    CHECK(allocated > 0);

    CHECK(ex::sync_wait(std::move(future)) == std::tuple{allocated});

    CHECK(rsc.allocated() == 0);
  }

  TEST_CASE(
    "The allocator provided directly to spawn_future overrides the allocator in the sender's "
    "environment",
    "[consumers][spawn_future]") {

    counting_resource rsc1;

    std::array<std::byte, 256> buffer{};
    std::pmr::monotonic_buffer_resource bumpAlloc(buffer.data(), buffer.size());

    counting_resource rsc2(bumpAlloc);

    std::pmr::polymorphic_allocator<> alloc1(&rsc1);
    std::pmr::polymorphic_allocator<> alloc2(&rsc2);

    REQUIRE(alloc1 != alloc2);

    scope_with_alloc scope{alloc1};

    CHECK(rsc1.allocated() == 0);
    CHECK(rsc2.allocated() == 0);

    auto future = ex::spawn_future(
      ex::read_env(ex::get_allocator) | ex::let_value([&](auto& envAlloc) noexcept {
        // the allocator in the environment should be the one provided to spawn_future
        // as an explicit argument and not the one provided by the scope
        CHECK(alloc1 != envAlloc);
        CHECK(alloc2 == envAlloc);

        return ex::just(rsc1.allocated(), rsc2.allocated());
      }),
      scope.get_token(),
      ex::prop(ex::get_allocator, alloc2));

    // we should have allocated some memory for the op from rsc2 but not from rsc
    auto allocated1 = rsc1.allocated();
    auto allocated2 = rsc2.allocated();
    CHECK(allocated1 == 0);
    CHECK(allocated2 > 0);

    CHECK(ex::sync_wait(std::move(future)) == std::tuple{allocated1, allocated2});

    CHECK(rsc1.allocated() == 0);
    CHECK(rsc2.allocated() == 0);
  }

  TEST_CASE("spawn_future tolerates throwing scope tokens", "[consumers][spawn_future]") {
    counting_resource rsc;
    std::pmr::polymorphic_allocator<> alloc(&rsc);

    struct throwing_token : null_token {
      const counting_resource* rsc;

      assoc try_associate() const {
        CHECK(rsc->allocated() > 0);
        throw std::runtime_error("nope");
      }
    };

    REQUIRE(rsc.allocated() == 0);

    bool threw = false;
    try {
      ex::spawn_future(ex::just(), throwing_token{{}, &rsc}, ex::prop(ex::get_allocator, alloc));
    } catch (const std::runtime_error& e) {
      threw = true;
      CHECK(std::string{"nope"} == e.what());
    }

    CHECK(threw);

    CHECK(rsc.allocated() == 0);
  }

  TEST_CASE("spawn_future tolerates expired scope tokens", "[consumers][spawn_future]") {
    struct expired_token : null_token { // inherit the wrap method template
      const counting_resource* rsc;
      bool* tried;

      struct assoc {
        constexpr explicit operator bool() const noexcept {
          return false;
        }

        constexpr assoc try_associate() const noexcept {
          return {};
        }
      };

      assoc try_associate() const {
        CHECK(rsc->allocated() > 0);
        *tried = true;
        return {};
      }
    };

    counting_resource rsc;
    std::pmr::polymorphic_allocator<> alloc(&rsc);

    REQUIRE(rsc.allocated() == 0);

    bool triedToAssociate = false;

    auto future = ex::spawn_future(
      ex::just(), expired_token{{}, &rsc, &triedToAssociate}, ex::prop(ex::get_allocator, alloc));

    CHECK(rsc.allocated() > 0);

    CHECK(!ex::sync_wait(std::move(future)).has_value());

    CHECK(rsc.allocated() == 0);
    CHECK(triedToAssociate);
  }

  TEST_CASE(
    "discarding a future sends a stop request to the spawned operation",
    "[adaptors][spawn_future]") {
    counting_resource rsc;
    std::pmr::polymorphic_allocator<> alloc(&rsc);

    REQUIRE(rsc.allocated() == 0);

    exec::static_thread_pool pool;

    ex::simple_counting_scope scope;

    std::atomic<bool> workStarted{false};
    std::atomic<bool> waitingForStopRequest{false};

    std::optional future = ex::spawn_future(
      ex::starts_on(
        pool.get_scheduler(),
        ex::read_env(ex::get_stop_token) | ex::then([&](auto stopToken) noexcept {
          auto callback = [&]() noexcept {
            waitingForStopRequest = true;
            waitingForStopRequest.notify_one();
          };

          using callback_t = ex::stop_callback_for_t<decltype(stopToken), decltype(callback)>;

	  [[maybe_unused]]
          callback_t registeredCallback(std::move(stopToken), std::move(callback));

          workStarted = true;
          workStarted.notify_one();

          waitingForStopRequest.wait(false);
        })),
      scope.get_token(),
      ex::prop(ex::get_allocator, alloc));

    CHECK(rsc.allocated() > 0);

    workStarted.wait(false);

    future.reset();

    ex::sync_wait(scope.join());

    CHECK(rsc.allocated() == 0);
  }

  struct never {
    using sender_concept = ex::sender_t;

    template <class Sender, class... Env>
    static consteval auto get_completion_signatures(Sender&&, Env&&...) noexcept
      -> ex::completion_signatures<ex::set_stopped_t()> {
      return {};
    }

    template <class Receiver>
    struct opstate {
      using operation_state_concept = ex::operation_state_t;

      template <ex::receiver Rcvr>
        requires std::constructible_from<Receiver, Rcvr>
      explicit opstate(Rcvr&& r) noexcept
        : rcvr_(std::forward<Rcvr>(r)) {
      }

      opstate(opstate&&) = delete;

      ~opstate() {
      }

      void start() & noexcept {
        std::construct_at(&callback_, ex::get_stop_token(ex::get_env(rcvr_)), callback{this});
      }

     private:
      struct callback {
        opstate* self_;

        void operator()() noexcept {
          std::destroy_at(&self_->callback_);
          ex::set_stopped(std::move(self_->rcvr_));
        }
      };

      using stoken_t = ex::stop_token_of_t<ex::env_of_t<Receiver>>;
      using callback_t = stoken_t::template callback_type<callback>;

      Receiver rcvr_;
      union {
        callback_t callback_;
      };
    };

    template <ex::receiver_of<ex::completion_signatures<ex::set_stopped_t()>> Receiver>
    auto connect(Receiver&& receiver) noexcept {
      return opstate<std::remove_cvref_t<Receiver>>{std::forward<Receiver>(receiver)};
    }
  };

  TEST_CASE("abandoning a future-wrapped never-sender works properly", "[adaptors][spawn_future]") {
    // this test exercises the part of __abandon that handles the spawned work completing
    // during the call __abandon makes to __stopSource_.request_stop()

    counting_resource rsc;
    std::pmr::polymorphic_allocator<> alloc(&rsc);

    REQUIRE(rsc.allocated() == 0);

    std::optional future =
      ex::spawn_future(never{}, null_token{}, ex::prop(ex::get_allocator, alloc));

    CHECK(rsc.allocated() > 0);

    future.reset();

    CHECK(rsc.allocated() == 0);
  }
} // namespace
