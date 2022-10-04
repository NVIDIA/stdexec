#include <catch2/catch.hpp>
#include <async_scope.hpp>
#include "test_common/schedulers.hpp"
#include "test_common/receivers.hpp"
#include "examples/schedulers/single_thread_context.hpp"

namespace ex = std::execution;
using _P2519::execution::async_scope;
using _P2300::this_thread::sync_wait;

TEST_CASE("calling request_stop will be visible in stop_source", "[async_scope][stop]") {
  async_scope scope;

  scope.request_stop();
  REQUIRE(scope.get_stop_source().stop_requested());
}
TEST_CASE("calling request_stop will be visible in stop_token", "[async_scope][stop]") {
  async_scope scope;

  scope.request_stop();
  REQUIRE(scope.get_stop_token().stop_requested());
}

TEST_CASE("cancelling the associated stop_source will cancel the async_scope object",
    "[async_scope][stop]") {
  bool empty = false;

  {
    impulse_scheduler sch;
    async_scope scope;
    bool called = false;

    // put work in the scope
    scope.spawn(ex::on(sch, ex::just())
              | ex::upon_stopped([&]{ called = true; }));
    REQUIRE_FALSE(called);

    // start a thread waiting on when the scope is empty:
    example::single_thread_context thread;
    auto thread_sch = thread.get_scheduler();
    ex::start_detached(ex::on(thread_sch, scope.on_empty())
                     | ex::then([&]{ empty = true; }));
    REQUIRE_FALSE(empty);

    // request the scope stop
    scope.get_stop_source().request_stop();

    // execute the work in the scope
    sch.start_next();

    // Should have completed with a stopped signal
    REQUIRE(called);
  } // blocks until the separate thread is joined

  REQUIRE(empty);
}

TEST_CASE(
    "cancelling the associated stop_source will be visible in stop_token", "[async_scope][stop]") {
  async_scope scope;

  scope.get_stop_source().request_stop();
  REQUIRE(scope.get_stop_token().stop_requested());
}
