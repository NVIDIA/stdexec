#include <catch2/catch.hpp>
#include <async_scope.hpp>
#include "test_common/schedulers.hpp"

namespace ex = std::execution;
using _P2519::execution::async_scope;
using _P2300::this_thread::sync_wait;

TEST_CASE("TODO: calling request_stop will cancel the async_scope object", "[async_scope][stop]") {
  async_scope scope;

  scope.request_stop();

  // TODO: reenable this
  // REQUIRE(P2519::__scope::empty(scope));
  impulse_scheduler sch;
  scope.spawn(ex::on(sch, ex::just()));
  // TODO: reenable this
  // REQUIRE(P2519::__scope::empty(scope));
}
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

TEST_CASE("TODO: cancelling the associated stop_source will cancel the async_scope object",
    "[async_scope][stop]") {
  async_scope scope;

  scope.get_stop_source().request_stop();

  // TODO: reenable this
  // REQUIRE(P2519::__scope::empty(scope));
  impulse_scheduler sch;
  scope.spawn(ex::on(sch, ex::just()));
  // TODO: the scope needs to be empty
  // REQUIRE(P2519::__scope::empty(scope));
  // REQUIRE_FALSE(P2519::__scope::empty(scope));

  // TODO: remove this after ensuring that the work is not in the scope anymore
  sch.start_next();
}
TEST_CASE(
    "cancelling the associated stop_source will be visible in stop_token", "[async_scope][stop]") {
  async_scope scope;

  scope.get_stop_source().request_stop();
  REQUIRE(scope.get_stop_token().stop_requested());
}
