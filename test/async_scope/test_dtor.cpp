#include <catch2/catch.hpp>
#include <async_scope.hpp>
#include "examples/schedulers/static_thread_pool.hpp"

namespace ex = std::execution;
using _P2519::execution::async_scope;
using _P2300::this_thread::sync_wait;

TEST_CASE("async_scope can be created and them immediately destructed", "[async_scope][dtor]") {
  async_scope scope;
  (void)scope;
}

TEST_CASE("async_scope destruction after spawning work into it", "[async_scope][dtor]") {
  example::static_thread_pool pool{4};
  ex::scheduler auto sch = pool.get_scheduler();
  std::atomic<int> counter{0};
  {
    async_scope scope;

    // Add some work into the scope
    for (int i = 0; i < 10; i++)
      scope.spawn(ex::on(sch, ex::just() | ex::then([&] { counter++; })));

    // Wait on the work, before calling destructor
    sync_wait(scope.empty());
  }
  // We should have all the work executed
  REQUIRE(counter == 10);
}
