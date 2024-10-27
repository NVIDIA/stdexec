#include <catch2/catch.hpp>
#include <exec/async_scope.hpp>
#include "exec/static_thread_pool.hpp"

namespace ex = stdexec;
using exec::async_scope;
using stdexec::sync_wait;

namespace {

  TEST_CASE("async_scope can be created and them immediately destructed", "[async_scope][dtor]") {
    async_scope scope;
    (void) scope;
  }

  TEST_CASE("async_scope destruction after spawning work into it", "[async_scope][dtor]") {
    exec::static_thread_pool pool{4};
    ex::scheduler auto sch = pool.get_scheduler();
    std::atomic<int> counter{0};
    {
      async_scope scope;

      // Add some work into the scope
      for (int i = 0; i < 10; i++)
        scope.spawn(ex::starts_on(sch, ex::just() | ex::then([&] { counter++; })));

      // Wait on the work, before calling destructor
      sync_wait(scope.on_empty());
    }
    // We should have all the work executed
    REQUIRE(counter == 10);
  }
} // namespace
