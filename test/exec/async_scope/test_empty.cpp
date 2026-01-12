#include "test_common/receivers.hpp"
#include "test_common/schedulers.hpp"
#include <catch2/catch.hpp>
#include <exec/async_scope.hpp>

namespace ex = STDEXEC;
using exec::async_scope;
using STDEXEC::sync_wait;

namespace {

  TEST_CASE("empty will complete immediately on an empty async_scope", "[async_scope][empty]") {
    async_scope scope;
    bool is_empty{false};

    ex::sender auto snd = scope.on_empty() | ex::then([&] { is_empty = true; });
    sync_wait(std::move(snd));
    REQUIRE(is_empty);
  }

  TEST_CASE("empty sender can properly connect a void receiver", "[async_scope][empty]") {
    async_scope scope;
    bool is_empty{false};

    scope.spawn(ex::just());

    ex::sender auto snd = scope.on_empty() | ex::then([&] { is_empty = true; });
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);
    REQUIRE(is_empty);
  }

  TEST_CASE("empty will complete after the work is done", "[async_scope][empty]") {
    impulse_scheduler sch;
    async_scope scope;

    // Add some work
    scope.spawn(ex::starts_on(sch, ex::just()));

    // The on_empty() sender cannot notify now
    bool is_empty{false};
    ex::sender auto snd = ex::starts_on(sch, scope.on_empty()) | ex::then([&] { is_empty = true; });
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);
    REQUIRE_FALSE(is_empty);

    // TODO: refactor this test
    sch.start_next();
    sch.start_next();
    // We should be notified now
    REQUIRE(is_empty);
  }

  TEST_CASE("TODO: empty can be used multiple times", "[async_scope][empty]") {
    impulse_scheduler sch;
    async_scope scope;

    // Add some work
    scope.spawn(ex::starts_on(sch, ex::just()));

    // The on_empty() sender cannot notify now
    bool is_empty{false};
    ex::sender auto snd = ex::starts_on(sch, scope.on_empty()) | ex::then([&] { is_empty = true; });
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);
    REQUIRE_FALSE(is_empty);

    // TODO: refactor this test
    sch.start_next();
    sch.start_next();
    // We should be notified now
    REQUIRE(is_empty);

    // Add some work
    scope.spawn(ex::starts_on(sch, ex::just()));

    // The on_empty() sender cannot notify now
    bool is_empty2{false};
    ex::sender auto snd2 = ex::starts_on(sch, scope.on_empty())
                         | ex::then([&] { is_empty2 = true; });
    auto op2 = ex::connect(std::move(snd2), expect_void_receiver{});
    ex::start(op2);
    REQUIRE_FALSE(is_empty2);

    // TODO: refactor this test
    sch.start_next();
    sch.start_next();
    // We should be notified now
    REQUIRE(is_empty2);
  }

// TODO: GCC-11 generates warnings (treated as errors) for the following test
#if !STDEXEC_GCC() || (STDEXEC_GCC_VERSION >= 12'00)
  TEST_CASE("waiting on work that spawns more work", "[async_scope][empty]") {
    impulse_scheduler sch;
    async_scope scope;

    bool work1_done{false};
    auto work1 = [&] {
      work1_done = true;
    };
    bool work2_done{false};
    auto work2 = [&] {
      // Spawn work
      scope.spawn(ex::starts_on(sch, ex::just() | ex::then(work1)));
      // We are done
      work2_done = true;
    };

    // Spawn work 2
    // No work is executed until the impulse scheduler dictates
    scope.spawn(ex::starts_on(sch, ex::just() | ex::then(work2)));

    // start an on_empty() sender
    bool is_empty{false};
    ex::sender auto snd = ex::starts_on(inline_scheduler{}, scope.on_empty())
                        | ex::then([&] { is_empty = true; });
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);
    REQUIRE_FALSE(work1_done);
    REQUIRE_FALSE(work2_done);
    REQUIRE_FALSE(is_empty);

    // Trigger the execution of work2
    // When work2 is done, work1 is not yet started
    sch.start_next();
    REQUIRE_FALSE(work1_done);
    REQUIRE(work2_done);
    REQUIRE_FALSE(is_empty);

    // Trigger the execution of work1
    // This will complete the on_empty() sender
    sch.start_next();
    REQUIRE(work1_done);
    REQUIRE(work2_done);
    REQUIRE(is_empty);
  }
#endif

// TODO: GCC-11 generates warnings (treated as errors) for the following test
#if !STDEXEC_GCC() || (STDEXEC_GCC_VERSION >= 12'00)
  TEST_CASE(
    "async_scope is empty after adding work when in cancelled state",
    "[async_scope][empty]") {
    impulse_scheduler sch;
    async_scope scope;

    bool is_empty1{false};
    ex::sender auto snd = ex::starts_on(inline_scheduler{}, scope.on_empty())
                        | ex::then([&] { is_empty1 = true; });
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);
    REQUIRE(is_empty1);

    // cancel & add work
    scope.request_stop();
    bool work_executed{false};
    scope.spawn(ex::starts_on(sch, ex::just()) | ex::upon_stopped([&] { work_executed = true; }));
    // note that we don't tell impulse sender to start the work

    bool is_empty2{false};
    ex::sender auto snd2 = ex::starts_on(inline_scheduler{}, scope.on_empty())
                         | ex::then([&] { is_empty2 = true; });
    auto op2 = ex::connect(std::move(snd2), expect_void_receiver{});
    ex::start(op2);
    REQUIRE_FALSE(is_empty2);

    REQUIRE_FALSE(work_executed);
    sch.start_next();
    REQUIRE(work_executed);
    REQUIRE(is_empty2);
  }
#endif
} // namespace
