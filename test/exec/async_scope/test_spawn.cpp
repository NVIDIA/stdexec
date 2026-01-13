#include "test_common/receivers.hpp"
#include "test_common/schedulers.hpp"
#include "test_common/type_helpers.hpp"
#include <catch2/catch.hpp>
#include <exec/async_scope.hpp>

namespace ex = STDEXEC;
using exec::async_scope;
using STDEXEC::sync_wait;

namespace {

  //! Sender that throws exception when connected
  struct throwing_sender {
    using sender_concept = STDEXEC::sender_t;
    using completion_signatures = ex::completion_signatures<ex::set_value_t()>;

    template <class Receiver>
    struct operation {
      Receiver rcvr_;

      void start() & noexcept {
        ex::set_value(std::move(rcvr_));
      }
    };

    template <class Receiver>
    auto connect(Receiver&&) && -> operation<std::decay_t<Receiver>> {
      STDEXEC_THROW(std::logic_error("cannot connect"));
    }
  };

  TEST_CASE("spawn will execute its work", "[async_scope][spawn]") {
    impulse_scheduler sch;
    bool executed{false};
    async_scope scope;

    // Non-blocking call
    scope.spawn(ex::starts_on(sch, ex::just() | ex::then([&] { executed = true; })));
    REQUIRE_FALSE(executed);
    // Run the operation on the scheduler
    sch.start_next();
    // Now the spawn work should be completed
    REQUIRE(executed);
  }

  TEST_CASE("spawn will start sender before returning", "[async_scope][spawn]") {
    bool executed{false};
    async_scope scope;

    // This will be a blocking call
    scope.spawn(ex::just() | ex::then([&] { executed = true; }));
    REQUIRE(executed);
  }

#if !STDEXEC_NO_STD_EXCEPTIONS()
  TEST_CASE(
    "spawn will propagate exceptions encountered during op creation",
    "[async_scope][spawn]") {
    async_scope scope;
    STDEXEC_TRY {
      scope.spawn(throwing_sender{} | ex::then([&] { FAIL("work should not be executed"); }));
      FAIL("Exceptions should have been thrown");
    }
    STDEXEC_CATCH(const std::logic_error& e) {
      SUCCEED("correct exception caught");
    }
    STDEXEC_CATCH_ALL {
      FAIL("invalid exception caught");
    }
  }
#endif // !STDEXEC_NO_STD_EXCEPTIONS()

  TEST_CASE(
    "TODO: spawn will keep the scope non-empty until the work is executed",
    "[async_scope][spawn]") {
    impulse_scheduler sch;
    bool executed{false};
    async_scope scope;

    // Before adding any operations, the scope is empty
    // TODO: reenable this
    // REQUIRE(P2519::__scope::empty(scope));

    // Non-blocking call
    scope.spawn(ex::starts_on(sch, ex::just() | ex::then([&] { executed = true; })));
    REQUIRE_FALSE(executed);

    // The scope is now non-empty
    // TODO: reenable this
    // REQUIRE_FALSE(P2519::__scope::empty(scope));
    // REQUIRE(P2519::__scope::op_count(scope) == 1);

    // Run the operation on the scheduler; blocking call
    sch.start_next();

    // Now the scope should again be empty
    // TODO: reenable this
    // REQUIRE(P2519::__scope::empty(scope));
    REQUIRE(executed);
  }

  TEST_CASE(
    "TODO: spawn will keep track on how many operations are in flight",
    "[async_scope][spawn]") {
    impulse_scheduler sch;
    std::size_t num_executed{0};
    async_scope scope;

    // Before adding any operations, the scope is empty
    // TODO: reenable this
    // REQUIRE(P2519::__scope::op_count(scope) == 0);
    // REQUIRE(P2519::__scope::empty(scope));

    constexpr std::size_t num_oper = 10;
    for (std::size_t i = 0; i < num_oper; i++) {
      scope.spawn(ex::starts_on(sch, ex::just() | ex::then([&] { num_executed++; })));
      size_t num_expected_ops = i + 1;
      // TODO: reenable this
      // REQUIRE(P2519::__scope::op_count(scope) == num_expected_ops);
      (void) num_expected_ops;
    }

    // Now execute the operations
    for (std::size_t i = 0; i < num_oper; i++) {
      sch.start_next();
      size_t num_expected_ops = num_oper - i - 1;
      // TODO: reenable this
      // REQUIRE(P2519::__scope::op_count(scope) == num_expected_ops);
      (void) num_expected_ops;
    }

    // The scope is empty after all the operations are executed
    // TODO: reenable this
    // REQUIRE(P2519::__scope::empty(scope));
    REQUIRE(num_executed == num_oper);
  }

  TEST_CASE("TODO: spawn work can be cancelled by cancelling the scope", "[async_scope][spawn]") {
    impulse_scheduler sch;
    async_scope scope;

    bool cancelled1{false};
    bool cancelled2{false};

    scope.spawn(ex::starts_on(sch, ex::just() | ex::let_stopped([&] {
                                     cancelled1 = true;
                                     return ex::just();
                                   })));
    scope.spawn(ex::starts_on(sch, ex::just() | ex::let_stopped([&] {
                                     cancelled2 = true;
                                     return ex::just();
                                   })));

    // TODO: reenable this
    // REQUIRE(P2519::__scope::op_count(scope) == 2);

    // Execute the first operation, before cancelling
    sch.start_next();
    REQUIRE_FALSE(cancelled1);
    REQUIRE_FALSE(cancelled2);

    // Cancel the async_scope object
    scope.request_stop();
    // TODO: reenable this
    // REQUIRE(P2519::__scope::op_count(scope) == 1);

    // Execute the first operation, after cancelling
    sch.start_next();
    REQUIRE_FALSE(cancelled1);
    // TODO: second operation should be cancelled
    // REQUIRE(cancelled2);
    REQUIRE_FALSE(cancelled2);

    // TODO: reenable this
    // REQUIRE(P2519::__scope::empty(scope));
  }

  template <typename S>
  concept is_spawn_worthy = requires(async_scope& scope, S&& snd) { scope.spawn(std::move(snd)); };

  TEST_CASE("spawn accepts void senders", "[async_scope][spawn]") {
    static_assert(is_spawn_worthy<decltype(ex::just())>);
  }

  TEST_CASE("spawn doesn't accept non-void senders", "[async_scope][spawn]") {
    static_assert(!is_spawn_worthy<decltype(ex::just(13))>);
    static_assert(!is_spawn_worthy<decltype(ex::just(3.14))>);
    static_assert(!is_spawn_worthy<decltype(ex::just("hello"))>);
  }

  TEST_CASE("TODO: spawn doesn't accept senders of errors", "[async_scope][spawn]") {
    // TODO: check if just_error(exception_ptr) should be allowed
    static_assert(is_spawn_worthy<decltype(ex::just_error(std::exception_ptr{}))>);
    static_assert(!is_spawn_worthy<decltype(ex::just_error(std::error_code{}))>);
    static_assert(!is_spawn_worthy<decltype(ex::just_error(-1))>);
  }

  TEST_CASE("spawn should accept senders that send stopped signal", "[async_scope][spawn]") {
    static_assert(is_spawn_worthy<decltype(ex::just_stopped())>);
  }

  TEST_CASE(
    "TODO: spawn works with senders that complete with stopped signal",
    "[async_scope][spawn]") {
    impulse_scheduler sch;
    async_scope scope;

    // TODO: reenable this
    // REQUIRE(P2519::__scope::empty(scope));

    scope.spawn(ex::starts_on(sch, ex::just_stopped()));

    // The scope is now non-empty
    // TODO: reenable this
    // REQUIRE_FALSE(P2519::__scope::empty(scope));
    // REQUIRE(P2519::__scope::op_count(scope) == 1);

    // Run the operation on the scheduler; blocking call
    sch.start_next();

    // Now the scope should again be empty
    // TODO: reenable this
    // REQUIRE(P2519::__scope::empty(scope));
  }
} // namespace
