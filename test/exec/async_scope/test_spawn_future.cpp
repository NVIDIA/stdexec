#include "test_common/receivers.hpp"
#include "test_common/schedulers.hpp"
#include <catch2/catch.hpp>
#include <exec/async_scope.hpp>
#include <exec/just_from.hpp>
#include <exec/static_thread_pool.hpp>

namespace ex = STDEXEC;
using exec::async_scope;
using ex::sync_wait;

namespace {
  void expect_empty(exec::async_scope& scope) {
    ex::run_loop loop;
    ex::scheduler auto sch = loop.get_scheduler();
    CHECK_FALSE(ex::execute_may_block_caller(sch));
    auto op = ex::connect(
      ex::then(
        scope.on_empty(),
        [&]() {
          loop.finish();
    }),
      expect_void_receiver{ex::prop{ex::get_scheduler, sch}});
    ex::start(op);
    loop.run();
  }

#if !STDEXEC_NO_STD_EXCEPTIONS()
  //! Sender that throws exception when connected
  struct throwing_sender {
    using sender_concept = ex::sender_t;
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
      throw std::logic_error("cannot connect");
    }
  };
#endif // !STDEXEC_NO_STD_EXCEPTIONS()

  TEST_CASE("spawn_future will execute its work", "[async_scope][spawn_future]") {
    impulse_scheduler sch;
    bool executed{false};
    async_scope scope;

    // Non-blocking call
    {
      ex::sender auto snd = scope.spawn_future(
        ex::starts_on(sch, ex::just() | ex::then([&] { executed = true; })));
      (void) snd;
    }
    REQUIRE_FALSE(executed);
    // Run the operation on the scheduler
    sch.start_next();
    // Now the spawn work should be completed
    REQUIRE(executed);
    expect_empty(scope);
  }

  TEST_CASE("spawn_future sender will complete", "[async_scope][spawn_future]") {
    impulse_scheduler sch;
    bool executed1{false};
    bool executed2{false};
    async_scope scope;

    // Non-blocking call
    ex::sender auto snd = scope.spawn_future(
      ex::starts_on(sch, ex::just() | ex::then([&] { executed1 = true; })));
    auto op = ex::connect(std::move(snd), expect_void_receiver_ex{executed2});
    ex::start(op);
    REQUIRE_FALSE(executed1);
    REQUIRE_FALSE(executed2);
    // Run the operation on the scheduler
    sch.start_next();
    // Now the work from `snd` should be completed
    REQUIRE(executed1);
    REQUIRE(executed2);
    expect_empty(scope);
  }

  TEST_CASE(
    "spawn_future sender will complete after the given sender completes",
    "[async_scope][spawn_future]") {
    impulse_scheduler sch;
    bool executed{false};
    async_scope scope;

    // Non-blocking call
    ex::sender auto snd = scope.spawn_future(
      ex::starts_on(sch, ex::just() | ex::then([&] { executed = true; })));
    ex::sender auto snd2 = std::move(snd) | ex::then([&] { REQUIRE(executed); });
    // Execute the given work
    sch.start_next();
    // Ensure `snd2` is complete
    wait_for_value(std::move(snd2));
    expect_empty(scope);
  }

  TEST_CASE("spawn_future returned sender can be dropped", "[async_scope][spawn_future]") {
    impulse_scheduler sch;
    std::atomic_bool executed{false};
    async_scope scope;

    // Non-blocking call; simply ignore the returned sender
    (void) scope.spawn_future(ex::starts_on(sch, ex::just() | ex::then([&] { executed = true; })));
    REQUIRE_FALSE(executed.load());
    // Execute the given work
    sch.start_next();
    REQUIRE(executed.load());
    expect_empty(scope);
  }

  TEST_CASE(
    "spawn_future returned sender can be captured and dropped",
    "[async_scope][spawn_future]") {
    impulse_scheduler sch;
    bool executed{false};
    async_scope scope;

    // Non-blocking call; simply ignore the returned sender
    {
      ex::sender auto snd = scope.spawn_future(
        ex::starts_on(sch, ex::just() | ex::then([&] { executed = true; })));
      (void) snd;
    }
    REQUIRE_FALSE(executed);
    // Execute the given work
    sch.start_next();
    REQUIRE(executed);
    expect_empty(scope);
  }

#if !STDEXEC_NO_STD_EXCEPTIONS()
  TEST_CASE("spawn_future with throwing copy", "[async_scope][spawn_future]") {
    async_scope scope;
    exec::static_thread_pool pool{2};

    struct throwing_copy {
      throwing_copy() = default;

      throwing_copy(const throwing_copy&) {
        throw std::logic_error("cannot copy");
      }
    };

    ex::sender auto snd = scope.spawn_future(
      ex::starts_on(pool.get_scheduler(), exec::just_from([](auto sink) {
                      return sink(throwing_copy());
                    })));
    STDEXEC_TRY {
      sync_wait(std::move(snd));
      FAIL("Exceptions should have been thrown");
    }
    STDEXEC_CATCH(const std::logic_error& e) {
      SUCCEED("correct exception caught");
    }
    STDEXEC_CATCH_ALL {
      FAIL("invalid exception caught");
    }
    sync_wait(scope.on_empty());
  }
#endif // !STDEXEC_NO_STD_EXCEPTIONS()

  TEST_CASE(
    "spawn_future returned sender can be connected but not started",
    "[async_scope][spawn_future]") {
    impulse_scheduler sch;
    bool executed{false};
    bool executed2{false};
    async_scope scope;

    // Non-blocking call; simply ignore the returned sender
    ex::sender auto snd = scope.spawn_future(
      ex::starts_on(sch, ex::just() | ex::then([&] { executed = true; })));
    auto op = ex::connect(std::move(snd), expect_void_receiver_ex{executed2});
    REQUIRE_FALSE(executed);
    REQUIRE_FALSE(executed2);
    // Execute the given work
    sch.start_next();
    REQUIRE(executed);
    // Our final receiver will not be notified (as `op` was not started)
    REQUIRE_FALSE(executed2);
    expect_empty(scope);
  }

  TEST_CASE("spawn_future will start sender before returning", "[async_scope][spawn_future]") {
    bool executed{false};
    async_scope scope;

    // This will be a blocking call
    {
      ex::sender auto snd = scope.spawn_future(ex::just() | ex::then([&] { executed = true; }));
      (void) snd;
    }
    REQUIRE(executed);
    expect_empty(scope);
  }

  TEST_CASE(
    "spawn_future returned sender can be started after given sender completed",
    "[async_scope][spawn_future]") {
    impulse_scheduler sch;
    bool executed{false};
    bool executed2{false};
    async_scope scope;

    ex::sender auto snd = scope.spawn_future(
      ex::starts_on(sch, ex::just() | ex::then([&] { executed = true; })));
    REQUIRE_FALSE(executed);
    // Execute the work given to spawn_future
    sch.start_next();
    REQUIRE_FALSE(executed2);
    // Now connect the returned sender
    auto op = ex::connect(std::move(snd), expect_void_receiver_ex{executed2});
    ex::start(op);
    REQUIRE(executed2);
    expect_empty(scope);
  }

#if !STDEXEC_NO_STD_EXCEPTIONS()
  TEST_CASE(
    "spawn_future will propagate exceptions encountered during op creation",
    "[async_scope][spawn_future]") {
    async_scope scope;
    STDEXEC_TRY {
      ex::sender auto snd = scope.spawn_future(
        throwing_sender{} | ex::then([&] { FAIL("work should not be executed"); }));
      (void) snd;
      FAIL("Exceptions should have been thrown");
    }
    STDEXEC_CATCH(const std::logic_error& e) {
      SUCCEED("correct exception caught");
    }
    STDEXEC_CATCH_ALL {
      FAIL("invalid exception caught");
    }
    expect_empty(scope);
  }
#endif // !STDEXEC_NO_STD_EXCEPTIONS()

  TEST_CASE(
    "TODO: spawn_future will keep the scope non-empty until the work is executed",
    "[async_scope][spawn_future]") {
    impulse_scheduler sch;
    bool executed{false};
    async_scope scope;

    // Before adding any operations, the scope is empty
    // TODO: reenable this
    // REQUIRE(P2519::__scope::empty(scope));

    // Non-blocking call
    {
      ex::sender auto snd = scope.spawn_future(
        ex::starts_on(sch, ex::just() | ex::then([&] { executed = true; })));
      (void) snd;
    }
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
    "TODO: spawn_future will keep track on how many operations are in flight",
    "[async_scope][spawn_future]") {
    impulse_scheduler sch;
    // std::size_t num_executed{0};
    async_scope scope;

    // Before adding any operations, the scope is empty
    // TODO: reenable this
    // REQUIRE(P2519::__scope::op_count(scope) == 0);
    // REQUIRE(P2519::__scope::empty(scope));

    // TODO: this will fail when running multiple iterations
    // constexpr std::size_t num_oper = 10;
    // for (std::size_t i = 0; i < num_oper; i++) {
    //     ex::sender auto snd =
    //             scope.spawn_future(ex::starts_on(sch, ex::just() | ex::then([&] { num_executed++; })));
    //     (void)snd;
    //     size_t num_expected_ops = i + 1;
    //     REQUIRE(P2519::__scope::op_count(scope) == num_expected_ops);
    // }

    // // Now execute the operations
    // for (std::size_t i = 0; i < num_oper; i++) {
    //     sch.start_next();
    //     size_t num_expected_ops = num_oper - i - 1;
    //     REQUIRE(P2519::__scope::op_count(scope) == num_expected_ops);
    // }

    // // The scope is empty after all the operations are executed
    // REQUIRE(P2519::__scope::empty(scope));
    // REQUIRE(num_executed == num_oper);

    expect_empty(scope);
  }

  TEST_CASE(
    "TODO: spawn_future work can be cancelled by cancelling the scope",
    "[async_scope][spawn_future]") {
    impulse_scheduler sch;
    async_scope scope;

    bool cancelled1{false};
    bool cancelled2{false};

    {
      ex::sender auto snd1 = scope
                               .spawn_future(ex::starts_on(sch, ex::just() | ex::let_stopped([&] {
                                                                  cancelled1 = true;
                                                                  return ex::just();
                                                                })));
      ex::sender auto snd2 = scope
                               .spawn_future(ex::starts_on(sch, ex::just() | ex::let_stopped([&] {
                                                                  cancelled2 = true;
                                                                  return ex::just();
                                                                })));
      (void) snd1;
      (void) snd2;
    }

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
    expect_empty(scope);
  }

  template <typename S>
  concept is_spawn_future_worthy = requires(async_scope& scope, S&& snd) {
    scope.spawn_future(std::move(snd));
  };

  TEST_CASE("spawn_future accepts void senders", "[async_scope][spawn_future]") {
    static_assert(is_spawn_future_worthy<decltype(ex::just())>);
  }

  TEST_CASE("spawn_future accepts non-void senders", "[async_scope][spawn_future]") {
    static_assert(is_spawn_future_worthy<decltype(ex::just(13))>);
    static_assert(is_spawn_future_worthy<decltype(ex::just(3.14))>);
    static_assert(is_spawn_future_worthy<decltype(ex::just("hello"))>);
  }

  TEST_CASE("spawn_future accepts senders of errors", "[async_scope][spawn_future]") {
    static_assert(is_spawn_future_worthy<decltype(ex::just_error(std::exception_ptr{}))>);
    static_assert(is_spawn_future_worthy<decltype(ex::just_error(std::error_code{}))>);
    static_assert(is_spawn_future_worthy<decltype(ex::just_error(-1))>);
  }

  TEST_CASE(
    "spawn_future should accept senders that send stopped signal",
    "[async_scope][spawn_future]") {
    static_assert(is_spawn_future_worthy<decltype(ex::just_stopped())>);
  }

  TEST_CASE(
    "TODO: spawn_future works with senders that complete with stopped signal",
    "[async_scope][spawn_future]") {
    impulse_scheduler sch;
    async_scope scope;

    // TODO: reenable this
    // REQUIRE(P2519::__scope::empty(scope));

    // TODO: make this work
    // ex::sender auto snd = scope.spawn_future(ex::starts_on(sch, ex::just_stopped()));
    // (void)snd;

    // // The scope is now non-empty
    // REQUIRE_FALSE(P2519::__scope::empty(scope));
    // REQUIRE(P2519::__scope::op_count(scope) == 1);

    // // Run the operation on the scheduler; blocking call
    // sch.start_next();

    // // Now the scope should again be empty
    // REQUIRE(P2519::__scope::empty(scope));

    expect_empty(scope);
  }

  TEST_CASE("spawn_future forwards value to returned sender", "[async_scope][spawn_future]") {
    impulse_scheduler sch;
    async_scope scope;

    // TODO: reenable this
    // REQUIRE(P2519::__scope::empty(scope));

    ex::sender auto snd = scope.spawn_future(ex::starts_on(sch, ex::just(13)));
    sch.start_next();
    wait_for_value(std::move(snd), 13);
    expect_empty(scope);
  }

  TEST_CASE("TODO: spawn_future forwards error to returned sender", "[async_scope][spawn_future]") {
    impulse_scheduler sch;
    async_scope scope;

    // TODO: reenable this
    // REQUIRE(P2519::__scope::empty(scope));

    // TODO: fix this
    // ex::sender auto snd = scope.spawn_future(ex::starts_on(sch, ex::just_error(-1)));
    // sch.start_next();
    // try
    // {
    //     sync_wait(std::move(snd));
    //     FAIL("Should not reach this point");
    // }
    // catch(int error)
    // {
    //     REQUIRE(error == -1);
    // }
    expect_empty(scope);
  }

  TEST_CASE(
    "TODO: spawn_future forwards stopped signal to returned sender",
    "[async_scope][spawn_future]") {
    // impulse_scheduler sch;
    async_scope scope;

    // TODO: reenable this
    // REQUIRE(P2519::__scope::empty(scope));

    // TODO: fix this
    // ex::sender auto snd = scope.spawn_future(ex::starts_on(sch, ex::just_stopped()));
    // sch.start_next();
    // auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
    // ex::start(op);
    expect_empty(scope);
  }
} // namespace
