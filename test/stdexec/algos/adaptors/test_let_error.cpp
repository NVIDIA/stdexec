/*
 * Copyright (c) 2022 Lucian Radu Teodorescu
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
#include <test_common/schedulers.hpp>
#include <test_common/senders.hpp>
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>
#include <exec/static_thread_pool.hpp>
#include <exec/env.hpp>

#include <chrono> // IWYU pragma: keep for chrono_literals

namespace ex = stdexec;

using namespace std::chrono_literals;

namespace {

  TEST_CASE("let_error returns a sender", "[adaptors][let_error]") {
    auto snd = ex::let_error(ex::just(), [](std::exception_ptr) { return ex::just(); });
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("let_error with environment returns a sender", "[adaptors][let_error]") {
    auto snd = ex::let_error(ex::just(), [](std::exception_ptr) { return ex::just(); });
    static_assert(ex::sender_in<decltype(snd), ex::env<>>);
    (void) snd;
  }

  TEST_CASE("let_error simple example", "[adaptors][let_error]") {
    bool called{false};
    auto snd = ex::let_error(ex::just_error(std::exception_ptr{}), [&](std::exception_ptr) {
      called = true;
      return ex::just();
    });
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);
    // The receiver checks that it's called
    // we also check that the function was invoked
    CHECK(called);
  }

  TEST_CASE("let_error simple example reference", "[adaptors][let_error]") {
    bool called{false};
    auto snd =
      ex::let_error(ex::split(ex::just_error(std::exception_ptr{})), [&](std::exception_ptr) {
        called = true;
        return ex::just();
      });
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);
    // The receiver checks that it's called
    // we also check that the function was invoked
    CHECK(called);
  }

  TEST_CASE("let_error can be piped", "[adaptors][let_error]") {
    ex::sender auto snd = ex::just() | ex::let_error([](std::exception_ptr) { return ex::just(); });
    (void) snd;
  }

  TEST_CASE(
    "let_error returning void can be waited on (error annihilation)",
    "[adaptors][let_error]") {
    ex::sender auto snd = ex::just_error(std::exception_ptr{})
                        | ex::let_error([](std::exception_ptr) { return ex::just(); });
    stdexec::sync_wait(std::move(snd));
  }

#if !STDEXEC_STD_NO_EXCEPTIONS()
  TEST_CASE("let_error can be used to produce values (error to value)", "[adaptors][let_error]") {
    ex::sender auto snd = ex::just() | ex::then([]() -> std::string {
                            throw std::logic_error{"error description"};
                          })
                        | ex::let_error([](std::exception_ptr eptr) {
                            STDEXEC_TRY {
                              std::rethrow_exception(eptr);
                            }
                            STDEXEC_CATCH(const std::exception& e) {
                              return ex::just(std::string{e.what()});
                            }
                          });
    wait_for_value(std::move(snd), std::string{"error description"});
    (void) snd;
  }

  TEST_CASE("let_error can be used to transform errors", "[adaptors][let_error]") {
    ex::sender auto snd = ex::just_error(1)
                        | ex::let_error(
                            [](int error_code) -> decltype(ex::just_error(std::exception_ptr{})) {
                              char buf[20];
                              std::snprintf(buf, 20, "%d", error_code);
                              throw std::logic_error(buf);
                            });

    auto op = ex::connect(std::move(snd), expect_error_receiver{});
    ex::start(op);
  }

  TEST_CASE("let_error can throw, and yield a different error type", "[adaptors][let_error]") {
    auto snd = ex::just_error(13) | ex::let_error([](int x) {
                 if (x % 2 == 0)
                   throw std::logic_error{"err"};
                 return ex::just_error(x);
               });
    auto op = ex::connect(std::move(snd), expect_error_receiver{13});
    ex::start(op);
  }
#endif // !STDEXEC_STD_NO_EXCEPTIONS()

  TEST_CASE("let_error can be used with just_stopped", "[adaptors][let_error]") {
    ex::sender auto snd = ex::just_stopped()
                        | ex::let_error([](std::exception_ptr) { return ex::just(17); });
    auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
    ex::start(op);
  }

  TEST_CASE("let_error function is not called on regular flow", "[adaptors][let_error]") {
    bool called{false};
    error_scheduler sched;
    ex::sender auto snd = ex::just() | ex::then([] { return 13; })
                        | ex::let_error([&](std::exception_ptr) {
                            called = true;
                            return ex::just(0);
                          });
    auto op = ex::connect(std::move(snd), expect_value_receiver{13});
    ex::start(op);
    CHECK_FALSE(called);
  }

  TEST_CASE("let_error function is not called when cancelled", "[adaptors][let_error]") {
    bool called{false};
    stopped_scheduler sched;
    ex::sender auto snd = ex::transfer_just(sched, 13) | ex::let_error([&](std::exception_ptr) {
                            called = true;
                            return ex::just(0);
                          });
    auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
    ex::start(op);
    CHECK_FALSE(called);
  }

  // Type that sets into a received boolean when the dtor is called
  struct my_type {
    bool* p_called_{nullptr};

    explicit my_type(bool* p_called)
      : p_called_(p_called) {
    }

    my_type(my_type&& rhs)
      : p_called_(rhs.p_called_) {
      rhs.p_called_ = nullptr;
    }

    auto operator=(my_type&& rhs) -> my_type& {
      if (p_called_)
        *p_called_ = true;
      p_called_ = rhs.p_called_;
      rhs.p_called_ = nullptr;
      return *this;
    }

    ~my_type() {
      if (p_called_)
        *p_called_ = true;
    }
  };

  TEST_CASE("let_error of just_error with custom type", "[adaptors][let_error]") {
    bool param_destructed{false};
    ex::sender auto snd = ex::just_error(my_type(&param_destructed))
                        | ex::let_error([&](const my_type&) { return ex::just(13); });

    {
      auto op = ex::connect(std::move(snd), expect_value_receiver{13});
      CHECK_FALSE(param_destructed);
      ex::start(op);
      CHECK_FALSE(param_destructed);
    }
    // the parameter is destructed once the operation_state object is destructed
    CHECK(param_destructed);
  }

  TEST_CASE(
    "let_error exposes a parameter that is destructed when the main operation is destructed ",
    "[adaptors][let_error]") {
    bool param_destructed{false};
    bool fun_called{false};
    impulse_scheduler sched;

    ex::sender auto s1 = ex::just_error(my_type(&param_destructed));
    ex::sender auto snd = ex::just_error(my_type(&param_destructed))
                        | ex::let_error([&](const my_type&) {
                            CHECK_FALSE(param_destructed);
                            fun_called = true;
                            return ex::transfer_just(sched, 13);
                          });
    int res{0};
    {
      auto op = ex::connect(std::move(snd), expect_value_receiver_ex{res});
      ex::start(op);
      // The function is called immediately after starting the operation
      CHECK(fun_called);
      // As the returned sender didn't complete yet, the parameter must still be alive
      CHECK_FALSE(param_destructed);
      CHECK(res == 0);

      // Now, tell the scheduler to execute the final operation
      sched.start_next();

      // As the main operation is still valid, the parameter is not yet destructed
      CHECK_FALSE(param_destructed);
    }

    // At this point everything can be destructed
    CHECK(param_destructed);
    CHECK(res == 13);
  }

  struct int_err_transform {
    auto operator()(std::exception_ptr ep) const -> fallible_just<int> {
      std::rethrow_exception(ep);
      return fallible_just{0};
    }

    auto operator()(int x) const -> fallible_just<int> {
      return fallible_just{x * 2 - 1};
    }
  };

  TEST_CASE("let_error works when changing threads", "[adaptors][let_error]") {
    exec::static_thread_pool pool{2};
    std::atomic<bool> called{false};
    {
      // lunch some work on the thread pool
      ex::sender auto snd = ex::starts_on(pool.get_scheduler(), ex::just_error(7))
                          | ex::let_error(int_err_transform{}) | ex::then([&](auto x) -> void {
                              CHECK(x == 13);
                              called.store(true);
                            });
      ex::start_detached(std::move(snd));
    }
    // wait for the work to be executed, with timeout
    // perform a poor-man's sync
    // NOTE: it's a shame that the `join` method in static_thread_pool is not public
    for (int i = 0; i < 1000 && !called.load(); i++)
      std::this_thread::sleep_for(1ms);
    // the work should be executed
    REQUIRE(called);
  }

  TEST_CASE(
    "let_error has the values_type from the input sender if returning error",
    "[adaptors][let_error]") {
    check_val_types<ex::__mset<pack<int>>>(
      fallible_just{7} | ex::let_error([](std::exception_ptr) { return ex::just_error(0); }));
    check_val_types<ex::__mset<pack<double>>>(
      fallible_just{3.14} | ex::let_error([](std::exception_ptr) { return ex::just_error(0); }));
    check_val_types<ex::__mset<pack<std::string>>>(
      fallible_just{std::string{"hello"}}
      | ex::let_error([](std::exception_ptr) { return ex::just_error(0); }));
  }

  TEST_CASE(
    "let_error adds to values_type the value types of the returned sender",
    "[adaptors][let_error]") {
    check_val_types<ex::__mset<pack<int>>>(
      fallible_just{1} | ex::let_error([](std::exception_ptr) { return ex::just(11); }));
    check_val_types<ex::__mset<pack<int>, pack<double>>>(
      fallible_just{1} | ex::let_error([](std::exception_ptr) { return ex::just(3.14); }));
    check_val_types<ex::__mset<pack<int>, pack<std::string>>>(
      fallible_just{1}
      | ex::let_error([](std::exception_ptr) { return ex::just(std::string{"hello"}); }));
  }

  TEST_CASE(
    "let_error overrides error_types from input sender (and adds std::exception_ptr)",
    "[adaptors][let_error]") {
    inline_scheduler sched1{};
    error_scheduler sched2{};
    error_scheduler<int> sched3{43};

    // Returning ex::just_error
    check_err_types<ex::__mset<>>(ex::transfer_just(sched1) | ex::let_error([](std::exception_ptr) {
                                    return ex::just_error(std::string{"err"});
                                  }));
    check_err_types<ex::__mset<std::exception_ptr, std::string>>(
      ex::transfer_just(sched2)
      | ex::let_error([](std::exception_ptr) { return ex::just_error(std::string{"err"}); }));
    check_err_types<ex::__mset<std::exception_ptr, std::string>>(
      ex::transfer_just(sched3) | ex::let_error([](ex::__one_of<int, std::exception_ptr> auto) {
        return ex::just_error(std::string{"err"});
      }));

    // Returning ex::just
    check_err_types<ex::__mset<>>(
      ex::transfer_just(sched1) | ex::let_error([](std::exception_ptr) { return ex::just(); }));
    check_err_types<ex::__mset<std::exception_ptr>>(
      ex::transfer_just(sched2) | ex::let_error([](std::exception_ptr) { return ex::just(); }));
    check_err_types<ex::__mset<std::exception_ptr>>(
      ex::transfer_just(sched3)
      | ex::let_error([](ex::__one_of<int, std::exception_ptr> auto) { return ex::just(); }));
  }

  TEST_CASE("let_error keeps sends_stopped from input sender", "[adaptors][let_error]") {
    inline_scheduler sched1{};
    error_scheduler sched2{};
    stopped_scheduler sched3{};

    check_sends_stopped<false>(
      ex::transfer_just(sched1) | ex::let_error([](std::exception_ptr) { return ex::just(); }));
    check_sends_stopped<true>(
      ex::transfer_just(sched2) | ex::let_error([](std::exception_ptr) { return ex::just(); }));
    check_sends_stopped<true>(
      ex::transfer_just(sched3) | ex::let_error([](std::exception_ptr) { return ex::just(); }));
  }

  // Return a different sender when we invoke this custom defined let_error implementation
  struct let_error_test_domain {
    template <class Sender>
      requires std::same_as<ex::tag_of_t<Sender>, ex::let_error_t>
    static auto transform_sender(Sender&&) {
      return ex::just(std::string{"what error?"});
    }
  };

  TEST_CASE("let_error can be customized", "[adaptors][let_error]") {
    // The customization will return a different value
    auto snd = ex::just(std::string{"hello"})
             | exec::write_attrs(ex::prop{ex::get_domain, let_error_test_domain{}})
             | ex::let_error([](std::exception_ptr) { return ex::just(std::string{"err"}); });
    wait_for_value(std::move(snd), std::string{"what error?"});
  }
} // namespace
