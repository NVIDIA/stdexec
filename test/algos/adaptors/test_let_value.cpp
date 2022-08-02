/*
 * Copyright (c) Lucian Radu Teodorescu
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
#if defined(__GNUC__) && !defined(__clang__)
#else

#include <catch2/catch.hpp>
#include <execution.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>
#include <examples/schedulers/static_thread_pool.hpp>

#include <chrono>

namespace ex = std::execution;

using namespace std::chrono_literals;

TEST_CASE("let_value returns a sender", "[adaptors][let_value]") {
  auto snd = ex::let_value(ex::just(), [] { return ex::just(); });
  static_assert(ex::sender<decltype(snd)>);
  (void)snd;
}
TEST_CASE("let_value with environment returns a sender", "[adaptors][let_value]") {
  auto snd = ex::let_value(ex::just(), [] { return ex::just(); });
  static_assert(ex::sender<decltype(snd), empty_env>);
  (void)snd;
}
TEST_CASE("let_value simple example", "[adaptors][let_value]") {
  bool called{false};
  auto snd = ex::let_value(ex::just(), [&] {
    called = true;
    return ex::just();
  });
  auto op = ex::connect(std::move(snd), expect_void_receiver{});
  ex::start(op);
  // The receiver checks that it's called
  // we also check that the function was invoked
  CHECK(called);
}

TEST_CASE("let_value can be piped", "[adaptors][let_value]") {
  ex::sender auto snd = ex::just() | ex::let_value([] { return ex::just(); });
  (void)snd;
}

TEST_CASE("let_value returning void can we waited on", "[adaptors][let_value]") {
  ex::sender auto snd = ex::just() | ex::let_value([] { return ex::just(); });
  std::this_thread::sync_wait(std::move(snd));
}

TEST_CASE("let_value can be used to produce values", "[adaptors][let_value]") {
  ex::sender auto snd = ex::just() | ex::let_value([] { return ex::just(13); });
  wait_for_value(std::move(snd), 13);
}

TEST_CASE("let_value can be used to transform values", "[adaptors][let_value]") {
  ex::sender auto snd = ex::just(13) | ex::let_value([](int& x) { return ex::just(x + 4); });
  wait_for_value(std::move(snd), 17);
}

TEST_CASE("let_value can be used with multiple parameters", "[adaptors][let_value]") {
  auto snd = ex::just(3, 0.1415) | ex::let_value([](int& x, double y) { return ex::just(x + y); });
  wait_for_value(std::move(snd), 3.1415);
}

TEST_CASE("let_value can be used to change the sender", "[adaptors][let_value]") {
  ex::sender auto snd = ex::just(13) | ex::let_value([](int& x) { return ex::just_error(x + 4); });
  auto op = ex::connect(std::move(snd), expect_error_receiver{});
  ex::start(op);
}

bool is_prime(int x) {
  if (x > 2 && (x % 2 == 0))
    return false;
  int d = 3;
  while (d * d < x) {
    if (x % d == 0)
      return false;
    d += 2;
  }
  return true;
}

TEST_CASE("let_value can be used for composition", "[adaptors][let_value]") {
  bool called1{false};
  bool called2{false};
  bool called3{false};
  auto f1 = [&](int& x) {
    called1 = true;
    return ex::just(2 * x);
  };
  auto f2 = [&](int& x) {
    called2 = true;
    return ex::just(x + 3);
  };
  auto f3 = [&](int& x) {
    called3 = true;
    if (!is_prime(x))
      throw std::logic_error("not prime");
    return ex::just(x);
  };
  ex::sender auto snd = ex::just(13)        //
                        | ex::let_value(f1) //
                        | ex::let_value(f2) //
                        | ex::let_value(f3) //
      ;
  wait_for_value(std::move(snd), 29);
  CHECK(called1);
  CHECK(called2);
  CHECK(called3);
}

TEST_CASE("let_value can throw, and set_error will be called", "[adaptors][let_value]") {
  auto snd = ex::just(13) //
             | ex::let_value([](int& x) {
                 throw std::logic_error{"err"};
                 return ex::just(x + 5);
               });
  auto op = ex::connect(std::move(snd), expect_error_receiver{});
  ex::start(op);
}

TEST_CASE("let_value can be used with just_error", "[adaptors][let_value]") {
  ex::sender auto snd = ex::just_error(std::string{"err"}) //
                        | ex::let_value([]() { return ex::just(17); });
  auto op = ex::connect(std::move(snd), expect_error_receiver{});
  ex::start(op);
}
TEST_CASE("let_value can be used with just_stopped", "[adaptors][let_value]") {
  ex::sender auto snd = ex::just_stopped() | //
                        ex::let_value([]() { return ex::just(17); });
  auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
  ex::start(op);
}

TEST_CASE("let_value function is not called on error", "[adaptors][let_value]") {
  bool called{false};
  error_scheduler sched;
  ex::sender auto snd = ex::transfer_just(sched, 13) //
                        | ex::let_value([&](int& x) {
                            called = true;
                            return ex::just(x + 5);
                          });
  auto op = ex::connect(std::move(snd), expect_error_receiver{});
  ex::start(op);
  CHECK_FALSE(called);
}
TEST_CASE("let_value function is not called when cancelled", "[adaptors][let_value]") {
  bool called{false};
  stopped_scheduler sched;
  ex::sender auto snd = ex::transfer_just(sched, 13) //
                        | ex::let_value([&](int& x) {
                            called = true;
                            return ex::just(x + 5);
                          });
  auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
  ex::start(op);
  CHECK_FALSE(called);
}

TEST_CASE("let_value exposes a parameter that is destructed when the main operation is destructed",
    "[adaptors][let_value]") {

  // Type that sets into a received boolean when the dtor is called
  struct my_type {
    bool* p_called_{nullptr};
    explicit my_type(bool* p_called)
        : p_called_(p_called) {}
    my_type(my_type&& rhs)
        : p_called_(rhs.p_called_) {
      rhs.p_called_ = nullptr;
    }
    my_type& operator=(my_type&& rhs) {
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

  bool param_destructed{false};
  bool fun_called{false};
  impulse_scheduler sched;

  ex::sender auto snd = ex::just(my_type(&param_destructed)) //
                        | ex::let_value([&](const my_type& obj) {
                            CHECK_FALSE(param_destructed);
                            fun_called = true;
                            return ex::transfer_just(sched, 13);
                          });

  {
    int res{0};
    auto op = ex::connect(std::move(snd), expect_value_receiver_ex<int>{&res});
    ex::start(op);
    // The function is called immediately after starting the operation
    CHECK(fun_called);
    // As the returned sender didn't complete yet, the parameter must still be alive
    CHECK_FALSE(param_destructed);
    CHECK(res == 0);

    // Now, tell the scheduler to execute the final operation
    sched.start_next();

    // The parameter is going to be destructed when the op is destructed; it should be valid now
    CHECK_FALSE(param_destructed);
    CHECK(res == 13);
  }

  // At this point everything can be destructed
  CHECK(param_destructed);
}

TEST_CASE("let_value works when changing threads", "[adaptors][let_value]") {
  example::static_thread_pool pool{2};
  bool called{false};
  {
    // lunch some work on the thread pool
    ex::sender auto snd = ex::transfer_just(pool.get_scheduler(), 7)                 //
                          | ex::let_value([](int& x) { return ex::just(x * 2 - 1); }) //
                          | ex::then([&](int x) {
                              CHECK(x == 13);
                              called = true;
                            });
    ex::start_detached(std::move(snd));
  }
  // wait for the work to be executed, with timeout
  // perform a poor-man's sync
  // NOTE: it's a shame that the `join` method in static_thread_pool is not public
  for (int i = 0; i < 1000 && !called; i++)
    std::this_thread::sleep_for(1ms);
  // the work should be executed
  REQUIRE(called);
}

TEST_CASE(
    "let_value has the values_type corresponding to the given values", "[adaptors][let_value]") {
  check_val_types<type_array<type_array<int>>>(
      ex::just() | ex::let_value([] { return ex::just(7); }));
  check_val_types<type_array<type_array<double>>>(
      ex::just() | ex::let_value([] { return ex::just(3.14); }));
  check_val_types<type_array<type_array<std::string>>>(
      ex::just() | ex::let_value([] { return ex::just(std::string{"hello"}); }));
}
TEST_CASE("let_value keeps error_types from input sender", "[adaptors][let_value]") {
  inline_scheduler sched1{};
  error_scheduler sched2{};
  error_scheduler<int> sched3{43};

  check_err_types<type_array<std::exception_ptr>>( //
      ex::transfer_just(sched1) | ex::let_value([] { return ex::just(); }));
  check_err_types<type_array<std::exception_ptr>>( //
      ex::transfer_just(sched2) | ex::let_value([] { return ex::just(); }));
  check_err_types<type_array<int, std::exception_ptr>>( //
      ex::transfer_just(sched3) | ex::let_value([] { return ex::just(); }));
}
TEST_CASE("let_value keeps sends_stopped from input sender", "[adaptors][let_value]") {
  inline_scheduler sched1{};
  error_scheduler sched2{};
  stopped_scheduler sched3{};

  check_sends_stopped<false>( //
      ex::transfer_just(sched1) | ex::let_value([] { return ex::just(); }));
  check_sends_stopped<true>( //
      ex::transfer_just(sched2) | ex::let_value([] { return ex::just(); }));
  check_sends_stopped<true>( //
      ex::transfer_just(sched3) | ex::let_value([] { return ex::just(); }));
}

// Return a different sender when we invoke this custom defined on implementation
using my_string_sender_t = decltype(ex::transfer_just(inline_scheduler{}, std::string{}));
template <typename Fun>
auto tag_invoke(ex::let_value_t, inline_scheduler sched, my_string_sender_t, Fun) {
  return ex::just(std::string{"hallo"});
}

TEST_CASE("let_value can be customized", "[adaptors][let_value]") {
  // The customization will return a different value
  auto snd = ex::transfer_just(inline_scheduler{}, std::string{"hello"}) //
             | ex::let_value([](std::string& x) { return ex::just(x + ", world"); });
  wait_for_value(std::move(snd), std::string{"hallo"});
}

#endif
