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

#include <catch2/catch.hpp>
#include <execution.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>

#include <chrono>

namespace ex = std::execution;

using namespace std::chrono_literals;

TEST_CASE("let_done returns a sender", "[adaptors][let_done]") {
  auto snd = ex::let_done(ex::just(), [] { return ex::just(); });
  static_assert(ex::sender<decltype(snd)>);
  (void)snd;
}
TEST_CASE("let_done returns a typed_sender", "[adaptors][let_done]") {
  auto snd = ex::let_done(ex::just(), [] { return ex::just(); });
  static_assert(ex::typed_sender<decltype(snd), empty_env>);
  (void)snd;
}
TEST_CASE("let_done simple example", "[adaptors][let_done]") {
  bool called{false};
  auto snd = ex::let_done(ex::just_done(), [&] {
    called = true;
    return ex::just();
  });
  auto op = ex::connect(std::move(snd), expect_void_receiver{}, empty_env{});
  ex::start(op);
  // The receiver checks that it's called
  // we also check that the function was invoked
  CHECK(called);
}

TEST_CASE("let_done can be piped", "[adaptors][let_done]") {
  ex::sender auto snd = ex::just() | ex::let_done([] { return ex::just(); });
  (void)snd;
}

TEST_CASE(
    "let_done returning void can we waited on (cancel annihilation)", "[adaptors][let_done]") {
  ex::sender auto snd = ex::just_done() | ex::let_done([] { return ex::just(); });
  std::this_thread::sync_wait(std::move(snd));
}

TEST_CASE("let_done can be used to produce values (cancel to value)", "[adaptors][let_done]") {
  ex::sender auto snd = ex::just_done() //
                        | ex::let_done([] { return ex::just(std::string{"cancelled"}); });
  wait_for_value(std::move(snd), std::string{"cancelled"});
}

TEST_CASE("let_done can throw, calling set_error", "[adaptors][let_done]") {
  auto snd = ex::just_done() //
             | ex::let_done([] {
                 throw std::logic_error{"err"};
                 return ex::just(1);
               });
  auto op = ex::connect(std::move(snd), expect_error_receiver{}, empty_env{});
  ex::start(op);
}

TEST_CASE("TODO: let_done can be used with just_error", "[adaptors][let_done]") {
  ex::sender auto snd = ex::just_error(1) //
                        | ex::let_done([] { return ex::just(17); });
  // TODO: check why this doesn't work
  // auto op = ex::connect(std::move(snd), expect_done_receiver{}, empty_env{});
  // ex::start(op);
  (void)snd;
}

TEST_CASE("let_done function is not called on regular flow", "[adaptors][let_done]") {
  bool called{false};
  error_scheduler sched;
  ex::sender auto snd = ex::just(13) //
                        | ex::let_done([&] {
                            called = true;
                            return ex::just(0);
                          });
  auto op = ex::connect(std::move(snd), expect_value_receiver<int>{13}, empty_env{});
  ex::start(op);
  CHECK_FALSE(called);
}
TEST_CASE("let_done function is not called on error flow", "[adaptors][let_done]") {
  bool called{false};
  error_scheduler<int> sched;
  ex::sender auto snd = ex::transfer_just(sched, 13) //
                        | ex::let_done([&] {
                            called = true;
                            return ex::just(0);
                          });
  auto op = ex::connect(std::move(snd), expect_error_receiver{}, empty_env{});
  ex::start(op);
  CHECK_FALSE(called);
}

TEST_CASE("let_done has the values_type from the input sender if returning error",
    "[adaptors][let_done]") {
  check_val_types<type_array<type_array<int>>>(ex::just(7) //
                                               | ex::let_done([] { return ex::just_error(0); }));
  check_val_types<type_array<type_array<double>>>(ex::just(3.14) //
                                                  | ex::let_done([] { return ex::just_error(0); }));
  check_val_types<type_array<type_array<std::string>>>(
      ex::just(std::string{"hello"}) //
      | ex::let_done([] { return ex::just_error(0); }));
}
TEST_CASE(
    "let_done adds to values_type the value types of the returned sender", "[adaptors][let_done]") {
  check_val_types<type_array<type_array<int>>>(ex::just(1) //
                                               | ex::let_done([] { return ex::just(11); }));
  check_val_types<type_array<type_array<int>, type_array<double>>>(
      ex::just(1) //
      | ex::let_done([] { return ex::just(3.14); }));
  check_val_types<type_array<type_array<int>, type_array<std::string>>>(
      ex::just(1) //
      | ex::let_done([] { return ex::just(std::string{"hello"}); }));
}
TEST_CASE("let_done has the error_type from the input sender if returning value",
    "[adaptors][let_done]") {
  check_err_types<type_array<std::exception_ptr, int>>( //
      ex::just_error(7)                                 //
      | ex::let_done([] { return ex::just(0); }));
  check_err_types<type_array<std::exception_ptr, double>>( //
      ex::just_error(3.14)                                 //
      | ex::let_done([] { return ex::just(0); }));
  check_err_types<type_array<std::exception_ptr, std::string>>( //
      ex::just_error(std::string{"hello"})                      //
      | ex::let_done([] { return ex::just(0); }));
}
TEST_CASE("let_done adds to error_type of the input sender", "[adaptors][let_done]") {
  check_err_types<type_array<std::exception_ptr, std::string, int>>( //
      ex::just_error(std::string{})                                  //
      | ex::let_done([] { return ex::just_error(0); }));
  check_err_types<type_array<std::exception_ptr, std::string, double>>( //
      ex::just_error(std::string{})                                     //
      | ex::let_done([] { return ex::just_error(3.14); }));
  check_err_types<type_array<std::exception_ptr, std::string>>( //
      ex::just_error(std::string{})                             //
      | ex::let_done([] { return ex::just_error(std::string{"err"}); }));
}

TEST_CASE("let_done overrides send_done from input sender", "[adaptors][let_done]") {
  inline_scheduler sched1{};
  error_scheduler sched2{};
  error_scheduler<int> sched3{43};

  // Returning ex::just
  check_sends_done<false>(      //
      ex::transfer_just(sched1) //
      | ex::let_done([] { return ex::just(); }));
  check_sends_done<false>(      //
      ex::transfer_just(sched2) //
      | ex::let_done([] { return ex::just(); }));
  check_sends_done<false>(      //
      ex::transfer_just(sched3) //
      | ex::let_done([] { return ex::just(); }));

  // Returning ex::just_done
  check_sends_done<true>(       //
      ex::transfer_just(sched1) //
      | ex::let_done([] { return ex::just_done(); }));
  check_sends_done<true>(       //
      ex::transfer_just(sched2) //
      | ex::let_done([] { return ex::just_done(); }));
  check_sends_done<true>(       //
      ex::transfer_just(sched3) //
      | ex::let_done([] { return ex::just_done(); }));
}

// Return a different sender when we invoke this custom defined on implementation
using my_string_sender_t = decltype(ex::transfer_just(inline_scheduler{}, std::string{}));
template <typename Fun>
auto tag_invoke(ex::let_done_t, inline_scheduler sched, my_string_sender_t, Fun) {
  return ex::just(std::string{"Don't stop me now"});
}

TEST_CASE("let_done can be customized", "[adaptors][let_done]") {
  // The customization will return a different value
  auto snd = ex::transfer_just(inline_scheduler{}, std::string{"hello"}) //
             | ex::let_done([] { return ex::just(std::string{"done"}); });
  wait_for_value(std::move(snd), std::string{"Don't stop me now"});
}
