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

namespace ex = std::execution;

// For testing `when_all_with_variant`, we just check a couple of examples, check customization, and
// we assume it's implemented in terms of `when_all`.

TEST_CASE("when_all returns a sender", "[adaptors][when_all]") {
  auto snd = ex::when_all(ex::just(3), ex::just(0.1415));
  static_assert(ex::sender<decltype(snd)>);
  (void)snd;
}
TEST_CASE("when_all returns a typed_sender", "[adaptors][when_all]") {
  auto snd = ex::when_all(ex::just(3), ex::just(0.1415));
  static_assert(ex::typed_sender<decltype(snd)>);
  (void)snd;
}
TEST_CASE("when_all simple example", "[adaptors][when_all]") {
  auto snd = ex::when_all(ex::just(3), ex::just(0.1415));
  auto snd1 = std::move(snd) | ex::then([](int x, double y) { return x + y; });
  auto op = ex::connect(std::move(snd1), expect_value_receiver<double>{3.1415});
  ex::start(op);
}

TEST_CASE("when_all returning two values can we waited on", "[adaptors][when_all]") {
  ex::sender auto snd = ex::when_all( //
      ex::just(2),                    //
      ex::just(3)                     //
  );
  wait_for_value(std::move(snd), 2, 3);
}

TEST_CASE("when_all with 5 senders", "[adaptors][when_all]") {
  ex::sender auto snd = ex::when_all( //
      ex::just(2),                    //
      ex::just(3),                    //
      ex::just(5),                    //
      ex::just(7),                    //
      ex::just(11)                    //
  );
  wait_for_value(std::move(snd), 2, 3, 5, 7, 11);
}

TEST_CASE("when_all with just one sender", "[adaptors][when_all]") {
  ex::sender auto snd = ex::when_all( //
      ex::just(2)                     //
  );
  wait_for_value(std::move(snd), 2);
}

TEST_CASE("TODO: when_all with no senders sender -- should fail", "[adaptors][when_all]") {
  auto snd = ex::when_all();
  static_assert(ex::typed_sender<decltype(snd)>);
  // TODO: calling `ex::when_all()` should be ill-formed
}

TEST_CASE("when_all when one sender sends void", "[adaptors][when_all]") {
  ex::sender auto snd = ex::when_all( //
      ex::just(2),                    //
      ex::just()                      //
  );
  wait_for_value(std::move(snd), 2);
}

TEST_CASE("TODO: when_all_with_variant basic example", "[adaptors][when_all]") {
  // TODO: when_all_with_variant doesn't work
  // ex::sender auto snd = ex::when_all_with_variant( //
  //     ex::just(2),                                 //
  //     ex::just(3.14)                               //
  // );
  // wait_for_value(std::move(snd), std::variant<int, double>{2}, std::variant<int, double>{3.14});
}

TEST_CASE("TODO: when_all_with_variant with same type", "[adaptors][when_all]") {
  // TODO: when_all_with_variant doesn't work
  // ex::sender auto snd = ex::when_all_with_variant( //
  //     ex::just(2),                                 //
  //     ex::just(3)                               //
  // );
  // wait_for_value(std::move(snd), std::variant<int>{2}, std::variant<int>{3});
}

TEST_CASE("when_all completes when children complete", "[adaptors][when_all]") {
  impulse_scheduler sched;
  bool called{false};
  ex::sender auto snd =                 //
      ex::when_all(                     //
          ex::transfer_just(sched, 11), //
          ex::transfer_just(sched, 13), //
          ex::transfer_just(sched, 17)  //
          )                             //
      | ex::then([&](int a, int b, int c) {
          called = true;
          return a + b + c;
        });
  auto op = ex::connect(std::move(snd), expect_value_receiver<int>{41});
  ex::start(op);
  // The when_all scheduler will complete only after 3 impulses
  CHECK_FALSE(called);
  sched.start_next();
  CHECK_FALSE(called);
  sched.start_next();
  CHECK_FALSE(called);
  sched.start_next();
  CHECK(called);
}

TEST_CASE("TODO: when_all can be used with just_*", "[adaptors][when_all]") {
  ex::sender auto snd = ex::when_all(       //
      ex::just(2),                          //
      ex::just_error(std::exception_ptr{}), //
      ex::just_done()                       //
  );
  // TODO: this should work
  // auto op = ex::connect(std::move(snd), expect_error_receiver{});
  // ex::start(op);
  // invalid check
  static_assert(!std::invocable<ex::connect_t, decltype(snd), expect_error_receiver>);
}

TEST_CASE(
    "when_all terminates with error if one child terminates with error", "[adaptors][when_all]") {
  error_scheduler sched;
  ex::sender auto snd = ex::when_all( //
      ex::just(2),                    //
      ex::transfer_just(sched, 5),    //
      ex::just(7)                     //
  );
  auto op = ex::connect(std::move(snd), expect_error_receiver{});
  ex::start(op);
}

TEST_CASE("when_all terminates with done if one child is cancelled", "[adaptors][when_all]") {
  done_scheduler sched;
  ex::sender auto snd = ex::when_all( //
      ex::just(2),                    //
      ex::transfer_just(sched, 5),    //
      ex::just(7)                     //
  );
  auto op = ex::connect(std::move(snd), expect_done_receiver{});
  ex::start(op);
}

TEST_CASE("when_all cancels remaining children if error is detected", "[adaptors][when_all]") {
  impulse_scheduler sched;
  error_scheduler err_sched;
  bool called1{false};
  bool called3{false};
  bool cancelled{false};
  ex::sender auto snd = ex::when_all(                                //
      ex::on(sched, ex::just()) | ex::then([&] { called1 = true; }), //
      ex::on(sched, ex::transfer_just(err_sched, 5)),                //
      ex::on(sched, ex::just())                                      //
          | ex::then([&] { called3 = true; })                        //
          | ex::let_done([&] {
              cancelled = true;
              return ex::just();
            }) //
  );
  auto op = ex::connect(std::move(snd), expect_error_receiver{});
  ex::start(op);
  // The first child will complete; the third one will be cancelled
  CHECK_FALSE(called1);
  CHECK_FALSE(called3);
  sched.start_next(); // start the first child
  CHECK(called1);
  sched.start_next(); // start the second child; this will generate an error
  CHECK_FALSE(called3);
  sched.start_next(); // start the third child
  CHECK_FALSE(called3);
  CHECK(cancelled);
}

TEST_CASE("when_all cancels remaining children if cancel is detected", "[adaptors][when_all]") {
  done_scheduler done_sched;
  impulse_scheduler sched;
  bool called1{false};
  bool called3{false};
  bool cancelled{false};
  ex::sender auto snd = ex::when_all(                                //
      ex::on(sched, ex::just()) | ex::then([&] { called1 = true; }), //
      ex::on(sched, ex::transfer_just(done_sched, 5)),               //
      ex::on(sched, ex::just())                                      //
          | ex::then([&] { called3 = true; })                        //
          | ex::let_done([&] {
              cancelled = true;
              return ex::just();
            }) //
  );
  auto op = ex::connect(std::move(snd), expect_done_receiver{});
  ex::start(op);
  // The first child will complete; the third one will be cancelled
  CHECK_FALSE(called1);
  CHECK_FALSE(called3);
  sched.start_next(); // start the first child
  CHECK(called1);
  sched.start_next(); // start the second child; this will call set_done
  CHECK_FALSE(called3);
  sched.start_next(); // start the third child
  CHECK_FALSE(called3);
  CHECK(cancelled);
}

TEST_CASE("when_all has the values_type based on the children", "[adaptors][when_all]") {
  check_val_types<type_array<type_array<int>>>(ex::when_all(ex::just(13)));
  check_val_types<type_array<type_array<double>>>(ex::when_all(ex::just(3.14)));
  check_val_types<type_array<type_array<int, double>>>(ex::when_all(ex::just(3, 0.14)));

  check_val_types<type_array<type_array<>>>(ex::when_all(ex::just()));

  check_val_types<type_array<type_array<int, double>>>(ex::when_all(ex::just(3), ex::just(0.14)));
  check_val_types<type_array<type_array<int, double, int, double>>>( //
      ex::when_all(                                                  //
          ex::just(3),                                               //
          ex::just(0.14),                                            //
          ex::just(1, 0.4142)                                        //
          )                                                          //
  );

  // if one child returns void, then the value is simply missing
  check_val_types<type_array<type_array<int, double>>>( //
      ex::when_all(                                     //
          ex::just(3),                                  //
          ex::just(),                                   //
          ex::just(0.14)                                //
          )                                             //
  );
}

TEST_CASE("when_all has the error_types based on the children", "[adaptors][when_all]") {
  check_err_types<type_array<std::exception_ptr, int>>(ex::when_all(ex::just_error(13)));
  check_err_types<type_array<std::exception_ptr, double>>(ex::when_all(ex::just_error(3.14)));

  check_err_types<type_array<std::exception_ptr>>(ex::when_all(ex::just()));

  check_err_types<type_array<std::exception_ptr, int, double>>(
      ex::when_all(ex::just_error(3), ex::just_error(0.14)));
  check_err_types<type_array<std::exception_ptr, int, double, std::string>>( //
      ex::when_all(                                                          //
          ex::just_error(3),                                                 //
          ex::just_error(0.14),                                              //
          ex::just_error(std::string{"err"})                                 //
          )                                                                  //
  );

  check_err_types<type_array<std::exception_ptr>>( //
      ex::when_all(                                //
          ex::just(13),                            //
          ex::just_error(std::exception_ptr{}),    //
          ex::just_done()                          //
          )                                        //
  );
}

TEST_CASE("when_all has the sends_done == true", "[adaptors][when_all]") {
  check_sends_done<true>(ex::when_all(ex::just(13)));
  check_sends_done<true>(ex::when_all(ex::just_error(-1)));
  check_sends_done<true>(ex::when_all(ex::just_done()));

  check_sends_done<true>(ex::when_all(ex::just(3), ex::just(0.14)));
  check_sends_done<true>(     //
      ex::when_all(           //
          ex::just(3),        //
          ex::just_error(-1), //
          ex::just_done()     //
          )                   //
  );
}

using my_string_sender_t = decltype(ex::transfer_just(inline_scheduler{}, std::string{}));

auto tag_invoke(ex::when_all_t, my_string_sender_t, my_string_sender_t) {
  // Return a different sender when we invoke this custom defined on implementation
  return ex::just(std::string{"first program"});
}

TEST_CASE("TODO: when_all can be customized", "[adaptors][when_all]") {
  // The customization will return a different value
  auto snd = ex::when_all(                                          //
      ex::transfer_just(inline_scheduler{}, std::string{"hello,"}), //
      ex::transfer_just(inline_scheduler{}, std::string{" world!"}) //
  );
  // TODO: check why function cannot be customized
  // wait_for_value(std::move(snd), std::string{"first program"});
  // Invalid check:
  wait_for_value(std::move(snd), std::string{"hello,"}, std::string{" world!"});
}

auto tag_invoke(ex::when_all_with_variant_t, my_string_sender_t, my_string_sender_t) {
  // Return a different sender when we invoke this custom defined on implementation
  return ex::just(std::string{"first program"});
}

// TODO: check when_all_with_variant
TEST_CASE("TODO: when_all_with_variant can be customized", "[adaptors][when_all]") {
  // // The customization will return a different value
  // auto snd = ex::when_all_with_variant(                             //
  //     ex::transfer_just(inline_scheduler{}, std::string{"hello,"}), //
  //     ex::transfer_just(inline_scheduler{}, std::string{" world!"}) //
  // );
  // // TODO: check why function cannot be customized
  // // wait_for_value(std::move(snd), std::string{"first program"});
  // // Invalid check:
  // wait_for_value(std::move(snd), std::variant<std::string>{std::string{"hello,"}},
  // std::variant<std::string>{std::string{" world!"}});
}
