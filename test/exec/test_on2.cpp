/*
 * Copyright (c) 2022 Lucian Radu Teodorescu
 * Copyright (c) 2022 NVIDIA Corporation
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
#include <test_common/receivers.hpp>
#include <exec/on.hpp>
#include <exec/env.hpp>

namespace ex = stdexec;

template <ex::scheduler Sched = inline_scheduler>
inline auto _with_scheduler(Sched sched = {}) {
  return exec::write(exec::with(ex::get_scheduler, std::move(sched)));
}

TEST_CASE(
  "exec::on transitions back to the receiver's scheduler when completing with a value",
  "[adaptors][exec::on]") {
  bool called{false};
  auto snd_base = ex::just() //
                | ex::then([&]() -> int {
                    called = true;
                    return 19;
                  });

  int recv_value{0};
  impulse_scheduler sched1;
  impulse_scheduler sched2;
  auto snd = exec::on(sched1, std::move(snd_base)) | _with_scheduler(sched2);
  auto op = ex::connect(std::move(snd), expect_value_receiver_ex{recv_value});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task
  // The base sender shouldn't be started
  CHECK_FALSE(called);

  // Tell sched1 to start executing one task
  sched1.start_next();

  // Now the base sender is called, and execution is transfered to sched2
  CHECK(called);
  CHECK(recv_value == 0);

  // Tell sched2 to start executing one task
  sched2.start_next();

  // Now the base sender is called, and a value is sent to the receiver
  CHECK(recv_value == 19);
}

TEST_CASE(
  "exec::on transitions back to the receiver's scheduler when completing with an error",
  "[adaptors][exec::on]") {
  bool called{false};
  auto snd_base = ex::just() //
                | ex::let_value([&]() {
                    called = true;
                    return ex::just_error(19);
                  });

  int recv_error{0};
  impulse_scheduler sched1;
  impulse_scheduler sched2;
  auto snd = exec::on(sched1, std::move(snd_base)) | _with_scheduler(sched2);
  auto op = ex::connect(std::move(snd), expect_error_receiver_ex{recv_error});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task
  // The base sender shouldn't be started
  CHECK_FALSE(called);

  // Tell sched1 to start executing one task
  sched1.start_next();

  // Now the base sender is called, and execution is transfered to sched2
  CHECK(called);
  CHECK(recv_error == 0);

  // Tell sched2 to start executing one task
  sched2.start_next();

  // Now the base sender is called, and an error is sent to the receiver
  CHECK(recv_error == 19);
}

TEST_CASE(
  "inner on transitions back to outer on's scheduler when completing with a value",
  "[adaptors][exec::on]") {
  bool called{false};
  auto snd_base = ex::just() //
                | ex::then([&]() -> int {
                    called = true;
                    return 19;
                  });

  int recv_value{0};
  impulse_scheduler sched1;
  impulse_scheduler sched2;
  impulse_scheduler sched3;
  auto snd = exec::on(sched1, exec::on(sched2, std::move(snd_base))) | _with_scheduler(sched3);
  auto op = ex::connect(std::move(snd), expect_value_receiver_ex{recv_value});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task
  // The base sender shouldn't be started
  CHECK_FALSE(called);

  // Tell sched1 to start executing one task. This will post
  // work to sched2
  sched1.start_next();

  // The base sender shouldn't be started
  CHECK_FALSE(called);

  // Tell sched2 to start executing one task. This will execute
  // the base sender and post work back to sched1
  sched2.start_next();

  // Now the base sender is called, and execution is transfered back
  // to sched1
  CHECK(called);
  CHECK(recv_value == 0);

  // Tell sched1 to start executing one task. This will post work to
  // sched3
  sched1.start_next();

  // The final receiver still hasn't been called
  CHECK(recv_value == 0);

  // Tell sched3 to start executing one task. It should call the
  // final receiver
  sched3.start_next();

  // Now the value is sent to the receiver
  CHECK(recv_value == 19);
}

TEST_CASE(
  "inner on transitions back to outer on's scheduler when completing with an error",
  "[adaptors][exec::on]") {
  bool called{false};
  auto snd_base = ex::just() //
                | ex::let_value([&]() {
                    called = true;
                    return ex::just_error(19);
                  });

  int recv_error{0};
  impulse_scheduler sched1;
  impulse_scheduler sched2;
  impulse_scheduler sched3;
  auto snd = exec::on(sched1, exec::on(sched2, std::move(snd_base))) | _with_scheduler(sched3);
  auto op = ex::connect(std::move(snd), expect_error_receiver_ex{recv_error});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task
  // The base sender shouldn't be started
  CHECK_FALSE(called);

  // Tell sched1 to start executing one task. This will post
  // work to sched2
  sched1.start_next();

  // The base sender shouldn't be started
  CHECK_FALSE(called);

  // Tell sched2 to start executing one task. This will execute
  // the base sender and post work back to sched1
  sched2.start_next();

  // Now the base sender is called, and execution is transfered back
  // to sched1
  CHECK(called);
  CHECK(recv_error == 0);

  // Tell sched1 to start executing one task. This will post work to
  // sched3
  sched1.start_next();

  // The final receiver still hasn't been called
  CHECK(recv_error == 0);

  // Tell sched3 to start executing one task. It should call the
  // final receiver
  sched3.start_next();

  // Now the error is sent to the receiver
  CHECK(recv_error == 19);
}

TEST_CASE(
  "exec::on(closure) transitions onto and back off of the scheduler when completing with a value",
  "[adaptors][exec::on]") {
  bool called{false};
  auto closure = ex::then([&]() -> int {
    called = true;
    return 19;
  });

  int recv_value{0};
  impulse_scheduler sched1;
  impulse_scheduler sched2;
  auto snd = ex::just() | exec::on(sched1, std::move(closure)) | _with_scheduler(sched2);
  auto op = ex::connect(std::move(snd), expect_value_receiver_ex{recv_value});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task
  // The closure shouldn't be started
  CHECK_FALSE(called);

  // Tell sched1 to start executing one task
  sched1.start_next();

  // Now the closure is called, and execution is transfered to sched2
  CHECK(called);
  CHECK(recv_value == 0);

  // Tell sched2 to start executing one task
  sched2.start_next();

  // Now the closure is called, and a value is sent to the receiver
  CHECK(recv_value == 19);
}

TEST_CASE(
  "exec::on(closure) transitions onto and back off of the scheduler when completing with "
  "an error",
  "[adaptors][exec::on]") {
  bool called{false};
  auto closure = ex::let_value([&]() {
    called = true;
    return ex::just_error(19);
  });

  int recv_error{0};
  impulse_scheduler sched1;
  impulse_scheduler sched2;
  auto snd = ex::just() | exec::on(sched1, std::move(closure)) | _with_scheduler(sched2);
  auto op = ex::connect(std::move(snd), expect_error_receiver_ex{recv_error});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task
  // The closure shouldn't be started
  CHECK_FALSE(called);

  // Tell sched1 to start executing one task
  sched1.start_next();

  // Now the closure is called, and execution is transfered to sched2
  CHECK(called);
  CHECK(recv_error == 0);

  // Tell sched2 to start executing one task
  sched2.start_next();

  // Now the closure is called, and a error is sent to the receiver
  CHECK(recv_error == 19);
}

TEST_CASE(
  "inner on(closure) transitions back to outer on's scheduler when completing with a value",
  "[adaptors][exec::on]") {
  bool called{false};
  auto closure = ex::then([&](int i) -> int {
    called = true;
    return i;
  });

  int recv_value{0};
  impulse_scheduler sched1;
  impulse_scheduler sched2;
  impulse_scheduler sched3;
  auto snd = exec::on(sched1, ex::just(19)) | exec::on(sched2, std::move(closure))
           | _with_scheduler(sched3);
  auto op = ex::connect(std::move(snd), expect_value_receiver_ex{recv_value});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task
  // The closure shouldn't be started
  CHECK_FALSE(called);

  // Tell sched1 to start executing one task. This will post
  // work to sched3
  sched1.start_next();

  // The closure shouldn't be started
  CHECK_FALSE(called);

  // Tell sched3 to start executing one task. This post work to
  // sched2.
  sched3.start_next();

  // The closure shouldn't be started
  CHECK_FALSE(called);

  // Tell sched2 to start executing one task. This will execute
  // the closure and post work back to sched3
  sched2.start_next();

  // Now the closure is called, and execution is transfered back
  // to sched3
  CHECK(called);
  CHECK(recv_value == 0);

  // Tell sched3 to start executing one task. This will call the
  // receiver
  sched3.start_next();

  // Now the value is sent to the receiver
  CHECK(recv_value == 19);
}

TEST_CASE(
  "inner on(closure) transitions back to outer on's scheduler when completing with an error",
  "[adaptors][exec::on]") {
  bool called{false};
  auto closure = ex::let_value([&](int i) {
    called = true;
    return ex::just_error(i);
  });

  int recv_error{0};
  impulse_scheduler sched1;
  impulse_scheduler sched2;
  impulse_scheduler sched3;
  auto snd = exec::on(sched1, ex::just(19)) | exec::on(sched2, std::move(closure))
           | _with_scheduler(sched3);
  auto op = ex::connect(std::move(snd), expect_error_receiver_ex{recv_error});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task
  // The closure shouldn't be started
  CHECK_FALSE(called);

  // Tell sched1 to start executing one task. This will post
  // work to sched3
  sched1.start_next();

  // The closure shouldn't be started
  CHECK_FALSE(called);

  // Tell sched3 to start executing one task. This post work to
  // sched2.
  sched3.start_next();

  // The closure shouldn't be started
  CHECK_FALSE(called);

  // Tell sched2 to start executing one task. This will execute
  // the closure and post work back to sched3
  sched2.start_next();

  // Now the closure is called, and execution is transfered back
  // to sched3
  CHECK(called);
  CHECK(recv_error == 0);

  // Tell sched3 to start executing one task. This will call the
  // receiver
  sched3.start_next();

  // Now the error is sent to the receiver
  CHECK(recv_error == 19);
}
