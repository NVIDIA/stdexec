/*
 * Copyright (c) Lucian Radu Teodorescu
 * Copyright (c) NVIDIA
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
#include <async_scope.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>

namespace ex = std::execution;
using _P2519::execution::async_scope;

TEST_CASE("ensure_started returns a sender", "[adaptors][ensure_started]") {
  auto snd = ex::ensure_started(ex::just(19));
  static_assert(ex::sender<decltype(snd)>);
  (void)snd;
}

TEST_CASE("ensure_started with environment returns a sender", "[adaptors][ensure_started]") {
  auto snd = ex::ensure_started(ex::just(19));
  static_assert(ex::sender<decltype(snd), empty_env>);
  (void)snd;
}

TEST_CASE("ensure_started simple example", "[adaptors][ensure_started]") {
  bool called{false};
  auto snd1 = ex::just() | ex::then([&] { called = true; });
  CHECK_FALSE(called);
  auto snd = ex::ensure_started(std::move(snd1));
  CHECK(called);
  auto op = ex::connect(std::move(snd), expect_void_receiver{});
  ex::start(op);
}

TEST_CASE("stopping ensure_started before the source completes calls set_stopped", "[adaptors][ensure_started]") {
  ex::in_place_stop_source stop_source;
  impulse_scheduler sch;
  bool called{false};
  auto snd = ex::on(sch, ex::just(19))
           | ex::write(ex::with(ex::get_stop_token, stop_source.get_token()))
           | ex::ensure_started();
  auto op = ex::connect(std::move(snd), expect_stopped_receiver_ex{called});
  ex::start(op);
  // request stop before the source yields a value
  stop_source.request_stop();
  // make the source yield the value
  sch.start_next();
  CHECK(called);
}

TEST_CASE("Dropping the sender without connecting it calls set_stopped", "[adaptors][ensure_started]") {
  impulse_scheduler sch;
  bool called = false;
  {
    auto snd =
        ex::on(sch, ex::just())
      | ex::upon_stopped([&] { called = true; })
      | ex::ensure_started();
  }
  // make the source yield the value
  sch.start_next();
  CHECK(called);
}

TEST_CASE("Dropping the opstate without starting it calls set_stopped", "[adaptors][ensure_started]") {
  impulse_scheduler sch;
  bool called = false;
  {
    auto snd =
        ex::on(sch, ex::just())
      | ex::upon_stopped([&] { called = true; })
      | ex::ensure_started();
  }
  // make the source yield the value
  sch.start_next();
  CHECK(called);
}
