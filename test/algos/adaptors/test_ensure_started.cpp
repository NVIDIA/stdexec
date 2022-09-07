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

// TODO: implement ensure_started
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
  async_scope scope;
  impulse_scheduler sch;
  auto snd = ex::on(sch, ex::just(19)) | ex::ensure_started();
  auto op = ex::connect(std::move(snd), expect_stopped_receiver_ex{});
  ex::start(op);
  // request stop before the source yields a value
  scope.get_stop_source().request_stop();
  // make the source yield the value
  sch.start_next();
}
