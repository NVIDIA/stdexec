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
#include <exec/env.hpp>
#include <exec/sequence.hpp>
#include <test_common/receivers.hpp>
#include <test_common/sequences.hpp>
#include <exec/static_thread_pool.hpp>

namespace ex = stdexec;
namespace P0TBD = _P0TBD::execution;

TEST_CASE("Simple test for iotas", "[factories][sequence][iotas]") {
  std::optional<exec::static_thread_pool> pool{2};
  ex::scheduler auto schd = pool->get_scheduler();
  auto r = make_expect_sequence_receiver(
    [](int v){return v > 0 && v <= 3;},
    exec::make_env(exec::with(ex::get_scheduler, schd)));
  auto s = P0TBD::iotas(1, 3);
  auto o1 = ex::connect(std::move(s), std::move(r));
  ex::start(o1);
  pool.reset();
}

TEST_CASE("Stack overflow test for iotas", "[factories][sequence][iotas]") {
  std::optional<exec::static_thread_pool> pool{2};
  ex::scheduler auto schd = pool->get_scheduler();
  auto o1 = ex::connect(
    P0TBD::iotas(1, 3000000),
    make_expect_sequence_receiver(
      [](int v){return v > 0 && v <= 3000000;},
      exec::make_env(exec::with(ex::get_scheduler, schd))));
  ex::start(o1);
  pool.reset();
}

TEST_CASE("iotas returns a sequence_sender", "[factories][sequence][iotas]") {
  exec::static_thread_pool pool{2};
  ex::scheduler auto schd = pool.get_scheduler();
  auto p = [](int v){return v > 0 && v <= 3;};
  using r = decltype(make_expect_sequence_receiver(
    std::move(p),
    exec::make_env(exec::with(ex::get_scheduler, schd))));
  using s = decltype(P0TBD::iotas(1, 3));
  static_assert(ex::sender_to<s, r>, "P0TBD::iotas must return a sender");
  REQUIRE(ex::sender_to<s, r>);
}
