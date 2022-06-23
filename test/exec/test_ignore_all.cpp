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
using expect_void = expect_void_receiver<exec::make_env_t<
  exec::with_t<ex::get_scheduler_t, decltype(exec::static_thread_pool{}.get_scheduler())&>>>;

TEST_CASE("Simple test for ignore_all", "[factories][sequence][ignore_all]") {
  std::optional<exec::static_thread_pool> pool{2};
  ex::scheduler auto schd = pool->get_scheduler();
  auto r = expect_void{exec::make_env(exec::with(ex::get_scheduler, schd))};
  auto s = P0TBD::ignore_all(P0TBD::iotas(1, 3));
  auto o1 = ex::connect(std::move(s), std::move(r));
  ex::start(o1);
  pool.reset();
}

TEST_CASE("Stack overflow test for ignore_all", "[factories][sequence][ignore_all]") {
  std::optional<exec::static_thread_pool> pool{2};
  ex::scheduler auto schd = pool->get_scheduler();
  auto o1 = ex::connect(
    P0TBD::ignore_all(P0TBD::iotas(1, 3000000)),
    expect_void{exec::make_env(exec::with(ex::get_scheduler, schd))});
  ex::start(o1);
  pool.reset();
}

TEST_CASE("ignore_all returns a sequence_sender", "[factories][sequence][ignore_all]") {
  using r = expect_void;
  using s = decltype(P0TBD::ignore_all(P0TBD::iotas(1, 3)));
  static_assert(ex::sender_to<s, r>, "P0TBD::ignore_all must return a sender");
  REQUIRE(ex::sender_to<s, r>);
}
