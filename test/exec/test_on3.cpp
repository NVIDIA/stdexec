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
#include <exec/on.hpp>
#include <exec/async_scope.hpp>

namespace ex = stdexec;

static const auto env = exec::make_env(exec::with(ex::get_scheduler, inline_scheduler{}));

TEST_CASE("Can pass exec::on sender to start_detached", "[adaptors][exec::on]") {
  ex::start_detached(exec::on(inline_scheduler{}, ex::just()), env);
}

TEST_CASE("Can pass exec::on sender to split", "[adaptors][exec::on]") {
  auto snd = ex::split(exec::on(inline_scheduler{}, ex::just()), env);
  (void) snd;
}

TEST_CASE("Can pass exec::on sender to ensure_started", "[adaptors][exec::on]") {
  auto snd = ex::ensure_started(exec::on(inline_scheduler{}, ex::just()), env);
  (void) snd;
}

TEST_CASE("Can pass exec::on sender to async_scope::spawn", "[adaptors][exec::on]") {
  exec::async_scope scope;
  impulse_scheduler sched;
  scope.spawn(exec::on(sched, ex::just()), env);
  sched.start_next();
  stdexec::sync_wait(scope.on_empty());
}

TEST_CASE("Can pass exec::on sender to async_scope::spawn_future", "[adaptors][exec::on]") {
  exec::async_scope scope;
  impulse_scheduler sched;
  auto fut = scope.spawn_future(exec::on(sched, ex::just(42)), env);
  sched.start_next();
  auto [i] = stdexec::sync_wait(std::move(fut)).value();
  CHECK(i == 42);
  stdexec::sync_wait(scope.on_empty());
}
