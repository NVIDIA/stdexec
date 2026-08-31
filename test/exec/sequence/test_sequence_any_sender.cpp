/*
 * Copyright (c) 2026 NVIDIA Corporation
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

// Regression test for issue #2101:
// exec::any_sender fails get_completion_signatures inside a sequence of 3+ senders.
// The root cause is that the recursive completion-signature computation in
// __seq::__sndr passes the first sender type through __copy_cvref_t, which
// can produce a reference type. any_sender's get_completion_signatures
// constraint (derived_from<Self, interface>) then fails for reference types.

#include <exec/sequence.hpp>
#include <exec/any_sender_of.hpp>
#include <stdexec/execution.hpp>

#include <test_common/catch2.hpp>

#include <exception>

namespace ex = STDEXEC;

TEST_CASE("sequence with 3 any_senders compiles and runs", "[sequence][any_sender]")
{
  // all but the last sender must be void senders
  using SigsVoid =
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr)>;
  using SigsInt =
    ex::completion_signatures<ex::set_value_t(int), ex::set_error_t(std::exception_ptr)>;

  using AnyVoid = exec::any_sender<exec::any_receiver<SigsVoid, exec::queries<>>>;
  using AnyInt  = exec::any_sender<exec::any_receiver<SigsInt, exec::queries<>>>;

  AnyVoid s1 = ex::just();
  AnyVoid s2 = ex::just();
  AnyInt  s3 = ex::just(42);

  // Before the fix this would fail to compile with:
  //   "no matching function for call to 'get_completion_signatures'"
  auto seq = exec::sequence(std::move(s1), std::move(s2), std::move(s3));

  auto [result] = *ex::sync_wait(std::move(seq));
  CHECK(result == 42);
}

TEST_CASE("sequence with 4 any_senders compiles and runs", "[sequence][any_sender]")
{
  using SigsVoid =
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr)>;
  using SigsInt =
    ex::completion_signatures<ex::set_value_t(int), ex::set_error_t(std::exception_ptr)>;

  using AnyVoid = exec::any_sender<exec::any_receiver<SigsVoid, exec::queries<>>>;
  using AnyInt  = exec::any_sender<exec::any_receiver<SigsInt, exec::queries<>>>;

  AnyVoid s1 = ex::just();
  AnyVoid s2 = ex::just();
  AnyVoid s3 = ex::just();
  AnyInt  s4 = ex::just(42);

  auto seq = exec::sequence(std::move(s1), std::move(s2), std::move(s3), std::move(s4));

  auto [result] = *ex::sync_wait(std::move(seq));
  CHECK(result == 42);
}

TEST_CASE(
  "sequence with mixed concrete and any_senders compiles and runs",
  "[sequence][any_sender]")
{
  using SigsVoid =
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr)>;
  using SigsInt =
    ex::completion_signatures<ex::set_value_t(int), ex::set_error_t(std::exception_ptr)>;

  using AnyVoid = exec::any_sender<exec::any_receiver<SigsVoid, exec::queries<>>>;
  using AnyInt  = exec::any_sender<exec::any_receiver<SigsInt, exec::queries<>>>;

  AnyVoid s1 = ex::just();
  auto    s2 = ex::just(); // concrete sender
  AnyInt  s3 = ex::just(42);

  auto seq = exec::sequence(std::move(s1), std::move(s2), std::move(s3));

  auto [result] = *ex::sync_wait(std::move(seq));
  CHECK(result == 42);
}
