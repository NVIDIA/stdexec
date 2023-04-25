/*
 * Copyright (c) 2023 Maikel Nadolski
 * Copyright (c) 2023 NVIDIA Corporation
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

#include "exec/sequence/iterate.hpp"

#include "exec/sequence/enumerate_each.hpp"
#include "exec/sequence/ignore_all.hpp"
#include "exec/sequence/then_each.hpp"

#include "catch2/catch.hpp"

#include <array>

TEST_CASE("iterate is sequence sender", "[sequence_senders][iterate]") {
  std::array<int, 1> a = {42};
  auto snd = exec::iterate(a);
  using Sender = decltype(snd);
  STATIC_REQUIRE(stdexec::sender<Sender>);
}

TEST_CASE("iterate is unstoppable if the environment is", "[sequence_senders][iterate]") {
  const std::array<int, 1> a = {42};
  auto snd = exec::iterate(std::ranges::views::all(a));
  using Sender = decltype(snd);
  using expected_compl = stdexec::completion_signatures<
    stdexec::set_value_t(const int&),
    stdexec::set_error_t(std::exception_ptr)>;
  using sender_compl = stdexec::completion_signatures_of_t<Sender, stdexec::empty_env>;
  STATIC_REQUIRE(std::same_as<sender_compl, expected_compl>);
}

TEST_CASE("sequence senders - iterate over array", "[sequence_senders][iterate]") {
  std::array<int, 3> a = {42, 43, 44};
  auto snd = exec::iterate(a)                                                      //
           | exec::enumerate_each()                                                //
           | exec::then_each([](int counter, int i) { CHECK(i == 42 + counter); }) //
           | exec::ignore_all();
  stdexec::sync_wait(snd);
}