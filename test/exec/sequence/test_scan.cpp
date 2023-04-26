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

#include "exec/sequence/scan.hpp"

#include "exec/sequence/empty_sequence.hpp"
#include "exec/sequence/once.hpp"
#include "exec/sequence/ignore_all.hpp"
#include "exec/sequence/then_each.hpp"
#include "exec/sequence/first_value.hpp"

#include <catch2/catch.hpp>

struct plus_t {
  template <class Tx, class Ty>
  auto operator()(Tx lhs, Ty rhs) -> decltype(lhs + rhs) {
    return lhs + rhs;
  }
};

inline constexpr plus_t plus{};

using namespace stdexec;

TEST_CASE("sequence_senders - scan returns a sender", "[sequence_senders][scan]") {
  auto snd = exec::scan(exec::empty_sequence(), 0, plus);
  STATIC_REQUIRE(sender<decltype(snd)>);
  STATIC_REQUIRE(same_as<
                 completion_signatures<set_error_t(std::exception_ptr)>,
                 completion_signatures_of_t<decltype(snd)>>);
}

TEST_CASE("sequence_senders - scan completion sigs", "[sequence_senders][scan]") {
  auto snd = exec::scan(exec::once(just(42)), 0, plus);
  STATIC_REQUIRE(same_as<
                 completion_signatures<set_error_t(std::exception_ptr), set_value_t(const int&)>,
                 completion_signatures_of_t<decltype(snd)>>);
}

TEST_CASE("sequence_senders - scan is sync_wait'able", "[sequence_senders][scan]") {
  auto snd = exec::scan(exec::once(just(42)), 0, plus);
  auto [value] = sync_wait(exec::first_value(snd)).value();
  CHECK(value == 42);
}