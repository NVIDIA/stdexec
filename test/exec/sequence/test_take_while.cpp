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

#include "exec/sequence/take_while.hpp"

#include "exec/sequence/any_sequence_sender_of.hpp"
#include "exec/sequence/empty_sequence.hpp"
#include "exec/sequence/enumerate_each.hpp"
#include "exec/sequence/ignore_all.hpp"
#include "exec/sequence/repeat.hpp"
#include "exec/sequence/once.hpp"
#include "exec/sequence/then_each.hpp"
#include "exec/sequence/first_value.hpp"

#include <catch2/catch.hpp>

using namespace stdexec;
using namespace exec;

TEST_CASE(
  "sequence_senders - take_while of empty_sequence does not eval predicate",
  "[sequence_senders][take_while]") {
  int num_evals = 0;
  auto sequence = empty_sequence() | take_while([&num_evals] {
                    ++num_evals;
                    return true;
                  });
  STATIC_REQUIRE(sender_in<decltype(sequence)>);
  STATIC_REQUIRE(same_as<
                 completion_signatures_of_t<decltype(sequence)>,
                 completion_signatures<set_stopped_t(), set_error_t(std::exception_ptr)>>);
  sync_wait(sequence | ignore_all());
  CHECK(num_evals == 0);
}

TEST_CASE(
  "sequence_senders - take_while of once evals once",
  "[sequence_senders][take_while][once]") {
  int num_evals = 0;
  auto sequence = once(just()) | take_while([&num_evals] {
                    ++num_evals;
                    return true;
                  });
  STATIC_REQUIRE(sender_in<decltype(sequence)>);
  STATIC_REQUIRE(
    same_as<
      completion_signatures_of_t<decltype(sequence)>,
      completion_signatures<set_stopped_t(), set_error_t(std::exception_ptr), set_value_t()>>);
  sync_wait(sequence | ignore_all());
  CHECK(num_evals == 1);
}

TEST_CASE(
  "sequence_sedners - take_while can be type-erased",
  "[sequence_senders][take_while][any_sequence_sender][first_value]") {
  auto sequence = once(just(42)) | take_while([](int) { return true; });
  STATIC_REQUIRE(sender_in<decltype(sequence)>);
  using sigs = completion_signatures_of_t<decltype(sequence)>;
  STATIC_REQUIRE(
    same_as<
      sigs,
      completion_signatures<set_stopped_t(), set_error_t(std::exception_ptr), set_value_t(int)>>);
  auto erased = any_sequence_receiver_ref<sigs>::any_sender<>(sequence);
  STATIC_REQUIRE(sender_in<decltype(erased)>);
  auto [val] = sync_wait(first_value(std::move(erased))).value();
  CHECK(val == 42);
}

TEST_CASE("sequence_senders - take_while stops a repeat", "[sequence_senders][take_while]") {
  int counter = 0;
  auto take_5 = repeat(just())    //
              | enumerate_each(1) //
              | then_each([&counter](int n) {
                  ++counter;
                  return n;
                })                                      //
              | take_while([](int n) { return n < 5; }) //
              | ignore_all();
  sync_wait(take_5);
  CHECK(counter == 5);
}
