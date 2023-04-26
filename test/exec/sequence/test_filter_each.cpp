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

#include "exec/sequence/filter_each.hpp"

#include "exec/sequence/any_sequence_sender_of.hpp"
#include "exec/sequence/empty_sequence.hpp"
#include "exec/sequence/enumerate_each.hpp"
#include "exec/sequence/ignore_all.hpp"
#include "exec/sequence/repeat.hpp"
#include "exec/sequence/once.hpp"
#include "exec/sequence/then_each.hpp"
#include "exec/sequence/first_value.hpp"
#include "exec/sequence/let_each.hpp"

#include <catch2/catch.hpp>

using namespace stdexec;
using namespace exec;

TEST_CASE(
  "sequence_senders - filter_each of empty_sequence does not eval predicate",
  "[sequence_senders][filter_each][empty_sequence]") {
  int num_evals = 0;
  auto sequence = empty_sequence() | filter_each([&num_evals] {
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
  "sequence_senders - filter_each of once evals once",
  "[sequence_senders][filter_each][once]") {
  int num_evals = 0;
  auto sequence = once(just()) | filter_each([&num_evals] {
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
  "sequence_sedners - filter_each can be type-erased",
  "[sequence_senders][filter_each][any_sequence_sender][first_value]") {
  auto sequence = once(just(42)) | filter_each([](int) { return true; });
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

TEST_CASE("sequence_senders - filter_each: fires once", "[sequence_senders][filter_each]") {
  auto always_true = [](auto&&...) {
    return true;
  };
  int count = 0;
  auto filtered = once(just(42))           //
                | filter_each(always_true) //
                | then_each([&](int value) {
                    CHECK(value == 42);
                    count += 1;
                  }) //
                | ignore_all();
  CHECK(count == 0);
  sync_wait(filtered);
  CHECK(count == 1);
}

TEST_CASE("sequence_senders - filter_each: fires none", "[sequence_senders][filter_each]") {
  auto always_false = [](auto&&...) {
    return false;
  };
  int count = 0;
  auto filtered = once(just(42))                            //
                | filter_each(always_false)                 //
                | then_each([&](int value) { count += 1; }) //
                | ignore_all();
  CHECK(count == 0);
  sync_wait(filtered);
  CHECK(count == 0);
}