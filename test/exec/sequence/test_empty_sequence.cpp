/*
 * Copyright (c) 2023 NVIDIA Corporation
 * Copyright (c) 2023 Maikel Nadolski
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

#include "exec/sequence/empty_sequence.hpp"

#include <catch2/catch.hpp>

using namespace stdexec;
using namespace exec;

namespace {

  TEST_CASE(
    "sequence_senders - empty_sequence is a sequence sender",
    "[sequence_senders][empty_sequence]") {
    using empty_t = decltype(empty_sequence());
    STATIC_REQUIRE(sequence_sender<empty_t, env<>>);
    STATIC_REQUIRE(
      same_as<
        __sequence_completion_signatures_of_t<empty_t, env<>>,
        completion_signatures<set_value_t()>
      >);
    STATIC_REQUIRE(
      same_as<completion_signatures_of_t<empty_t>, completion_signatures<set_value_t()>>);
    STATIC_REQUIRE(same_as<item_types_of_t<empty_t, env<>>, item_types<>>);
  }

  struct count_set_next_receiver_t {
    using receiver_concept = stdexec::receiver_t;
    int& count_invocations_;

    friend auto
      tag_invoke(set_next_t, count_set_next_receiver_t& __self, auto /* item */) noexcept {
      ++__self.count_invocations_;
      return just();
    }

    void set_value() noexcept {
    }
  };

  TEST_CASE(
    "sequence_senders - empty_sequence is a sequence sender to a minimal receiver of set_value_t()",
    "[sequence_senders][empty_sequence]") {
    using empty_t = decltype(empty_sequence());
    STATIC_REQUIRE(receiver_of<count_set_next_receiver_t, completion_signatures<set_value_t()>>);
    STATIC_REQUIRE(sequence_sender_to<empty_t, count_set_next_receiver_t>);

    int count{0};
    auto op = subscribe(empty_sequence(), count_set_next_receiver_t{count});
    stdexec::start(op);
    CHECK(count == 0);
  }
} // namespace
