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

#include "exec/sequence/transform_each.hpp"

#include "exec/sequence/empty_sequence.hpp"
#include "exec/sequence/ignore_all.hpp"

#include <catch2/catch.hpp>

using namespace stdexec;
using namespace exec;

struct next_receiver {
  template <class Item>
  friend auto tag_invoke(set_next_t, next_receiver&, Item&& item) noexcept {
    return then(static_cast<Item&&>(item), []() noexcept {});
  }

  friend void tag_invoke(set_value_t, next_receiver&&) noexcept {
    CHECK(true);
  }

  friend empty_env tag_invoke(get_env_t, next_receiver) noexcept { return {}; }
};

TEST_CASE("transform_each - using then with sender works", "[sequence_senders][transform_each]") {
  auto sender = just(42);
  auto sequence = transform_each(sender, then([](int value) noexcept { CHECK(value == 42); }));
  using Sequence = decltype(sequence);
  STATIC_REQUIRE(sequence_sender<Sequence>);
  using completion_sigs = __sequence_completion_signatures_of_t<Sequence, empty_env>;
  using sequence_sigs = completion_signatures_of_t<Sequence, empty_env>;
  STATIC_REQUIRE(same_as<sequence_sigs, completion_signatures<set_value_t()>>);
  STATIC_REQUIRE(same_as<completion_sigs, completion_signatures<set_value_t()>>);
  STATIC_REQUIRE(sequence_sender_to<Sequence, next_receiver>);
  auto op = sequence_connect(sequence, next_receiver{});
  start(op);
}


TEST_CASE("transform_each - using then with empty_sequence works", "[sequence_senders][transform_each][empty_sequence]") {
  auto sequence = transform_each(empty_sequence(), then([]() noexcept { CHECK(false); }));
  using Sequence = decltype(sequence);
  STATIC_REQUIRE(sequence_sender<Sequence>);
  using sequence_sigs = completion_signatures_of_t<Sequence, empty_env>;
  STATIC_REQUIRE(same_as<sequence_sigs, completion_signatures<>>);
  STATIC_REQUIRE(sequence_sender_to<Sequence, next_receiver>);
  auto op = sequence_connect(sequence, next_receiver{});
  start(op);
}

TEST_CASE("transform_each - using then with sender works with ignore_all", "[sequence_senders][transform_each][ignore_all]") {
  auto sender = just(42);
  auto sequence = transform_each(sender, then([](int value) noexcept { CHECK(value == 42); }));
  auto result = ignore_all(sequence);
  CHECK(sync_wait(result));
}