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

#include "exec/sequence/once.hpp"

#include "test_common/type_helpers.hpp"

#include <catch2/catch.hpp>

using namespace stdexec;
using namespace exec;

template <__completion_signature... _Sigs>
struct test_receiver {
  template <class _Tag, class... _Args>
    requires __one_of<_Tag(_Args...), _Sigs...>
  friend void tag_invoke(_Tag, test_receiver&&, _Args&&...) noexcept {}

  friend empty_env tag_invoke(get_env_t, test_receiver) noexcept { return {}; }
};

template <__completion_signature... _Sigs>
struct count_receiver {
  int& counter;

  template <sender_to<test_receiver<_Sigs...>> Item>
  friend auto tag_invoke(set_next_t, count_receiver& self, Item&& item) {
    return then(static_cast<Item&&>(item), [&self](auto&&...) noexcept { ++self.counter; });
  }

  friend void tag_invoke(set_value_t, count_receiver&& self) noexcept {
    CHECK(self.counter == 1);
  }

  friend empty_env tag_invoke(get_env_t, const count_receiver&) noexcept {
    return {};
  }
};


TEST_CASE("sequence_senders - single - Test for concepts", "[sequence_senders][single]") {
  using just_t = decltype(just());
  using just_single_t = decltype(single(just()));
  STATIC_REQUIRE(sender<just_t>);
  STATIC_REQUIRE(sender<just_single_t>);
  STATIC_REQUIRE(same_as<completion_signatures_of_t<just_t>, completion_signatures_of_t<just_single_t>>);
  STATIC_REQUIRE(sender_to<just_t, count_receiver<>>);
  STATIC_REQUIRE_FALSE(sender_to<just_single_t, count_receiver<set_value_t()>>);
  STATIC_REQUIRE(sequence_sender_to<just_single_t, count_receiver<set_value_t()>>);
}

TEST_CASE("sequence_senders - single - fires once", "[sequence_senders][single]") {
  int counter = 0;
  auto sequence = single(just(42));
  auto receiver = count_receiver<set_value_t(int)>{counter};
  auto op = sequence_connect(sequence, receiver);
  start(op);
  CHECK(counter == 1);
}

TEST_CASE("sequence_senders - single - fires once movable", "[sequence_senders][single]") {
  int counter = 0;
  auto sequence = single(just(::movable(42)));
  auto receiver = count_receiver<set_value_t(::movable)>{counter};
  using sequence_t = __decay_t<decltype(sequence)>;
  STATIC_REQUIRE_FALSE(sequence_sender_to<const sequence_t&, count_receiver<set_value_t(::movable)>>);
  STATIC_REQUIRE(sequence_sender_to<sequence_t&&, count_receiver<set_value_t(::movable)>>);
  auto op = sequence_connect(static_cast<sequence_t&&>(sequence), receiver);
  start(op);
  CHECK(counter == 1);
}