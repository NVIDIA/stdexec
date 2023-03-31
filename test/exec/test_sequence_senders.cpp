/*
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

#include "exec/sequence_senders.hpp"
#include "exec/variant_sender.hpp"

#include <catch2/catch.hpp>

using namespace stdexec;
using namespace exec;

struct next_receiver {
  template <sender _Item>
  friend _Item tag_invoke(set_next_t, next_receiver&, _Item&& __item) noexcept {
    return __item;
  }

  friend void tag_invoke(set_value_t, next_receiver&&) noexcept {}

  friend void tag_invoke(set_stopped_t, next_receiver&&) noexcept {}

  template <class E>
  friend void tag_invoke(set_error_t, next_receiver&&, E&&) noexcept {}

  friend empty_env tag_invoke(get_env_t, const next_receiver&) noexcept {
    return {};
  }
};

TEST_CASE("sequence_senders - Test missing next signature", "[sequence_senders]") {
  using just_t = decltype(just());
  STATIC_REQUIRE(receiver<next_receiver>);
  STATIC_REQUIRE(sequence_receiver_of<next_receiver, completion_signatures<set_value_t(int)>>);
  STATIC_REQUIRE(sequence_receiver_of<next_receiver, completion_signatures<set_value_t(), set_stopped_t()>>);
  STATIC_REQUIRE(sender_to<just_t, next_receiver>);
  STATIC_REQUIRE_FALSE(sequence_sender_to<just_t, next_receiver>);
}

TEST_CASE("sequence_senders - repeat_each", "[sequence_senders]") {
  auto r = repeat_each(just_stopped()) | join_all();
  using join_t = decltype(r);
  STATIC_REQUIRE(sender<join_t>);
  STATIC_REQUIRE_FALSE(sequence_sender_to<join_t, next_receiver>);
  using sigs = completion_signatures_of_t<join_t, empty_env>;
  STATIC_REQUIRE(sequence_receiver_of<next_receiver, sigs>);
  auto result = sync_wait(r | then([] { return true; }));
  REQUIRE(result);
  CHECK(std::get<0>(result.value()));
}

TEST_CASE("sequence_senders - let_value_each", "[sequence_senders]") {
  auto r = repeat_each(just());
  int count = 0;
  auto fun = [&count]() {
    ++count;
    return just_stopped();
  };
  auto l = let_value_each(r, fun);
  using let_t = decltype(l);
  STATIC_REQUIRE(sender_in<let_t, empty_env>);
  STATIC_REQUIRE(sequence_sender_to<let_t, next_receiver>);
  sync_wait(join_all(l));
  CHECK(count == 1);
}

TEST_CASE("sequence_senders - let_stopped_each", "[sequence_senders]") {
  auto r = repeat_each(just_stopped());
  int count = 0;
  auto fun = [&count]() {
    ++count;
    return just_stopped();
  };
  auto l = let_stopped_each(r, fun);
  using let_t = decltype(l);
  STATIC_REQUIRE(sender_in<let_t, empty_env>);
  STATIC_REQUIRE(sequence_sender_to<let_t, next_receiver>);
  sync_wait(join_all(l));
  CHECK(count == 1);
}

TEST_CASE("sequence_senders - enumerate_each", "[sequence_senders]") {
  using just_t = decltype(just());
  using just_stopped_t = decltype(just_stopped());
  auto e = enumerate_each(repeat_each(just()));
  using enumerate_t = decltype(e);
  auto fun = [](int) -> exec::variant_sender<just_t, just_stopped_t> {
    return just();
  };
  auto l = let_value_each(e, fun);
  using let_t = decltype(l);
  STATIC_REQUIRE(sender_in<let_t, empty_env>);
  STATIC_REQUIRE(sequence_sender_to<let_t, next_receiver>);
  using enumerate_t = decltype(e);
  STATIC_REQUIRE(sender_in<enumerate_t, empty_env>);
  STATIC_REQUIRE(sequence_sender_to<enumerate_t, next_receiver>);
  int count = 0;
  sync_wait(
    repeat_each(just()) //
    | enumerate_each()  //
    | let_value_each([&count](int counter) -> exec::variant_sender<just_t, just_stopped_t> {
        if (counter < 10) {
          ++count;
          return just();
        } else {
          return just_stopped();
        }
      }) //
    | join_all());
  CHECK(count == 10);
}