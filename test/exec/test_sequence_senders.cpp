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

#include <catch2/catch.hpp>

using namespace stdexec;
using namespace exec;

struct next_receiver {
  template <sender_of<set_value_t(int)> _Item>
  friend _Item tag_invoke(set_next_t, next_receiver&, _Item&& __item) noexcept {
    return __item;
  }

  friend void tag_invoke(set_value_t, next_receiver&&) noexcept {}

  friend empty_env tag_invoke(get_env_t, const next_receiver&) noexcept {
    return {};
  }
};

TEST_CASE("sequence_senders - Test missing next signature", "[sequence_senders]") {
  STATIC_REQUIRE(receiver<next_receiver>);
  STATIC_REQUIRE(sequence_receiver_of<next_receiver, completion_signatures<set_value_t(int)>>);
  STATIC_REQUIRE_FALSE(sequence_receiver_of<next_receiver, completion_signatures<set_value_t()>>);
}

TEST_CASE("sequence_senders - repeat_effect", "[sequence_senders]") {
  auto r = join_all(repeat_effect(just()));
  using join_t = decltype(r);
  STATIC_REQUIRE(sender<join_t>);
  STATIC_REQUIRE(sender_of<join_t, set_value_t()>);
  stdexec::__debug_sender(r);
}