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

#include "exec/sequence/then_each.hpp"
#include "exec/sequence/ignore_all.hpp"

#include <catch2/catch.hpp>

using namespace stdexec;
using namespace exec;

template <__completion_signature... _Sigs>
struct test_receiver {
  using is_receiver = void;

  template <class _Tag, class... _Args>
    requires __one_of<_Tag(_Args...), _Sigs...>
  friend void tag_invoke(_Tag, test_receiver&&, _Args&&...) noexcept {}

  friend empty_env tag_invoke(get_env_t, test_receiver) noexcept { return {}; }
};

template <__completion_signature... _Sigs>
struct next_receiver {
  template <sender_to<test_receiver<_Sigs...>> _Item>
  friend _Item tag_invoke(set_next_t, next_receiver&, _Item&& __item) noexcept {
    return __item;
  }

  friend void tag_invoke(set_value_t, next_receiver&&) noexcept {
  }

  friend void tag_invoke(set_stopped_t, next_receiver&&) noexcept {
  }

  template <class E>
  friend void tag_invoke(set_error_t, next_receiver&&, E&&) noexcept {
  }

  friend empty_env tag_invoke(get_env_t, const next_receiver&) noexcept {
    return {};
  }
};


TEST_CASE("sequence_senders - once - Test for concepts", "[sequence_senders][once]") {
  using just_t = decltype(just());
  using just_once_t = decltype(once(just()));
  STATIC_REQUIRE(sender<just_t>);
  STATIC_REQUIRE(sender<just_once_t>);
  STATIC_REQUIRE(same_as<completion_signatures_of_t<just_t>, completion_signatures_of_t<just_once_t>>);
  STATIC_REQUIRE(sender_to<just_t, next_receiver<>>);
  STATIC_REQUIRE_FALSE(sender_to<just_once_t, next_receiver<set_value_t()>>);
  STATIC_REQUIRE(sequence_sender_to<just_once_t, next_receiver<set_value_t()>>);
}

TEST_CASE("sequence_senders - once - fires once", "[sequence_sneders][once]") {
  int counter = 0;
  sync_wait(
    once(just()) | then_each([&counter] { ++counter; }) | ignore_all()
  );
  CHECK(counter == 1);
}