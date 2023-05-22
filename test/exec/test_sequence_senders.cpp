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

#include "exec/sequence_senders.hpp"

#include <catch2/catch.hpp>

using namespace stdexec;
using namespace exec;

struct nop_operation {
  friend void tag_invoke(start_t, nop_operation&) noexcept {
  }
};

TEST_CASE("sequence_senders - nop_operation is an operation state", "[sequence_senders]") {
  STATIC_REQUIRE(operation_state<nop_operation>);
}

template <__completion_signature... _Sigs>
struct some_sender_of {
  using is_sender = void;
  using completion_signatures = stdexec::completion_signatures<_Sigs...>;

  template <class R>
  friend nop_operation tag_invoke(connect_t, some_sender_of self, R&& rcvr);
};

TEST_CASE("sequence_senders - some_sender_of is a sender", "[sequence_senders]") {
  STATIC_REQUIRE(sender<some_sender_of<set_value_t()>>);
  STATIC_REQUIRE(sender_in<some_sender_of<set_value_t()>, empty_env>);
  STATIC_REQUIRE(same_as<
                 completion_signatures_of_t<some_sender_of<set_value_t()>>,
                 completion_signatures<set_value_t()>>);
  STATIC_REQUIRE(same_as<
                 completion_signatures_of_t<some_sender_of<set_value_t(int)>>,
                 completion_signatures<set_value_t(int)>>);
}

template <__completion_signature... _Sigs>
struct test_receiver {
  using is_receiver = void;

  template <class _Tag, class... _Args>
    requires __one_of<_Tag(_Args...), _Sigs...>
  friend void tag_invoke(_Tag, test_receiver&&, _Args&&...) noexcept {
  }

  friend empty_env tag_invoke(get_env_t, test_receiver) noexcept {
    return {};
  }
};

TEST_CASE("sequence_senders - test_receiver is a receiver of its Sigs", "[sequence_senders]") {
  STATIC_REQUIRE(receiver<test_receiver<>>);
  STATIC_REQUIRE(receiver_of<test_receiver<set_value_t()>, completion_signatures<set_value_t()>>);
  STATIC_REQUIRE_FALSE(
    receiver_of<test_receiver<set_value_t()>, completion_signatures<set_value_t(int)>>);
  STATIC_REQUIRE_FALSE(
    receiver_of<test_receiver<set_value_t()>, completion_signatures<set_error_t(int)>>);
  STATIC_REQUIRE_FALSE(
    receiver_of<test_receiver<set_value_t()>, completion_signatures<set_stopped_t()>>);
  STATIC_REQUIRE(sender_to<some_sender_of<set_value_t()>, test_receiver<set_value_t()>>);
  STATIC_REQUIRE_FALSE(sender_to<
                       some_sender_of<set_value_t(int), set_stopped_t()>,
                       test_receiver<set_value_t(), set_stopped_t()>>);
}

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

TEST_CASE("sequence_senders - Test missing next signature", "[sequence_senders]") {
  using just_t = decltype(just());
  using next_receiver_t = next_receiver<set_value_t(int)>;
  STATIC_REQUIRE(receiver<next_receiver_t>);
  STATIC_REQUIRE(sequence_receiver_of< next_receiver_t, completion_signatures<set_value_t(int)>>);
  STATIC_REQUIRE_FALSE(sequence_receiver_of<
                       next_receiver_t,
                       completion_signatures<set_value_t(int), set_stopped_t()>>);
  STATIC_REQUIRE_FALSE(
    sequence_receiver_of< next_receiver_t, completion_signatures<set_value_t()>>);
  STATIC_REQUIRE(sender_to<just_t, next_receiver_t>);
}

template <__completion_signature... _Sigs>
struct some_sequence_sender_of {
  using is_sender = sequence_tag;
  using completion_signatures = stdexec::completion_signatures<set_value_t()>;
  using sequence_signatures = stdexec::completion_signatures<_Sigs...>;

  template <receiver R>
  friend nop_operation tag_invoke(connect_t, some_sequence_sender_of self, R&& rcvr);
};

TEST_CASE("sequence_senders - Test for sequence_connect_t", "[sequence_senders]") {
  using next_receiver_t = next_receiver<set_value_t(int), set_stopped_t()>;
  using seq_sender_t = some_sequence_sender_of<set_value_t(int), set_stopped_t()>;
  STATIC_REQUIRE(sender<seq_sender_t>);
  STATIC_REQUIRE(sequence_sender<seq_sender_t>);
  STATIC_REQUIRE(sender_to<seq_sender_t, next_receiver_t>);
  STATIC_REQUIRE(sender_to<some_sequence_sender_of<set_value_t(int)>, next_receiver_t>);
}
