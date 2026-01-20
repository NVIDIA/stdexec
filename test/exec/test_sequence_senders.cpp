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

using namespace STDEXEC;
using namespace exec;

namespace {

  struct nop_operation {
    void start() & noexcept {
    }
  };

  TEST_CASE("sequence_senders - nop_operation is an operation state", "[sequence_senders]") {
    STATIC_REQUIRE(operation_state<nop_operation>);
  }

  template <__completion_signature... _Sigs>
  struct some_sender_of {
    using sender_concept = STDEXEC::sender_t;
    using completion_signatures = STDEXEC::completion_signatures<_Sigs...>;

    template <class R>
    auto connect(R&&) const -> nop_operation {
      return {};
    }
  };

  TEST_CASE("sequence_senders - some_sender_of is a sender", "[sequence_senders]") {
    STATIC_REQUIRE(sender<some_sender_of<set_value_t()>>);
    STATIC_REQUIRE(sender_in<some_sender_of<set_value_t()>, env<>>);
    STATIC_REQUIRE(
      std::same_as<
        completion_signatures_of_t<some_sender_of<set_value_t()>>,
        completion_signatures<set_value_t()>
      >);
    STATIC_REQUIRE(
      std::same_as<
        completion_signatures_of_t<some_sender_of<set_value_t(int)>>,
        completion_signatures<set_value_t(int)>
      >);
  }

  template <__completion_signature... _Sigs>
  struct test_receiver {
    using receiver_concept = STDEXEC::receiver_t;

    template <class... _Args>
      requires __one_of<set_value_t(_Args...), _Sigs...>
    void set_value(_Args&&...) noexcept {
    }

    template <class E>
      requires __one_of<set_error_t(E), _Sigs...>
    void set_error(E&&) noexcept {
    }

    void set_stopped() noexcept
      requires __one_of<set_stopped_t(), _Sigs...>
    {
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
    // Fails because `connect` is no longer constrained:
    // STATIC_REQUIRE_FALSE(
    //   sender_to<
    //     some_sender_of<set_value_t(int), set_stopped_t()>,
    //     test_receiver<set_value_t(), set_stopped_t()>
    //   >);
  }

  template <__completion_signature... _Sigs>
  struct next_receiver {
    using receiver_concept = STDEXEC::receiver_t;

    template <sender_to<test_receiver<_Sigs...>> _Item>
    auto set_next(_Item&& __item) & noexcept -> _Item {
      return __item;
    }

    void set_value() noexcept {
    }

    void set_stopped() noexcept {
    }

    template <class E>
    void set_error(E&&) noexcept {
    }
  };

  TEST_CASE("sequence_senders - Test missing next signature", "[sequence_senders]") {
    using just_t = decltype(just());
    using just_int_t = decltype(just(0));
    using next_receiver_t = next_receiver<set_value_t(int)>;
    STATIC_REQUIRE(receiver<next_receiver_t>);
    STATIC_REQUIRE(sequence_receiver_of<next_receiver_t, item_types<just_int_t>>);
    STATIC_REQUIRE_FALSE(sequence_receiver_of<next_receiver_t, item_types<just_t>>);
    STATIC_REQUIRE(sender_to<just_t, next_receiver_t>);
  }

  template <__completion_signature... _Sigs>
  struct some_sequence_sender_of {
    using sender_concept = sequence_sender_t;
    using completion_signatures = STDEXEC::completion_signatures<set_value_t()>;
    using item_types = exec::item_types<some_sender_of<_Sigs...>>;

    template <receiver R>
    auto subscribe(R&&) const -> nop_operation {
      return {};
    }
  };

  TEST_CASE("sequence_senders - Test for subscribe", "[sequence_senders]") {
    using next_receiver_t = next_receiver<set_value_t(int), set_stopped_t()>;
    using seq_sender_t = some_sequence_sender_of<set_value_t(int), set_stopped_t()>;
    STATIC_REQUIRE(sender<seq_sender_t>);
    STATIC_REQUIRE(sequence_sender<seq_sender_t, env<>>);
    STATIC_REQUIRE(sequence_sender_to<seq_sender_t, next_receiver_t>);
    STATIC_REQUIRE(sequence_sender_to<some_sequence_sender_of<set_value_t(int)>, next_receiver_t>);
  }
} // namespace
