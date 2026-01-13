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
#include "exec/sequence/ignore_all_values.hpp"

#include <catch2/catch.hpp>

namespace {

  TEST_CASE("ignore_all_values - ignore empty sequence", "[sequence_senders][ignore_all_values]") {
    auto sndr = exec::ignore_all_values(exec::empty_sequence());
    using Sender = decltype(sndr);
    STATIC_REQUIRE(STDEXEC::sender_in<Sender, STDEXEC::env<>>);
    STATIC_REQUIRE(
      std::same_as<
        STDEXEC::completion_signatures<STDEXEC::set_value_t()>,
        STDEXEC::completion_signatures_of_t<Sender, STDEXEC::env<>>
      >);
    STATIC_REQUIRE(STDEXEC::sender_expr_for<Sender, exec::ignore_all_values_t>);
    CHECK(STDEXEC::sync_wait(sndr));
  }

  TEST_CASE("ignore_all_values - ignore just(42)", "[sequence_senders][ignore_all_values]") {
    auto sndr = exec::ignore_all_values(STDEXEC::just(42));
    using Sender = decltype(sndr);
    STATIC_REQUIRE(STDEXEC::sender_in<Sender, STDEXEC::env<>>);
    STATIC_REQUIRE(
      std::same_as<
        STDEXEC::completion_signatures<STDEXEC::set_value_t()>,
        STDEXEC::completion_signatures_of_t<Sender, STDEXEC::env<>>
      >);
    CHECK(STDEXEC::sync_wait(sndr));
  }

  TEST_CASE("ignore_all_values - ignore just()", "[sequence_senders][ignore_all_values]") {
    auto sndr = exec::ignore_all_values(STDEXEC::just());
    using Sender = decltype(sndr);
    STATIC_REQUIRE(STDEXEC::sender_in<Sender, STDEXEC::env<>>);
    STATIC_REQUIRE(
      std::same_as<
        STDEXEC::completion_signatures<STDEXEC::set_value_t()>,
        STDEXEC::completion_signatures_of_t<Sender, STDEXEC::env<>>
      >);
    CHECK(STDEXEC::sync_wait(sndr));
  }

  TEST_CASE("ignore_all_values - ignore just_stopped()", "[sequence_senders][ignore_all_values]") {
    auto sndr = exec::ignore_all_values(STDEXEC::just_stopped());
    using Sender = decltype(sndr);
    STATIC_REQUIRE(STDEXEC::sender_in<Sender, STDEXEC::env<>>);
    STATIC_REQUIRE(
      STDEXEC::__mset_eq<
        STDEXEC::__mset<STDEXEC::set_value_t(), STDEXEC::set_stopped_t()>,
        STDEXEC::completion_signatures_of_t<Sender, STDEXEC::env<>>
      >);
    CHECK_FALSE(STDEXEC::sync_wait(sndr));
  }

#if !STDEXEC_NO_STD_EXCEPTIONS()
  TEST_CASE("ignore_all_values - ignore just_error()", "[sequence_senders][ignore_all_values]") {
    auto sndr = exec::ignore_all_values(
      STDEXEC::just_error(std::make_exception_ptr(std::runtime_error("test"))));
    using Sender = decltype(sndr);
    STATIC_REQUIRE(STDEXEC::sender_in<Sender, STDEXEC::env<>>);
    STATIC_REQUIRE(
      STDEXEC::__mset_eq<
        STDEXEC::__mset<STDEXEC::set_value_t(), STDEXEC::set_error_t(std::exception_ptr)>,
        STDEXEC::completion_signatures_of_t<Sender, STDEXEC::env<>>
      >);
    CHECK_THROWS(STDEXEC::sync_wait(sndr));
  }
#endif // !STDEXEC_NO_STD_EXCEPTIONS()

  struct sequence_op {
    void start() & noexcept {
    }
  };

  template <class Item>
  struct sequence {
    using sender_concept = exec::sequence_sender_t;

    using completion_signatures =
      STDEXEC::completion_signatures<STDEXEC::set_value_t(), STDEXEC::set_error_t(int)>;

    using item_types = exec::item_types<Item>;

    template <class R>
    auto subscribe(R&&) const noexcept -> sequence_op {
      return sequence_op{};
    }
  };

  TEST_CASE("ignore_all_values - Merge error and stop signatures from sequence and items") {
    using just_t = decltype(STDEXEC::just_error(
      std::make_exception_ptr(std::runtime_error("test"))));
    sequence<just_t> seq;
    auto ignore = exec::ignore_all_values(seq);
    using ActualSigs = STDEXEC::completion_signatures_of_t<decltype(ignore), STDEXEC::env<>>;
    using ExpectedSigs = STDEXEC::__mset<
      STDEXEC::set_value_t(),
      STDEXEC::set_error_t(int),
      STDEXEC::set_error_t(std::exception_ptr)
    >;
    STATIC_REQUIRE(STDEXEC::__mset_eq<ExpectedSigs, ActualSigs>);
  }
} // namespace
