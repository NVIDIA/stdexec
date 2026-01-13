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

#include <exec/any_sender_of.hpp>
#include <exec/env.hpp>
#include <exec/inline_scheduler.hpp>
#include <exec/static_thread_pool.hpp>
#include <exec/when_any.hpp>
#include <stdexec/stop_token.hpp>

#include <test_common/receivers.hpp>
#include <test_common/schedulers.hpp>

#include <catch2/catch.hpp>

using namespace STDEXEC;
using namespace exec;

// NOLINTBEGIN(misc-unused-using-decls)
using STDEXEC::completion_signatures;
using STDEXEC::set_error;
using STDEXEC::set_error_t;
using STDEXEC::set_stopped;
using STDEXEC::set_stopped_t;
using STDEXEC::set_value;
using STDEXEC::set_value_t;
// NOLINTEND(misc-unused-using-decls)

namespace {
  TEST_CASE("", "[utilities][completion_signatures]") {
    STATIC_REQUIRE(completion_signatures{} == completion_signatures{});
    STATIC_REQUIRE_FALSE(completion_signatures{} != completion_signatures{});
  }

  // Additional tests for completion_signatures
  TEST_CASE("completion_signatures_basic", "[utilities][completion_signatures]") {
    constexpr auto cs_empty = completion_signatures<>{};
    constexpr auto cs_value = completion_signatures<set_value_t(int)>{};
    constexpr auto cs_error = completion_signatures<set_error_t(float)>{};
    // constexpr auto cs_stopped = completion_signatures<set_stopped_t()>{};
    constexpr auto cs_all =
      completion_signatures<set_value_t(int), set_error_t(float), set_stopped_t()>{};

    // Test size
    STATIC_REQUIRE(cs_empty.__size() == 0);
    STATIC_REQUIRE(cs_value.__size() == 1);
    STATIC_REQUIRE(cs_all.__size() == 3);

    // Test contains
    STATIC_REQUIRE(cs_value.__contains(static_cast<set_value_t (*)(int)>(nullptr)));
    STATIC_REQUIRE_FALSE(cs_value.__contains(static_cast<set_error_t (*)(float)>(nullptr)));
    STATIC_REQUIRE(cs_all.__contains(static_cast<set_stopped_t (*)()>(nullptr)));

    // Test count
    STATIC_REQUIRE(cs_all.__count(set_value) == 1);
    STATIC_REQUIRE(cs_all.__count(set_error) == 1);
    STATIC_REQUIRE(cs_all.__count(set_stopped) == 1);

    // Test operator==
    STATIC_REQUIRE(cs_value == cs_value);
    STATIC_REQUIRE_FALSE(cs_value == cs_error);
    STATIC_REQUIRE(cs_empty == cs_empty);
    STATIC_REQUIRE(cs_all == cs_all);
    STATIC_REQUIRE(
      completion_signatures<set_value_t(int), set_error_t(float), set_value_t(int)>{}
      == completion_signatures<set_error_t(float), set_value_t(int)>{});

    // Test operator!=
    STATIC_REQUIRE(cs_value != cs_error);
    STATIC_REQUIRE_FALSE(cs_all != cs_all);

    // // Test operator+
    // STATIC_REQUIRE(
    //   (cs_value + cs_error) == completion_signatures<set_value_t(int), set_error_t(float)>{});
    // STATIC_REQUIRE((cs_empty + cs_value) == cs_value);
    // STATIC_REQUIRE((cs_value + cs_empty) == cs_value);

    // // Test operator-
    // STATIC_REQUIRE(
    //   (cs_all - cs_value) == completion_signatures<set_error_t(float), set_stopped_t()>{});
    // STATIC_REQUIRE(
    //   (cs_all - cs_error) == completion_signatures<set_value_t(int), set_stopped_t()>{});
    // STATIC_REQUIRE(
    //   (cs_all - cs_stopped) == completion_signatures<set_value_t(int), set_error_t(float)>{});
    // STATIC_REQUIRE((cs_all - cs_all) == completion_signatures<>{});
    // STATIC_REQUIRE((cs_value - cs_error) == cs_value);
  }

  // Test select
  TEST_CASE("completion_signatures_select", "[utilities][completion_signatures]") {
    constexpr auto cs =
      completion_signatures<set_value_t(int), set_error_t(float), set_stopped_t()>{};

    // select(set_value) should return only set_value_t(int)
    constexpr auto v = cs.__select(set_value);
    STATIC_REQUIRE(v.__size() == 1);
    STATIC_REQUIRE(v.__contains<set_value_t(int)>());

    // select(set_error) should return only set_error_t(float)
    constexpr auto e = cs.__select(set_error);
    STATIC_REQUIRE(e.__size() == 1);
    STATIC_REQUIRE(e.__contains<set_error_t(float)>());

    // select(set_stopped) should return only set_stopped_t()
    constexpr auto s = cs.__select(set_stopped);
    STATIC_REQUIRE(s.__size() == 1);
    STATIC_REQUIRE(s.__contains<set_stopped_t()>());
  }

  // Test filter
  struct filter_value_only {
    template <class Sig>
    constexpr bool operator()(Sig*) const noexcept {
      return STDEXEC::__detail::__tag_of_sig_t<Sig>::__disposition
          == STDEXEC::__disposition::__value;
    }
  };

  TEST_CASE("completion_signatures_filter", "[utilities][completion_signatures]") {
    constexpr auto cs =
      completion_signatures<set_value_t(int), set_error_t(float), set_stopped_t()>{};
    constexpr auto filtered = cs.__filter(filter_value_only{});
    STATIC_REQUIRE(filtered.__size() == 1);
    STATIC_REQUIRE(filtered.__contains<set_value_t(int)>());
  }

  // Test apply
  struct count_signatures {
    template <class... Sigs>
    constexpr int operator()(Sigs*...) const noexcept {
      return sizeof...(Sigs);
    }
  };

  TEST_CASE("completion_signatures_apply", "[utilities][completion_signatures]") {
    constexpr auto cs =
      completion_signatures<set_value_t(int), set_error_t(float), set_stopped_t()>{};
    constexpr int count = cs.__apply(count_signatures{});
    STATIC_REQUIRE(count == 3);
  }
} // namespace
