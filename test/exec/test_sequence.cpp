/*
 * Copyright (c) 2024 NVIDIA Corporation
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

#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>
#include <exec/sequence.hpp>
#include <test_common/type_helpers.hpp>
#include <test_common/receivers.hpp>

#include <memory>
#include <exception>

namespace {
  struct big {
    std::unique_ptr<int[]> p{new int[1000]};

    big() = default;

    auto operator==(const big&) const noexcept -> bool {
      return true;
    }
  };

  struct connect_exception : std::exception {
    connect_exception() = default;

    [[nodiscard]]
    auto what() const noexcept -> const char* override {
      return "connect";
    }
  };

  struct throwing_connect {
    using sender_concept = ex::sender_t;
    using completion_signatures = ex::completion_signatures<ex::set_value_t()>;

    struct op {
      using operation_state_concept = ex::operation_state_t;

      void start() & noexcept {
      }
    };

    [[nodiscard]]
    auto connect(ex::__ignore) const -> op {
      STDEXEC_THROW(connect_exception{});
    }
  };
} // namespace

TEST_CASE("sequence produces a sender", "[sequence]") {
  // The sequence algorithm requires at least one sender.
  STATIC_REQUIRE(!ex::__callable<exec::sequence_t>);

  auto s0 = exec::sequence(ex::just(42));
  STATIC_REQUIRE(ex::sender<decltype(s0)>);
  STATIC_REQUIRE(ex::sender_in<decltype(s0)>);
  check_val_types<ex::__mset<pack<int>>>(s0);
  // sequence with one argument doesn't add an exception_ptr error completion
  check_err_types<ex::__mset<>>(s0);
  check_sends_stopped<false>(s0);

  using env_t = ex::prop<ex::get_allocator_t, std::allocator<void>>;
  auto s1 = exec::sequence(ex::just_error(42), ex::read_env(ex::get_allocator));
  STATIC_REQUIRE(ex::sender<decltype(s1)>);
  STATIC_REQUIRE(!ex::sender_in<decltype(s1)>);
  STATIC_REQUIRE(ex::sender_in<decltype(s1), env_t>);
  check_val_types<ex::__mset<pack<const std::allocator<void>&>>, env_t>(s1);
  check_err_types<ex::__mset<std::exception_ptr, int>, env_t>(s1);
  check_sends_stopped<false, env_t>(s1);
}

TEST_CASE("sequence with one argument works", "[sequence]") {
  SECTION("value completion") {
    auto sndr = exec::sequence(ex::just(42));
    auto op = ex::connect(std::move(sndr), expect_value_receiver{42});
    ex::start(op);
  }
  SECTION("error completion") {
    auto sndr = exec::sequence(ex::just_error(42));
    auto op = ex::connect(std::move(sndr), expect_error_receiver{42});
    ex::start(op);
  }
  SECTION("stopped completion") {
    auto sndr = exec::sequence(ex::just_stopped());
    auto op = ex::connect(std::move(sndr), expect_stopped_receiver{});
    ex::start(op);
  }
}

TEST_CASE("sequence with two arguments works", "[sequence]") {
  SECTION("value completion") {
    auto sndr = exec::sequence(ex::just(big{}), ex::just(big{}, 4, 6, 8));
    auto op = ex::connect(std::move(sndr), expect_value_receiver{big{}, 4, 6, 8});
    ex::start(op);
  }
  SECTION("error completion 1") {
    auto sndr = exec::sequence(ex::just_error(big{}), ex::just(big{}, 4, 6, 8));
    auto op = ex::connect(std::move(sndr), expect_error_receiver{big{}});
    ex::start(op);
  }
  SECTION("error completion 2") {
    auto sndr = exec::sequence(ex::just(big{}, 4, 6, 8), ex::just_error(big{}));
    auto op = ex::connect(std::move(sndr), expect_error_receiver{big{}});
    ex::start(op);
  }
  SECTION("stopped completion 1") {
    auto stop = ex::just(big{}) | ex::let_value([](auto&) { return ex::just_stopped(); });
    auto sndr = exec::sequence(std::move(stop), ex::just(big{}, 4, 6, 8));
    auto op = ex::connect(std::move(sndr), expect_stopped_receiver{});
    ex::start(op);
  }
  SECTION("stopped completion 2") {
    auto stop = ex::just(big{}) | ex::let_value([](auto&) { return ex::just_stopped(); });
    auto sndr = exec::sequence(ex::just(big{}, 4, 6, 8), std::move(stop));
    auto op = ex::connect(std::move(sndr), expect_stopped_receiver{});
    ex::start(op);
  }
}

#if !STDEXEC_STD_NO_EXCEPTIONS()
TEST_CASE("sequence with sender with throwing connect", "[sequence]") {
  auto err = std::make_exception_ptr(connect_exception{});
  auto sndr = exec::sequence(ex::just(big{}), throwing_connect{}, ex::just(big{}, 42));
  auto op = ex::connect(std::move(sndr), expect_error_receiver{err});
  ex::start(op);
}
#endif // !STDEXEC_STD_NO_EXCEPTIONS()
