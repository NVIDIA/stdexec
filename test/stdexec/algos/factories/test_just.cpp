/*
 * Copyright (c) 2022 Lucian Radu Teodorescu
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
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ex = STDEXEC;

namespace {

  constexpr int test_constexpr() noexcept {
    struct receiver {
      using receiver_concept = ex::receiver_t;
      constexpr void set_value(const int i) && noexcept {
        this->i = i;
      }
      int& i;
    };
    int i = 0;
    auto op = ex::connect(ex::just(5), receiver{i});
    ex::start(op);
    return i;
  }
  static_assert(test_constexpr() == 5);

  TEST_CASE("Simple test for just", "[factories][just]") {
    auto o1 = ex::connect(ex::just(1), expect_value_receiver(1));
    ex::start(o1);
    auto o2 = ex::connect(ex::just(2), expect_value_receiver(2));
    ex::start(o2);
    auto o3 = ex::connect(ex::just(3), expect_value_receiver(3));
    ex::start(o3);

    auto o4 =
      ex::connect(ex::just(std::string("this")), expect_value_receiver(std::string("this")));
    ex::start(o4);
    auto o5 =
      ex::connect(ex::just(std::string("that")), expect_value_receiver(std::string("that")));
    ex::start(o5);
  }

  TEST_CASE("just returns a sender", "[factories][just]") {
    using t = decltype(ex::just(1));
    static_assert(ex::sender<t>, "ex::just must return a sender");
    REQUIRE(ex::sender<t>);
    REQUIRE(ex::enable_sender<t>);
  }

  TEST_CASE("just can handle multiple values", "[factories][just]") {
    bool executed{false};
    auto f = [&](int x, double d) {
      CHECK(x == 3);
      CHECK(d == 0.14);
      executed = true;
    };
    auto op = ex::connect(ex::just(3, 0.14), make_fun_receiver(std::move(f)));
    ex::start(op);
    CHECK(executed);
  }

  TEST_CASE("value types are properly set for just", "[factories][just]") {
    check_val_types<ex::__mset<pack<int>>>(ex::just(1));
    check_val_types<ex::__mset<pack<double>>>(ex::just(3.14));
    check_val_types<ex::__mset<pack<std::string>>>(ex::just(std::string{}));

    check_val_types<ex::__mset<pack<int, double>>>(ex::just(1, 3.14));
    check_val_types<ex::__mset<pack<int, double, std::string>>>(ex::just(1, 3.14, std::string{}));
  }

  TEST_CASE("error types are properly set for just", "[factories][just]") {
    check_err_types<ex::__mset<>>(ex::just(1));
  }

  TEST_CASE("just cannot call set_stopped", "[factories][just]") {
    check_sends_stopped<false>(ex::just(1));
  }

  TEST_CASE("just works with value type", "[factories][just]") {
    auto snd = ex::just(std::string{"hello"});

    // Check reported type
    check_val_types<ex::__mset<pack<std::string>>>(snd);

    // Check received value
    std::string res;
    typecat cat{typecat::undefined};
    auto op =
      ex::connect(std::move(snd), typecat_receiver<std::string>{.value_ = &res, .cat_ = &cat});
    ex::start(op);
    CHECK(res == "hello");
    CHECK(cat == typecat::rvalref);
  }

  TEST_CASE("just works with ref type", "[factories][just]") {
    std::string original{"hello"};
    auto snd = ex::just(original);

    // Check reported type
    check_val_types<ex::__mset<pack<std::string>>>(snd);

    // Check received value
    std::string res;
    typecat cat{typecat::undefined};
    auto op =
      ex::connect(std::move(snd), typecat_receiver<std::string>{.value_ = &res, .cat_ = &cat});
    ex::start(op);
    CHECK(res == original);
    CHECK(cat == typecat::rvalref);
  }

  TEST_CASE("just works with const-ref type", "[factories][just]") {
    const std::string original{"hello"};
    auto snd = ex::just(original);

    // Check reported type
    check_val_types<ex::__mset<pack<std::string>>>(snd);

    // Check received value
    std::string res;
    typecat cat{typecat::undefined};
    auto op =
      ex::connect(std::move(snd), typecat_receiver<std::string>{.value_ = &res, .cat_ = &cat});
    ex::start(op);
    CHECK(res == original);
    CHECK(cat == typecat::rvalref);
  }

  TEST_CASE("just works with types with throwing move", "[factories][just]") {
    struct throwing_move {
      explicit throwing_move(std::size_t& throws_after) noexcept
        : throws_after_(throws_after) {
      }

      throwing_move(throwing_move&& other) // NOLINT(bugprone-exception-escape)
        : throws_after_(other.throws_after_) {
        if (throws_after_) {
          --throws_after_;
        } else {
          throw std::runtime_error("Throwing as requested");
        }
      }
     private:
      std::size_t& throws_after_;
    };

    std::size_t throws_after = 0;
    const auto repeat_until_succeeds = [&](auto f) noexcept -> decltype(auto) {
      struct guard {
        ~guard() noexcept {
          CHECK(threw);
        }
        bool threw{false};
      };
      guard g;
      for (;;) {
        auto orig = throws_after;
        try {
          return f();
        } catch (...) {
          g.threw = true;
          ++orig;
          throws_after = orig;
        }
      }
    };
    auto sender = repeat_until_succeeds(
      [&]() { return ::STDEXEC::just(throwing_move(throws_after)); });
    CHECK(throws_after == 0);
    std::size_t invoked = 0;
    auto op = repeat_until_succeeds([&]() {
      return ::STDEXEC::connect(std::move(sender), make_fun_receiver([&](throwing_move&&) noexcept {
                                  ++invoked;
                                }));
    });
    CHECK(throws_after == 0);
    CHECK(invoked == 0);
    ::STDEXEC::start(op);
    CHECK(invoked == 1);
  }
} // namespace
