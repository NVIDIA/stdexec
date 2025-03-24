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

namespace ex = stdexec;

namespace {

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
} // namespace
