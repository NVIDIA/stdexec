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

#include "exec/sequence/transform_each.hpp"

#include "exec/sender_for.hpp"
#include "exec/sequence/empty_sequence.hpp"
#include "exec/sequence/ignore_all_values.hpp"
#include "exec/sequence/iterate.hpp"
#include <test_common/catch2.hpp>

#include <test_common/receivers.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/senders.hpp>
#include <test_common/type_helpers.hpp>

#include <array>
#include <exception>
#include <ranges>
#include <utility>

namespace
{

  struct next_rcvr
  {
    using receiver_concept = ex::receiver_tag;

    auto set_next(auto item)
    {
      return item;
    }

    void set_value() noexcept {}
  };

  TEST_CASE("transform_each - subscribe is noexcept for a nonthrowing child",
            "[sequence_senders][transform_each]")
  {
    auto transformed = exec::transform_each(exec::empty_sequence(), ex::then([]() noexcept {}));

    STATIC_REQUIRE(noexcept(exec::subscribe(transformed, next_rcvr{})));
    STATIC_REQUIRE(noexcept(exec::subscribe(std::as_const(transformed), next_rcvr{})));
    STATIC_REQUIRE(noexcept(exec::subscribe(std::move(transformed), next_rcvr{})));

    auto op = exec::subscribe(std::move(transformed), next_rcvr{});
    ex::start(op);
  }

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  struct range_error
  {};

  struct throwing_copy_value
  {
    throwing_copy_value() = default;

    throwing_copy_value(throwing_copy_value const &)
    {
      throw range_error{};
    }

    throwing_copy_value(throwing_copy_value &&) noexcept = default;
  };

  TEST_CASE("transform_each - subscribe propagates adaptor copy exceptions",
            "[sequence_senders][transform_each]")
  {
    auto adaptor     = ex::then([capture = throwing_copy_value{}]() noexcept { (void) capture; });
    auto transformed = exec::transform_each(exec::empty_sequence(), std::move(adaptor));

    STATIC_REQUIRE_FALSE(noexcept(exec::subscribe(transformed, next_rcvr{})));
    CHECK_THROWS_AS(exec::subscribe(transformed, next_rcvr{}), range_error);
  }

  template <bool ThrowFromBegin>
  struct throwing_range
  {
    using iterator = std::array<int, 1>::iterator;

    auto begin() noexcept(!ThrowFromBegin) -> iterator
    {
      if constexpr (ThrowFromBegin)
      {
        throw range_error{};
      }
      return values_.begin();
    }

    auto end() noexcept(ThrowFromBegin) -> iterator
    {
      if constexpr (!ThrowFromBegin)
      {
        throw range_error{};
      }
      return values_.end();
    }

    std::array<int, 1> values_{0};
  };

  struct range_rcvr : next_rcvr
  {
    void set_error(std::exception_ptr) noexcept {}

    void set_stopped() noexcept {}
  };

  TEMPLATE_TEST_CASE("transform_each - subscribe propagates range exceptions",
                     "[sequence_senders][transform_each][iterate]",
                     throwing_range<true>,
                     throwing_range<false>)
  {
    TestType range;
    auto     transformed = exec::transform_each(exec::iterate(std::views::all(range)),
                                            ex::then([](int) noexcept {}));

    SECTION("lvalue sender")
    {
      STATIC_REQUIRE_FALSE(noexcept(exec::subscribe(transformed, range_rcvr{})));
      CHECK_THROWS_AS(exec::subscribe(transformed, range_rcvr{}), range_error);
    }

    SECTION("const lvalue sender")
    {
      STATIC_REQUIRE_FALSE(noexcept(exec::subscribe(std::as_const(transformed), range_rcvr{})));
      CHECK_THROWS_AS(exec::subscribe(std::as_const(transformed), range_rcvr{}), range_error);
    }

    SECTION("rvalue sender")
    {
      STATIC_REQUIRE_FALSE(noexcept(exec::subscribe(std::move(transformed), range_rcvr{})));
      CHECK_THROWS_AS(exec::subscribe(std::move(transformed), range_rcvr{}), range_error);
    }
  }
#endif  // !STDEXEC_NO_STDCPP_EXCEPTIONS()

  TEST_CASE("transform_each - transform sender applies adaptor to no elements",
            "[sequence_senders][transform_each][empty_sequence]")
  {
    int  counter     = 0;
    auto transformed = exec::transform_each(exec::empty_sequence(),
                                            ex::then([&counter]() noexcept { ++counter; }));
    auto op          = exec::subscribe(transformed, next_rcvr{});
    ex::start(op);
    CHECK(counter == 0);
  }

  TEST_CASE("transform_each - transform sender applies adaptor to a sender",
            "[sequence_senders][transform_each]")
  {
    int  value       = 0;
    auto transformed = exec::transform_each(ex::just(42),
                                            ex::then([&value](int x) noexcept { value = x; }));
    auto op          = exec::subscribe(transformed, next_rcvr{});
    ex::start(op);
    CHECK(value == 42);
  }

  TEST_CASE("transform_each - transform sender applies adaptor to a sender and ignores all values",
            "[sequence_senders][transform_each][ignore_all_values]")
  {
    int  value       = 0;
    auto transformed = exec::transform_each(ex::just(42), ex::then([&value](int x) { value = x; }))
                     | exec::ignore_all_values();
    ex::sync_wait(transformed);
    CHECK(value == 42);
  }

  TEST_CASE("transform_each - transform sender applies adaptor to each item",
            "[sequence_senders][transform_each][iterate]")
  {
    auto range = [](auto from, auto to)
    {
      return exec::iterate(std::views::iota(from, to));
    };
    auto then_each = [](auto f)
    {
      return exec::transform_each(ex::then(f));
    };
    int  total = 0;
    auto sum   = range(0, 10) | then_each([&total](int x) noexcept { total += x; });
    ex::sync_wait(exec::ignore_all_values(sum));
    CHECK(total == 45);
  }

  struct my_domain
  {
    template <exec::sender_for<ex::then_t> Sender, class Env>
    static auto transform_sender(STDEXEC::set_value_t, Sender &&, Env const &)
    {
      return ex::just(int{42});
    }
  };

  TEST_CASE("transform_each - can be customized late", "[transform_each][ignore_all_values]")
  {
    // The customization will return a different value
    basic_inline_scheduler<my_domain> sched;
    int                               result = 0;
    auto                              start  = ex::just(std::string{"hello"});
    auto with_scheduler = ex::write_env(ex::prop{ex::get_scheduler, inline_scheduler()});
    auto adaptor        = ex::on(sched, ex::then([](std::string x) { return x + ", world"; }))
                 | with_scheduler;
    auto snd = start | exec::transform_each(adaptor)
             | exec::transform_each(ex::then([&](int x) { result = x; }))
             | exec::ignore_all_values();
    ex::sync_wait(snd);
    CHECK(result == 42);
  }
}  // namespace
