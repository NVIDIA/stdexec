/*
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#include <exec/env.hpp>
#include <stdexec/execution.hpp>
#include <test_common/catch2.hpp>

namespace
{
#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  struct missing_query : STDEXEC::__query<missing_query>
  {
    using STDEXEC::__query<missing_query>::operator();
  };

  struct default_move_error
  {};

  struct throwing_default
  {
    explicit throwing_default(bool* throw_on_move) noexcept
      : throw_on_move_{throw_on_move}
    {}

    throwing_default(throwing_default const &) noexcept = default;

    throwing_default(throwing_default&& other)
      : throw_on_move_{other.throw_on_move_}
    {
      if (*throw_on_move_)
      {
        throw default_move_error{};
      }
    }

    bool* throw_on_move_;
  };

  struct read_with_default_receiver
  {
    using receiver_concept = STDEXEC::receiver_tag;

    template <class _Value>
    void set_value(_Value&&) noexcept
    {}

    void set_error(std::exception_ptr) noexcept {}
    void set_stopped() noexcept {}

    auto get_env() const noexcept -> STDEXEC::env<>
    {
      return {};
    }
  };
#endif  // !STDEXEC_NO_STDCPP_EXCEPTIONS()

  // Two dummy properties:
  constexpr struct Foo
    : STDEXEC::__query<Foo>
    , STDEXEC::forwarding_query_t
  {
    using STDEXEC::__query<Foo>::operator();
  } foo{};

  constexpr struct Bar : STDEXEC::__query<Bar>
  {
    static constexpr auto query(STDEXEC::forwarding_query_t) noexcept -> bool
    {
      return true;
    }
  } bar{};

  TEST_CASE("Test make_env works", "[env]")
  {
    auto e = STDEXEC::prop{foo, 42};
    CHECK(foo(e) == 42);

    auto e2 = exec::make_env(e, STDEXEC::prop{bar, 43});
    CHECK(foo(e2) == 42);
    CHECK(bar(e2) == 43);

    auto e3 = exec::make_env(e2, STDEXEC::prop{foo, 44});
    CHECK(foo(e3) == 44);
    CHECK(bar(e3) == 43);

    auto e4 = exec::without(e3, foo);
    STATIC_REQUIRE(!std::invocable<Foo, decltype(e4)>);
    CHECK(bar(e4) == 43);
  }

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  TEST_CASE("read_with_default connect propagates default move exceptions", "[env]")
  {
    bool throw_on_move = false;
    auto sndr          = exec::read_with_default(missing_query{}, throwing_default{&throw_on_move});
    throw_on_move      = true;

    STATIC_REQUIRE_FALSE(noexcept(std::move(sndr).connect(read_with_default_receiver{})));
    CHECK_THROWS_AS(std::move(sndr).connect(read_with_default_receiver{}), default_move_error);
  }
#endif  // !STDEXEC_NO_STDCPP_EXCEPTIONS()
}  // namespace
