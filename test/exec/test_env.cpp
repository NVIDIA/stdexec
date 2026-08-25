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

#include <type_traits>
#include <utility>

namespace
{
#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  struct missing_query : STDEXEC::__query<missing_query>
  {
    using STDEXEC::__query<missing_query>::operator();
  };

  struct default_move_error
  {};

  struct receiver_move_error
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

  struct throwing_receiver
  {
    using receiver_concept = STDEXEC::receiver_tag;

    explicit throwing_receiver(bool* throw_on_move) noexcept
      : throw_on_move_{throw_on_move}
    {}

    throwing_receiver(throwing_receiver const &) noexcept = default;

    throwing_receiver(throwing_receiver&& other)
      : throw_on_move_{other.throw_on_move_}
    {
      if (*throw_on_move_)
      {
        throw receiver_move_error{};
      }
    }

    template <class _Value>
    void set_value(_Value&&) noexcept
    {}

    void set_error(std::exception_ptr) noexcept {}
    void set_stopped() noexcept {}

    auto get_env() const noexcept -> STDEXEC::env<>
    {
      return {};
    }

    bool* throw_on_move_;
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

  TEST_CASE("read_with_default connect propagates receiver move exceptions", "[env]")
  {
    bool              throw_on_move = false;
    auto              sndr          = exec::read_with_default(missing_query{}, 42);
    throwing_receiver rcvr{&throw_on_move};
    throw_on_move = true;

    STATIC_REQUIRE_FALSE(noexcept(std::move(sndr).connect(rcvr)));
    CHECK_THROWS_AS(std::move(sndr).connect(rcvr), receiver_move_error);
  }
#endif  // !STDEXEC_NO_STDCPP_EXCEPTIONS()

  TEST_CASE("without propagates environment move exceptions", "[env]")
  {
    struct missing_query : STDEXEC::__query<missing_query>
    {
      using STDEXEC::__query<missing_query>::operator();
    };

    struct without_move_error
    {};

    struct throwing_env
    {
      explicit throwing_env(bool* throw_on_move) noexcept
        : throw_on_move_{throw_on_move}
      {}

      throwing_env(throwing_env const & other) noexcept
        : throw_on_move_{other.throw_on_move_}
      {}

      throwing_env(throwing_env&& other)
        : throw_on_move_{other.throw_on_move_}
      {
#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
        if (*throw_on_move_)
        {
          throw without_move_error{};
        }
#endif
      }

      int query(Foo) const noexcept
      {
        return 42;
      }

      bool* throw_on_move_;
    };

    bool         throw_on_move = false;
    throwing_env missing_env{&throw_on_move};
    throwing_env queried_env{&throw_on_move};

    STATIC_REQUIRE_FALSE(noexcept(exec::without(std::move(missing_env), missing_query{})));
    STATIC_REQUIRE_FALSE(noexcept(exec::without(std::move(queried_env), foo)));
    STATIC_REQUIRE(noexcept(exec::without(missing_env, missing_query{})));
    STATIC_REQUIRE(noexcept(exec::without(queried_env, foo)));

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
    throw_on_move = true;
    CHECK_THROWS_AS(exec::without(std::move(missing_env), missing_query{}), without_move_error);
    CHECK_THROWS_AS(exec::without(std::move(queried_env), foo), without_move_error);
#endif

    struct throwing_copy_error
    {};

    struct throwing_copy_env
    {
      explicit throwing_copy_env(bool* throw_on_copy) noexcept
        : throw_on_copy_{throw_on_copy}
      {}

      throwing_copy_env(throwing_copy_env const & other)
        : throw_on_copy_{other.throw_on_copy_}
      {
#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
        if (*throw_on_copy_)
        {
          throw throwing_copy_error{};
        }
#endif
      }

      throwing_copy_env(throwing_copy_env&&) noexcept = default;

      bool* throw_on_copy_;
    };

    bool              throw_on_copy = false;
    throwing_copy_env copy_env{&throw_on_copy};

    STATIC_REQUIRE(
      std::is_same_v<decltype(exec::without(copy_env, missing_query{})), throwing_copy_env&>);
    STATIC_REQUIRE(noexcept(exec::without(copy_env, missing_query{})));

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
    throw_on_copy = true;
    auto&& result = exec::without(copy_env, missing_query{});
    CHECK(&result == &copy_env);
#endif
  }
}  // namespace
