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

#include "exec/just_from.hpp"
#include "test_common/receivers.hpp"
#include "test_common/tuple.hpp"
#include "test_common/type_helpers.hpp"

#include <test_common/catch2.hpp>

namespace
{
  constinit int global_int = 0;

  struct throwing_move_callable
  {
    explicit throwing_move_callable(bool& should_throw) noexcept
      : should_throw_(&should_throw)
    {}

    throwing_move_callable(throwing_move_callable const &other) noexcept
      : should_throw_(other.should_throw_)
    {}

    throwing_move_callable(throwing_move_callable&& other) noexcept(false)
      : should_throw_(other.should_throw_)
    {
#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
      if (*should_throw_)
      {
        throw 42;
      }
#endif
    }

    template <class Sink>
    auto operator()(Sink sink) noexcept
    {
      return sink();
    }

    bool* should_throw_;
  };

  TEST_CASE("just_from is a sender", "[just_from]")
  {
    SECTION("potentially throwing")
    {
      auto s  = exec::just_from([](auto sink) { return sink(42); });
      using S = decltype(s);
      STATIC_REQUIRE(ex::sender<S>);
      STATIC_REQUIRE(ex::sender_in<S>);
      ::check_val_types<ex::__mset<pack<int>>>(s);
      ::check_err_types<ex::__mset<std::exception_ptr>>(s);
      ::check_sends_stopped<false>(s);
    }

    SECTION("not potentially throwing")
    {
      auto s  = exec::just_from([](auto sink) noexcept { return sink(42); });
      using S = decltype(s);
      STATIC_REQUIRE(ex::sender<S>);
      STATIC_REQUIRE(ex::sender_in<S>);
      ::check_val_types<ex::__mset<pack<int>>>(s);
      ::check_err_types<ex::__mset<>>(s);
      ::check_sends_stopped<false>(s);
    }
  }

  TEST_CASE("just_from basically works", "[just_from]")
  {
    auto s = exec::just_from([](auto sink) noexcept { return sink(42, 43, 44); });
    ::check_val_types<ex::__mset<pack<int, int, int>>>(s);
    ::check_err_types<ex::__mset<>>(s);
    ::check_sends_stopped<false>(s);

    auto [a, b, c] = ex::sync_wait(s).value();
    CHECK(a == 42);
    CHECK(b == 43);
    CHECK(c == 44);
  }

  TEST_CASE("just_from is conditionally noexcept when storing the callable", "[just_from]")
  {
    auto nothrow_fn = [](auto sink) noexcept
    {
      return sink();
    };
    STATIC_REQUIRE(noexcept(exec::just_from(nothrow_fn)));

    bool                   should_throw = false;
    throwing_move_callable fn{should_throw};
    STATIC_REQUIRE_FALSE(noexcept(exec::just_from(fn)));

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
    should_throw = true;
    CHECK_THROWS_AS(exec::just_from(fn), int);
#endif
  }

  TEST_CASE("just_from submit is conditionally noexcept", "[just_from]")
  {
    bool should_throw = false;
    auto s            = exec::just_from(throwing_move_callable{should_throw});

    STATIC_REQUIRE_FALSE(noexcept(static_cast<decltype(s)&&>(s).submit(empty_recv::recv0{})));
    STATIC_REQUIRE(noexcept(s.submit(empty_recv::recv0{})));

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
    should_throw = true;
    CHECK_THROWS_AS(static_cast<decltype(s)&&>(s).submit(empty_recv::recv0{}), int);
#endif
  }

  TEST_CASE("just_from with multiple completions", "[just_from]")
  {
    auto fn = [](auto sink) noexcept
    {
      if (sizeof(sink) == ~0ul)
      {
        sink(42);
      }
      else
      {
        sink(43, 44);
      }
      return ex::completion_signatures<ex::set_value_t(int), ex::set_value_t(int, int)>{};
    };
    auto s = exec::just_from(fn);
    ::check_val_types<ex::__mset<pack<int>, pack<int, int>>>(s);
    ::check_err_types<ex::__mset<>>(s);
    ::check_sends_stopped<false>(s);

    auto var = ex::sync_wait_with_variant(s).value();
    std::visit(
      []<class Tupl>(Tupl tupl)
      {
        constexpr auto N = std::tuple_size_v<Tupl>;
        CHECK(N == 2);
        if constexpr (N == 2)
        {
          CHECK_TUPLE(tupl == std::tuple{43, 44});
        }
      },
      var);
  }

  TEST_CASE("just_from can send references", "[just_from]")
  {
    global_int = 42;
    auto s     = exec::just_from([](auto sink) noexcept { return sink(global_int); })
           | ex::then(
               [](int &i) noexcept
               {
                 CHECK(&i == &global_int);
                 return std::ref(i);
               });
    ::check_val_types<ex::__mset<pack<std::reference_wrapper<int>>>>(s);
    auto [iref] = ex::sync_wait(s).value();
    CHECK(&iref.get() == &global_int);
  }
}  // anonymous namespace
