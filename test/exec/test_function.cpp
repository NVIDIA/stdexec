/*
 * Copyright (c) 2026 Ian Petersen
 * Copyright (c) 2026 NVIDIA Corporation
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

#include <exec/function.hpp>

#include <catch2/catch_all.hpp>

#include <stdexec/execution.hpp>

#include <memory>

namespace ex = STDEXEC;

namespace
{
  TEST_CASE("exec::function is constructible", "[types][function]")
  {
    SECTION("void()")
    {
      exec::function<void()> sndr([]() noexcept { return ex::just(); });
      STATIC_REQUIRE(STDEXEC::sender<decltype(sndr)>);
    }

    SECTION("int()")
    {
      exec::function<int()> sndr([]() noexcept { return ex::just(42); });
      STATIC_REQUIRE(STDEXEC::sender<decltype(sndr)>);
    }

    SECTION("void(int, double&)")
    {
      double                             d = 4.;
      exec::function<void(int, double&)> sndr(5,
                                              d,
                                              [](int, double&) noexcept { return ex::just(); });
      STATIC_REQUIRE(STDEXEC::sender<decltype(sndr)>);
    }

    SECTION("void() noexcept")
    {
      exec::function<void() noexcept> sndr([]() noexcept { return ex::just(); });
      STATIC_REQUIRE(STDEXEC::sender<decltype(sndr)>);
    }

    SECTION("int() noexcept")
    {
      exec::function<int() noexcept> sndr([]() noexcept { return ex::just(42); });
      STATIC_REQUIRE(STDEXEC::sender<decltype(sndr)>);
    }

    SECTION("sender_tag() with only set_value_t(int)")
    {
      exec::function<ex::sender_tag(), ex::completion_signatures<ex::set_value_t(int)>> sndr(
        []() noexcept { return ex::just(42); });
      STATIC_REQUIRE(STDEXEC::sender<decltype(sndr)>);
    }

    SECTION("sender_tag() with only set_stopped_t()")
    {
      exec::function<ex::sender_tag(), ex::completion_signatures<ex::set_stopped_t()>> sndr(
        []() noexcept { return ex::just_stopped(); });
      STATIC_REQUIRE(STDEXEC::sender<decltype(sndr)>);
    }

    SECTION("void() with trivial custom environment")
    {
      exec::function<void(), exec::queries<>> sndr([]() noexcept { return ex::just(); });
      STATIC_REQUIRE(STDEXEC::sender<decltype(sndr)>);
    }

    SECTION("sender_tag(int) with only set_value_t() and trivial environment")
    {
      exec::function<ex::sender_tag(int),
                     ex::completion_signatures<ex::set_value_t()>,
                     exec::queries<>>
        sndr(5, [](int) noexcept { return ex::just(); });
      STATIC_REQUIRE(STDEXEC::sender<decltype(sndr)>);
    }
  }

  TEST_CASE("exec::function is connectable", "[types][function]")
  {
    SECTION("int() noexcept from just(42)")
    {
      exec::function<int() noexcept> sndr([]() noexcept { return ex::just(42); });

      auto [fortytwo] = ex::sync_wait(std::move(sndr)).value();

      REQUIRE(fortytwo == 42);
    }

    SECTION("void() from throwing factory")
    {
      exec::function<void()> sndr([]() -> decltype(ex::just()) { throw "oops"; });

      REQUIRE_THROWS(ex::sync_wait(std::move(sndr)));
    }

    SECTION("void() from throwing then")
    {
      exec::function<void()> sndr([]() noexcept
                                  { return ex::just() | ex::then([] { throw "oops"; }); });

      REQUIRE_THROWS(ex::sync_wait(std::move(sndr)));
    }

    SECTION("void() from just_stopped()")
    {
      exec::function<void()> sndr([]() noexcept { return ex::just_stopped(); });

      auto ret = ex::sync_wait(std::move(sndr));

      REQUIRE_FALSE(ret.has_value());
    }

    SECTION("custom completions from just_error(42)")
    {
      exec::function<ex::sender_tag(),
                     ex::completion_signatures<ex::set_value_t(), ex::set_error_t(int)>>
        sndr([]() noexcept { return ex::just_error(42); });

      REQUIRE_THROWS_AS(ex::sync_wait(std::move(sndr)), int);
    }
  }

  TEST_CASE("exec::function forwards get_frame_allocator", "[types][function]")
  {
    // TODO: you probably shouldn't have to specify the frame allocator query like this
    using Queries = exec::queries<std::pmr::polymorphic_allocator<std::byte>(
      exec::get_frame_allocator_t) noexcept>;

    exec::function<bool() noexcept, Queries> sndr(
      []() noexcept
      {
        return ex::read_env(exec::get_frame_allocator)
             | ex::then(
                 [](auto alloc) noexcept
                 {
                   return std::same_as<std::pmr::polymorphic_allocator<std::byte>, decltype(alloc)>;
                 });
      });

    std::pmr::polymorphic_allocator<std::byte> alloc;

    auto [ret] = ex::sync_wait(std::move(sndr)
                               | ex::write_env(ex::prop(exec::get_frame_allocator, alloc)))
                   .value();

    REQUIRE(ret);
  }

  TEST_CASE("exec::function is conditionally lvalue connectable", "[types][function]")
  {
    exec::function<int()> sndr([]() noexcept { return ex::just(42); });

    auto [ret] = ex::sync_wait(sndr).value();

    REQUIRE(ret == 42);
  }

  TEST_CASE("exec::function accepts lvalue callables", "[types][function]")
  {
    exec::function<int(int) noexcept> sndr(42, ex::just);

    auto [ret] = ex::sync_wait(sndr).value();

    REQUIRE(ret == 42);
  }
}  // namespace
