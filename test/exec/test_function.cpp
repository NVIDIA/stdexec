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

namespace ex = STDEXEC;

namespace
{

  TEST_CASE("exec::function is constructible", "[types][function]")
  {
    exec::function<void()> voidSndr([]() noexcept { return ex::just(); });

    exec::function<int()> intSndr([]() noexcept { return ex::just(42); });

    double                             d = 4.;
    exec::function<void(int, double&)> binarySndr(5,
                                                  d,
                                                  [](int, double&) noexcept { return ex::just(); });

    exec::function<void() noexcept> nothrowSndr([]() noexcept { return ex::just(); });
    exec::function<int() noexcept>  nothrowIntSndr([]() noexcept { return ex::just(42); });

    STATIC_REQUIRE(STDEXEC::sender<decltype(voidSndr)>);
    STATIC_REQUIRE(STDEXEC::sender<decltype(intSndr)>);
    STATIC_REQUIRE(STDEXEC::sender<decltype(binarySndr)>);
    STATIC_REQUIRE(STDEXEC::sender<decltype(nothrowSndr)>);
    STATIC_REQUIRE(STDEXEC::sender<decltype(nothrowIntSndr)>);
  }

  TEST_CASE("exec::function is connectable", "[types][function]")
  {
    exec::function<int()> sndr([]() noexcept { return ex::just(42); });

    auto [fortytwo] = ex::sync_wait(std::move(sndr)).value();

    REQUIRE(fortytwo == 42);
  }
}  // namespace
