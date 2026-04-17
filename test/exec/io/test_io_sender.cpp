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

#include <exec/io/io_sender.hpp>

#include <catch2/catch_all.hpp>

#include <stdexec/execution.hpp>

namespace ex = STDEXEC;

namespace
{

  TEST_CASE("exec::io_sender is constructible", "[types][io_sender]")
  {
    exec::io_sender<void()> voidSndr([]() noexcept { return ex::just(); });

    exec::io_sender<int()> intSndr([]() noexcept { return ex::just(42); });

    double                              d = 4.;
    exec::io_sender<void(int, double&)> binarySndr(5,
                                                   d,
                                                   [](int, double&) noexcept
                                                   { return ex::just(); });

    STATIC_REQUIRE(STDEXEC::sender<decltype(voidSndr)>);
    STATIC_REQUIRE(STDEXEC::sender<decltype(intSndr)>);
    STATIC_REQUIRE(STDEXEC::sender<decltype(binarySndr)>);
  }

  TEST_CASE("exec::io_sender is connectable", "[types][io_sender]")
  {
    exec::io_sender<int()> sndr([]() noexcept { return ex::just(42); });

    auto [fortytwo] = ex::sync_wait(std::move(sndr)).value();

    REQUIRE(fortytwo == 42);
  }
}  // namespace
