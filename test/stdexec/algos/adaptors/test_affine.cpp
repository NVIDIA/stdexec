/*
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
#include <catch2/catch_all.hpp>

#include <stdexec/execution.hpp>

#include <test_common/receivers.hpp>

namespace ex = STDEXEC;

namespace
{
  // [exec.adapt.obj]/4: a pipeable sender adaptor object that accepts only one
  // argument is a pipeable sender adaptor closure object.
  TEST_CASE("affine is a sender adaptor closure object", "[adaptors][affine]")
  {
    STATIC_REQUIRE(ex::__sender_adaptor_closure<ex::affine_t>);
    STATIC_REQUIRE(std::same_as<decltype(ex::affine()), ex::affine_t>);

    ex::sender auto snd = ex::just(42) | ex::affine;
    wait_for_value(std::move(snd), 42);
  }

  TEST_CASE("affine can still be called directly and piped with parentheses", "[adaptors][affine]")
  {
    wait_for_value(ex::affine(ex::just(42)), 42);
    wait_for_value(ex::just(42) | ex::affine(), 42);
  }

  TEST_CASE("affine composes with other closures without parentheses", "[adaptors][affine]")
  {
    auto            clsur = ex::then([](int i) { return i + 1; }) | ex::affine;
    ex::sender auto snd   = ex::just(41) | clsur;
    wait_for_value(std::move(snd), 42);
  }
}  // namespace
