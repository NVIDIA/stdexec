/*
 * Copyright (c) 2025 Ian Petersen
 * Copyright (c) 2025 NVIDIA Corporation
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
#include <test_common/scope_tokens.hpp>
#include <test_common/scope_helpers.hpp>

#include <array>
#include <memory_resource>
#include <stdexcept>

namespace ex = STDEXEC;

namespace {
  TEST_CASE("Trivial spawns compile", "[consumers][spawn]") {
    ex::spawn(ex::just(), null_token{});
    ex::spawn(ex::just_stopped(), null_token{});
  }

  TEST_CASE("spawn doesn't leak", "[consumers][spawn]") {
    counting_resource rsc;
    std::pmr::polymorphic_allocator<> alloc(&rsc);

    REQUIRE(rsc.allocated() == 0);

    ex::spawn(
      ex::read_env(ex::get_allocator) | ex::then([&](auto&& envAlloc) noexcept {
        // check that the allocator provided to spawn is in our environment
        REQUIRE(alloc == envAlloc);
        // check that we actually allocated something to run this op
        REQUIRE(rsc.allocated() > 0);
      }),
      null_token{},
      ex::prop(ex::get_allocator, alloc));

    REQUIRE(rsc.allocated() == 0);
  }

  TEST_CASE("spawn reads an allocator from the sender's environment", "[consumers][spawn]") {
    counting_resource rsc;
    std::pmr::polymorphic_allocator<> alloc(&rsc);

    scope_with_alloc scope{alloc};

    REQUIRE(rsc.allocated() == 0);

    ex::spawn(
      ex::read_env(ex::get_allocator) | ex::then([&](auto&& envAlloc) noexcept {
        // we should've pulled the scope's allocator into our environment
        REQUIRE(alloc == envAlloc);

        // we should've allocated some memory for this operation
        REQUIRE(rsc.allocated() > 0);
      }),
      scope.get_token());

    REQUIRE(rsc.allocated() == 0);
  }

  TEST_CASE(
    "The allocator provided directly to spawn overrides the allocator in the sender's environment",
    "[consumers][spawn]") {

    counting_resource rsc1;

    std::array<std::byte, 256> buffer{};
    std::pmr::monotonic_buffer_resource bumpAlloc(buffer.data(), buffer.size());

    counting_resource rsc2(bumpAlloc);

    std::pmr::polymorphic_allocator<> alloc1(&rsc1);
    std::pmr::polymorphic_allocator<> alloc2(&rsc2);

    REQUIRE(alloc1 != alloc2);

    scope_with_alloc scope{alloc1};

    REQUIRE(rsc1.allocated() == 0);
    REQUIRE(rsc2.allocated() == 0);

    ex::spawn(
      ex::read_env(ex::get_allocator) | ex::then([&](auto& envAlloc) noexcept {
        // the allocator in the environment should be the one provided to spawn
        // as an explicit argument and not the one provided by the scope
        REQUIRE(alloc1 != envAlloc);
        REQUIRE(alloc2 == envAlloc);

        // we should have allocated some memory for the op from rsc2 but not from rsc
        REQUIRE(rsc1.allocated() == 0);
        REQUIRE(rsc2.allocated() > 0);
      }),
      scope.get_token(),
      ex::prop(ex::get_allocator, alloc2));

    REQUIRE(rsc1.allocated() == 0);
    REQUIRE(rsc2.allocated() == 0);
  }

  TEST_CASE("spawn tolerates throwing scope tokens", "[consumers][spawn]") {
    counting_resource rsc;
    std::pmr::polymorphic_allocator<std::byte> alloc(&rsc);

    struct throwing_token : null_token {
      const counting_resource* rsc;

      assoc try_associate() const {
        REQUIRE(rsc->allocated() > 0);
        throw std::runtime_error("nope");
      }
    };

    REQUIRE(rsc.allocated() == 0);

    bool threw = false;
    try {
      ex::spawn(ex::just(), throwing_token{{}, &rsc}, ex::prop(ex::get_allocator, alloc));
    } catch (const std::runtime_error& e) {
      threw = true;
      REQUIRE(std::string{"nope"} == e.what());
    }

    REQUIRE(threw);

    REQUIRE(rsc.allocated() == 0);
  }

  TEST_CASE("spawn tolerates expired scope tokens", "[consumers][spawn]") {
    struct expired_token : null_token { // inherit the wrap method template
      const counting_resource* rsc;
      bool* tried;

      struct assoc {
        constexpr explicit operator bool() const noexcept {
          return false;
        }

        constexpr assoc try_associate() const noexcept {
          return {};
        }
      };

      assoc try_associate() const {
        REQUIRE(rsc->allocated() > 0);
        *tried = true;
        return {};
      }
    };

    counting_resource rsc;
    std::pmr::polymorphic_allocator<std::byte> alloc(&rsc);

    REQUIRE(rsc.allocated() == 0);

    bool triedToAssociate = false;

    ex::spawn(
      ex::just(), expired_token{{}, &rsc, &triedToAssociate}, ex::prop(ex::get_allocator, alloc));

    REQUIRE(rsc.allocated() == 0);
    REQUIRE(triedToAssociate);
  }
} // namespace
