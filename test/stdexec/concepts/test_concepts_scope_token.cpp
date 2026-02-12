/*
 * Copyright (c) 2022 Ian Petersen
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

namespace ex = STDEXEC;

namespace {

  TEST_CASE("Scope token helpers are correctly defined", "[concepts][scope_token]") {
    // check the test-sender and test-env definitions are appropriate
    STATIC_REQUIRE(ex::sender<ex::__scope_concepts::__test_sender>);
    STATIC_REQUIRE(ex::sender_in<ex::__scope_concepts::__test_sender, ex::env<>>);
    STATIC_REQUIRE(ex::operation_state<ex::__scope_concepts::__test_sender::__op>);
  }

  // a "null" token that can always create new associations
  struct null_token {
    // the always-truthy association type
    struct assoc {
      // this need not be explicit, although it should be
      constexpr operator bool() const noexcept {
        return true;
      }

      // this may throw, although it need not
      constexpr assoc try_associate() const noexcept {
        return {};
      }
    };

    constexpr assoc try_associate() const noexcept {
      return {};
    }

    template <ex::sender Sender>
    Sender&& wrap(Sender&& snd) const noexcept {
      return std::forward<Sender>(snd);
    }
  };

  struct throwing_try_associate : null_token {
    constexpr assoc try_associate() const noexcept(false) {
      return {};
    }
  };

  struct throwing_wrap : null_token {
    template <ex::sender Sender>
    constexpr Sender&& wrap(Sender&& snd) const noexcept(false) {
      return std::forward<Sender>(snd);
    }
  };

  struct wrapping_wrap : null_token {
    template <ex::sender Sender>
    struct wrapper : Sender {
      wrapper(const Sender& snd)
        : Sender(snd) {
      }

      wrapper(Sender&& snd)
        : Sender(std::move(snd)) {
      }
    };

    template <ex::sender Sender>
    auto wrap(Sender&& snd) const noexcept {
      return wrapper{std::forward<Sender>(snd)};
    }
  };

  TEST_CASE("Scope token concept accepts basic token types", "[concepts][scope_token]") {
    // scope_token should accept the basic null_token
    STATIC_REQUIRE(ex::scope_token<null_token>);

    // it's ok for try_associate to throw
    STATIC_REQUIRE(ex::scope_token<throwing_try_associate>);

    // it's ok for wrap to throw
    STATIC_REQUIRE(ex::scope_token<throwing_wrap>);

    // it's ok for wrap to change the type of its argument
    STATIC_REQUIRE(ex::scope_token<wrapping_wrap>);
  }

  struct move_only : null_token {
    move_only() = default;
    move_only(move_only&&) = default;
    ~move_only() = default;

    move_only& operator=(move_only&&) = default;
  };

  struct non_const_try_associate : null_token {
    assoc try_associate() noexcept {
      return {};
    }
  };

  struct non_const_wrap : null_token {
    template <ex::sender Sender>
    Sender&& wrap(Sender&& snd) noexcept {
      return std::forward<Sender>(snd);
    }
  };

  TEST_CASE("Scope token concept rejects non-token types", "[concepts][scope_token]") {
    STATIC_REQUIRE(!ex::scope_token<int>);

    // tokens must be copyable
    STATIC_REQUIRE(!ex::scope_token<move_only>);

    // try_associate must be const-qualified
    STATIC_REQUIRE(!ex::scope_token<non_const_try_associate>);

    // wrap must be const-qualified
    STATIC_REQUIRE(!ex::scope_token<non_const_wrap>);
  }
} // namespace
