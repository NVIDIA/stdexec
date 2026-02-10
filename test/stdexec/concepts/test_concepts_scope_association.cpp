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

  // a "null" association that is always truthy and for which try_associate() always succeeds
  struct null_association {
    // this need not be explicit, although it should be
    constexpr operator bool() const noexcept {
      return true;
    }

    // this may throw, although it need not
    constexpr null_association try_associate() const noexcept {
      return {};
    }
  };

  // a CRTP base that lets us produce variations on null_associaton with the right return type on try_associate
  template <class Derived>
  struct crtp_association : null_association {
    constexpr Derived try_associate() const noexcept {
      return {};
    }
  };

  struct throwing_specials : crtp_association<throwing_specials> {
    // it's ok for the non-move operators to throw
    throwing_specials() noexcept(false) {
      // nvcc doesn't respect the noexcept(false) with = default
    }
    throwing_specials(const throwing_specials&) noexcept(false) {
      // nvcc doesn't respect the noexcept(false) with = default
    }
    throwing_specials(throwing_specials&&) noexcept = default;
    ~throwing_specials() = default;

    throwing_specials& operator=(const throwing_specials&) noexcept(false) {
      // nvcc doesn't respect the noexcept(false) with = default
      return *this;
    }
    throwing_specials& operator=(throwing_specials&&) noexcept = default;
  };

  struct move_only : crtp_association<move_only> {
    // copy operations are not required
    move_only() = default;
    move_only(move_only&&) = default;
    ~move_only() = default;

    move_only& operator=(move_only&&) = default;
  };

  struct explicit_bool : crtp_association<explicit_bool> {
    // the bool conversion may be explicit
    constexpr explicit operator bool() const noexcept {
      return true;
    }
  };

  struct throwing_reassociate : crtp_association<throwing_reassociate> {
    // try_associate may throw
    constexpr throwing_reassociate try_associate() const noexcept(false) {
      return {};
    }
  };

  TEST_CASE(
    "Scope association concept accepts basic association types",
    "[concepts][scope_association]") {
    // scope_association should accept the basic null_association
    STATIC_REQUIRE(ex::scope_association<null_association>);

    // the default constructor and copy operations may throw
    STATIC_REQUIRE(ex::scope_association<throwing_specials>);
    // double check that we're testing what we think we are
    STATIC_REQUIRE(!ex::__nothrow_constructible_from<throwing_specials>);
    STATIC_REQUIRE(!ex::__nothrow_constructible_from<throwing_specials, const throwing_specials&>);
    STATIC_REQUIRE(!ex::__nothrow_assignable_from<throwing_specials, const throwing_specials&>);

    // copy operations are not required
    STATIC_REQUIRE(ex::scope_association<move_only>);

    // the bool conversion may be explicit
    STATIC_REQUIRE(ex::scope_association<explicit_bool>);

    // try_associate may throw
    STATIC_REQUIRE(ex::scope_association<throwing_reassociate>);
  }

  // invalid association because of immovability
  struct immovable : crtp_association<immovable> {
    immovable(immovable&&) = delete;
  };

  // conditionally-invalid association because of throwing move operations
  template <bool ThrowingCtor, bool ThrowingAssign>
  struct throwing_moves : crtp_association<throwing_moves<ThrowingCtor, ThrowingAssign>> {
    throwing_moves() = default;
    throwing_moves(throwing_moves&&) noexcept(ThrowingCtor) {
      // nvcc doesn't respect the noexcept(false) with = default
    }
    ~throwing_moves() = default;

    throwing_moves& operator=(throwing_moves&&) noexcept(ThrowingAssign) {
      // nvcc doesn't respect the noexcept(false) with = default
      return *this;
    }
  };

  // invalid assocation because of a throwing move constructor
  using throwing_move_ctor = throwing_moves<true, false>;
  // invalid assocation because of a throwing move assignment operator
  using throwing_move_assign = throwing_moves<false, true>;

  // invalid assocation because of a missing default constructor
  struct missing_ctor : crtp_association<missing_ctor> {
    missing_ctor() = delete;
  };

  // invalid assocation because of a throwing conversion to bool
  struct throwing_boolish : crtp_association<throwing_boolish> {
    constexpr explicit operator bool() const noexcept(false) {
      return true;
    }
  };

  // invalid assocation because try_associate returns the wrong type
  struct cannot_reassociate : null_association { };

  // invalid association because operator bool is non-const
  struct non_const_boolish : crtp_association<non_const_boolish> {
    constexpr explicit operator bool() noexcept {
      return true;
    }
  };

  // invalid association because try_associate is non-const
  struct non_const_try_associate : null_association {
    constexpr non_const_try_associate try_associate() noexcept {
      return {};
    };
  };

  TEST_CASE(
    "Scope association concept rejects non-association types",
    "[concepts][scope_association]") {
    STATIC_REQUIRE(!ex::scope_association<int>);

    // movability is required
    STATIC_REQUIRE(!ex::scope_association<immovable>);

    // the move operations must be non-throwing
    STATIC_REQUIRE(!ex::scope_association<throwing_move_ctor>);
    STATIC_REQUIRE(!ex::scope_association<throwing_move_assign>);

    // default initialization is required
    STATIC_REQUIRE(!ex::scope_association<missing_ctor>);

    // conversion to bool must not throw
    STATIC_REQUIRE(!ex::scope_association<throwing_boolish>);

    // try_associate must return an association
    STATIC_REQUIRE(!ex::scope_association<cannot_reassociate>);

    // operator bool must be const qualified
    STATIC_REQUIRE(!ex::scope_association<non_const_boolish>);

    // try_associate must be const qualified
    STATIC_REQUIRE(!ex::scope_association<non_const_try_associate>);
  }
} // namespace
