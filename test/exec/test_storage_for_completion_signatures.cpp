/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *                         Copyright (c) 2025 Robert Leahy. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <exec/storage_for_completion_signatures.hpp>

#include <catch2/catch_all.hpp>
#include <stdexec/execution.hpp>

#include <cstddef>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>

#include "../test_common/receivers.hpp"
#include "../test_common/type_helpers.hpp"

using namespace exec;

namespace {

template<typename Storage, typename Value>
concept can_arrive_set_value_with =
  requires(Storage& s, Value&& value) {
    s.arrive(::STDEXEC::set_value, static_cast<Value&&>(value));
  };

struct copyable_value {
  copyable_value() = default;
  copyable_value(const copyable_value&) noexcept = default;
  copyable_value(copyable_value&&) noexcept = default;
};

using copyable_value_storage = storage_for_completion_signatures<
  ::STDEXEC::completion_signatures<
    ::STDEXEC::set_value_t(copyable_value)>>;

static_assert(!can_arrive_set_value_with<copyable_value_storage, copyable_value&>);
static_assert(can_arrive_set_value_with<copyable_value_storage, copyable_value>);

TEST_CASE("Storing no completion signatures works", "[storage_for_completion_signatures]") {
  storage_for_completion_signatures<
    ::STDEXEC::completion_signatures<>> storage;
  static_assert(
    std::is_same_v<
      decltype(storage)::completion_signatures,
      ::STDEXEC::completion_signatures<>>);
  CHECK(!std::move(storage).visit([&](auto&&...) {
    FAIL("Unexpected invocation of visitor");
  }));
  struct receiver {
    using receiver_concept = ::STDEXEC::receiver_t;
    explicit receiver(bool& moved) noexcept
      : moved_(&moved) {
    }
    receiver(receiver&& other) noexcept
      : moved_(std::exchange(other.moved_, nullptr)) {
      *moved_ = true;
    }
    void set_value() noexcept {
      FAIL("Unexpected value");
    }
    bool* moved_;
  };
  bool moved = false;
  CHECK(!std::move(storage).complete(receiver{moved}));
  CHECK(!moved);
}

TEST_CASE("Storing simple completion signatures and then visiting them works", "[storage_for_completion_signatures]") {
  using completion_signatures = ::STDEXEC::completion_signatures<
      ::STDEXEC::set_value_t(int),
      ::STDEXEC::set_stopped_t(),
      ::STDEXEC::set_error_t(std::error_code)>;
  using storage = storage_for_completion_signatures<
    completion_signatures>;
  static_assert(
    set_equivalent<
      completion_signatures,
      storage::completion_signatures>);
  CHECK(!storage{}.visit([&](auto&&...) {
    FAIL("Unexpected invocation of visitor");
  }));
  struct base {
    void operator()(::STDEXEC::set_stopped_t&&) && {
      FAIL("Unexpected stop");
    }
    void operator()(::STDEXEC::set_value_t&&, int&&) && {
      FAIL("Unexpected value");
    }
    void operator()(::STDEXEC::set_error_t&&, std::error_code&&) && {
      FAIL("Unexpected error");
    }
    bool invoked{false};
  protected:
    void invoke_() {
      CHECK(!invoked);
      invoked = true;
    }
  };
  {
    storage s;
    static_assert(
      noexcept(
        s.arrive(::STDEXEC::set_stopped)));
    s.arrive(::STDEXEC::set_stopped);
    static_assert(
      noexcept(
        std::move(s).visit([](auto&&...) noexcept {})));
    struct : base {
      using base::operator();
      void operator()(::STDEXEC::set_stopped_t&&) && {
        invoke_();
      }
    } visitor;
    static_assert(!noexcept(std::move(s).visit(visitor)));
    CHECK(std::move(s).visit(std::move(visitor)));
    CHECK(visitor.invoked);
  }
  {
    storage s;
    static_assert(
      noexcept(
        s.arrive(::STDEXEC::set_value, 5)));
    s.arrive(::STDEXEC::set_value, 5);
    struct : base {
      using base::operator();
      void operator()(::STDEXEC::set_value_t&&, int&& i) && {
        invoke_();
        CHECK(i == 5);
      }
    } visitor;
    static_assert(!noexcept(std::move(s).visit(visitor)));
    CHECK(std::move(s).visit(std::move(visitor)));
    CHECK(visitor.invoked);
  }
  {
    storage s;
    static_assert(
      noexcept(
        s.arrive(::STDEXEC::set_error, std::error_code{})));
    s.arrive(
      ::STDEXEC::set_error,
      make_error_code(std::errc::no_such_file_or_directory));
    struct : base {
      using base::operator();
      void operator()(::STDEXEC::set_error_t&&, std::error_code&& ec) && {
        invoke_();
        CHECK(ec == make_error_code(std::errc::no_such_file_or_directory));
      }
    } visitor;
    static_assert(!noexcept(std::move(s).visit(visitor)));
    CHECK(std::move(s).visit(std::move(visitor)));
    CHECK(visitor.invoked);
  }
}

TEST_CASE("Storing simple completion signatures and then completing a receiver therewith works", "[storage_for_completion_signatures]") {
  using storage_type =
    storage_for_completion_signatures<
      ::STDEXEC::completion_signatures<
        ::STDEXEC::set_value_t(),
        ::STDEXEC::set_stopped_t(),
        ::STDEXEC::set_error_t(std::exception_ptr)>>;
  {
    storage_type storage;
    storage.arrive(::STDEXEC::set_value);
    CHECK(std::move(storage).complete(expect_void_receiver{}));
  }
  {
    std::optional<storage_type> storage(std::in_place);
    storage->arrive(
      ::STDEXEC::set_error,
      std::make_exception_ptr(std::logic_error("TEST")));
    std::exception_ptr ex;
    struct receiver {
      using receiver_concept = ::STDEXEC::receiver_t;
      void set_value() noexcept {
        FAIL("Unexpected value invocation");
      }
      void set_stopped() noexcept {
        FAIL("Unexpected stopped invocation");
      }
      void set_error(std::exception_ptr&& ex) noexcept {
        //  This ensures that the exception_ptr is moved onto the stack
        CHECK(storage_);
        storage_.reset();
        CHECK(!ex_);
        ex_ = std::move(ex);
      }
      std::optional<storage_type>& storage_;
      std::exception_ptr& ex_;
    };
    CHECK(std::move(*storage).complete(receiver{storage, ex}));
    REQUIRE(ex);
    bool threw = false;
    try {
      std::rethrow_exception(std::move(ex));
    } catch (const std::logic_error& ex) {
      threw = true;
      CHECK(ex.what() == std::string_view("TEST"));
    }
    CHECK(threw);
  }
}

TEST_CASE("When moving a stored value onto the stack throws complete reports std::exception_ptr", "[storage_for_completion_signatures]") {
  struct throws_on_second_move {
    explicit throws_on_second_move(int& moves) noexcept
      : moves_(&moves) {
    }
    throws_on_second_move(throws_on_second_move&& other) {
      moves_ = other.moves_;
      ++*moves_;
      if (*moves_ == 2) {
        throw std::runtime_error("Throwing as requested");
      }
    }
    int* moves_;
  };

  using storage = storage_for_completion_signatures<
    ::STDEXEC::completion_signatures<
      ::STDEXEC::set_value_t(throws_on_second_move)>>;

  struct receiver {
    using receiver_concept = ::STDEXEC::receiver_t;
    void set_value(throws_on_second_move&&) noexcept {
      FAIL("Unexpected value invocation");
    }
    void set_error(std::exception_ptr&& ex) noexcept {
      CHECK(!ex_);
      ex_ = std::move(ex);
    }
    std::exception_ptr& ex_;
  };

  int moves = 0;
  storage s;
  s.arrive(::STDEXEC::set_value, throws_on_second_move{moves});
  CHECK(moves == 1);

  std::exception_ptr ex;
  CHECK(std::move(s).complete(receiver{ex}));
  REQUIRE(ex);
  bool threw = false;
  try {
    std::rethrow_exception(std::move(ex));
  } catch (const std::runtime_error& ex) {
    threw = true;
    CHECK(ex.what() == std::string_view("Throwing as requested"));
  }
  CHECK(threw);
}

TEST_CASE("When storing a completion signature would throw it is simply coalesced to std::exception_ptr", "[storage_for_completion_signatures]") {
  struct maybe_throws_on_move {
    maybe_throws_on_move() = default;
    maybe_throws_on_move(maybe_throws_on_move&& other) {
      if (other.throws) {
        throw std::runtime_error("Throwing as requested");
      }
    }
    bool throws{false};
  };
  {
    using storage = storage_for_completion_signatures<
      ::STDEXEC::completion_signatures<
        ::STDEXEC::set_value_t(maybe_throws_on_move)>>;
    static_assert(storage::nothrow_arrive);
    static_assert(
      set_equivalent<
        ::STDEXEC::completion_signatures<
          ::STDEXEC::set_value_t(maybe_throws_on_move),
          ::STDEXEC::set_error_t(std::exception_ptr)>,
        storage::completion_signatures>);
    struct base {
      void operator()(::STDEXEC::set_value_t&&, maybe_throws_on_move&&) & {
        FAIL("Unexpected value invocation");
      }
      void operator()(::STDEXEC::set_error_t&&, std::exception_ptr&&) & {
        FAIL("Unexpected error invocation");
      }
      bool invoked{false};
    protected:
      void invoke_() {
        CHECK(!invoked);
        invoked = true;
      }
    };
    {
      storage s;
      static_assert(
        noexcept(
          s.arrive(::STDEXEC::set_value, maybe_throws_on_move{})));
      maybe_throws_on_move obj;
      obj.throws = true;
      s.arrive(::STDEXEC::set_value, std::move(obj));
      struct : base {
        using base::operator();
        void operator()(::STDEXEC::set_error_t&&, std::exception_ptr&& ex) & {
          invoke_();
          REQUIRE(ex);
          //  TODO?
        }
      } visitor;
      CHECK(std::move(s).visit(visitor));
      CHECK(visitor.invoked);
    }
    {
      storage s;
      s.arrive(::STDEXEC::set_value, maybe_throws_on_move{});
      struct : base {
        using base::operator();
        void operator()(::STDEXEC::set_value_t&&, maybe_throws_on_move&&) & {
          invoke_();
        }
      } visitor;
      CHECK(std::move(s).visit(visitor));
      CHECK(visitor.invoked);
    }
  }
  //  Important that the below cases don't add the std::exception_ptr completion
  //  since propagating a reference can't throw
  {
    using signatures = ::STDEXEC::completion_signatures<
      ::STDEXEC::set_value_t(maybe_throws_on_move&)>;
    using storage = storage_for_completion_signatures<signatures>;
    static_assert(
      std::is_same_v<
        storage::completion_signatures,
        signatures>);
    maybe_throws_on_move obj;
    storage s;
    s.arrive(::STDEXEC::set_value, obj);
    bool invoked = false;
    CHECK(std::move(s).visit([&](::STDEXEC::set_value_t, maybe_throws_on_move& stored) {
      CHECK(!invoked);
      invoked = true;
      CHECK(&obj == &stored);
    }));
    CHECK(invoked);
  }
  {
    using signatures = ::STDEXEC::completion_signatures<
      ::STDEXEC::set_value_t(maybe_throws_on_move&&)>;
    using storage = storage_for_completion_signatures<signatures>;
    static_assert(
      std::is_same_v<
        storage::completion_signatures,
        signatures>);
    maybe_throws_on_move obj;
    storage s;
    s.arrive(::STDEXEC::set_value, std::move(obj));
    bool invoked = false;
    CHECK(std::move(s).visit([&](::STDEXEC::set_value_t, maybe_throws_on_move&& stored) {
      CHECK(!invoked);
      invoked = true;
      CHECK(&obj == &stored);
    }));
    CHECK(invoked);
  }
}

TEST_CASE("Storage can propagate persistence exceptions instead of coalescing them", "[storage_for_completion_signatures]") {
  struct maybe_throws_on_move {
    maybe_throws_on_move() = default;
    maybe_throws_on_move(maybe_throws_on_move&& other) {
      if (other.throws) {
        throw std::runtime_error("Throwing as requested");
      }
      other.moved = true;
    }
    bool throws{false};
    bool moved{false};
  };

  using signatures = ::STDEXEC::completion_signatures<
    ::STDEXEC::set_value_t(maybe_throws_on_move)>;
  using storage = storage_for_completion_signatures<
    signatures,
    storage_for_completion_signatures_error_policy::__propagate>;

  static_assert(!storage::nothrow_arrive);
  static_assert(std::is_same_v<storage::completion_signatures, signatures>);
  static_assert(!noexcept(
    std::declval<storage&>().arrive(::STDEXEC::set_value, std::declval<maybe_throws_on_move>())));

  storage s;
  maybe_throws_on_move obj;
  obj.throws = true;
  CHECK_THROWS_AS(s.arrive(::STDEXEC::set_value, std::move(obj)), std::runtime_error);
  CHECK(!obj.moved);
  CHECK(!std::move(s).visit([](auto&&...) {
    FAIL("Unexpected invocation of visitor");
  }));
}

TEST_CASE("Propagating storage remains noexcept when persistence is noexcept", "[storage_for_completion_signatures]") {
  using signatures = ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(int)>;
  using storage = storage_for_completion_signatures<
    signatures,
    storage_for_completion_signatures_error_policy::__propagate>;

  static_assert(storage::nothrow_arrive);
  static_assert(std::is_same_v<storage::completion_signatures, signatures>);
  storage s;
  static_assert(noexcept(s.arrive(::STDEXEC::set_value, 42)));
  s.arrive(::STDEXEC::set_value, 42);

  bool invoked = false;
  CHECK(std::move(s).visit([&](::STDEXEC::set_value_t, int value) noexcept {
    CHECK(!invoked);
    invoked = true;
    CHECK(value == 42);
  }));
  CHECK(invoked);
}

} // unnamed namespace
