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
#include <test_common/receivers.hpp>
#include <test_common/scope_tokens.hpp>

#include <concepts>
#include <memory>

namespace ex = STDEXEC;

namespace {
  TEST_CASE("associate returns a sender", "[adaptors][associate]") {
    using snd_t = decltype(ex::associate(ex::just(), null_token{}));

    STATIC_REQUIRE(ex::sender<snd_t>);
    STATIC_REQUIRE(ex::sender<snd_t&>);
    STATIC_REQUIRE(ex::sender<const snd_t&>);
  }

  TEST_CASE("associate is appropriately noexcept", "[adaptors][associate]") {
    // double-check our dependencies
    STATIC_REQUIRE(noexcept(ex::just()));
    STATIC_REQUIRE(noexcept(null_token{}));

    // null_token is no-throw default constructible and tokens must be no-throw
    // copyable and movable so this whole thing had better be no-throw
    STATIC_REQUIRE(noexcept(ex::associate(null_token{})));

    // constructing and passing in a no-throw sender should let the whole
    // expression be no-throw
    STATIC_REQUIRE(noexcept(ex::associate(ex::just(), null_token{})));
    STATIC_REQUIRE(noexcept(ex::just() | ex::associate(null_token{})));

    // conversely, trafficking in senders with potentially-throwing copy
    // constructors should lead to the whole expression becoming potentially-throwing
    const auto justString = ex::just(std::string{"Copying strings is potentially-throwing"});
    STATIC_REQUIRE(!noexcept(ex::associate(justString, null_token{})));
    STATIC_REQUIRE(!noexcept(justString | ex::associate(null_token{})));
    (void) justString;
  }

  template <class Sender, class... CompSig>
  constexpr bool expected_completion_signatures() {
    using expected_sigs = ex::completion_signatures<CompSig...>;
    using actual_sigs = ex::completion_signatures_of_t<Sender>;
    return expected_sigs{} == actual_sigs{};
  }

  TEST_CASE("associate has appropriate completion signatures", "[adaptors][associate]") {
    {
      using snd_t = decltype(ex::associate(ex::just(), null_token{}));

      STATIC_REQUIRE(
        expected_completion_signatures<snd_t, ex::set_value_t(), ex::set_stopped_t()>());

      STATIC_REQUIRE(
        expected_completion_signatures<snd_t&, ex::set_value_t(), ex::set_stopped_t()>());

      STATIC_REQUIRE(
        expected_completion_signatures<const snd_t&, ex::set_value_t(), ex::set_stopped_t()>());
    }

    {
      using snd_t = decltype(ex::associate(ex::just(std::string{}), null_token{}));

      STATIC_REQUIRE(
        expected_completion_signatures<snd_t, ex::set_value_t(std::string), ex::set_stopped_t()>());

      STATIC_REQUIRE(
        expected_completion_signatures<snd_t&, ex::set_value_t(std::string), ex::set_stopped_t()>());

      STATIC_REQUIRE(
        expected_completion_signatures<
          const snd_t&,
          ex::set_value_t(std::string),
          ex::set_stopped_t()
        >());
    }

    {
      using snd_t = decltype(ex::associate(ex::just_stopped(), null_token{}));

      STATIC_REQUIRE(expected_completion_signatures<snd_t, ex::set_stopped_t()>());

      STATIC_REQUIRE(expected_completion_signatures<snd_t&, ex::set_stopped_t()>());

      STATIC_REQUIRE(expected_completion_signatures<const snd_t&, ex::set_stopped_t()>());
    }

    {
      using snd_t = decltype(ex::associate(ex::just_error(5), null_token{}));

      STATIC_REQUIRE(
        expected_completion_signatures<snd_t, ex::set_error_t(int), ex::set_stopped_t()>());

      STATIC_REQUIRE(
        expected_completion_signatures<snd_t&, ex::set_error_t(int), ex::set_stopped_t()>());

      STATIC_REQUIRE(
        expected_completion_signatures<const snd_t&, ex::set_error_t(int), ex::set_stopped_t()>());
    }

    {
      int i = 42;
      using snd_t = decltype(ex::associate(ex::just(std::ref(i)), null_token{}));

      STATIC_REQUIRE(
        expected_completion_signatures<
          snd_t,
          ex::set_value_t(std::reference_wrapper<int>),
          ex::set_stopped_t()
        >());

      STATIC_REQUIRE(
        expected_completion_signatures<
          snd_t&,
          ex::set_value_t(std::reference_wrapper<int>),
          ex::set_stopped_t()
        >());

      STATIC_REQUIRE(
        expected_completion_signatures<
          const snd_t&,
          ex::set_value_t(std::reference_wrapper<int>),
          ex::set_stopped_t()
        >());
    }
  }

  TEST_CASE("associate is the identity with null_token", "[adaptors][associate]") {
    auto checkForIdentity = []<class... V>(ex::sender auto&& snd, V... values) {
      // wait_for_values wants prvalue expected values
      wait_for_value(snd, V(values)...);
      wait_for_value(std::as_const(snd), V(values)...);
      wait_for_value(std::move(snd), V(values)...);
    };

    // nullary set_value
    checkForIdentity(ex::just() | ex::associate(null_token{}));

    // unary set_value
    checkForIdentity(ex::just(42) | ex::associate(null_token{}), 42);

    // binary set_value
    checkForIdentity(ex::just(42, 67) | ex::associate(null_token{}), 42, 67);

    // set_value of a reference
    int i = 42;
    checkForIdentity(ex::just(std::ref(i)) | ex::associate(null_token{}), std::ref(i));

    // passing set_value(int) through to another adaptor
    checkForIdentity(
      ex::just(42) | ex::associate(null_token{}) | ex::then([](int i) noexcept { return i; }), 42);

    // passing set_error(int) through to another adaptor
    checkForIdentity(
      ex::just_error(42) | ex::associate(null_token{}) | ex::upon_error([](int i) { return i; }),
      42);

    // passing set_stopped() through to another adaptor
    checkForIdentity(
      ex::just_stopped() | ex::associate(null_token{})
        | ex::upon_stopped([]() noexcept { return 42; }),
      42);
  }

  struct expired_token {
    struct assoc {
      constexpr operator bool() const noexcept {
        return false;
      }

      constexpr assoc try_associate() const noexcept {
        return {};
      }
    };

    template <ex::sender Sender>
    constexpr Sender&& wrap(Sender&& sndr) const noexcept {
      return std::forward<Sender>(sndr);
    }

    constexpr assoc try_associate() const noexcept {
      return {};
    }
  };

  TEST_CASE("associate is just_stopped with expired_token", "[adaptors][associate]") {
    wait_for_value(
      ex::just(true) | ex::associate(expired_token{})
        | ex::upon_stopped([]() noexcept { return false; }),
      false);
  }

  struct scope {
    bool open{true};

    struct assoc {
      constexpr operator bool() const noexcept {
        return !!scope_;
      }

      constexpr assoc try_associate() const noexcept {
        return assoc{scope_ && scope_->open ? scope_ : nullptr};
      }

      const scope* scope_;
    };

    struct token {
      template <ex::sender Sender>
      constexpr Sender&& wrap(Sender&& sndr) const noexcept {
        return std::forward<Sender>(sndr);
      }

      constexpr assoc try_associate() const noexcept {
        return assoc{scope_->open ? scope_ : nullptr};
      }

      const scope* scope_;
    };

    constexpr token get_token() const noexcept {
      return token{this};
    }
  };

  TEST_CASE(
    "copying an associate-sender re-queries for a new association",
    "[adaptors][associate]") {
    STATIC_REQUIRE(ex::scope_token<scope::token>);
    STATIC_REQUIRE(ex::scope_association<scope::assoc>);

    scope s;

    auto snd = ex::associate(ex::just(42), s.get_token());

    // expect this copy of snd to complete with a value because the scope is still open
    wait_for_value(snd | ex::upon_stopped([]() noexcept { return 67; }), 42);

    // close the scope
    s.open = false;

    // now expect the copy to complete with stopped because we closed the scope
    wait_for_value(snd | ex::upon_stopped([]() noexcept { return 67; }), 67);

    // the original should complete with a value even though it's started after closing the scope
    wait_for_value(std::move(snd), 42);
  }

  TEST_CASE(
    "the sender argument is eagerly destroyed when try_associate fails",
    "[adaptors][associate]") {
    bool deleted = false;
    using deleter_t = decltype([](bool* p) noexcept { *p = true; });
    std::unique_ptr<bool, deleter_t> ptr(&deleted);

    auto snd = ex::just(std::move(ptr)) | ex::associate(expired_token{});

    REQUIRE(deleted == true);

    STATIC_REQUIRE(!std::copy_constructible<decltype(snd)>);
    (void) snd;
  }

  // TODO: check the pass-through nature of __sync_attrs
  // TODO: check the pass-through stop request behaviour
  // TODO: confirm timing of destruction of opstate relative to release of association
  // TODO: confirm that the TODO list is exhaustive
} // namespace
