/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Robert Leahy. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <exec/started_by.hpp>

#include <stdexec/execution.hpp>
#include <test_common/catch2.hpp>
#include <test_common/type_helpers.hpp>

#include <concepts>
#include <exception>
#include <type_traits>
#include <utility>

namespace
{
  namespace ex = ::STDEXEC;

  using value_signatures = ex::completion_signatures<ex::set_value_t(int)>;

  struct value_receiver
  {
    using receiver_concept = ex::receiver_tag;

    void set_value(int value) && noexcept
    {
      *result = value;
    }

    void set_error(std::exception_ptr) && noexcept
    {
      FAIL_CHECK("Unexpected error completion");
    }

    constexpr auto get_env() const noexcept -> ex::env<>
    {
      return {};
    }

    int* result;
  };

  struct completion_control
  {
    void* operation{};
    void (*complete)(void*, int) noexcept {};

    void set_value(int value) const noexcept
    {
      complete(operation, value);
    }
  };

  struct continuation
  {
    explicit continuation(int value) noexcept
      : value(value)
    {}

    continuation(continuation&&) = delete;

    virtual void complete(ex::set_value_t, int&&) noexcept = 0;

    int value;
  };

  struct multi_continuation
  {
    virtual void complete(ex::set_value_t, int&&) noexcept                = 0;
    virtual void complete(ex::set_error_t, std::exception_ptr&&) noexcept = 0;
  };

  struct throwing_base
  {
    explicit throwing_base(int) {}
  };

  struct incomplete_base
  {
    explicit incomplete_base(int) noexcept {}

    virtual void missing() noexcept = 0;
  };

  struct aggregate_base
  {
    int value;
  };

  struct large_receiver
  {
    using receiver_concept = ex::receiver_tag;

    void set_value() && noexcept {}

    void set_error(std::exception_ptr) && noexcept {}

    constexpr auto get_env() const noexcept -> ex::env<>
    {
      return {};
    }

    char padding[256]{};
  };

  struct receiver_dependent_noexcept
  {
    template <class _Operation>
    void operator()(_Operation&) noexcept(sizeof(_Operation) > 128)
    {}
  };

  TEST_CASE("started_by advertises its explicit completion signatures", "[started_by]")
  {
    auto sender = ::exec::started_by<value_signatures>([](auto&) noexcept {});

    STATIC_REQUIRE(ex::sender_in<decltype(sender)>);
    STATIC_REQUIRE_FALSE(ex::dependent_sender<decltype(sender)>);
    STATIC_REQUIRE(ex::sender_in<decltype(sender), ex::env<>>);
    STATIC_REQUIRE(
      set_equivalent<
        ex::completion_signatures_of_t<decltype(sender)>,
        ex::completion_signatures<ex::set_value_t(int), ex::set_error_t(std::exception_ptr)>>);
    STATIC_REQUIRE(set_equivalent<ex::completion_signatures_of_t<decltype(sender), ex::env<>>,
                                  value_signatures>);
  }

  TEST_CASE("started_by adds exception_ptr when initiation can throw", "[started_by]")
  {
    auto sender = ::exec::started_by<value_signatures>([](auto&) {});

    STATIC_REQUIRE(ex::sender_in<decltype(sender), ex::env<>>);
    STATIC_REQUIRE(
      set_equivalent<
        ex::completion_signatures_of_t<decltype(sender), ex::env<>>,
        ex::completion_signatures<ex::set_value_t(int), ex::set_error_t(std::exception_ptr)>>);
  }

  TEST_CASE("started_by initiates on start and completes asynchronously", "[started_by]")
  {
    completion_control control;
    int                result{};

    auto sender = ::exec::started_by<value_signatures>(
      [&]<class _Operation>(_Operation& operation) noexcept
      {
        control.operation = &operation;
        control.complete  = [](void* pointer, int value) noexcept
        {
          static_cast<_Operation*>(pointer)->complete(ex::set_value, std::move(value));
        };
      });

    auto operation = ex::connect(sender, value_receiver{&result});
    CHECK(control.operation == nullptr);
    ex::start(operation);
    REQUIRE(control.operation != nullptr);
    CHECK(result == 0);
    control.set_value(42);
    CHECK(result == 42);
  }

  TEST_CASE("started_by converts a thrown initiation into set_error", "[started_by]")
  {
    bool caught{};
    struct receiver
    {
      using receiver_concept = ex::receiver_tag;

      void set_value(int) && noexcept
      {
        FAIL_CHECK("Unexpected value completion");
      }

      void set_error(std::exception_ptr error) && noexcept
      {
        try
        {
          std::rethrow_exception(std::move(error));
        }
        catch (int value)
        {
          CHECK(value == 17);
          *caught = true;
        }
        catch (...)
        {
          FAIL_CHECK("Unexpected exception type");
        }
      }

      constexpr auto get_env() const noexcept -> ex::env<>
      {
        return {};
      }

      bool* caught;
    };

    auto sender    = ::exec::started_by<value_signatures>([](auto&) { throw 17; });
    auto operation = ex::connect(sender, receiver{&caught});
    ex::start(operation);
    CHECK(caught);
  }

  TEST_CASE("started_by constructs and publicly exposes its operation base", "[started_by]")
  {
    continuation* erased{};
    int           observed{};
    int           result{};

    auto sender = ::exec::started_by<value_signatures, continuation>(
      [&](auto& operation) noexcept
      {
        observed = operation.value;
        erased   = &operation;
      },
      23);

    auto operation = ex::connect(sender, value_receiver{&result});
    STATIC_REQUIRE(std::derived_from<decltype(operation), continuation>);
    STATIC_REQUIRE(ex::__nothrow_connectable<decltype(sender), value_receiver>);
    CHECK(operation.value == 23);
    ex::start(operation);
    CHECK(observed == 23);
    REQUIRE(erased != nullptr);
    erased->complete(ex::set_value, 29);
    CHECK(result == 29);
  }

  TEST_CASE("started_by checks base construction in the sender cvref", "[started_by]")
  {
    auto valid   = ::exec::started_by<value_signatures, throwing_base>([](auto&) noexcept {}, 1);
    auto invalid = ::exec::started_by<value_signatures, throwing_base>([](auto&) noexcept {},
                                                                       nullptr);

    STATIC_REQUIRE(ex::sender_in<decltype(valid), ex::env<>>);
    STATIC_REQUIRE_FALSE(ex::sender_in<decltype(invalid), ex::env<>>);
    STATIC_REQUIRE_FALSE(ex::__nothrow_connectable<decltype(valid), value_receiver>);
  }

  TEST_CASE("started_by checks construction and concreteness of an abstract base", "[started_by]")
  {
    auto invalid_args = ::exec::started_by<value_signatures, continuation>([](auto&) noexcept {},
                                                                           nullptr);
    auto incomplete   = ::exec::started_by<value_signatures, incomplete_base>([](auto&) noexcept {},
                                                                            1);

    STATIC_REQUIRE_FALSE(ex::sender_in<decltype(invalid_args), ex::env<>>);
    STATIC_REQUIRE_FALSE(ex::sender_in<decltype(incomplete), ex::env<>>);
  }

  TEST_CASE("started_by constructs an aggregate operation base", "[started_by]")
  {
    auto sender = ::exec::started_by<value_signatures, aggregate_base>([](auto&) noexcept {}, 31);
    auto operation = ex::connect(sender, value_receiver{});

    CHECK(operation.value == 31);
  }

  TEST_CASE("started_by implements every abstract completion overload", "[started_by]")
  {
    using signatures =
      ex::completion_signatures<ex::set_value_t(int), ex::set_error_t(std::exception_ptr)>;

    multi_continuation* erased{};
    int                 result{};
    auto sender = ::exec::started_by<signatures, multi_continuation>([&](auto& operation) noexcept
                                                                     { erased = &operation; });
    auto operation = ex::connect(sender, value_receiver{&result});

    ex::start(operation);
    REQUIRE(erased != nullptr);
    erased->complete(ex::set_value, 37);
    CHECK(result == 37);
  }

  TEST_CASE("started_by checks initiation against an archetype receiver", "[started_by]")
  {
    auto invalid = ::exec::started_by<value_signatures>([](int) noexcept {});
    STATIC_REQUIRE(ex::sender_in<decltype(invalid)>);
    STATIC_REQUIRE_FALSE(ex::sender_in<decltype(invalid), ex::env<>>);
  }

  TEST_CASE("started_by rejects receiver-dependent initiation throwingness", "[started_by]")
  {
    using signatures = ex::completion_signatures<ex::set_value_t()>;
    auto sender      = ::exec::started_by<signatures>(receiver_dependent_noexcept{});

    STATIC_REQUIRE(ex::sender_in<decltype(sender), ex::env<>>);
    STATIC_REQUIRE_FALSE(ex::sender_to<decltype(sender), large_receiver>);
  }
}  // namespace
