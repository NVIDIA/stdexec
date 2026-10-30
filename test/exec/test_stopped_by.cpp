/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Robert Leahy. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <exec/stopped_by.hpp>

#include <stdexec/execution.hpp>
#include <test_common/catch2.hpp>
#include <test_common/type_helpers.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace
{
  namespace ex = ::STDEXEC;

  using stop_env = ex::prop<ex::get_stop_token_t, ex::inplace_stop_token>;

  struct main_control
  {
    void* operation{};
    void (*set_value)(void*, int) noexcept {};
    bool started{};

    void complete(int value) const noexcept
    {
      set_value(operation, value);
    }
  };

  struct main_sender
  {
    using sender_concept = ex::sender_tag;

    template <class...>
    static consteval auto
    get_completion_signatures() noexcept -> ex::completion_signatures<ex::set_value_t(int)>
    {
      return {};
    }

    template <class _Receiver>
    struct operation
    {
      using operation_state_concept = ex::operation_state_tag;

      void start() & noexcept
      {
        control->started   = true;
        control->operation = this;
        control->set_value = [](void* pointer, int value) noexcept
        {
          auto& self = *static_cast<operation*>(pointer);
          ex::set_value(static_cast<_Receiver&&>(self.receiver), std::move(value));
        };
      }

      main_control* control;
      _Receiver     receiver;
    };

    template <class _Receiver>
    auto connect(_Receiver receiver) const noexcept -> operation<_Receiver>
    {
      return {control, static_cast<_Receiver&&>(receiver)};
    }

    main_control* control;
  };

  struct stop_control
  {
    void* operation{};
    void (*set_value)(void*) noexcept {};
    bool started{};
    bool destroyed{};

    void complete() const noexcept
    {
      set_value(operation);
    }
  };

  struct stop_sender
  {
    using sender_concept = ex::sender_tag;

    template <class...>
    static consteval auto
    get_completion_signatures() noexcept -> ex::completion_signatures<ex::set_value_t()>
    {
      return {};
    }

    template <class _Receiver>
    struct operation
    {
      using operation_state_concept = ex::operation_state_tag;

      void start() & noexcept
      {
        control->started   = true;
        control->operation = this;
        control->set_value = [](void* pointer) noexcept
        {
          auto& self = *static_cast<operation*>(pointer);
          ex::set_value(static_cast<_Receiver&&>(self.receiver));
        };
      }

      ~operation()
      {
        control->destroyed = true;
      }

      stop_control* control;
      _Receiver     receiver;
    };

    template <class _Receiver>
    auto connect(_Receiver receiver) const noexcept -> operation<_Receiver>
    {
      return {control, static_cast<_Receiver&&>(receiver)};
    }

    stop_control* control;
  };

  struct invalid_stop_factory
  {};

  struct throwing_move
  {
    throwing_move() = default;
    throwing_move(throwing_move&&) noexcept(false);
  };

  struct throwing_value_sender
  {
    using sender_concept = ex::sender_tag;

    template <class...>
    static consteval auto get_completion_signatures() noexcept
      -> ex::completion_signatures<ex::set_value_t(throwing_move)>
    {
      return {};
    }
  };

  inline constexpr auto stop_sender_sends_stopped =
    ::exec::stopped_by(main_sender{nullptr}, [](auto&) noexcept { return ex::just_stopped(); });

  inline constexpr auto stop_factory_can_throw = ::exec::stopped_by(main_sender{nullptr},
                                                                    [](auto&)
                                                                    { return ex::just(); });

  inline constexpr auto stop_factory_returns_non_sender = ::exec::stopped_by(main_sender{nullptr},
                                                                             [](auto&) noexcept
                                                                             { return 42; });

  inline constexpr auto classifier_can_throw = ::exec::stopped_by(
    main_sender{nullptr},
    [](auto&) noexcept { return ex::just(); },
    [](ex::set_value_t, int const &) { return false; });

  TEST_CASE("stopped_by validates its stop factory and classifier", "[stopped_by]")
  {
    STATIC_REQUIRE(ex::sender_in<decltype(stop_sender_sends_stopped)>);
    STATIC_REQUIRE(ex::sender_in<decltype(stop_factory_can_throw)>);
    STATIC_REQUIRE(ex::sender_in<decltype(stop_factory_returns_non_sender)>);
    STATIC_REQUIRE(ex::sender_in<decltype(classifier_can_throw)>);
    STATIC_REQUIRE_FALSE(ex::sender_in<decltype(stop_sender_sends_stopped), stop_env>);
    STATIC_REQUIRE_FALSE(ex::sender_in<decltype(stop_factory_can_throw), stop_env>);
    STATIC_REQUIRE_FALSE(ex::sender_in<decltype(stop_factory_returns_non_sender), stop_env>);
    STATIC_REQUIRE_FALSE(ex::sender_in<decltype(classifier_can_throw), stop_env>);
  }

  TEST_CASE("stopped_by has conservative environment-independent signatures", "[stopped_by]")
  {
    auto unclassified = ::exec::stopped_by(main_sender{nullptr},
                                           [](auto&) noexcept { return ex::just(); });
    auto classified = ::exec::stopped_by(
      main_sender{nullptr},
      [](auto&) noexcept { return ex::just(); },
      [](ex::set_value_t, int const&) noexcept { return false; });
    auto throwing_storage = ::exec::stopped_by(throwing_value_sender{},
                                               [](auto&) noexcept { return ex::just(); });

    STATIC_REQUIRE(ex::sender_in<decltype(unclassified)>);
    STATIC_REQUIRE_FALSE(ex::dependent_sender<decltype(unclassified)>);
    STATIC_REQUIRE(set_equivalent<ex::completion_signatures_of_t<decltype(unclassified)>,
                                  ex::completion_signatures<ex::set_value_t(int)>>);
    STATIC_REQUIRE(
      set_equivalent<
        ex::completion_signatures_of_t<decltype(classified)>,
        ex::completion_signatures<ex::set_value_t(int), ex::set_stopped_t()>>);
    STATIC_REQUIRE(
      set_equivalent<
        ex::completion_signatures_of_t<decltype(throwing_storage)>,
        ex::completion_signatures<ex::set_value_t(throwing_move),
                                  ex::set_error_t(std::exception_ptr)>>);
  }

  enum class result_kind
  {
    none,
    value,
    stopped
  };

  struct result_receiver
  {
    using receiver_concept = ex::receiver_tag;

    void set_value(int value) && noexcept
    {
      *result       = result_kind::value;
      *result_value = value;
    }

    void set_error(std::exception_ptr) && noexcept
    {
      FAIL_CHECK("Unexpected error completion");
    }

    void set_stopped() && noexcept
    {
      *result = result_kind::stopped;
    }

    auto get_env() const noexcept -> stop_env
    {
      return {ex::get_stop_token, source->get_token()};
    }

    result_kind*             result;
    int*                     result_value;
    ex::inplace_stop_source* source;
  };

  TEST_CASE("stopped_by is transparent in an unstoppable environment", "[stopped_by]")
  {
    auto sender = ex::just(42) | ::exec::stopped_by(invalid_stop_factory{});

    STATIC_REQUIRE(set_equivalent<ex::completion_signatures_of_t<decltype(sender), ex::env<>>,
                                  ex::completion_signatures<ex::set_value_t(int)>>);
  }

  TEST_CASE("stopped_by forwards an ordinary completion", "[stopped_by]")
  {
    main_control            main;
    stop_control            stop;
    ex::inplace_stop_source source;
    result_kind             result = result_kind::none;
    int                     value{};

    auto sender = ::exec::stopped_by(main_sender{&main},
                                     [&](auto&) noexcept { return stop_sender{&stop}; });
    STATIC_REQUIRE(set_equivalent<ex::completion_signatures_of_t<decltype(sender), stop_env>,
                                  ex::completion_signatures<ex::set_value_t(int)>>);

    auto operation = ex::connect(sender, result_receiver{&result, &value, &source});
    ex::start(operation);
    CHECK(main.started);
    CHECK_FALSE(stop.started);

    main.complete(42);
    CHECK(result == result_kind::value);
    CHECK(value == 42);
  }

  TEST_CASE("stopped_by coordinates a child that completes inline during start", "[stopped_by]")
  {
    ex::inplace_stop_source source;
    result_kind             result = result_kind::none;
    int                     value{};
    bool                    stop_started{};

    auto sender = ::exec::stopped_by(ex::just(43),
                                     [&](auto&) noexcept
                                     {
                                       stop_started = true;
                                       return ex::just();
                                     });

    auto operation = ex::connect(sender, result_receiver{&result, &value, &source});
    ex::start(operation);

    CHECK_FALSE(stop_started);
    CHECK(result == result_kind::value);
    CHECK(value == 43);
  }

  TEST_CASE("stopped_by waits for both operations after stop is requested", "[stopped_by]")
  {
    main_control            main;
    stop_control            stop;
    ex::inplace_stop_source source;
    result_kind             result = result_kind::none;
    int                     value{};
    void*                   target{};

    auto sender = main_sender{&main}
                | ::exec::stopped_by(
                    [&](auto& operation) noexcept
                    {
                      target = &operation;
                      return stop_sender{&stop};
                    });

    auto operation = ex::connect(sender, result_receiver{&result, &value, &source});
    ex::start(operation);

    source.request_stop();
    CHECK(stop.started);
    CHECK(target == main.operation);

    main.complete(17);
    CHECK(result == result_kind::none);
    stop.complete();
    CHECK(result == result_kind::value);
    CHECK(value == 17);
  }

  TEST_CASE("stopped_by completes when the main operation is last", "[stopped_by]")
  {
    main_control            main;
    stop_control            stop;
    ex::inplace_stop_source source;
    result_kind             result = result_kind::none;
    int                     value{};

    auto sender = ::exec::stopped_by(main_sender{&main},
                                     [&](auto&) noexcept { return stop_sender{&stop}; });

    auto operation = ex::connect(sender, result_receiver{&result, &value, &source});
    ex::start(operation);

    source.request_stop();
    CHECK_FALSE(stop.destroyed);
    stop.complete();
    CHECK(result == result_kind::none);
    CHECK(stop.destroyed);
    main.complete(29);
    CHECK(result == result_kind::value);
    CHECK(value == 29);
  }

  TEST_CASE("stopped_by supports an inline stop operation", "[stopped_by]")
  {
    main_control            main;
    ex::inplace_stop_source source;
    result_kind             result = result_kind::none;
    int                     value{};
    bool                    stop_started{};

    auto sender = ::exec::stopped_by(main_sender{&main},
                                     [&](auto&) noexcept
                                     {
                                       stop_started = true;
                                       return ex::just();
                                     });

    auto operation = ex::connect(sender, result_receiver{&result, &value, &source});
    ex::start(operation);

    source.request_stop();
    CHECK(stop_started);
    CHECK(result == result_kind::none);
    main.complete(31);
    CHECK(result == result_kind::value);
    CHECK(value == 31);
  }

  TEST_CASE("stopped_by starts the main operation before servicing a pending stop request",
            "[stopped_by]")
  {
    main_control            main;
    stop_control            stop;
    ex::inplace_stop_source source;
    result_kind             result = result_kind::none;
    int                     value{};
    bool                    main_was_started{};
    source.request_stop();

    auto sender = ::exec::stopped_by(main_sender{&main},
                                     [&](auto&) noexcept
                                     {
                                       main_was_started = main.started;
                                       return stop_sender{&stop};
                                     });

    auto operation = ex::connect(sender, result_receiver{&result, &value, &source});
    ex::start(operation);

    CHECK(main_was_started);
    CHECK(stop.started);
    main.complete(3);
    stop.complete();
    CHECK(result == result_kind::value);
  }

  TEST_CASE("stopped_by invokes its classifier for every completion", "[stopped_by]")
  {
    main_control            main;
    ex::inplace_stop_source source;
    result_kind             result = result_kind::none;
    int                     value{};
    int                     classifications{};

    auto sender = ::exec::stopped_by(
      main_sender{&main},
      [](auto&) noexcept { return ex::just(); },
      [&](ex::set_value_t, int const & completed_value) noexcept
      {
        ++classifications;
        return completed_value == 42;
      });
    STATIC_REQUIRE(
      set_equivalent<ex::completion_signatures_of_t<decltype(sender), stop_env>,
                     ex::completion_signatures<ex::set_value_t(int), ex::set_stopped_t()>>);

    auto operation = ex::connect(sender, result_receiver{&result, &value, &source});
    ex::start(operation);
    main.complete(42);

    CHECK(classifications == 1);
    CHECK(result == result_kind::stopped);
  }
}  // namespace
