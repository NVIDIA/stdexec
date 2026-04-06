/*
 * Copyright (c) 2023 NVIDIA Corporation
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

#include <exec/any_sender_of.hpp>
#include <exec/env.hpp>
#include <exec/inline_scheduler.hpp>
#include <exec/split.hpp>
#include <exec/static_thread_pool.hpp>
#include <exec/when_any.hpp>
#include <stdexec/stop_token.hpp>

#include <test_common/receivers.hpp>
#include <test_common/schedulers.hpp>

#include <catch2/catch.hpp>

#include <iostream>

namespace ex = STDEXEC;

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_MSVC(4702)  // unreachable code
STDEXEC_PRAGMA_IGNORE_MSVC(4996)  // 'function' was declared deprecated
STDEXEC_PRAGMA_IGNORE_GNU("-Wdeprecated-declarations")
STDEXEC_PRAGMA_IGNORE_EDG(deprecated_entity)
STDEXEC_PRAGMA_IGNORE_EDG(deprecated_entity_with_custom_message)

namespace
{
  struct get_address_t : ex::__query<get_address_t>
  {};

  inline constexpr get_address_t get_address;

  struct env
  {
    void const            *object_{nullptr};
    ex::inplace_stop_token token_{};

    [[nodiscard]]
    auto query(get_address_t) const noexcept -> void const *
    {
      return object_;
    }

    [[nodiscard]]
    auto query(ex::get_stop_token_t) const noexcept -> ex::inplace_stop_token
    {
      return token_;
    }
  };

  struct sink_receiver
  {
    using receiver_concept = ex::receiver_tag;

    std::variant<std::monostate, int, std::exception_ptr, ex::set_stopped_t> value_{};

    void set_value(int value) noexcept
    {
      value_ = value;
    }

    void set_error(std::exception_ptr e) noexcept
    {
      value_ = e;
    }

    void set_stopped() noexcept
    {
      value_ = ex::set_stopped_t();
    }

    [[nodiscard]]
    auto get_env() const noexcept -> env
    {
      return {.object_ = static_cast<void const *>(this)};
    }
  };

  TEST_CASE("exec::any_receiver_ref is constructible from receivers", "[types][any_sender]")
  {
    using Sigs = ex::completion_signatures<ex::set_value_t(int)>;
    sink_receiver                rcvr;
    exec::any_receiver_ref<Sigs> ref{rcvr};
    STATIC_REQUIRE(ex::receiver<decltype(ref)>);
    STATIC_REQUIRE(ex::receiver_of<decltype(ref), Sigs>);
    STATIC_REQUIRE(!ex::receiver_of<decltype(ref), ex::completion_signatures<ex::set_value_t()>>);
    STATIC_REQUIRE(std::is_copy_assignable_v<exec::any_receiver_ref<Sigs>>);
    STATIC_REQUIRE(std::is_constructible_v<exec::any_receiver_ref<Sigs>, sink_receiver const &>);
    STATIC_REQUIRE(!std::is_constructible_v<exec::any_receiver_ref<Sigs>, sink_receiver &&>);
    STATIC_REQUIRE(
      !std::is_constructible_v<exec::any_receiver_ref<ex::completion_signatures<ex::set_value_t()>>,
                               sink_receiver const &>);
  }

  TEST_CASE("exec::any_receiver_ref is queryable", "[types][any_sender]")
  {
    using Sigs         = ex::completion_signatures<ex::set_value_t(int)>;
    using receiver_ref = exec::any_receiver_ref<Sigs, get_address.signature<void const *()>>;
    sink_receiver rcvr1{};
    sink_receiver rcvr2{};
    receiver_ref  ref1{rcvr1};
    receiver_ref  ref2{rcvr2};
    CHECK(get_address(ex::get_env(ref1)) == &rcvr1);
    CHECK(get_address(ex::get_env(ref2)) == &rcvr2);
    {
      receiver_ref copied_ref = ref2;
      CHECK(get_address(ex::get_env(copied_ref)) != &ref2);
      CHECK(get_address(ex::get_env(copied_ref)) == &rcvr2);
      ref1 = copied_ref;
      CHECK(get_address(ex::get_env(ref1)) != &rcvr1);
      CHECK(get_address(ex::get_env(ref1)) != &copied_ref);
      CHECK(get_address(ex::get_env(ref1)) == &rcvr2);
      copied_ref = rcvr1;
      CHECK(get_address(ex::get_env(ref1)) == &rcvr2);
      CHECK(get_address(ex::get_env(copied_ref)) == &rcvr1);
    }
    CHECK(get_address(ex::get_env(ref1)) == &rcvr2);
  }

  TEST_CASE("exec::any_receiver_ref calls ex::receiver methods", "[types][any_sender]")
  {
    using Sigs         = ex::completion_signatures<ex::set_value_t(int),
                                                   ex::set_error_t(std::exception_ptr),
                                                   ex::set_stopped_t()>;
    using receiver_ref = exec::any_receiver_ref<Sigs>;
    REQUIRE(ex::receiver_of<receiver_ref, Sigs>);
    sink_receiver value{};
    sink_receiver error{};
    sink_receiver stopped{};

    // Check set value
    CHECK(value.value_.index() == 0);
    receiver_ref ref = value;
    ex::set_value(static_cast<receiver_ref &&>(ref), 42);
    CHECK(value.value_.index() == 1);
    CHECK(std::get<1>(value.value_) == 42);
    // Check set error
    CHECK(error.value_.index() == 0);
    ref = error;
    ex::set_error(static_cast<receiver_ref &&>(ref), std::make_exception_ptr(42));
    CHECK(error.value_.index() == 2);
#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
    // MSVC issues a warning about unreachable code in this block, hence the warning
    // suppression at the top of the file.
    CHECK_THROWS_AS(std::rethrow_exception(std::get<2>(error.value_)), int);
#endif
    // Check set stopped
    CHECK(stopped.value_.index() == 0);
    ref = stopped;
    ex::set_stopped(static_cast<receiver_ref &&>(ref));
    CHECK(stopped.value_.index() == 3);
  }

  TEST_CASE("exec::any_receiver_ref is connectable with exec::when_any", "[types][any_sender]")
  {
    using Sigs         = ex::completion_signatures<ex::set_value_t(int), ex::set_stopped_t()>;
    using receiver_ref = exec::any_receiver_ref<Sigs>;
    REQUIRE(ex::receiver_of<receiver_ref, Sigs>);
    sink_receiver rcvr{};
    receiver_ref  ref = rcvr;

    auto sndr = exec::when_any(ex::just(42));
    CHECK(rcvr.value_.index() == 0);
    auto op = ex::connect(std::move(sndr), std::move(ref));
    ex::start(op);
    CHECK(rcvr.value_.index() == 1);
    CHECK(std::get<1>(rcvr.value_) == 42);
  }

  template <class... Ts>
  using any_sender_of =
    exec::any_receiver_ref<ex::completion_signatures<Ts...>>::template any_sender<>;

  TEST_CASE("any sender is a sender", "[types][any_sender]")
  {
    STATIC_REQUIRE(ex::sender<any_sender_of<ex::set_value_t()>>);
    STATIC_REQUIRE(std::is_move_assignable_v<any_sender_of<ex::set_value_t()>>);
    STATIC_REQUIRE(std::is_nothrow_move_assignable_v<any_sender_of<ex::set_value_t()>>);
    STATIC_REQUIRE(!std::is_copy_assignable_v<any_sender_of<ex::set_value_t()>>);
    any_sender_of<ex::set_value_t()> sndr = ex::just();
  }

  TEST_CASE("ex::sync_wait works on any_sender_of", "[types][any_sender]")
  {
    int                              value = 0;
    any_sender_of<ex::set_value_t()> sndr  = ex::just(42)
                                          | ex::then([&](int v) noexcept { value = v; });
    CHECK(std::same_as<ex::completion_signatures_of_t<any_sender_of<ex::set_value_t()>>,
                       ex::completion_signatures<ex::set_value_t()>>);
    ex::sync_wait(std::move(sndr));
    CHECK(value == 42);
  }

  TEST_CASE("construct any_sender_of recursively from ex::when_all", "[types][any_sender]")
  {
    any_sender_of<ex::set_value_t(), ex::set_stopped_t(), ex::set_error_t(std::exception_ptr)>
      sndr = ex::just();
    using sender_t =
      any_sender_of<ex::set_value_t(), ex::set_stopped_t(), ex::set_error_t(std::exception_ptr)>;
    using when_all_t = decltype(ex::when_all(std::move(sndr)));
    STATIC_REQUIRE(std::is_constructible_v<sender_t, when_all_t &&>);
  }

  TEST_CASE("ex::sync_wait returns value", "[types][any_sender]")
  {
    any_sender_of<ex::set_value_t(int)> sndr = ex::just(21)
                                             | ex::then([&](int v) noexcept { return 2 * v; });
    CHECK(std::same_as<ex::completion_signatures_of_t<any_sender_of<ex::set_value_t(int)>>,
                       ex::completion_signatures<ex::set_value_t(int)>>);
    auto [value1] = *ex::sync_wait(std::move(sndr));
    CHECK(value1 == 42);
  }

  template <class... Vals>
  using my_sender_of = any_sender_of<ex::set_value_t(Vals)..., ex::set_error_t(std::exception_ptr)>;

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  TEST_CASE("ex::sync_wait returns value and exception", "[types][any_sender]")
  {
    my_sender_of<int> sndr = ex::just(21) | ex::then([&](int v) { return 2 * v; });
    auto [value]           = *ex::sync_wait(std::move(sndr));
    CHECK(value == 42);

    sndr = ex::just(21) | ex::then([&](int) -> int { throw 420; });
    CHECK_THROWS_AS(ex::sync_wait(std::move(sndr)), int);
  }
#endif  // !STDEXEC_NO_STDCPP_EXCEPTIONS()

  TEST_CASE("any_sender is connectable with exec::any_receiver_ref", "[types][any_sender]")
  {
    using Sigs         = ex::completion_signatures<ex::set_value_t(int), ex::set_stopped_t()>;
    using receiver_ref = exec::any_receiver_ref<Sigs>;
    using sender       = receiver_ref::any_sender<>;
    REQUIRE(ex::receiver_of<receiver_ref, Sigs>);
    sender sndr = ex::just_stopped();
    {
      sink_receiver rcvr{};
      receiver_ref  ref = rcvr;
      auto          op  = ex::connect(std::move(sndr), std::move(ref));
      CHECK(rcvr.value_.index() == 0);
      ex::start(op);
      CHECK(rcvr.value_.index() == 3);
    }
    sndr = ex::just(42);
    {
      sink_receiver rcvr{};
      receiver_ref  ref = rcvr;
      auto          op  = ex::connect(std::move(sndr), std::move(ref));
      CHECK(rcvr.value_.index() == 0);
      ex::start(op);
      CHECK(rcvr.value_.index() == 1);
    }
    sndr = exec::when_any(ex::just(42));
    {
      sink_receiver rcvr{};
      receiver_ref  ref = rcvr;
      auto          op  = ex::connect(std::move(sndr), std::move(ref));
      CHECK(rcvr.value_.index() == 0);
      ex::start(op);
      CHECK(rcvr.value_.index() == 1);
    }
  }

  template <class... Vals>
  using my_stoppable_sender_of = any_sender_of<ex::set_value_t(Vals)...,
                                               ex::set_error_t(std::exception_ptr),
                                               ex::set_stopped_t()>;

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  TEST_CASE("any_sender uses overload rules for completion signatures", "[types][any_sender]")
  {
    auto split_sender = exec::split(ex::just(42));
    STATIC_REQUIRE(
      ex::sender_of<decltype(split_sender), ex::set_error_t(std::exception_ptr const &)>);
    STATIC_REQUIRE(ex::sender_of<decltype(split_sender), ex::set_value_t(int const &)>);
    my_stoppable_sender_of<int> sndr = split_sender;

    auto [value] = *ex::sync_wait(std::move(sndr));
    CHECK(value == 42);

    sndr = ex::just(21) | ex::then([&](int) -> int { throw 420; });
    CHECK_THROWS_AS(ex::sync_wait(std::move(sndr)), int);
  }
#endif  // !STDEXEC_NO_STDCPP_EXCEPTIONS()

  class stopped_token
  {
   private:
    bool stopped_{true};

    struct __callback_type
    {
      template <class Fn>
      explicit __callback_type(stopped_token t, Fn &&f) noexcept
      {
        if (t.stopped_)
        {
          static_cast<Fn &&>(f)();
        }
      }
    };
   public:
    constexpr stopped_token() noexcept = default;

    explicit constexpr stopped_token(bool stopped) noexcept
      : stopped_{stopped}
    {}

    template <class>
    using callback_type = __callback_type;

    static auto stop_requested() noexcept -> std::true_type
    {
      return {};
    }

    static auto stop_possible() noexcept -> std::true_type
    {
      return {};
    }

    auto operator==(stopped_token const &) const noexcept -> bool = default;
  };

  template <class Token>
  struct stopped_receiver_base
  {
    using receiver_concept = ex::receiver_tag;
    Token stop_token_{};
  };

  template <class Token>
  struct stopped_receiver_env
  {
    stopped_receiver_base<Token> const *receiver_;

    [[nodiscard]]
    auto query(ex::get_stop_token_t) const noexcept -> Token
    {
      return receiver_->stop_token_;
    }
  };

  template <class Token>
  struct stopped_receiver : stopped_receiver_base<Token>
  {
    stopped_receiver(Token token, bool expect_stop)
      : stopped_receiver_base<Token>{token}
      , expect_stop_{expect_stop}
    {}

    bool expect_stop_{false};

    template <class... Args>
    void set_value(Args &&...) noexcept
    {
      CHECK(!expect_stop_);
    }

    void set_stopped() noexcept
    {
      CHECK(expect_stop_);
    }

    [[nodiscard]]
    auto get_env() const noexcept -> stopped_receiver_env<Token>
    {
      return {this};
    }
  };

  template <class Token>
  stopped_receiver(Token, bool) -> stopped_receiver<Token>;

  static_assert(
    ex::receiver_of<stopped_receiver<ex::inplace_stop_token>,
                    ex::completion_signatures<ex::set_value_t(int), ex::set_stopped_t()>>);

  TEST_CASE("any_sender - does connect with stop token", "[types][any_sender]")
  {
    using completions_t = ex::completion_signatures<ex::set_value_t(int), ex::set_stopped_t()>;
    constexpr auto stop_token_query =
      ex::get_stop_token.signature<ex::inplace_stop_token() noexcept>;

    using stoppable_sender = exec::any_receiver_ref<completions_t, stop_token_query>::any_sender<>;
    stoppable_sender        sndr = exec::when_any(ex::just(21));
    ex::inplace_stop_source stop_source{};
    stopped_receiver        rcvr{stop_source.get_token(), true};
    stop_source.request_stop();
    auto do_check = ex::connect(std::move(sndr), std::move(rcvr));
    // This CHECKS whether set_value is called
    ex::start(do_check);
  }

  TEST_CASE("any_sender - does connect with an user-defined stop token", "[types][any_sender]")
  {
    using completions_t = ex::completion_signatures<ex::set_value_t(int), ex::set_stopped_t()>;
    constexpr auto stop_token_query =
      ex::get_stop_token.signature<ex::inplace_stop_token() noexcept>;

    using stoppable_sender = exec::any_receiver_ref<completions_t, stop_token_query>::any_sender<>;
    stoppable_sender sndr  = exec::when_any(ex::just(21));

    SECTION("stopped true")
    {
      stopped_token    token{true};
      stopped_receiver rcvr{token, true};
      auto             do_check = ex::connect(std::move(sndr), std::move(rcvr));
      // This CHECKS whether set_value is called
      ex::start(do_check);
    }

    SECTION("stopped false")
    {
      stopped_token    token{false};
      stopped_receiver rcvr{token, false};
      auto             do_check = ex::connect(std::move(sndr), std::move(rcvr));
      // This CHECKS whether set_value is called
      ex::start(do_check);
    }
  }

  TEST_CASE("any_sender - does connect with stop token if the get_stop_token query is registered "
            "with ex::inplace_stop_token",
            "[types][any_sender]")
  {
    using completions_t = ex::completion_signatures<ex::set_value_t(int), ex::set_stopped_t()>;
    constexpr auto stop_token_query =
      ex::get_stop_token.signature<ex::inplace_stop_token() noexcept>;

    using stoppable_sender = exec::any_receiver_ref<completions_t, stop_token_query>::any_sender<>;
    stoppable_sender        sndr = exec::when_any(ex::just(21));
    ex::inplace_stop_source stop_source{};
    stopped_receiver        rcvr{stop_source.get_token(), true};
    stop_source.request_stop();
    auto do_check = ex::connect(std::move(sndr), std::move(rcvr));
    // This CHECKS whether a set_stopped is called
    ex::start(do_check);
  }

  TEST_CASE("any_sender - does connect with stop token if the ex::get_stop_token query is "
            "registered with ex::never_stop_token",
            "[types][any_sender]")
  {
    using completions_t = ex::completion_signatures<ex::set_value_t(int), ex::set_stopped_t()>;
    constexpr auto stop_token_query = ex::get_stop_token.signature<ex::never_stop_token() noexcept>;

    using unstoppable_sender =
      exec::any_receiver_ref<completions_t, stop_token_query>::any_sender<>;
    unstoppable_sender      sndr = exec::when_any(ex::just(21));
    ex::inplace_stop_source stop_source{};
    stopped_receiver        rcvr{stop_source.get_token(), false};
    stop_source.request_stop();
    auto do_check = ex::connect(std::move(sndr), std::move(rcvr));
    // This CHECKS whether a set_stopped is called
    ex::start(do_check);
  }

  TEST_CASE("any_sender - get_completion_signatures is constrained with respect to stop-token"
            "ex::receiver stop_token queries",
            "[types][any_sender]")
  {
    using completions_t = ex::completion_signatures<ex::set_value_t(), ex::set_stopped_t()>;
    constexpr auto stop_token_query =
      ex::get_stop_token.signature<ex::inplace_stop_token() noexcept>;

    using sender = exec::any_receiver_ref<completions_t, stop_token_query>::any_sender<>;
    using env    = ex::prop<ex::get_stop_token_t, ex::inplace_stop_token>;

    STATIC_REQUIRE(ex::dependent_sender<sender>);
    STATIC_REQUIRE(requires {
      { ex::get_completion_signatures<sender, env>() } -> std::same_as<completions_t>;
    });
  }

  TEST_CASE("queryable any_scheduler with inline_scheduler", "[types][any_sender]")
  {
    using completions_t        = ex::completion_signatures<ex::set_value_t()>;
    using any_sender_t         = exec::any_sender<exec::any_receiver<completions_t>>;
    using fwd_progress_query_t = ex::forward_progress_guarantee(
      ex::get_forward_progress_guarantee_t) noexcept;
    using my_scheduler2 = exec::any_scheduler<any_sender_t, exec::queries<fwd_progress_query_t>>;

    STATIC_REQUIRE(ex::scheduler<my_scheduler2>);
    my_scheduler2 scheduler = ex::inline_scheduler();
    my_scheduler2 copied    = scheduler;
    CHECK(copied == scheduler);

    auto sndr = ex::schedule(scheduler);
    STATIC_REQUIRE(ex::sender<decltype(sndr)>);
    std::same_as<my_scheduler2> auto get_sched = ex::get_completion_scheduler<ex::set_value_t>(
      ex::get_env(sndr));
    CHECK(get_sched == scheduler);

    CHECK(ex::get_forward_progress_guarantee(scheduler)
          == ex::get_forward_progress_guarantee(ex::inline_scheduler()));

    bool called = false;
    ex::sync_wait(std::move(sndr) | ex::then([&] { called = true; }));
    CHECK(called);
  }

  TEST_CASE("any_scheduler adds ex::set_value_t() completion sig",
            "[types][any_scheduler][any_sender]")
  {
    using namespace exec;
    using completions_t = ex::completion_signatures<ex::set_value_t()>;
    using scheduler_t   = any_scheduler<any_sender<any_receiver<completions_t>>>;
    using sender_t      = decltype(ex::schedule(std::declval<scheduler_t>()));
    STATIC_REQUIRE(std::is_same_v<ex::completion_signatures_of_t<sender_t>, completions_t>);
    STATIC_REQUIRE(ex::scheduler<scheduler_t>);
  }

  TEST_CASE("User-defined completion_scheduler<ex::set_value_t> is ignored",
            "[types][any_scheduler][any_sender]")
  {
    using namespace exec;
    using completions_t = ex::completion_signatures<ex::set_value_t()>;
    using sender_queries_t =
      queries<ex::inline_scheduler(ex::get_completion_scheduler_t<ex::set_value_t>) noexcept>;
    using weird_scheduler_t =
      any_scheduler<any_sender<any_receiver<completions_t>, sender_queries_t>>;
    using completion_scheduler_t = decltype(ex::get_completion_scheduler<ex::set_value_t>(
      std::declval<weird_scheduler_t>()));
    STATIC_REQUIRE(ex::scheduler<weird_scheduler_t>);
    STATIC_REQUIRE(std::same_as<completion_scheduler_t, weird_scheduler_t>);
  }

  template <class... Queries>
  using stoppable_scheduler =
    exec::any_scheduler<exec::any_sender<exec::any_receiver<
                          ex::completion_signatures<ex::set_value_t(), ex::set_stopped_t()>>>,
                        exec::queries<Queries...>>;

  TEST_CASE("any scheduler with static_thread_pool", "[types][any_sender]")
  {
    exec::static_thread_pool pool(1);
    stoppable_scheduler<>    scheduler = pool.get_scheduler();
    auto                     copied    = scheduler;
    CHECK(copied == scheduler);

    auto sndr = ex::schedule(scheduler);
    STATIC_REQUIRE(ex::sender<decltype(sndr)>);
    std::same_as<stoppable_scheduler<>> auto get_sched =
      ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr));
    CHECK(get_sched == scheduler);

    bool called = false;
    ex::sync_wait(std::move(sndr) | ex::then([&] { called = true; }));
    CHECK(called);
  }

  TEST_CASE("queryable any_scheduler with static_thread_pool", "[types][any_sender]")
  {
    using my_scheduler = stoppable_scheduler<ex::forward_progress_guarantee(
      ex::get_forward_progress_guarantee_t) noexcept>;

    exec::static_thread_pool pool(1);
    my_scheduler             scheduler = pool.get_scheduler();
    auto                     copied    = scheduler;
    CHECK(copied == scheduler);

    auto sndr = ex::schedule(scheduler);
    STATIC_REQUIRE(ex::sender<decltype(sndr)>);
    std::same_as<my_scheduler> auto completion_sched =
      ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr));
    CHECK(completion_sched == scheduler);

    auto fwd_progress = scheduler.query(ex::get_forward_progress_guarantee);
    STATIC_REQUIRE(std::same_as<decltype(fwd_progress), ex::forward_progress_guarantee>);

    CHECK(ex::get_forward_progress_guarantee(scheduler)
          == ex::get_forward_progress_guarantee(pool.get_scheduler()));

    bool called = false;
    ex::sync_wait(std::move(sndr) | ex::then([&] { called = true; }));
    CHECK(called);
  }

  TEST_CASE("Scheduler with error handling and set_stopped", "[types][any_scheduler][any_sender]")
  {
    using receiver_t =
      exec::any_receiver<ex::completion_signatures<ex::set_value_t(),
                                                   ex::set_stopped_t(),
                                                   ex::set_error_t(std::exception_ptr)>>;
    using sender_t        = exec::any_sender<receiver_t>;
    using scheduler_t     = exec::any_scheduler<sender_t>;
    scheduler_t scheduler = ex::inline_scheduler();
    {
      auto op = ex::connect(ex::schedule(scheduler), expect_void_receiver{});
      ex::start(op);
    }
    scheduler = stopped_scheduler();
    {
      auto op = ex::connect(ex::schedule(scheduler), expect_stopped_receiver{});
      ex::start(op);
    }
    scheduler = error_scheduler<>{std::make_exception_ptr(std::logic_error("test"))};
    {
      auto op = ex::connect(ex::schedule(scheduler), expect_error_receiver<>{});
      ex::start(op);
    }
  }

  TEST_CASE("any_scheduler sender lifetime", "[types][any_scheduler][any_sender]")
  {
    using receiver_t =
      exec::any_receiver<ex::completion_signatures<ex::set_value_t(),
                                                   ex::set_stopped_t(),
                                                   ex::set_error_t(std::exception_ptr)>>;
    using sender_t        = exec::any_sender<receiver_t>;
    using scheduler_t     = exec::any_scheduler<sender_t>;
    scheduler_t scheduler = ex::inline_scheduler();
    auto        sndr      = ex::schedule(scheduler);
    scheduler             = stopped_scheduler();
    {
      auto op = ex::connect(ex::schedule(scheduler), expect_stopped_receiver{});
      ex::start(op);
    }
    {
      auto op = ex::connect(std::move(sndr), expect_void_receiver{});
      ex::start(op);
    }
  }

  // A scheduler that counts how many instances are extant.
  struct counting_scheduler
  {
    static int count;

    counting_scheduler() noexcept
    {
      ++count;
    }

    counting_scheduler(counting_scheduler const &) noexcept
    {
      ++count;
    }

    counting_scheduler(counting_scheduler &&) noexcept
    {
      ++count;
    }

    ~counting_scheduler()
    {
      --count;
    }

    auto operator==(counting_scheduler const &) const noexcept -> bool = default;

   private:
    template <class R>
    struct operation : immovable
    {
      R recv_;

      void start() & noexcept
      {
        ex::set_value(static_cast<R &&>(recv_));
      }
    };

    struct sender
    {
      using sender_concept        = ex::sender_tag;
      using completion_signatures = ex::completion_signatures<ex::set_value_t()>;

      template <ex::receiver R>
      auto connect(R r) const -> operation<R>
      {
        return {{}, static_cast<R &&>(r)};
      }

      [[nodiscard]]
      auto
      query(ex::get_completion_scheduler_t<ex::set_value_t>) const noexcept -> counting_scheduler
      {
        return {};
      }

      [[nodiscard]]
      auto get_env() const noexcept -> sender const &
      {
        return *this;
      }
    };

   public:
    [[nodiscard]]
    auto schedule() const noexcept -> sender
    {
      return {};
    }
  };

  int counting_scheduler::count = 0;

  TEST_CASE("check that any_scheduler cleans up all resources",
            "[types][any_scheduler][any_sender]")
  {
    using receiver_t  = exec::any_receiver<ex::completion_signatures<ex::set_value_t()>>;
    using sender_t    = exec::any_sender<receiver_t>;
    using scheduler_t = exec::any_scheduler<sender_t>;
    {
      scheduler_t scheduler = ex::inline_scheduler{};
      scheduler             = counting_scheduler{};
      CHECK(counting_scheduler::count == 1);
      {
        auto op = ex::connect(ex::schedule(scheduler), expect_value_receiver<>{});
        ex::start(op);
      }
    }
    CHECK(counting_scheduler::count == 0);
  }

  ///////////////////////////////////////////////////////////////////////////////
  //                                                                any_scheduler

  template <auto... Queries>
  using my_scheduler = any_sender_of<ex::set_value_t()>::any_scheduler<Queries...>;

  TEST_CASE("any scheduler with inline_scheduler", "[types][any_sender]")
  {
    STATIC_REQUIRE(ex::scheduler<my_scheduler<>>);
    my_scheduler<> scheduler = ex::inline_scheduler();
    my_scheduler<> copied    = scheduler;
    CHECK(copied == scheduler);

    auto sndr = ex::schedule(scheduler);
    STATIC_REQUIRE(ex::sender<decltype(sndr)>);
    std::same_as<my_scheduler<>> auto get_sched = ex::get_completion_scheduler<ex::set_value_t>(
      ex::get_env(sndr));
    CHECK(get_sched == scheduler);

    bool called = false;
    ex::sync_wait(std::move(sndr) | ex::then([&] { called = true; }));
    CHECK(called);
  }

  template <class _Sigs>
  struct sink;

  template <class... _Sigs>
  struct sink<ex::completion_signatures<_Sigs...>>
  {
    using receiver_concept = ex::receiver_tag;

    template <class... _Args>
    void set_value(_Args &&...__args) noexcept
      requires ex::__one_of<ex::set_value_t(_Args...), _Sigs...>
    {
      std::printf("set_value called");
      ((std::cout << ' ' << __args), ...);
      std::cout << '\n';
    }
    template <class Error>
    void set_error(Error &&) noexcept
      requires ex::__one_of<ex::set_error_t(Error), _Sigs...>
    {
      std::puts("set_error called");
    }
    void set_stopped() noexcept
      requires ex::__one_of<ex::set_stopped_t(), _Sigs...>
    {
      std::puts("set_stopped called");
    }
    struct env
    {
      [[nodiscard]]
      constexpr auto query(ex::get_stop_token_t) const noexcept -> ex::inplace_stop_token
      {
        return {};
      }
    };
    [[nodiscard]]
    constexpr env get_env() const noexcept
    {
      return {};
    }
  };

  TEST_CASE("any_receiver", "[types][any_sender]")
  {
    using completions_t = ex::completion_signatures<ex::set_value_t(int),
                                                    ex::set_error_t(std::exception_ptr),
                                                    ex::set_stopped_t()>;
    using queries_t     = exec::queries<ex::inplace_stop_token(ex::get_stop_token_t) noexcept>;

    using any_receiver_t = exec::any_receiver<completions_t, queries_t>;
    any_receiver_t rcvr  = sink<completions_t>{};
    std::move(rcvr).set_value(42);
    std::move(rcvr).set_error(std::make_exception_ptr(std::runtime_error("error")));
    std::move(rcvr).set_stopped();

    [[maybe_unused]]
    auto tok = ex::get_stop_token(ex::get_env(rcvr));
    STATIC_REQUIRE(std::same_as<decltype(tok), ex::inplace_stop_token>);

    using any_sender_t = exec::any_sender<any_receiver_t>;
    any_sender_t sndr  = ex::just(42);
    auto         op    = ex::connect(std::move(sndr), sink<completions_t>{});
    ex::start(op);

    using inline_completions_t                     = ex::completion_signatures<ex::set_value_t()>;
    using inline_any_receiver_t                    = exec::any_receiver<inline_completions_t>;
    using inline_any_sender_t                      = exec::any_sender<inline_any_receiver_t>;
    exec::any_scheduler<inline_any_sender_t> sch   = ex::inline_scheduler{};
    auto                                     sndr2 = ex::schedule(sch);
    auto op2 = ex::connect(std::move(sndr2), sink<inline_completions_t>{});
    ex::start(op2);
  }
}  // namespace

STDEXEC_PRAGMA_POP()
