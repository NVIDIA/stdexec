/*
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

#include "exec/fork_join.hpp"
#include "exec/just_from.hpp"
#include "exec/single_thread_context.hpp"
#include "exec/static_thread_pool.hpp"
#include "test_common/schedulers.hpp"
#include "test_common/type_helpers.hpp"

#include <test_common/catch2.hpp>

#include <atomic>
#include <exception>
#include <stdexcept>
#include <thread>

using namespace STDEXEC;

namespace
{
  TEST_CASE("fork_join is a sender", "[adaptors][fork_join]")
  {
    auto sndr = exec::fork_join(just(), then([] {}));
    STATIC_REQUIRE(sender<decltype(sndr)>);
  }

  TEST_CASE("fork_join is a sender in empty env", "[adaptors][fork_join]")
  {
    auto sndr = exec::fork_join(just(), then([] {}));
    STATIC_REQUIRE(sender_in<decltype(sndr), env<>>);
    STATIC_REQUIRE(
      set_equivalent<completion_signatures_of_t<decltype(sndr), env<>>,
                     completion_signatures<set_value_t(), set_error_t(std::exception_ptr)>>);
  }

  TEST_CASE("fork_join coalesces empty and unary calls", "[adaptors][fork_join]")
  {
    /// Empty (no closure given).
    STDEXEC::sender auto empty = exec::fork_join(STDEXEC::just());
    using empty_t              = decltype(empty);
    STATIC_REQUIRE(std::same_as<empty_t, decltype(STDEXEC::just())>);
    STATIC_REQUIRE(!exec::sender_for<empty_t, exec::fork_join_t>);
    STATIC_REQUIRE(noexcept(exec::fork_join(STDEXEC::just())));

    auto then = STDEXEC::then([]() noexcept {});

    /// Unary closure.
    STDEXEC::sender auto unary = exec::fork_join(STDEXEC::just(), then);
    using unary_t              = decltype(unary);
    STATIC_REQUIRE(std::same_as<unary_t, decltype(STDEXEC::just() | then)>);
    STATIC_REQUIRE(!exec::sender_for<unary_t, exec::fork_join_t>);
    STATIC_REQUIRE(noexcept(exec::fork_join(STDEXEC::just(), then)));

    /// Multiple closures.
    STDEXEC::sender auto multiple = STDEXEC::just() | exec::fork_join(then, then);
    STATIC_REQUIRE(exec::sender_for<decltype(multiple), exec::fork_join_t>);
    STATIC_REQUIRE(noexcept(exec::fork_join(STDEXEC::just(), then, then)));
  }

  struct ForwardingThen
  {
    template <typename Value>
    [[maybe_unused]]
    constexpr decltype(auto) operator()(Value &&value) const noexcept
    {
      return std::forward<Value>(value);
    }

    constexpr void operator()() const noexcept {}
  };

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  struct throwing_copy
  {
    throwing_copy() = default;

    throwing_copy(throwing_copy const &) noexcept(false)
    {
      throw 42;
    }

    int value = 0;
  };
#endif

  //! Count how many copies and moves are performed.
  struct counter
  {
    inline static std::atomic<unsigned int> copy_constructions{0};
    inline static std::atomic<unsigned int> copy_assignments{0};
    inline static std::atomic<unsigned int> move_constructions{0};
    inline static std::atomic<unsigned int> move_assignments{0};

    counter() = default;
    counter(counter const &) noexcept
    {
      copy_constructions.fetch_add(1, std::memory_order_relaxed);
    }
    counter &operator=(counter const &) noexcept
    {
      copy_assignments.fetch_add(1, std::memory_order_relaxed);
      return *this;
    }
    counter(counter &&) noexcept
    {
      move_constructions.fetch_add(1, std::memory_order_relaxed);
    }
    counter &operator=(counter &&) noexcept
    {
      move_assignments.fetch_add(1, std::memory_order_relaxed);
      return *this;
    }
  };

  template <char ID>
  struct identifiable_domain : public STDEXEC::default_domain
  {};

  template <class Sndr, class Env>
  concept has_value_completion_scheduler = requires(Sndr const &sndr, Env const &env) {
    STDEXEC::get_completion_scheduler<STDEXEC::set_value_t>(STDEXEC::get_env(sndr), env);
  };

  TEST_CASE("fork_join completion domain and scheduler", "[adaptors][fork_join]")
  {
    basic_inline_scheduler<identifiable_domain<'A'>> sched_A;
    basic_inline_scheduler<identifiable_domain<'B'>> sched_B;
    basic_inline_scheduler<identifiable_domain<'C'>> sched_C;

    auto env = STDEXEC::__root_env{};

    // The results are sent where the last closure's sender completes. These complete inline,
    // where the sender of the cached results completes, which is where the child completed:
    // on sched_A.
    auto sndr1 = STDEXEC::schedule(sched_A) | STDEXEC::then(ForwardingThen{})
               | exec::fork_join(STDEXEC::then(ForwardingThen{}), STDEXEC::then(ForwardingThen{}));

    auto domain1 = STDEXEC::get_completion_domain<STDEXEC::set_value_t>(STDEXEC::get_env(sndr1),
                                                                        env);
    static_assert(std::same_as<decltype(domain1), identifiable_domain<'A'>>);

    auto sched1 = STDEXEC::get_completion_scheduler<STDEXEC::set_value_t>(STDEXEC::get_env(sndr1),
                                                                          env);
    static_assert(std::same_as<decltype(sched1), decltype(sched_A)>);
    CHECK(sched1 == sched_A);

    // The on() closures make the sender dependent: where on() returns depends on the
    // environment. Each returns to where the sender of the cached results completes, so both
    // complete in sched_A's domain. But when_all cannot know that they complete on the same
    // scheduler, so there is no completion scheduler.
    auto sndr2 = STDEXEC::schedule(sched_A) | STDEXEC::then(ForwardingThen{})
               | exec::fork_join(STDEXEC::on(sched_B, STDEXEC::then(ForwardingThen{})),
                                 STDEXEC::on(sched_C, STDEXEC::then(ForwardingThen{})))
               | STDEXEC::then(ForwardingThen{});

    auto domain2 = STDEXEC::get_completion_domain<STDEXEC::set_value_t>(STDEXEC::get_env(sndr2),
                                                                        env);
    static_assert(std::same_as<decltype(domain2), identifiable_domain<'A'>>);
    static_assert(!has_value_completion_scheduler<decltype(sndr2), decltype(env)>);
  }

  TEST_CASE("fork_join completion domain and scheduler when the closures continue on other "
            "schedulers",
            "[adaptors][fork_join]")
  {
    basic_inline_scheduler<identifiable_domain<'A'>> sched_A;
    basic_inline_scheduler<identifiable_domain<'B'>> sched_B;
    basic_inline_scheduler<identifiable_domain<'C'>> sched_C;

    auto env = STDEXEC::__root_env{};

    // The results are sent where the last closure's sender completes: on sched_B, or on
    // sched_C. So there is no completion scheduler, and the completion domain is one of
    // theirs.
    auto sndr = STDEXEC::schedule(sched_A) | STDEXEC::then(ForwardingThen{})
              | exec::fork_join(STDEXEC::continues_on(sched_B) | STDEXEC::then(ForwardingThen{}),
                                STDEXEC::continues_on(sched_C) | STDEXEC::then(ForwardingThen{}))
              | STDEXEC::then(ForwardingThen{});

    auto domain = STDEXEC::get_completion_domain<STDEXEC::set_value_t>(STDEXEC::get_env(sndr), env);
    static_assert(
      std::same_as<
        decltype(domain),
        STDEXEC::indeterminate_domain<identifiable_domain<'B'>, identifiable_domain<'C'>>>);
    static_assert(!has_value_completion_scheduler<decltype(sndr), decltype(env)>);
  }

  TEST_CASE("fork_join completes where the last of its closures' senders completes",
            "[adaptors][fork_join]")
  {
    exec::single_thread_context stc_a{}, stc_b{}, stc_c{};

    std::thread::id who;

    auto sndr = STDEXEC::schedule(stc_a.get_scheduler()) | STDEXEC::then(ForwardingThen{})
              | exec::fork_join(STDEXEC::continues_on(stc_b.get_scheduler())
                                  | STDEXEC::then(ForwardingThen{}),
                                STDEXEC::continues_on(stc_c.get_scheduler())
                                  | STDEXEC::then(ForwardingThen{}))
              | STDEXEC::then([&who]() { who = std::this_thread::get_id(); });

    // It completes on stc_b or on stc_c, whichever closure's sender completes last, so there
    // is no completion scheduler.
    auto env    = STDEXEC::__root_env{};
    auto domain = STDEXEC::get_completion_domain<STDEXEC::set_value_t>(STDEXEC::get_env(sndr), env);
    static_assert(std::same_as<decltype(domain), STDEXEC::default_domain>);
    static_assert(!has_value_completion_scheduler<decltype(sndr), decltype(env)>);

    CHECK(stc_a.get_thread_id() != stc_b.get_thread_id());
    CHECK(stc_a.get_thread_id() != stc_c.get_thread_id());

    STDEXEC::sync_wait(std::move(sndr));

    CHECK((who == stc_b.get_thread_id() || who == stc_c.get_thread_id()));
  }

  TEST_CASE("fork_join broadcasts results to multiple continuations", "[adaptors][fork_join]")
  {
    auto fn = [](auto sink)
    {
      sink(42);
      return completion_signatures<set_value_t(int),
                                   set_value_t(int, int),
                                   set_value_t(int, int, int)>{};
    };
    auto sndr =
      exec::fork_join(exec::just_from(fn),
                      then(
                        [](auto &&...is)
                        {
                          CHECK(sizeof...(is) == 1);
                          CHECK(((is == 42) && ...));
                          STATIC_REQUIRE((std::is_same_v<decltype(is), int const &> && ...));
                          return (is + ...);
                        }),
                      then(
                        [](auto &&...is)
                        {
                          CHECK(sizeof...(is) == 1);
                          CHECK(((is == 42) && ...));
                          STATIC_REQUIRE((std::is_same_v<decltype(is), int const &> && ...));
                          return (is + ...);
                        }));
    STATIC_REQUIRE(sender_in<decltype(sndr), env<>>);
    STATIC_REQUIRE(set_equivalent<
                   completion_signatures_of_t<decltype(sndr), env<>>,
                   completion_signatures<set_value_t(int, int), set_error_t(std::exception_ptr)>>);

    auto [i1, i2] = sync_wait(sndr).value();
    CHECK(i1 == 42);
    CHECK(i2 == 42);
  }

  TEST_CASE("fork_join with empty value channel", "[adaptors][fork_join]")
  {
    auto sndr = ::STDEXEC::just() | ::STDEXEC::then([]() noexcept -> void {})
              | exec::fork_join(::STDEXEC::then([]() noexcept -> void {}),
                                ::STDEXEC::then([]() noexcept -> void {}));

    ::STDEXEC::sync_wait(std::move(sndr));
  }

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  TEST_CASE("fork_join reports failures while caching results", "[adaptors][fork_join]")
  {
    std::atomic<int> witness{0};

    auto sndr = exec::fork_join(exec::just_from(
                                  [](auto sink) noexcept
                                  {
                                    static throwing_copy value;
                                    return sink(value);
                                  }),
                                then([&witness](throwing_copy const &) noexcept { ++witness; }),
                                then([&witness](throwing_copy const &) noexcept { ++witness; }));

    CHECK_THROWS_AS(sync_wait(std::move(sndr)), int);

    CHECK(witness == 0);
  }
#endif

  TEST_CASE("fork_join caches without copying when results are movable and replays by reference to "
            "children",
            "[adaptors][fork_join]")
  {
    std::atomic<int> witness{0};

    auto sndr = exec::fork_join(exec::just_from([](auto sink) noexcept { return sink(counter{}); }),
                                STDEXEC::then([&witness](counter const &) noexcept { ++witness; }),
                                STDEXEC::then([&witness](counter const &) noexcept { ++witness; }));

    STDEXEC::sync_wait(std::move(sndr));

    CHECK(counter::copy_constructions == 0);
    CHECK(counter::copy_assignments == 0);
    CHECK(counter::move_constructions == 1);
    CHECK(counter::move_assignments == 0);

    CHECK(witness == 2);
  }

  TEST_CASE("fork_join can be nested", "[adaptors][fork_join]")
  {
    std::atomic<int> witness = 0;

    auto make_then = [&witness]()
    {
      return ::STDEXEC::then([&witness]() noexcept { ++witness; });
    };

    auto sndr = ::STDEXEC::just() | make_then()
              | exec::fork_join(make_then(), exec::fork_join(make_then(), make_then()));

    ::STDEXEC::sync_wait(std::move(sndr));

    CHECK(witness == 4);
  }

  struct customize_fork_join_domain : public STDEXEC::default_domain
  {
    template <STDEXEC::__sender_for<exec::fork_join_t> Sndr, class Env>
    constexpr auto transform_sender(STDEXEC::set_value_t, Sndr &&, Env const &) const noexcept
    {
      return STDEXEC::just(std::string("congrats on customizing fork_join_t"));
    }
  };

  TEST_CASE("fork_join is customizable", "[adaptors][fork_join]")
  {
    basic_inline_scheduler<customize_fork_join_domain> sched{};

    auto sndr = ::STDEXEC::schedule(sched) | ::STDEXEC::then(ForwardingThen{})
              | exec::fork_join(::STDEXEC::then(ForwardingThen{}),
                                ::STDEXEC::then(ForwardingThen{}));

    CHECK(std::get<0>(STDEXEC::sync_wait(std::move(sndr)).value())
          == "congrats on customizing fork_join_t");
  }

  TEST_CASE("fork_join following a continues_on", "[adaptors][fork_join]")
  {
    std::atomic<int> witness = 0;

    auto make_then = [&witness]()
    {
      return ::STDEXEC::then([&witness]() noexcept { ++witness; });
    };

    auto sndr = ::STDEXEC::just() | ::STDEXEC::continues_on(::STDEXEC::inline_scheduler{})
              | exec::fork_join(make_then(), make_then());

    ::STDEXEC::sync_wait(std::move(sndr));

    CHECK(witness == 2);
  }

  TEST_CASE("fork_join with exec::static_thread_pool", "[adaptors][fork_join]")
  {
    std::atomic<int> witness{0};

    ::exec::static_thread_pool pool{4};

    auto sndr = ::STDEXEC::schedule(pool.get_scheduler())
              | ::STDEXEC::then(
                  [&witness]() noexcept -> int
                  {
                    ++witness;
                    return witness;
                  })
              | ::exec::fork_join(::STDEXEC::then([&witness](int const received) noexcept
                                                  { witness += received; }),
                                  ::STDEXEC::then([&witness](int const received) noexcept
                                                  { witness += received; }));

    static_assert(::STDEXEC::dependent_sender<decltype(sndr)>);

    ::STDEXEC::sync_wait(std::move(sndr));

    CHECK(witness == 3);
  }

  // Can the sender complete with set_error(exception_ptr)?
  template <class Sndr>
  constexpr bool has_exception_ptr_error_v =
    STDEXEC::__mapply<STDEXEC::__mcontains<set_error_t(std::exception_ptr)>,
                      completion_signatures_of_t<Sndr, env<>>>::value;

  TEST_CASE("fork_join can be connected without throwing if its child and its closures' senders "
            "can",
            "[adaptors][fork_join]")
  {
    // (The function is noexcept, so an exception could come only from connecting.)
    auto fn = []() noexcept
    {
      return exec::fork_join(just(), then([]() noexcept {}), then([]() noexcept {}));
    };
    STATIC_REQUIRE(
      STDEXEC::__nothrow_connectable<decltype(fn()), STDEXEC::__receiver_archetype<env<>>>);

    // So an algorithm that connects it when another sender completes has no exception to
    // report.
    STATIC_REQUIRE(!has_exception_ptr_error_v<decltype(let_value(just(), fn))>);
    CHECK(sync_wait(let_value(just(), fn)).has_value());
  }

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  // A sender whose connect can throw
  struct throws_on_connect
  {
    using sender_concept        = sender_tag;
    using completion_signatures = STDEXEC::completion_signatures<set_value_t()>;

    template <class Receiver>
    auto connect(Receiver) const -> connect_result_t<decltype(just()), Receiver>
    {
      throw std::logic_error("connect");
    }
  };

  // A closure that ignores the sender of the cached results, and gives a sender whose connect
  // can throw
  struct to_throws_on_connect
  {
    template <class Sndr>
    auto operator()(Sndr) const noexcept -> throws_on_connect
    {
      return {};
    }
  };

  TEST_CASE("fork_join can throw when connected if its child's or a closure's sender can",
            "[adaptors][fork_join]")
  {
    // (The functions are noexcept, so the exceptions can come only from connecting.)
    auto child_throws = []() noexcept
    {
      return exec::fork_join(throws_on_connect{}, then([]() noexcept {}), then([]() noexcept {}));
    };
    auto fork_throws = []() noexcept
    {
      return exec::fork_join(just(), to_throws_on_connect{}, then([]() noexcept {}));
    };
    STATIC_REQUIRE(!STDEXEC::__nothrow_connectable<decltype(child_throws()),
                                                   STDEXEC::__receiver_archetype<env<>>>);
    STATIC_REQUIRE(!STDEXEC::__nothrow_connectable<decltype(fork_throws()),
                                                   STDEXEC::__receiver_archetype<env<>>>);

    // So an algorithm that connects it when another sender completes reports the exception.
    STATIC_REQUIRE(has_exception_ptr_error_v<decltype(let_value(just(), child_throws))>);
    STATIC_REQUIRE(has_exception_ptr_error_v<decltype(let_value(just(), fork_throws))>);
    CHECK_THROWS_AS(sync_wait(let_value(just(), child_throws)), std::logic_error);
    CHECK_THROWS_AS(sync_wait(let_value(just(), fork_throws)), std::logic_error);
  }
#endif  // !STDEXEC_NO_STDCPP_EXCEPTIONS()
}  // namespace
