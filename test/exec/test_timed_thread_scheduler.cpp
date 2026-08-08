/*
 * Copyright (c) 2024 Maikel Nadolski
 * Copyright (c) 2024 NVIDIA Corporation
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

#include <exec/timed_thread_scheduler.hpp>

#include "test_common/catch2.hpp"

#include <exec/async_scope.hpp>
#include <exec/when_any.hpp>

#include <concepts>
#include <utility>

// Avoid a TSAN bug in GCC 11 and earlier
#if STDEXEC_GCC() && STDEXEC_GCC_VERSION < 1200 && defined(__SANITIZE_THREAD__)
// nothing
#else
namespace
{
  struct schedule_after_only_scheduler
  {
    using scheduler_concept = STDEXEC::scheduler_tag;
    using time_point       = std::chrono::steady_clock::time_point;
    using duration         = time_point::duration;

    struct sender;

    constexpr auto operator==(schedule_after_only_scheduler const &) const noexcept -> bool =
      default;

    [[nodiscard]]
    auto now() const noexcept -> time_point
    {
      return std::chrono::steady_clock::now();
    }

    [[nodiscard]]
    auto schedule() const noexcept -> sender;

    [[nodiscard]]
    auto schedule_after(duration) const noexcept -> sender;
  };

  struct schedule_after_only_scheduler::sender
  {
    using sender_concept = STDEXEC::sender_tag;
    using completion_signatures =
      STDEXEC::completion_signatures<STDEXEC::set_value_t()>;

    struct attrs
    {
      [[nodiscard]]
      auto query(STDEXEC::get_completion_scheduler_t<STDEXEC::set_value_t>) const noexcept
        -> schedule_after_only_scheduler
      {
        return {};
      }
    };

    template <class Receiver>
    struct operation
    {
      void start() & noexcept
      {
        STDEXEC::set_value(static_cast<Receiver &&>(receiver_));
      }

      Receiver receiver_;
    };

    template <class Receiver>
    auto connect(Receiver receiver) const -> operation<Receiver>
    {
      return {static_cast<Receiver &&>(receiver)};
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> attrs
    {
      return {};
    }
  };

  auto schedule_after_only_scheduler::schedule() const noexcept -> sender
  {
    return {};
  }

  auto schedule_after_only_scheduler::schedule_after(duration) const noexcept -> sender
  {
    return {};
  }

  template <class CompletionScheduler>
  struct sender_with_completion_scheduler
  {
    using sender_concept = STDEXEC::sender_tag;
    using completion_signatures =
      STDEXEC::completion_signatures<STDEXEC::set_value_t()>;

    struct attrs
    {
      [[nodiscard]]
      auto query(STDEXEC::get_completion_scheduler_t<STDEXEC::set_value_t>) const noexcept
        -> CompletionScheduler
      {
        return {};
      }
    };

    template <class Receiver>
    struct operation
    {
      void start() & noexcept
      {
        STDEXEC::set_value(static_cast<Receiver &&>(receiver_));
      }

      Receiver receiver_;
    };

    template <class Receiver>
    auto connect(Receiver receiver) const -> operation<Receiver>
    {
      return {static_cast<Receiver &&>(receiver)};
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> attrs
    {
      return {};
    }
  };

  struct schedule_after_delegating_scheduler
  {
    using scheduler_concept = STDEXEC::scheduler_tag;
    using time_point       = std::chrono::steady_clock::time_point;
    using duration         = time_point::duration;
    using sender           = sender_with_completion_scheduler<schedule_after_only_scheduler>;

    constexpr auto operator==(schedule_after_delegating_scheduler const &) const noexcept -> bool =
      default;

    [[nodiscard]]
    auto now() const noexcept -> time_point
    {
      return std::chrono::steady_clock::now();
    }

    [[nodiscard]]
    auto schedule() const noexcept -> sender
    {
      return {};
    }

    [[nodiscard]]
    auto schedule_after(duration) const noexcept -> sender
    {
      return {};
    }
  };

  struct schedule_at_delegating_scheduler
  {
    using scheduler_concept = STDEXEC::scheduler_tag;
    using time_point       = std::chrono::steady_clock::time_point;
    using duration         = time_point::duration;
    using sender           = sender_with_completion_scheduler<schedule_after_only_scheduler>;

    constexpr auto operator==(schedule_at_delegating_scheduler const &) const noexcept -> bool =
      default;

    [[nodiscard]]
    auto now() const noexcept -> time_point
    {
      return std::chrono::steady_clock::now();
    }

    [[nodiscard]]
    auto schedule() const noexcept -> sender
    {
      return {};
    }

    [[nodiscard]]
    auto schedule_at(time_point) const noexcept -> sender
    {
      return {};
    }
  };

  TEST_CASE("timed_thread_scheduler - unused context",
            "[types][timed_thread_scheduler][schedulers]")
  {
    using scheduler_t = exec::timed_thread_scheduler;

    static_assert(exec::__timed_scheduler<scheduler_t>);
    static_assert(std::same_as<exec::schedule_after_result_t<scheduler_t>,
                               decltype(exec::schedule_after(
                                 std::declval<scheduler_t>(),
                                 std::declval<exec::duration_of_t<scheduler_t> const &>()))>);
    static_assert(std::same_as<exec::schedule_at_result_t<scheduler_t>,
                               decltype(exec::schedule_at(
                                 std::declval<scheduler_t>(),
                                 std::declval<exec::time_point_of_t<scheduler_t> const &>()))>);
    exec::timed_thread_context context;
  }

  TEST_CASE("timed_thread_scheduler - now", "[timed_thread_scheduler][now]")
  {
    exec::timed_thread_context   context;
    exec::timed_thread_scheduler scheduler = context.get_scheduler();
    auto                         tp        = exec::now(scheduler);
    REQUIRE(tp.time_since_epoch().count() > 0);
  }

  TEST_CASE("timed_thread_scheduler - schedule", "[timed_thread_scheduler][schedule]")
  {
    exec::timed_thread_context   context;
    exec::timed_thread_scheduler scheduler = context.get_scheduler();
    CHECK(STDEXEC::sync_wait(STDEXEC::schedule(scheduler)));
  }

  TEST_CASE("timed_thread_scheduler - schedule_at", "[timed_thread_scheduler][schedule_at]")
  {
    exec::timed_thread_context   context;
    exec::timed_thread_scheduler scheduler = context.get_scheduler();
    auto                         now       = exec::now(scheduler);
    auto                         tp        = now + std::chrono::milliseconds(10);
    CHECK(STDEXEC::sync_wait(exec::schedule_at(scheduler, tp)));
  }

  TEST_CASE("timed_thread_scheduler - schedule_after", "[timed_thread_scheduler][schedule_at]")
  {
    exec::timed_thread_context   context;
    exec::timed_thread_scheduler scheduler = context.get_scheduler();
    auto                         duration  = std::chrono::milliseconds(10);
    CHECK(STDEXEC::sync_wait(exec::schedule_after(scheduler, duration)));
  }

  TEST_CASE("timed scheduler fallbacks advertise completion schedulers",
            "[timed_scheduler][completion_scheduler]")
  {
    static_assert(exec::timed_scheduler<schedule_after_only_scheduler>);

    exec::timed_thread_context   context;
    exec::timed_thread_scheduler timed_scheduler = context.get_scheduler();
    auto                         after_sender = exec::schedule_after(
      timed_scheduler, std::chrono::milliseconds(10));

    CHECK(STDEXEC::get_completion_scheduler<STDEXEC::set_value_t>(
            STDEXEC::get_env(after_sender))
          == timed_scheduler);

    schedule_after_only_scheduler after_only_scheduler;
    auto at_sender = exec::schedule_at(after_only_scheduler, exec::now(after_only_scheduler));

    CHECK(STDEXEC::get_completion_scheduler<STDEXEC::set_value_t>(STDEXEC::get_env(at_sender))
          == after_only_scheduler);
    CHECK(STDEXEC::sync_wait(std::move(at_sender)));
  }

  TEST_CASE("timed scheduler fallbacks do not invent completion schedulers",
            "[timed_scheduler][completion_scheduler]")
  {
    using completion_scheduler_query =
      STDEXEC::get_completion_scheduler_t<STDEXEC::set_value_t>;

    static_assert(exec::timed_scheduler<schedule_after_delegating_scheduler>);
    using at_sender_t = decltype(exec::schedule_at(
      std::declval<schedule_after_delegating_scheduler>(),
      std::declval<exec::time_point_of_t<schedule_after_delegating_scheduler> const &>()));
    static_assert(!STDEXEC::__callable<completion_scheduler_query,
                                        STDEXEC::env_of_t<at_sender_t>>);

    static_assert(exec::timed_scheduler<schedule_at_delegating_scheduler>);
    using after_sender_t = decltype(exec::schedule_after(
      std::declval<schedule_at_delegating_scheduler>(),
      std::declval<exec::duration_of_t<schedule_at_delegating_scheduler> const &>()));
    static_assert(!STDEXEC::__callable<completion_scheduler_query,
                                        STDEXEC::env_of_t<after_sender_t>>);

    schedule_after_delegating_scheduler after_scheduler;
    CHECK(STDEXEC::sync_wait(exec::schedule_at(after_scheduler, exec::now(after_scheduler))));

    schedule_at_delegating_scheduler at_scheduler;
    CHECK(STDEXEC::sync_wait(exec::schedule_after(at_scheduler, std::chrono::milliseconds(0))));
  }

  TEST_CASE("timed_thread_scheduler - when_any", "[timed_thread_scheduler][when_any]")
  {
    exec::timed_thread_context   context;
    exec::timed_thread_scheduler scheduler = context.get_scheduler();
    auto                         duration1 = std::chrono::milliseconds(10);
    auto                         duration2 = std::chrono::seconds(5);
    auto                         shorter = exec::when_any(exec::schedule_after(scheduler, duration1)
                                    | STDEXEC::then([] { return 1; }),
                                  exec::schedule_after(scheduler, duration2)
                                    | STDEXEC::then([] { return 2; }));
    auto                         t0      = std::chrono::steady_clock::now();
    auto [n]                             = STDEXEC::sync_wait(std::move(shorter)).value();
    auto t1                              = std::chrono::steady_clock::now();
    auto duration                        = t1 - t0;
    CHECK(duration1 <= duration);
    CHECK(n == 1);
  }

  TEST_CASE("timed_thread_scheduler - more when_any", "[timed_thread_scheduler][when_any]")
  {
    exec::timed_thread_context   context;
    exec::timed_thread_scheduler scheduler = context.get_scheduler();
    auto                         duration1 = std::chrono::milliseconds(10);
    auto                         duration2 = std::chrono::seconds(5);
    auto                         shorter =
      exec::when_any(exec::schedule_after(scheduler, duration1) | STDEXEC::then([] { return 1; }),
                     exec::schedule_after(scheduler, duration2) | STDEXEC::then([] { return 2; }),
                     exec::schedule_after(scheduler, duration2) | STDEXEC::then([] { return 3; }),
                     exec::schedule_after(scheduler, duration2) | STDEXEC::then([] { return 4; }),
                     exec::schedule_after(scheduler, duration2) | STDEXEC::then([] { return 5; }),
                     exec::schedule_after(scheduler, duration2) | STDEXEC::then([] { return 6; }),
                     exec::schedule_after(scheduler, duration2) | STDEXEC::then([] { return 7; }));
    auto t0       = std::chrono::steady_clock::now();
    auto [n]      = STDEXEC::sync_wait(std::move(shorter)).value();
    auto t1       = std::chrono::steady_clock::now();
    auto duration = t1 - t0;
    CHECK(duration1 <= duration);
    CHECK(n == 1);
  }

  TEST_CASE("timed_thread_scheduler - many timers with async scope",
            "[timed_thread_scheduler][async_scope]")
  {
    exec::timed_thread_context   context;
    exec::timed_thread_scheduler scheduler = context.get_scheduler();
    exec::async_scope            scope;
    int                          counter  = 0;
    int                          ntimers  = 1'000;
    auto                         now      = exec::now(scheduler);
    auto                         deadline = now + std::chrono::milliseconds(100);
    auto                         t0       = std::chrono::steady_clock::now();
    for (int i = 0; i < ntimers; ++i)
    {
      scope.spawn(exec::schedule_at(scheduler, deadline)
                  | STDEXEC::then([&counter] { ++counter; }));
    }
    CHECK(STDEXEC::sync_wait(scope.on_empty()));
    auto t1 = std::chrono::steady_clock::now();
    CHECK(counter == ntimers);
    auto duration = t1 - t0;
    CHECK(duration > std::chrono::milliseconds(100));
  }
}  // namespace
#endif
