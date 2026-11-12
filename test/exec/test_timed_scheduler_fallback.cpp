/*
 * Copyright (c) 2026 NVIDIA Corporation
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

// Regression tests for #2165: the `schedule_after`/`schedule_at` fallbacks must
// probe the native sender through the same const-qualification the deferred
// invocation uses. The fallback's let_value lambda captures the scheduler by
// copy and is not mutable, so it invokes the native CPO with a `const S&`. The
// probe in the fallback deduced the native sender type through a non-const
// lvalue instead, so a scheduler whose const-lvalue and non-const-lvalue
// overloads return differing senders advertises a completion scheduler that
// its deferred invocation never resolves to.

#include <exec/timed_scheduler.hpp>

#include <test_common/catch2.hpp>

#include <chrono>
#include <utility>

namespace
{
  // A scheduler with no `schedule_after` member and lvalue-ref-qualified
  // `schedule_at` members. `schedule_after` on it goes through the fallback
  // (into `schedule_at`), and the two lvalue ref-qualified `schedule_at`
  // overloads return senders that differ in what their attributes advertise.
  struct ref_qualified_scheduler
  {
    using scheduler_concept = STDEXEC::scheduler_tag;
    using time_point        = std::chrono::steady_clock::time_point;
    using duration          = time_point::duration;

    struct sender;

    struct const_lvalue_sender;

    constexpr auto operator==(ref_qualified_scheduler const &) const noexcept -> bool = default;

    [[nodiscard]]
    auto now() const noexcept -> time_point
    {
      return {};
    }

    [[nodiscard]]
    auto schedule() const noexcept -> sender;

    // The overload the probe reaches: a non-const lvalue. Its sender does not
    // advertise a completion scheduler.
    [[nodiscard]]
    auto schedule_at(time_point) & noexcept -> sender;

    // The overload the fallback's deferred invocation reaches: the lambda
    // holds the scheduler as a `const S&`. Its sender advertises this
    // scheduler as the completion scheduler.
    [[nodiscard]]
    auto schedule_at(time_point) const & noexcept -> const_lvalue_sender;
  };

  // The native sender resolved through the non-const-lvalue overload. Its
  // attributes do not advertise a completion scheduler.
  struct ref_qualified_scheduler::sender
  {
    using sender_concept        = STDEXEC::sender_tag;
    using completion_signatures = STDEXEC::completion_signatures<STDEXEC::set_value_t()>;

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
  };

  // The native sender resolved through the const-lvalue overload. Its
  // attributes advertise this scheduler as the completion scheduler.
  struct ref_qualified_scheduler::const_lvalue_sender
  {
    using sender_concept        = STDEXEC::sender_tag;
    using completion_signatures = STDEXEC::completion_signatures<STDEXEC::set_value_t()>;

    struct attrs
    {
      [[nodiscard]]
      auto query(STDEXEC::get_completion_scheduler_t<STDEXEC::set_value_t>) const noexcept
        -> ref_qualified_scheduler
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

  auto ref_qualified_scheduler::schedule() const noexcept -> sender
  {
    return {};
  }

  auto ref_qualified_scheduler::schedule_at(time_point) & noexcept -> sender
  {
    return {};
  }

  auto ref_qualified_scheduler::schedule_at(time_point) const & noexcept -> const_lvalue_sender
  {
    return {};
  }

  static_assert(exec::timed_scheduler<ref_qualified_scheduler>);
}  // namespace

TEST_CASE("timed scheduler fallback probes through const-lvalue scheduler",
          "[timed_scheduler][completion_scheduler]")
{
  using completion_scheduler_query = STDEXEC::get_completion_scheduler_t<STDEXEC::set_value_t>;

  // Passing the scheduler as a prvalue: the fallback must probe the native
  // sender the same way its deferred lambda will invoke it, i.e. through a
  // `const S&`.
  auto after    = exec::schedule_after(ref_qualified_scheduler{}, std::chrono::milliseconds(10));
  using after_t = decltype(after);
  static_assert(STDEXEC::__callable<completion_scheduler_query, STDEXEC::env_of_t<after_t>>);

  // Same behavior regardless of the value category at the call site.
  ref_qualified_scheduler sched;
  auto after_from_lvalue    = exec::schedule_after(sched, std::chrono::milliseconds(10));
  using after_from_lvalue_t = decltype(after_from_lvalue);
  static_assert(
    STDEXEC::__callable<completion_scheduler_query, STDEXEC::env_of_t<after_from_lvalue_t>>);

  CHECK(STDEXEC::get_completion_scheduler<STDEXEC::set_value_t>(STDEXEC::get_env(after))
        == ref_qualified_scheduler{});
  CHECK(STDEXEC::get_completion_scheduler<STDEXEC::set_value_t>(STDEXEC::get_env(after_from_lvalue))
        == sched);
}
