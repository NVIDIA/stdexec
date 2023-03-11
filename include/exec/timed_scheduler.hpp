/*
 * Copyright (c) 2023 Maikel Nadolski
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
#pragma once

#include "../stdexec/execution.hpp"

namespace exec {
  namespace __schedule_after {
    using namespace stdexec;
    using std::chrono::duration;

    struct schedule_after_t {
      template <class _Scheduler, class _Rep, class _Period>
        requires tag_invocable<schedule_after_t, _Scheduler, const duration<_Rep, _Period>&>
              && sender<
                   tag_invoke_result_t<schedule_after_t, _Scheduler, const duration<_Rep, _Period>&>>
      auto operator()(_Scheduler&& __sched, const duration<_Rep, _Period>& __duration) const
        noexcept(
          nothrow_tag_invocable<schedule_after_t, _Scheduler, const duration<_Rep, _Period>&>)
          -> tag_invoke_result_t<schedule_after_t, _Scheduler, const duration<_Rep, _Period>&> {
        return tag_invoke(schedule_after_t{}, (_Scheduler&&) __sched, __duration);
      }
    };
  }

  using __schedule_after::schedule_after_t;
  inline constexpr schedule_after_t schedule_after{};

  namespace __schedule_at {
    using namespace stdexec;
    using std::chrono::time_point;

    struct schedule_at_t {
      template <class _Scheduler, class _Clock, class _Duration>
        requires tag_invocable<schedule_at_t, _Scheduler, const time_point<_Clock, _Duration>&>
              && sender<tag_invoke_result_t<
                schedule_at_t,
                _Scheduler,
                const time_point<_Clock, _Duration>&>>
      auto operator()(_Scheduler&& __sched, const time_point<_Clock, _Duration>& __time_point) const
        noexcept(
          nothrow_tag_invocable< schedule_at_t, _Scheduler, const time_point<_Clock, _Duration>&>)
          -> tag_invoke_result_t< schedule_at_t, _Scheduler, const time_point<_Clock, _Duration>&> {
        return tag_invoke(schedule_at_t{}, (_Scheduler&&) __sched, __time_point);
      }
    };
  }

  using __schedule_at::schedule_at_t;
  inline constexpr schedule_at_t schedule_at{};

  template <class _Scheduler, class _Duration>
  concept __has_schedule_after = requires(_Scheduler&& __sched, const _Duration& __duration) {
    { schedule_after((_Scheduler&&) __sched, __duration) } -> stdexec::sender;
  };

  template <class _Scheduler, class _TimePoint>
  concept __has_schedule_at = requires(_Scheduler&& __sched, const _TimePoint& __time_point) {
    { schedule_after((_Scheduler&&) __sched, __time_point) } -> stdexec::sender;
  };

  // TODO: Add more requirements such as __has_schedule_at or __has_now
  template <class _Scheduler, class _Clock = std::chrono::system_clock>
  concept timed_scheduler = std::chrono::is_clock_v<_Clock> && stdexec::scheduler<_Scheduler>
                         && __has_schedule_after<_Scheduler, typename _Clock::duration>
                         && __has_schedule_at<_Scheduler, typename _Clock::time_point>;

  template <timed_scheduler _Scheduler>
  using schedule_after_result_t = stdexec::__call_result_t<schedule_after_t, _Scheduler>;
}