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
  namespace __now {
    using namespace stdexec;

    template <class _Tp>
    concept time_point = //
      regular<_Tp> &&    //
      totally_ordered<_Tp> && requires(_Tp __tp, const _Tp __ctp, typename _Tp::duration __dur) {
        { __ctp + __dur } -> same_as<_Tp>;
        { __ctp - __dur } -> same_as<_Tp>;
        { __ctp - __ctp } -> same_as<typename _Tp::duration>;
        { __tp += __dur } -> same_as<_Tp&>;
        { __tp -= __dur } -> same_as<_Tp&>;
      };

    struct now_t {
      template <class _Scheduler>
        requires tag_invocable<now_t, const _Scheduler&>
      auto operator()(const _Scheduler& __sched) const
        noexcept(nothrow_tag_invocable<now_t, const _Scheduler&>)
          -> tag_invoke_result_t<now_t, const _Scheduler&> {
        static_assert(time_point<tag_invoke_result_t<now_t, const _Scheduler&>>);
        return tag_invoke(now_t{}, __sched);
      }
    };
  }

  using __now::now_t;
  inline constexpr now_t now{};

  template <class _TimeScheduler>
  concept __time_scheduler = //
    stdexec::scheduler<_TimeScheduler>
    && requires(_TimeScheduler&& __sched) { now((_TimeScheduler&&) __sched); };

  template <__time_scheduler _TimeScheduler>
  using time_point_of_t = //
    decltype(now(stdexec::__declval<_TimeScheduler>()));

  template <__time_scheduler _TimeScheduler>
  using duration_of_t = //
    typename time_point_of_t<_TimeScheduler>::duration;

  namespace __schedule_after {
    using namespace stdexec;

    struct schedule_after_t {
      template <class _Scheduler>
        requires tag_invocable<schedule_after_t, _Scheduler, const duration_of_t<_Scheduler>&>
              && sender< tag_invoke_result_t<
                schedule_after_t,
                _Scheduler,
                const duration_of_t<_Scheduler>&>>
      auto operator()(_Scheduler&& __sched, const duration_of_t<_Scheduler>& __duration) const
        noexcept(
          nothrow_tag_invocable<schedule_after_t, _Scheduler, const duration_of_t<_Scheduler>&>)
          -> tag_invoke_result_t<schedule_after_t, _Scheduler, const duration_of_t<_Scheduler>&> {
        return tag_invoke(schedule_after_t{}, (_Scheduler&&) __sched, __duration);
      }
    };
  }

  using __schedule_after::schedule_after_t;
  inline constexpr schedule_after_t schedule_after{};

  namespace __schedule_at {
    using namespace stdexec;

    struct schedule_at_t {
      template <class _Scheduler>
        requires tag_invocable<schedule_at_t, _Scheduler, const time_point_of_t<_Scheduler>&>
              && sender<tag_invoke_result_t<
                schedule_at_t,
                _Scheduler,
                const time_point_of_t<_Scheduler>&>>
      auto operator()(_Scheduler&& __sched, const time_point_of_t<_Scheduler>& __time_point) const
        noexcept(
          nothrow_tag_invocable< schedule_at_t, _Scheduler, const time_point_of_t<_Scheduler>&>)
          -> tag_invoke_result_t< schedule_at_t, _Scheduler, const time_point_of_t<_Scheduler>&> {
        return tag_invoke(schedule_at_t{}, (_Scheduler&&) __sched, __time_point);
      }
    };
  }

  using __schedule_at::schedule_at_t;
  inline constexpr schedule_at_t schedule_at{};

  template <class _Scheduler>
  concept __has_schedule_after = //
    requires(_Scheduler&& __sched, const duration_of_t<_Scheduler>& __duration) {
      { schedule_after((_Scheduler&&) __sched, __duration) } -> stdexec::sender;
    };

  template <class _Scheduler>
  concept __has_schedule_at = //
    requires(_Scheduler&& __sched, const time_point_of_t<_Scheduler>& __time_point) {
      { schedule_at((_Scheduler&&) __sched, __time_point) } -> stdexec::sender;
    };

  // TODO: Add more requirements such as __has_schedule_at or __has_now
  template <class _Scheduler, class _Clock = std::chrono::system_clock>
  concept timed_scheduler =             //
    __time_scheduler<_Scheduler> &&     //
    __has_schedule_after<_Scheduler> && //
    __has_schedule_at<_Scheduler>;

  template <timed_scheduler _Scheduler>
  using schedule_after_result_t = //
    stdexec::__call_result_t<schedule_after_t, _Scheduler>;
}
