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
    concept time_point =      //
      regular<_Tp> &&         //
      totally_ordered<_Tp> && //
      requires(_Tp __tp, const _Tp __ctp, typename _Tp::duration __dur) {
        { __ctp + __dur } -> same_as<_Tp>;
        { __dur + __ctp } -> same_as<_Tp>;
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

  template <class _TimedScheduler>
  concept __timed_scheduler =              //
    stdexec::scheduler<_TimedScheduler> && //
    requires(_TimedScheduler&& __sched) {  //
      now((_TimedScheduler&&) __sched);
    };

  template <__timed_scheduler _TimedScheduler>
  using time_point_of_t = //
    decltype(now(stdexec::__declval<_TimedScheduler>()));

  template <__timed_scheduler _TimedScheduler>
  using duration_of_t = //
    typename stdexec::__decay_t<time_point_of_t<_TimedScheduler>>::duration;

  namespace __schedule_after {
    struct schedule_after_t;
  }

  using __schedule_after::schedule_after_t;
  extern const schedule_after_t schedule_after;

  namespace __schedule_at {
    struct schedule_at_t;
  }

  using __schedule_at::schedule_at_t;
  extern const schedule_at_t schedule_at;

  template <class _TimedScheduler>
  concept __has_custom_schedule_after =   //
    __timed_scheduler<_TimedScheduler> && //
    stdexec::tag_invocable<schedule_after_t, _TimedScheduler, const duration_of_t<_TimedScheduler>&>;

  template <__has_custom_schedule_after _TimedScheduler>
  using __custom_schedule_after_sender_t = //
    stdexec::
      tag_invoke_result_t<schedule_after_t, _TimedScheduler, const duration_of_t<_TimedScheduler>&>;

  template <class _TimedScheduler>
  concept __has_custom_schedule_at =      //
    __timed_scheduler<_TimedScheduler> && //
    stdexec::tag_invocable<schedule_at_t, _TimedScheduler, const time_point_of_t<_TimedScheduler>&>;

  template <__has_custom_schedule_at _TimedScheduler>
  using __custom_schedule_at_sender_t = //
    stdexec::
      tag_invoke_result_t<schedule_at_t, _TimedScheduler, const time_point_of_t<_TimedScheduler>&>;

  namespace __schedule_after {
    using namespace stdexec;

    struct schedule_after_t {
      template <class _Scheduler>
        requires __has_custom_schedule_after<_Scheduler>
      auto operator()(_Scheduler&& __sched, const duration_of_t<_Scheduler>& __duration) const
        noexcept(noexcept(tag_invoke(schedule_after, (_Scheduler&&) __sched, __duration)))
          -> __custom_schedule_after_sender_t<_Scheduler> {
        static_assert(sender<__custom_schedule_after_sender_t<_Scheduler>>);
        return tag_invoke(schedule_after, (_Scheduler&&) __sched, __duration);
      }

      template <class _Scheduler>
        requires(!__has_custom_schedule_after<_Scheduler>) && //
                __has_custom_schedule_at<_Scheduler>
      auto operator()(_Scheduler&& __sched, const duration_of_t<_Scheduler>& __duration) const
        noexcept(
          noexcept(tag_invoke(schedule_at, (_Scheduler&&) __sched, now(__sched) + __duration)))
          -> __custom_schedule_at_sender_t<_Scheduler> {
        static_assert(sender<__custom_schedule_at_sender_t<_Scheduler>>);
        auto __time_point = now(__sched) + __duration;
        return tag_invoke(schedule_at, (_Scheduler&&) __sched, __time_point);
      }
    };
  }

  inline constexpr schedule_after_t schedule_after{};

  namespace __schedule_at {
    using namespace stdexec;

    struct schedule_at_t {
      template <class _Scheduler>
        requires __has_custom_schedule_at<_Scheduler>
      auto operator()(_Scheduler&& __sched, const time_point_of_t<_Scheduler>& __time_point) const
        noexcept(noexcept(tag_invoke(schedule_at, (_Scheduler&&) __sched, __time_point)))
          -> __custom_schedule_at_sender_t<_Scheduler> {
        static_assert(sender<__custom_schedule_at_sender_t<_Scheduler>>);
        return tag_invoke(schedule_at, (_Scheduler&&) __sched, __time_point);
      }

      template <class _Scheduler>
        requires(!__has_custom_schedule_at<_Scheduler>) && //
                __has_custom_schedule_after<_Scheduler>
      auto operator()(_Scheduler&& __sched, const duration_of_t<_Scheduler>& __time_point) const
        noexcept(
          noexcept(tag_invoke(schedule_after, (_Scheduler&&) __sched, __time_point - now(__sched))))
          -> __custom_schedule_after_sender_t<_Scheduler> {
        static_assert(sender<__custom_schedule_after_sender_t<_Scheduler>>);
        auto __duration = __time_point - now(__sched);
        return tag_invoke(schedule_after, (_Scheduler&&) __sched, __duration);
      }
    };
  }

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

  template <class _Scheduler, class _Clock = std::chrono::system_clock>
  concept timed_scheduler =             //
    __timed_scheduler<_Scheduler> &&    //
    __has_schedule_after<_Scheduler> && //
    __has_schedule_at<_Scheduler>;

  template <timed_scheduler _Scheduler>
  using schedule_after_result_t = //
    stdexec::__call_result_t<schedule_after_t, _Scheduler>;

  template <timed_scheduler _Scheduler>
  using schedule_at_result_t = //
    stdexec::__call_result_t<schedule_at_t, _Scheduler>;
}
