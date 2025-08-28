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

#include <chrono>

namespace exec {
  namespace __now {
    using namespace stdexec;

    template <class _Tp>
    concept time_point = regular<_Tp> && totally_ordered<_Tp>
                      && requires(_Tp __tp, const _Tp __ctp, typename _Tp::duration __dur) {
                           { __ctp + __dur } -> same_as<_Tp>;
                           { __dur + __ctp } -> same_as<_Tp>;
                           { __ctp - __dur } -> same_as<_Tp>;
                           { __ctp - __ctp } -> same_as<typename _Tp::duration>;
                           { __tp += __dur } -> same_as<_Tp&>;
                           { __tp -= __dur } -> same_as<_Tp&>;
                         };

    template <class _Scheduler>
    concept __has_now = requires(const _Scheduler& __sched) { __sched.now(); };

    struct now_t {
      template <class _Scheduler>
        requires __has_now<_Scheduler>
      auto operator()(const _Scheduler& __sched) const noexcept(noexcept(__sched.now()))
        -> __decay_t<decltype(__sched.now())> {
        static_assert(time_point<__decay_t<decltype(__sched.now())>>);
        return __sched.now();
      }

      template <class _Scheduler>
        requires(!__has_now<_Scheduler>) && tag_invocable<now_t, const _Scheduler&>
      auto operator()(const _Scheduler& __sched) const
        noexcept(nothrow_tag_invocable<now_t, const _Scheduler&>)
          -> __decay_t<tag_invoke_result_t<now_t, const _Scheduler&>> {
        static_assert(time_point<__decay_t<tag_invoke_result_t<now_t, const _Scheduler&>>>);
        return tag_invoke(now_t{}, __sched);
      }
    };
  } // namespace __now

  using __now::now_t;
  inline constexpr now_t now{};

  template <class _TimedScheduler>
  concept __timed_scheduler = stdexec::scheduler<_TimedScheduler>
                           && requires(_TimedScheduler&& __sched) {
                                now(static_cast<_TimedScheduler &&>(__sched));
                              };

  template <__timed_scheduler _TimedScheduler>
  using time_point_of_t = decltype(now(stdexec::__declval<_TimedScheduler>()));

  template <__timed_scheduler _TimedScheduler>
  using duration_of_t = typename stdexec::__decay_t<time_point_of_t<_TimedScheduler>>::duration;

  namespace __schedule_after {
    struct __schedule_after_base_t;
    struct schedule_after_t;
  } // namespace __schedule_after

  using __schedule_after::__schedule_after_base_t;
  using __schedule_after::schedule_after_t;
  extern const schedule_after_t schedule_after;

  namespace __schedule_at {
    struct __schedule_at_base_t;
    struct schedule_at_t;
  } // namespace __schedule_at

  using __schedule_at::__schedule_at_base_t;
  using __schedule_at::schedule_at_t;
  extern const schedule_at_t schedule_at;

  namespace __schedule_after {
    using namespace stdexec;

    template <class _Scheduler>
    concept __has_schedule_after_member =
      requires(_Scheduler&& __sched, const duration_of_t<_Scheduler>& __duration) {
        __sched.schedule_after(__duration);
      };

    struct __schedule_after_base_t {
      template <class _Scheduler>
        requires __has_schedule_after_member<_Scheduler>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(_Scheduler&& __sched, const duration_of_t<_Scheduler>& __duration) const
        noexcept(noexcept(__sched.schedule_after(__duration)))
          -> decltype(__sched.schedule_after(__duration)) {
        static_assert(sender<decltype(__sched.schedule_after(__duration))>);
        return __sched.schedule_after(__duration);
      }

      template <class _Scheduler>
        requires(!__has_schedule_after_member<_Scheduler>)
             && tag_invocable<schedule_after_t, _Scheduler, const duration_of_t<_Scheduler>&>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(_Scheduler&& __sched, const duration_of_t<_Scheduler>& __duration) const
        noexcept(
          nothrow_tag_invocable<schedule_after_t, _Scheduler, const duration_of_t<_Scheduler>&>)
          -> tag_invoke_result_t<schedule_after_t, _Scheduler, const duration_of_t<_Scheduler>&> {
        static_assert(
          sender<
            tag_invoke_result_t<schedule_after_t, _Scheduler, const duration_of_t<_Scheduler>&>
          >);
        return tag_invoke(schedule_after, static_cast<_Scheduler&&>(__sched), __duration);
      }
    };

    struct schedule_after_t : __schedule_after_base_t {
#if !STDEXEC_CLANG() || STDEXEC_CLANG_VERSION >= 16'00
      using __schedule_after_base_t::operator();
#else
      // clang prior to 16 is not able to find the correct overload in the
      // __schedule_after_base_t class.
      template <class _Scheduler>
        requires __callable<__schedule_after_base_t, _Scheduler, const duration_of_t<_Scheduler>&>
      auto operator()(_Scheduler&& __sched, const duration_of_t<_Scheduler>& __time_point) const
        noexcept(
          __nothrow_callable<__schedule_after_base_t, _Scheduler, const duration_of_t<_Scheduler>&>)
          -> __call_result_t<__schedule_after_base_t, _Scheduler, const duration_of_t<_Scheduler>&> {
        return __schedule_after_base_t{}(static_cast<_Scheduler&&>(__sched), __time_point);
      }
#endif

      template <class _Scheduler>
        requires(!__callable<__schedule_after_base_t, _Scheduler, const duration_of_t<_Scheduler>&>)
             && __callable<__schedule_at_base_t, _Scheduler, const time_point_of_t<_Scheduler>&>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(_Scheduler&& __sched, const duration_of_t<_Scheduler>& __duration)
        const noexcept {
        // TODO get_completion_scheduler<set_value_t>
        return let_value(
          just(),
          [__sched, __duration]() noexcept(
            __nothrow_callable<schedule_at_t, _Scheduler, time_point_of_t<_Scheduler>>&&
              __nothrow_callable<now_t, const _Scheduler&>) {
            return schedule_at(__sched, now(__sched) + __duration);
          });
      }
    };
  } // namespace __schedule_after

  inline constexpr schedule_after_t schedule_after{};

  namespace __schedule_at {
    using namespace stdexec;

    template <class _Scheduler>
    concept __has_schedule_at_member =
      requires(_Scheduler&& __sched, const time_point_of_t<_Scheduler>& __time_point) {
        __sched.schedule_at(__time_point);
      };

    struct __schedule_at_base_t {
      template <class _Scheduler>
        requires __has_schedule_at_member<_Scheduler>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(_Scheduler&& __sched, const time_point_of_t<_Scheduler>& __time_point) const
        noexcept(noexcept(__sched.schedule_at(__time_point)))
          -> decltype(__sched.schedule_at(__time_point)) {
        static_assert(sender<decltype(__sched.schedule_at(__time_point))>);
        return __sched.schedule_at(__time_point);
      }

      template <class _Scheduler>
        requires(!__has_schedule_at_member<_Scheduler>)
             && tag_invocable<schedule_at_t, _Scheduler, const time_point_of_t<_Scheduler>&>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(_Scheduler&& __sched, const time_point_of_t<_Scheduler>& __time_point) const
        noexcept(
          nothrow_tag_invocable<schedule_at_t, _Scheduler, const time_point_of_t<_Scheduler>&>)
          -> tag_invoke_result_t<schedule_at_t, _Scheduler, const time_point_of_t<_Scheduler>&> {
        static_assert(
          sender<
            tag_invoke_result_t<schedule_at_t, _Scheduler, const time_point_of_t<_Scheduler>&>
          >);
        return tag_invoke(schedule_at, static_cast<_Scheduler&&>(__sched), __time_point);
      }
    };

    struct schedule_at_t : __schedule_at_base_t {
#if !STDEXEC_CLANG() || STDEXEC_CLANG_VERSION >= 16'00
      using __schedule_at_base_t::operator();
#else
      // clang prior to 16 is not able to find the correct overload in the
      // __schedule_at_base_t class.
      template <class _Scheduler>
        requires __callable<__schedule_at_base_t, _Scheduler, const time_point_of_t<_Scheduler>&>
      auto operator()(_Scheduler&& __sched, const time_point_of_t<_Scheduler>& __time_point) const
        noexcept(
          __nothrow_callable<__schedule_at_base_t, _Scheduler, const time_point_of_t<_Scheduler>&>)
          -> __call_result_t<__schedule_at_base_t, _Scheduler, const time_point_of_t<_Scheduler>&> {
        return __schedule_at_base_t{}(static_cast<_Scheduler&&>(__sched), __time_point);
      }
#endif

      template <class _Scheduler>
        requires(!__callable<__schedule_at_base_t, _Scheduler, const time_point_of_t<_Scheduler>&>)
             && __callable<__schedule_after_base_t, _Scheduler, const duration_of_t<_Scheduler>&>
      auto operator()(_Scheduler&& __sched, const time_point_of_t<_Scheduler>& __time_point) const
        noexcept(noexcept(schedule_after(__sched, __time_point - now(__sched)))) {
        // TODO get_completion_scheduler<set_value_t>
        return let_value(
          just(),
          [__sched, __time_point]() noexcept(
            noexcept(schedule_after(__sched, __time_point - now(__sched)))) {
            return schedule_after(__sched, __time_point - now(__sched));
          });
      }
    };
  } // namespace __schedule_at

  inline constexpr schedule_at_t schedule_at{};

  template <class _Scheduler>
  concept __has_schedule_after =
    requires(_Scheduler&& __sched, const duration_of_t<_Scheduler>& __duration) {
      { schedule_after(static_cast<_Scheduler &&>(__sched), __duration) } -> stdexec::sender;
    };

  template <class _Scheduler>
  concept __has_schedule_at =
    requires(_Scheduler&& __sched, const time_point_of_t<_Scheduler>& __time_point) {
      { schedule_at(static_cast<_Scheduler &&>(__sched), __time_point) } -> stdexec::sender;
    };

  template <class _Scheduler, class _Clock = std::chrono::system_clock>
  concept timed_scheduler = __timed_scheduler<_Scheduler> && __has_schedule_after<_Scheduler>
                         && __has_schedule_at<_Scheduler>;

  template <timed_scheduler _Scheduler>
  using schedule_after_result_t = stdexec::__call_result_t<schedule_after_t, _Scheduler>;

  template <timed_scheduler _Scheduler>
  using schedule_at_result_t = stdexec::__call_result_t<schedule_at_t, _Scheduler>;
} // namespace exec
