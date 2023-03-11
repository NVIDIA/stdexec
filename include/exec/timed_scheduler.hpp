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
    struct schedule_after_t {
      template <class _Scheduler, class _Duration>
        requires stdexec::tag_invocable<schedule_after_t, _Scheduler, _Duration>
              && stdexec::sender<
                   stdexec::tag_invoke_result_t<schedule_after_t, _Scheduler, _Duration>>
      auto operator()(_Scheduler&& __sched, _Duration&& __duration) const
        noexcept(stdexec::nothrow_tag_invocable<schedule_after_t, _Scheduler, _Duration>)
          -> stdexec::tag_invoke_result_t<schedule_after_t, _Scheduler, _Duration> {
        return tag_invoke(schedule_after_t{}, (_Scheduler&&) __sched, (_Duration&&) __duration);
      }
    };
  }

  using __schedule_after::schedule_after_t;
  inline constexpr schedule_after_t schedule_after{};

  template <class _Scheduler, class _Duration>
  concept __has_schedule_after = requires(_Scheduler&& __sched, _Duration&& __duration) {
    { schedule_after((_Scheduler&&) __sched, __duration) } -> stdexec::sender;
  };

  // TODO: Add more requirements such as __has_schedule_at or __has_now
  template <class _Scheduler>
  concept timed_scheduler =
    stdexec::scheduler<_Scheduler> && __has_schedule_after<_Scheduler, std::chrono::seconds>;

  template <timed_scheduler _Scheduler>
  using schedule_after_result_t = stdexec::__call_result_t<schedule_after_t, _Scheduler>;
}