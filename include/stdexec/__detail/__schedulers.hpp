/*
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "__concepts.hpp"
#include "__cpo.hpp"
#include "__env.hpp"
#include "__senders.hpp"
#include "__tag_invoke.hpp"

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.schedule]
  namespace __sched {
    struct schedule_t {
      template <class _Scheduler>
        requires tag_invocable<schedule_t, _Scheduler>
      STDEXEC_ATTRIBUTE((host, device))
      auto
        operator()(_Scheduler&& __sched) const
        noexcept(nothrow_tag_invocable<schedule_t, _Scheduler>) {
        static_assert(sender<tag_invoke_result_t<schedule_t, _Scheduler>>);
        return tag_invoke(schedule_t{}, static_cast<_Scheduler&&>(__sched));
      }

      constexpr STDEXEC_MEMFN_DECL(auto forwarding_query)(this schedule_t) -> bool {
        return false;
      }
    };
  } // namespace __sched

  using __sched::schedule_t;
  inline constexpr schedule_t schedule{};

  template <class _Scheduler>
  concept __has_schedule = //
    requires(_Scheduler&& __sched) {
      { schedule(static_cast<_Scheduler&&>(__sched)) } -> sender;
    };

  namespace __detail {
    using _GetComplSched = get_completion_scheduler_t<set_value_t>;
  } // namespace __detail

  template <class _Scheduler>
  concept __sender_has_completion_scheduler = requires(_Scheduler&& __sched) {
    {
      tag_invoke(__detail::_GetComplSched(), get_env(schedule(static_cast<_Scheduler&&>(__sched))))
    } -> same_as<__decay_t<_Scheduler>>;
  };

  template <class _Scheduler>
  concept scheduler =                                //
    __has_schedule<_Scheduler>                       //
    && __sender_has_completion_scheduler<_Scheduler> //
    && equality_comparable<__decay_t<_Scheduler>>    //
    && copy_constructible<__decay_t<_Scheduler>>;

  template <scheduler _Scheduler>
  using schedule_result_t = __call_result_t<schedule_t, _Scheduler>;

  template <class _SchedulerProvider>
  concept __scheduler_provider = //
    requires(const _SchedulerProvider& __sp) {
      { get_scheduler(__sp) } -> scheduler;
    };
} // namespace stdexec
