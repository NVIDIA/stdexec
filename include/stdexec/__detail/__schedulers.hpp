/*
 * Copyright (c) 2021-2024 NVIDIA Corporation
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
#include "__env.hpp"
#include "__senders_core.hpp"
#include "__tag_invoke.hpp"

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.schedule]
  namespace __sched {
    template <class _Scheduler>
    concept __has_schedule_member = requires(_Scheduler&& __sched) {
      static_cast<_Scheduler &&>(__sched).schedule();
    };

    struct schedule_t {
      template <class _Scheduler>
        requires __has_schedule_member<_Scheduler>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      auto operator()(_Scheduler&& __sched) const
        noexcept(noexcept(static_cast<_Scheduler&&>(__sched).schedule()))
          -> decltype(static_cast<_Scheduler&&>(__sched).schedule()) {
        static_assert(
          sender<decltype(static_cast<_Scheduler&&>(__sched).schedule())>,
          "schedule() member functions must return a sender");
        return static_cast<_Scheduler&&>(__sched).schedule();
      }

      template <class _Scheduler>
        requires(!__has_schedule_member<_Scheduler>) && tag_invocable<schedule_t, _Scheduler>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      auto operator()(_Scheduler&& __sched) const
        noexcept(nothrow_tag_invocable<schedule_t, _Scheduler>)
          -> tag_invoke_result_t<schedule_t, _Scheduler> {
        static_assert(sender<tag_invoke_result_t<schedule_t, _Scheduler>>);
        return tag_invoke(*this, static_cast<_Scheduler&&>(__sched));
      }

      static constexpr auto query(forwarding_query_t) noexcept -> bool {
        return false;
      }
    };
  } // namespace __sched

  using __sched::schedule_t;
  inline constexpr schedule_t schedule{};

  struct scheduler_t { };

  template <class _Scheduler>
  concept __has_schedule = requires(_Scheduler&& __sched) {
    { schedule(static_cast<_Scheduler &&>(__sched)) } -> sender;
  };

  template <class _Scheduler>
  concept __sender_has_completion_scheduler = requires(_Scheduler&& __sched) {
    {
      stdexec::__decay_copy(
        get_completion_scheduler<set_value_t>(
          get_env(schedule(static_cast<_Scheduler &&>(__sched)))))
    } -> same_as<__decay_t<_Scheduler>>;
  };

  template <class _Scheduler>
  concept scheduler = __has_schedule<_Scheduler> && __sender_has_completion_scheduler<_Scheduler>
                   && equality_comparable<__decay_t<_Scheduler>>
                   && copy_constructible<__decay_t<_Scheduler>>;

  template <scheduler _Scheduler>
  using schedule_result_t = __call_result_t<schedule_t, _Scheduler>;

  template <class _SchedulerProvider>
  concept __scheduler_provider = requires(const _SchedulerProvider& __sp) {
    { get_scheduler(__sp) } -> scheduler;
  };

  namespace __queries {
    template <class _Env>
    STDEXEC_ATTRIBUTE(always_inline, host, device)
    constexpr void get_scheduler_t::__validate() noexcept {
      static_assert(__nothrow_callable<get_scheduler_t, const _Env&>);
      static_assert(scheduler<__call_result_t<get_scheduler_t, const _Env&>>);
    }

    template <class _Env>
    STDEXEC_ATTRIBUTE(always_inline, host, device)
    constexpr void get_delegation_scheduler_t::__validate() noexcept {
      static_assert(__nothrow_callable<get_delegation_scheduler_t, const _Env&>);
      static_assert(scheduler<__call_result_t<get_delegation_scheduler_t, const _Env&>>);
    }

    template <__completion_tag _Tag>
    template <class _Env>
    STDEXEC_ATTRIBUTE(always_inline, host, device)
    constexpr void get_completion_scheduler_t<_Tag>::__validate() noexcept {
      static_assert(__nothrow_callable<get_completion_scheduler_t<_Tag>, const _Env&>);
      static_assert(scheduler<__call_result_t<get_completion_scheduler_t<_Tag>, const _Env&>>);
    }
  } // namespace __queries
} // namespace stdexec
