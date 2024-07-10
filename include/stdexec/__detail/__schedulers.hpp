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
#include "__cpo.hpp"
#include "__env.hpp"
#include "__senders.hpp"
#include "__tag_invoke.hpp"

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.schedule]
  namespace __sched {
    struct schedule_t {
      template <__same_as<schedule_t> _Self, class _Scheduler>
      STDEXEC_ATTRIBUTE((host, device, always_inline))
      friend auto
        tag_invoke(_Self, _Scheduler&& __sched) //
        noexcept(noexcept(static_cast<_Scheduler&&>(__sched).schedule()))
          -> decltype(static_cast<_Scheduler&&>(__sched).schedule()) {
        static_assert(
          sender<decltype(static_cast<_Scheduler&&>(__sched).schedule())>,
          "schedule() member functions must return a sender");
        return static_cast<_Scheduler&&>(__sched).schedule();
      }

      template <class _Scheduler>
        requires tag_invocable<schedule_t, _Scheduler>
      STDEXEC_ATTRIBUTE((host, device))
      auto
        operator()(_Scheduler&& __sched) const
        noexcept(nothrow_tag_invocable<schedule_t, _Scheduler>) {
        static_assert(sender<tag_invoke_result_t<schedule_t, _Scheduler>>);
        return tag_invoke(schedule_t{}, static_cast<_Scheduler&&>(__sched));
      }

      static constexpr auto query(forwarding_query_t) noexcept -> bool {
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

  template <class _Scheduler>
  concept __sender_has_completion_scheduler = requires(_Scheduler&& __sched) {
    {
      get_completion_scheduler<set_value_t>(get_env(schedule(static_cast<_Scheduler&&>(__sched))))
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

  namespace __queries {
    template <class _Env>
      requires tag_invocable<get_scheduler_t, const _Env&>
    inline auto get_scheduler_t::operator()(const _Env& __env) const noexcept
      -> tag_invoke_result_t<get_scheduler_t, const _Env&> {
      static_assert(nothrow_tag_invocable<get_scheduler_t, const _Env&>);
      static_assert(scheduler<tag_invoke_result_t<get_scheduler_t, const _Env&>>);
      return tag_invoke(get_scheduler_t{}, __env);
    }

    template <class _Env>
      requires tag_invocable<get_delegatee_scheduler_t, const _Env&>
    inline auto get_delegatee_scheduler_t::operator()(const _Env& __env) const noexcept
      -> tag_invoke_result_t<get_delegatee_scheduler_t, const _Env&> {
      static_assert(nothrow_tag_invocable<get_delegatee_scheduler_t, const _Env&>);
      static_assert(scheduler<tag_invoke_result_t<get_delegatee_scheduler_t, const _Env&>>);
      return tag_invoke(get_delegatee_scheduler_t{}, __env);
    }

    template <__completion_tag _Tag>
    template <__has_completion_scheduler_for<_Tag> _Env>
    auto get_completion_scheduler_t<_Tag>::operator()(const _Env& __env) const noexcept
      -> tag_invoke_result_t<get_completion_scheduler_t<_Tag>, const _Env&> {
      static_assert(
        nothrow_tag_invocable<get_completion_scheduler_t<_Tag>, const _Env&>,
        "get_completion_scheduler<_Tag> should be noexcept");
      static_assert(scheduler<tag_invoke_result_t<get_completion_scheduler_t<_Tag>, const _Env&>>);
      return tag_invoke(*this, __env);
    }
  } // namespace __queries

  namespace __detail {
    // A handy utility for augmenting an environment with a scheduler.
    template <class _Env, class _Scheduler>
    STDEXEC_ATTRIBUTE((always_inline))
    auto
      __mkenv_sched(_Env&& __env, _Scheduler __sched) {
      auto __env2 = __env::__join(
        __env::__with(__sched, get_scheduler),
        __env::__without(static_cast<_Env&&>(__env), get_domain));
      using _Env2 = decltype(__env2);

      struct __env_t : _Env2 { };

      return __env_t{static_cast<_Env2&&>(__env2)};
    }
  } // namespace __detail
} // namespace stdexec
