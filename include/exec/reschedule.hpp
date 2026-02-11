/*
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
#pragma once

#include "../stdexec/execution.hpp"

namespace exec {
  struct _CANNOT_RESCHEDULE_ { };
  using STDEXEC::_THE_CURRENT_EXECUTION_ENVIRONMENT_DOESNT_HAVE_A_SCHEDULER_;

  namespace __resched {
    using namespace STDEXEC;

    struct reschedule_t;
    template <class _Env>
    using __no_scheduler_error = __mexception<
      _WHAT_(_CANNOT_RESCHEDULE_),
      _WHY_(_THE_CURRENT_EXECUTION_ENVIRONMENT_DOESNT_HAVE_A_SCHEDULER_),
      _WHERE_(_IN_ALGORITHM_, reschedule_t),
      _WITH_ENVIRONMENT_(_Env)
    >;

    template <class _Env>
    using __schedule_sender_t = schedule_result_t<__call_result_t<get_scheduler_t, _Env>>;

    template <class _Env>
    using __try_schedule_sender_t =
      __minvoke<__mtry_catch_q<__schedule_sender_t, __q<__no_scheduler_error>>, _Env>;

    template <class _Env>
    using __completions =
      __minvoke_q<__completion_signatures_of_t, __try_schedule_sender_t<_Env>, _Env>;

    struct __scheduler {
      struct __sender {
        using sender_concept = sender_t;

        template <class _Self, class _Env>
        static consteval auto get_completion_signatures() -> __completions<_Env> {
          return {};
        }

        template <receiver _Receiver>
          requires receiver_of<_Receiver, __completions<env_of_t<_Receiver>>>
        auto connect(_Receiver __rcvr) const
          -> connect_result_t<__schedule_sender_t<env_of_t<_Receiver>>, _Receiver> {
          auto __sched = get_scheduler(STDEXEC::get_env(__rcvr));
          return STDEXEC::connect(STDEXEC::schedule(__sched), static_cast<_Receiver&&>(__rcvr));
        }

        [[nodiscard]]
        constexpr auto get_env() const noexcept {
          return STDEXEC::prop{get_completion_scheduler<set_value_t>, __scheduler()};
        }
      };

      [[nodiscard]]
      constexpr auto schedule() const noexcept -> __sender {
        return {};
      }

      constexpr auto operator==(const __scheduler&) const noexcept -> bool = default;
    };

    struct reschedule_t {
      template <sender _Sender>
      constexpr auto operator()(_Sender&& __sndr) const {
        return STDEXEC::continues_on(static_cast<_Sender&&>(__sndr), __resched::__scheduler{});
      }

      constexpr auto operator()() const {
        return STDEXEC::continues_on(__resched::__scheduler{});
      }
    };
  } // namespace __resched

  inline constexpr auto reschedule = __resched::reschedule_t{};
} // namespace exec
