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
  namespace __resched {

    using namespace stdexec;

    template <
      __mstring _Where = "In reschedule() ..."_mstr,
      __mstring _What =
        "The execution environment does not have a current scheduler on which to reschedule."_mstr
    >
    struct _INVALID_RESCHEDULE_NO_SCHEDULER_ { };

    template <class _Env>
    using __no_scheduler_error =
      __mexception<_INVALID_RESCHEDULE_NO_SCHEDULER_<>, _WITH_ENVIRONMENT_<_Env>>;

    template <class _Env>
    using __schedule_sender_t = schedule_result_t<__call_result_t<get_scheduler_t, _Env>>;

    template <class _Env>
    using __try_schedule_sender_t =
      __minvoke<__mtry_catch_q<__schedule_sender_t, __q<__no_scheduler_error>>, _Env>;

    template <class _Env>
    using __completions =
      __meval<__completion_signatures_of_t, __try_schedule_sender_t<_Env>, _Env>;

    struct __scheduler {
      struct __sender {
        using sender_concept = sender_t;

        template <class _Env>
        auto get_completion_signatures(_Env&&) noexcept -> __completions<_Env> {
          return {};
        }

        template <receiver _Receiver>
          requires receiver_of<_Receiver, __completions<env_of_t<_Receiver>>>
        auto connect(_Receiver __rcvr) const
          -> connect_result_t<__schedule_sender_t<env_of_t<_Receiver>>, _Receiver> {
          auto __sched = get_scheduler(stdexec::get_env(__rcvr));
          return stdexec::connect(stdexec::schedule(__sched), static_cast<_Receiver&&>(__rcvr));
        }

        [[nodiscard]]
        auto get_env() const noexcept {
          return stdexec::prop{get_completion_scheduler<set_value_t>, __scheduler()};
        }
      };

      [[nodiscard]]
      auto schedule() const noexcept -> __sender {
        return {};
      }

      auto operator==(const __scheduler&) const noexcept -> bool = default;
    };

    struct __reschedule_t {
      template <sender _Sender>
      auto operator()(_Sender&& __sndr) const {
        return stdexec::continues_on(static_cast<_Sender&&>(__sndr), __resched::__scheduler{});
      }

      auto operator()() const {
        return stdexec::continues_on(__resched::__scheduler{});
      }
    };
  } // namespace __resched

  inline constexpr auto reschedule = __resched::__reschedule_t{};
} // namespace exec
