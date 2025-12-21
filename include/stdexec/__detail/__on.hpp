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
#include "__basic_sender.hpp"
#include "__continues_on.hpp"
#include "__diagnostics.hpp"
#include "__env.hpp"
#include "__inline_scheduler.hpp"
#include "__meta.hpp"
#include "__schedulers.hpp"
#include "__senders_core.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__sender_introspection.hpp"
#include "__utility.hpp"
#include "__write_env.hpp"

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.on]
  namespace __on {
    struct on_t;

    struct _CANNOT_RESTORE_THE_EXECUTION_CONTEXT_AFTER_ON;
    struct _THE_ENVIRONMENT_OF_THE_RECEIVER_DOES_NOT_HAVE_A_SCHEDULER_FOR_ON_TO_RETURN_TO;
    struct _THE_PREDECESSOR_SENDER_DOES_NOT_KNOW_THE_SCHEDULER_ON_WHICH_IT_COMPLETES;

    template <class _Scheduler, class _Closure>
    struct __on_data {
      _Scheduler __sched_;
      _Closure __clsur_;
    };

    template <class _Scheduler, class _Closure>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __on_data(_Scheduler, _Closure) //
      ->__on_data<_Scheduler, _Closure>;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    struct on_t {
      template <scheduler _Scheduler, sender _Sender>
      auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const -> __well_formed_sender auto {
        return __make_sexpr<on_t>(
          static_cast<_Scheduler&&>(__sched), static_cast<_Sender&&>(__sndr));
      }

      template <sender _Sender, scheduler _Scheduler, __sender_adaptor_closure_for<_Sender> _Closure>
      auto operator()(_Sender&& __sndr, _Scheduler&& __sched, _Closure&& __clsur) const
        -> __well_formed_sender auto {
        return __make_sexpr<on_t>(
          __on_data{static_cast<_Scheduler&&>(__sched), static_cast<_Closure&&>(__clsur)},
          static_cast<_Sender&&>(__sndr));
      }

      template <scheduler _Scheduler, __sender_adaptor_closure _Closure>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(_Scheduler&& __sched, _Closure&& __clsur) const {
        return __closure(
          *this, static_cast<_Scheduler&&>(__sched), static_cast<_Closure&&>(__clsur));
      }

      template <class _Sender, class _OldSched, class _NewSched>
      static auto __reschedule(_Sender&& __sndr, _OldSched&& __old_sched, _NewSched&& __new_sched) {
        return continues_on(
          write_env(static_cast<_Sender&&>(__sndr), __sched_env{__old_sched}),
          static_cast<_NewSched&&>(__new_sched));
      }

      template <class _Sender, class _Env>
      STDEXEC_ATTRIBUTE(always_inline)
      static auto __transform_sender_fn(const _Env& __env) noexcept {
        return [&]<class _Data, class _Child>(__ignore, _Data&& __data, _Child&& __child) {

          if constexpr (scheduler<_Data>) {
            using __not_a_scheduler = stdexec::__not_a_scheduler<
              _WHAT_<>(_CANNOT_RESTORE_THE_EXECUTION_CONTEXT_AFTER_ON),
              _WHY_(_THE_ENVIRONMENT_OF_THE_RECEIVER_DOES_NOT_HAVE_A_SCHEDULER_FOR_ON_TO_RETURN_TO),
              _WHERE_(_IN_ALGORITHM_, on_t),
              _WITH_ENVIRONMENT_<_Env>,
              _WITH_SENDER_<_Child>
            >;

            // If __is_root_env<_Env> is true, then this sender has no parent, so there is
            // no need to restore the execution context. We can use the inline scheduler
            // as the scheduler if __env does not have one.
            using __end_sched_t = __if_c<__is_root_env<_Env>, inline_scheduler, __not_a_scheduler>;

            // This branch handles the case where `on` was called like `on(sch, sndr)`. In
            // this case, we find the old scheduler by looking in the receiver's
            // environment.
            auto __old = __with_default{get_scheduler, __end_sched_t()}(__env);

            return continues_on(
              starts_on(static_cast<_Data&&>(__data), static_cast<_Child&&>(__child)),
              static_cast<decltype(__old)&&>(__old));
          } else {
            using __not_a_scheduler = stdexec::__not_a_scheduler<
              _WHAT_<>(_CANNOT_RESTORE_THE_EXECUTION_CONTEXT_AFTER_ON),
              _WHY_(_THE_PREDECESSOR_SENDER_DOES_NOT_KNOW_THE_SCHEDULER_ON_WHICH_IT_COMPLETES),
              _WHERE_(_IN_ALGORITHM_, on_t),
              _WITH_SENDER_<_Child>,
              _WITH_ENVIRONMENT_<_Env>
            >;

            // If __is_root_env<_Env> is true, then this sender has no parent, so there is
            // no need to restore the execution context. We can use the inline scheduler
            // as the scheduler if __env does not have one.
            using __end_sched_t = __if_c<__is_root_env<_Env>, inline_scheduler, __not_a_scheduler>;

            // This branch handles the case where `on` was called like `sndr | on(sch,
            // clsur)`. In this case, __child is a predecessor sender, so the scheduler we
            // want to restore is the completion scheduler of __child.
            constexpr auto __get_old_sched =
              __with_default{get_completion_scheduler<set_value_t>, __end_sched_t()};
            auto __old = __get_old_sched(get_env(__child), __env);

            auto& [__sched, __clsur] = __data;
            auto __pred = __reschedule(static_cast<_Child&&>(__child), __old, __sched);
            return __reschedule(
              __forward_like<_Data>(__clsur)(std::move(__pred)),
              std::move(__sched),
              std::move(__old));
          }
        };
      }

      template <class _Sender, class _Env>
      STDEXEC_ATTRIBUTE(always_inline)
      static auto transform_sender(set_value_t, _Sender&& __sndr, const _Env& __env) {
        return __sexpr_apply(static_cast<_Sender&&>(__sndr), __transform_sender_fn<_Sender>(__env));
      }
    };
  } // namespace __on

  using __on::on_t;
  inline constexpr on_t on{};

  namespace v2 {
    using on_t [[deprecated("use stdexec::on_t instead")]] = stdexec::on_t;
    [[deprecated("use stdexec::on instead")]]
    inline constexpr stdexec::on_t const & on = stdexec::on;
  } // namespace v2

  template <>
  struct __sexpr_impl<on_t> : __sexpr_defaults {
    static constexpr auto get_completion_signatures =
      []<class _Sender, class _Env>(_Sender&&, const _Env&) noexcept
      -> __completion_signatures_of_t<transform_sender_result_t<_Sender, _Env>, _Env> {
      return {};
    };
  };
} // namespace stdexec
