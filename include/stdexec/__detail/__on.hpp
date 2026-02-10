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
#include "__completion_signatures_of.hpp"
#include "__continues_on.hpp"
#include "__diagnostics.hpp"
#include "__env.hpp"
#include "__inline_scheduler.hpp"
#include "__meta.hpp"
#include "__schedulers.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__sender_introspection.hpp"
#include "__utility.hpp"

namespace STDEXEC {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.on]
  namespace __on {
    struct on_t;
    struct _CANNOT_RESTORE_EXECUTION_CONTEXT_AFTER_ON_ { };

    template <class _Sender, class _Env>
    struct __no_scheduler_in_environment {
      using sender_concept = sender_t;

      template <class>
      static consteval auto get_completion_signatures() {
        return STDEXEC::__throw_compile_time_error<
          _WHAT_(_CANNOT_RESTORE_EXECUTION_CONTEXT_AFTER_ON_),
          _WHY_(_THE_CURRENT_EXECUTION_ENVIRONMENT_DOESNT_HAVE_A_SCHEDULER_),
          _WHERE_(_IN_ALGORITHM_, on_t),
          _WITH_PRETTY_SENDER_<_Sender>,
          _WITH_ENVIRONMENT_(_Env)
        >();
      }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    struct on_t {
      template <scheduler _Scheduler, sender _Sender>
      constexpr auto
        operator()(_Scheduler&& __sched, _Sender&& __sndr) const -> __well_formed_sender auto {
        return __make_sexpr<on_t>(
          static_cast<_Scheduler&&>(__sched), static_cast<_Sender&&>(__sndr));
      }

      template <sender _Sender, scheduler _Scheduler, __sender_adaptor_closure_for<_Sender> _Closure>
      constexpr auto operator()(_Sender&& __sndr, _Scheduler&& __sched, _Closure&& __clsur) const
        -> __well_formed_sender auto {
        return __make_sexpr<on_t>(
          __tuple{static_cast<_Scheduler&&>(__sched), static_cast<_Closure&&>(__clsur)},
          static_cast<_Sender&&>(__sndr));
      }

      template <scheduler _Scheduler, __sender_adaptor_closure _Closure>
      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()(_Scheduler&& __sched, _Closure&& __clsur) const {
        return __closure(
          *this, static_cast<_Scheduler&&>(__sched), static_cast<_Closure&&>(__clsur));
      }

      // This transform_sender overload handles the case where `on` was called like
      // `on(sch, sndr)`. In this case, we find the old scheduler by looking in the
      // receiver's environment.
      template <__decay_copyable _Sender, class _Env>
        requires scheduler<__data_of<_Sender>>
      STDEXEC_ATTRIBUTE(always_inline)
      static auto transform_sender(set_value_t, _Sender&& __sndr, const _Env& __env) {
        auto& [__tag, __sched, __child] = __sndr;
        auto __old = __with_default(get_scheduler, __end_sched_t<_Sender, _Env>())(__env);

        return continues_on(
          starts_on(
            STDEXEC::__forward_like<_Sender>(__sched), STDEXEC::__forward_like<_Sender>(__child)),
          std::move(__old));
      }

      // This transform_sender overload handles the case where `on` was called like
      // `sndr | on(sch, clsur)` or `on(sndr, sch, clsur)`. In this case, __child is a
      // predecessor sender, so the scheduler we want to restore is the completion
      // scheduler of __child.
      template <__decay_copyable _Sender, class _Env>
        requires(!scheduler<__data_of<_Sender>>)
      STDEXEC_ATTRIBUTE(always_inline)
      static auto transform_sender(set_value_t, _Sender&& __sndr, const _Env& __env) {
        auto& [__tag, __data, __child] = __sndr;
        auto& [__sched, __clsur] = __data;

        auto __old = __with_default(
          get_completion_scheduler<set_value_t>,
          __end_sched_t<_Sender, _Env>())(get_env(__child), __env);

        auto __pred = __reschedule(STDEXEC::__forward_like<_Sender>(__child), __old, __sched);
        return __reschedule(
          STDEXEC::__forward_like<_Sender>(__clsur)(std::move(__pred)),
          std::move(__sched),
          std::move(__old));
      }

      template <class _Sender, class _Env>
      static auto transform_sender(set_value_t, _Sender&&, const _Env&) {
        return __not_a_sender<_SENDER_TYPE_IS_NOT_DECAY_COPYABLE_, _WITH_PRETTY_SENDER_<_Sender>>{};
      }

     private:
      // If __is_root_env<_Env> is true, then this sender has no parent, so there is
      // no need to restore the execution context. We can use the inline scheduler
      // as the scheduler if __env does not have one.
      template <class _Sender, class _Env>
      using __end_sched_t = __if_c<
        __is_root_env<_Env>,
        inline_scheduler,
        __not_a_scheduler<__no_scheduler_in_environment<_Sender, _Env>>
      >;

      template <class _Sender, class _OldSched, class _NewSched>
      static constexpr auto __reschedule(
        _Sender&& __sndr,
        [[maybe_unused]] _OldSched&& __old_sched,
        _NewSched&& __new_sched) {
        // BUGBUG TODO(ericniebler): FIXME
        // return continues_on(
        //   write_env(static_cast<_Sender&&>(__sndr), __sched_env{__old_sched}),
        //   static_cast<_NewSched&&>(__new_sched));
        return continues_on(static_cast<_Sender&&>(__sndr), static_cast<_NewSched&&>(__new_sched));
      }
    };
  } // namespace __on

  using __on::on_t;
  inline constexpr on_t on{};

  namespace v2 {
    using on_t [[deprecated("use STDEXEC::on_t instead")]] = STDEXEC::on_t;
    [[deprecated("use STDEXEC::on instead")]]
    inline constexpr STDEXEC::on_t const & on = STDEXEC::on;
  } // namespace v2

  template <>
  struct __sexpr_impl<on_t> : __sexpr_defaults {
    template <class _Sender, class _Env>
    static constexpr auto get_completion_signatures() {
      using __sndr_t = __detail::__transform_sender_result_t<on_t, set_value_t, _Sender, _Env>;
      return STDEXEC::get_completion_signatures<__sndr_t, _Env>();
    }
  };
} // namespace STDEXEC
