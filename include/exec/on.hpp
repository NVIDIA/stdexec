/*
 * Copyright (c) 2022 NVIDIA Corporation
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
#include "env.hpp"
#include "__detail/__sender_facade.hpp"

namespace exec {
  /////////////////////////////////////////////////////////////////////////////
  // A scoped version of [execution.senders.adaptors.on]
  namespace __on {
    using namespace stdexec;

    enum class on_kind { start_on, continue_on };

    template <on_kind>
      struct on_t;

    template <class _Env, class _Sender>
      struct _ENVIRONMENT_HAS_NO_SCHEDULER_FOR_THE_ON_ADAPTOR_TO_TRANSITION_BACK_TO {};

    template <class _Env, class _SchedulerId>
      struct __with_sched_env : _Env {
        using _Scheduler = stdexec::__t<_SchedulerId>;
        _Scheduler __sched_;
        friend _Scheduler tag_invoke(get_scheduler_t, const __with_sched_env& __self) noexcept {
          return __self.__sched_;
        }
      };

    template <class _Env>
      struct __with_connect_transform_env : _Env {
        friend std::true_type tag_invoke(__use_connect_transform_t, const __with_connect_transform_env&) noexcept {
          return {};
        }
      };

    template <class _Scheduler>
      struct __with_sched_kernel : __default_kernel {
        _Scheduler __sched_;
        __with_sched_kernel(_Scheduler __sched)
          : __sched_((_Scheduler&&) __sched)
        {}

        template <class _Env>
          __with_sched_env<_Env, __id<_Scheduler>> get_env(_Env __env) {
            return {(_Env&&) __env, __sched_};
          }

        template <class _Env, class _OtherSchedulerId>
          _Env get_env(__with_sched_env<_Env, _OtherSchedulerId> __env) {
            return (_Env&&) __env;
          }
      };

    template <class _SenderId, class _Scheduler>
      struct __with_sched
        : __sender_facade<
            __with_sched<_SenderId, _Scheduler>,
            __t<_SenderId>,
            __with_sched_kernel<_Scheduler>>
      {};

    template <class _Sender>
      using __pretty_print =
        __minvoke<__with_default<__id_<>, _Sender>, _Sender>;

    template <class _SenderId, class _Scheduler>
      struct __start_on;

    template <class _Scheduler>
      struct __start_on_kernel : __default_kernel {
        _Scheduler __sched_;
        explicit __start_on_kernel(_Scheduler __sched)
          : __sched_((_Scheduler&&) __sched)
        {}

        template <class... _Ts>
          using __self_t = __make_dependent_on<__start_on_kernel, _Ts...>;

        template <class _Sender, class _OldScheduler>
          auto transform_sender_(_Sender&& __sndr, _OldScheduler __old_sched) {
            return stdexec::on(__sched_, (_Sender&&) __sndr)
              | transfer(__old_sched);
          }

        template <class _Sender, class _Receiver, class... _Ts>
          using __new_sender_t =
            decltype(__declval<__self_t<_Ts...>&>().transform_sender_(
              __declval<_Sender>(),
              __declval<__current_scheduler_t<_Receiver>>()));

        template <class _Sender>
          using __on_sender_t =
            __member_t<_Sender, __start_on<__pretty_print<decay_t<_Sender>>, _Scheduler>>;

        template <class _Sender, class _Receiver>
          using __diagnostic_t =
            _FAILURE_TO_CONNECT_::_WHAT_<
              _ENVIRONMENT_HAS_NO_SCHEDULER_FOR_THE_ON_ADAPTOR_TO_TRANSITION_BACK_TO<
                env_of_t<_Receiver>, __on_sender_t<_Sender>>>;

        template <class _Sender, class _Receiver>
          using __result_t =
            __minvoke<
              __if_c<
                __valid<__new_sender_t, _Sender, _Receiver>,
                __q<__new_sender_t>,
                __q<__diagnostic_t>>,
              _Sender,
              _Receiver>;

        template <class _Sender, class _Receiver>
          auto transform_sender(_Sender&& __sndr, __ignore, _Receiver& __rcvr)
            -> __result_t<_Sender, _Receiver> {
            if constexpr (__valid<__new_sender_t, _Sender, _Receiver>) {
              auto __sched = get_scheduler(stdexec::get_env(__rcvr));
              return transform_sender_((_Sender&&) __sndr, __sched);
            } else {
              return {};
            }
          }

        template <class _Env>
          __with_connect_transform_env<_Env> get_env(_Env __env) {
            return {(_Env&&) __env};
          }
      };

    template <class _SenderId, class _Scheduler>
      struct __start_on
        : __sender_facade<
            __start_on<_SenderId, _Scheduler>,
            __t<_SenderId>,
            __start_on_kernel<_Scheduler>>
      {};

    template <class _Sender, class _Scheduler>
      using __start_on_t =
        __t<__start_on<__id<decay_t<_Sender>>, _Scheduler>>;

    template <>
      struct on_t<on_kind::start_on> {
        template <scheduler _Scheduler, sender _Sender>
            requires constructible_from<decay_t<_Sender>, _Sender>
          auto operator()(_Scheduler __sched, _Sender&& __sndr) const
            -> __start_on_t<_Sender, _Scheduler> {
            return {(_Sender&&) __sndr, (_Scheduler&&) __sched};
          }
      };

    template <class _SenderId, class _Scheduler, class _Closure>
      struct __continue_on;

    template <class _Scheduler, class _Closure>
      struct __continue_on_kernel : __default_kernel {
        _Scheduler __sched_;
        _Closure __closure_;
        __continue_on_kernel(_Scheduler __sched, _Closure __closure)
          : __sched_((_Scheduler&&) __sched)
          , __closure_((_Closure&&) __closure)
        {}

        template <class... _Ts>
          using __self_t = __make_dependent_on<__continue_on_kernel, _Ts...>;

        template <class _Sender, class _OldScheduler>
          auto transform_sender_(_Sender&& __sndr, _OldScheduler __old_sched) {
            using __sender_t = __t<__with_sched<__id<decay_t<_Sender>>, _OldScheduler>>;
            return __sender_t{(_Sender&&) __sndr, __old_sched}
              | transfer(__sched_)
              | (__member_t<_Sender, _Closure>&&) __closure_
              | transfer(__old_sched);
          }

        template <class _Sender, class _Receiver, class... _Ts>
          using __new_sender_t =
            decltype(__declval<__self_t<_Ts...>&>().transform_sender_(
              __declval<_Sender>(),
              __declval<__current_scheduler_t<_Receiver>>()));

        template <class _Sender>
          using __on_sender_t =
            __member_t<_Sender, __continue_on<__pretty_print<decay_t<_Sender>>, _Scheduler, _Closure>>;

        template <class _Sender, class _Receiver>
          using __diagnostic_t =
            _FAILURE_TO_CONNECT_::_WHAT_<
              _ENVIRONMENT_HAS_NO_SCHEDULER_FOR_THE_ON_ADAPTOR_TO_TRANSITION_BACK_TO<
                env_of_t<_Receiver>, __on_sender_t<_Sender>>>;

        template <class _Sender, class _Receiver>
          using __result_t =
            __minvoke<
              __if_c<
                __valid<__new_sender_t, _Sender, _Receiver>,
                __q<__new_sender_t>,
                __q<__diagnostic_t>>,
              _Sender,
              _Receiver>;

        template <class _Sender, class _Receiver>
          auto transform_sender(_Sender&& __sndr, __ignore, _Receiver& __rcvr)
            -> __result_t<_Sender, _Receiver> {
            if constexpr (__valid<__new_sender_t, _Sender, _Receiver>) {
              auto __sched = get_scheduler(stdexec::get_env(__rcvr));
              return transform_sender_((_Sender&&) __sndr, __sched);
            } else {
              return {};
            }
          }

        template <class _Env>
          __with_sched_env<_Env, __id<_Scheduler>> get_env(_Env __env) {
            return {(_Env&&) __env, __sched_};
          }
      };

    template <class _SenderId, class _Scheduler, class _Closure>
      struct __continue_on
        : __sender_facade<
            __continue_on<_SenderId, _Scheduler, _Closure>,
            __t<_SenderId>,
            __continue_on_kernel<_Scheduler, _Closure>>
      {};

    template <class _Sender, class _Scheduler, class _Closure>
      using __continue_on_t =
        __t<__continue_on<__id<decay_t<_Sender>>, _Scheduler, _Closure>>;

    template <>
      struct on_t<on_kind::continue_on> {
        template <sender _Sender, scheduler _Scheduler, __sender_adaptor_closure_for<_Sender> _Closure>
            requires constructible_from<decay_t<_Sender>, _Sender>
          auto operator()(_Sender&& __sndr, _Scheduler __sched, _Closure __closure) const
            -> __continue_on_t<_Sender, _Scheduler, _Closure> {
            return {
              (_Sender&&) __sndr,
              (_Scheduler&&) __sched,
              (_Closure&&) __closure
            };
          }

        template <scheduler _Scheduler, __sender_adaptor_closure _Closure>
          auto operator()(_Scheduler __sched, _Closure __closure) const
            -> __binder_back<on_t, _Scheduler, _Closure> {
            return {{}, {}, {(_Scheduler&&) __sched, (_Closure&&) __closure}};
          }
      };

    struct __on_t
      : on_t<on_kind::start_on>
      , on_t<on_kind::continue_on> {
      using on_t<on_kind::start_on>::operator();
      using on_t<on_kind::continue_on>::operator();
    };
  } // namespace __on

  using __on::on_kind;
  using __on::on_t;
  inline constexpr __on::__on_t on{};
} // namespace exec
