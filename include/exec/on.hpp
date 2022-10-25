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

namespace exec {
  /////////////////////////////////////////////////////////////////////////////
  // A scoped version of [execution.senders.adaptors.on]
  namespace __on {
    using namespace stdexec;

    enum class on_kind { start_on, continue_on };

    template <on_kind>
      struct on_t;

    template <class _Scheduler, class _Sender>
      struct __start_fn {
        _Scheduler __sched_;
        _Sender __sndr_;

        template <class _Self, class _OldSched>
          static auto __call(_Self&& __self, _OldSched __old_sched) {
            return std::move(((_Self&&) __self).__sndr_)
              | exec::write(exec::with(get_scheduler, __self.__sched_))
              | transfer(__old_sched)
              ;
          }

        template <scheduler _OldSched>
          auto operator()(_OldSched __old_sched) && {
            return __call(std::move(*this), __old_sched);
          }
        template <scheduler _OldSched>
          auto operator()(_OldSched __old_sched) const & {
            return __call(*this, __old_sched);
          }
      };
    template <class _Scheduler, class _Sender>
      __start_fn(_Scheduler, _Sender)
        -> __start_fn<_Scheduler, _Sender>;

    template <class _Env, class _Sender>
      struct _ENVIRONMENT_HAS_NO_SCHEDULER_FOR_THE_ON_ADAPTOR_TO_TRANSITION_BACK_TO {};

    template <class _SchedulerId, class _SenderId>
      struct __start_on_sender {
        using _Scheduler = __t<_SchedulerId>;
        using _Sender = __t<_SenderId>;

        _Scheduler __sched_;
        _Sender __sndr_;

        template <class _Self>
          static auto __call(_Self&& __self) {
            return let_value(
              stdexec::read(get_scheduler)
                | transfer(__self.__sched_),
              __start_fn{__self.__sched_, ((_Self&&) __self).__sndr_});
          }

        template <class _Self>
          using __inner_t = decltype(__call(__declval<_Self>()));

        template <__decays_to<__start_on_sender> _Self, receiver _Receiver>
            requires constructible_from<_Sender, __member_t<_Self, _Sender>> &&
              sender_to<__inner_t<_Self>, _Receiver>
          friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __receiver)
            -> connect_result_t<__inner_t<_Self>, _Receiver> {
            return connect(__call((_Self&&) __self), (_Receiver&&) __receiver);
          }

        template <__decays_to<__start_on_sender> _Self, class _Env>
          friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
            -> completion_signatures_of_t<__inner_t<_Self>, _Env>;

        template <__decays_to<__start_on_sender> _Self, __none_of<no_env> _Env>
            requires (!__callable<get_scheduler_t, _Env>)
          friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
            -> _ENVIRONMENT_HAS_NO_SCHEDULER_FOR_THE_ON_ADAPTOR_TO_TRANSITION_BACK_TO<_Env, _Sender>;

        // forward sender queries:
        template <tag_category<forwarding_sender_query> _Tag, class... _As _NVCXX_CAPTURE_PACK(_As)>
            requires __callable<_Tag, const _Sender&, _As...>
          friend auto tag_invoke(_Tag __tag, const __start_on_sender& __self, _As&&... __as)
            noexcept(__nothrow_callable<_Tag, const _Sender&, _As...>)
            -> __call_result_if_t<tag_category<_Tag, forwarding_sender_query>, _Tag, const _Sender&, _As...> {
            _NVCXX_EXPAND_PACK_RETURN(_As, __as,
              return ((_Tag&&) __tag)(__self.__sndr_, (_As&&) __as...);
            )
          }
      };
    template <class _Scheduler, class _Sender>
      __start_on_sender(_Scheduler, _Sender)
        -> __start_on_sender<__x<_Scheduler>, __x<_Sender>>;

    template <>
      struct on_t<on_kind::start_on> {
        template <scheduler _Scheduler, sender _Sender>
            requires constructible_from<decay_t<_Sender>, _Sender>
          auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const {
            // connect-based customization will remove the need for this check
            using __has_customizations =
              __call_result_t<__has_algorithm_customizations_t, _Scheduler>;
            static_assert(
              !__has_customizations{},
              "For now the default exec::on implementation doesn't support scheduling "
              "onto schedulers that customize algorithms.");
            return __start_on_sender{(_Scheduler&&) __sched, (_Sender&&) __sndr};
          }
      };

    template <class _Sender, class _Scheduler, class _Closure>
      struct __continue_fn {
        _Sender __sndr_;
        _Scheduler __sched_;
        _Closure __closure_;

        template <class _Self, class _OldSched>
          static auto __call(_Self&& __self, _OldSched __old_sched) {
            return ((_Self&&) __self).__sndr_
              | exec::write(exec::with(get_scheduler, __old_sched))
              | transfer(__self.__sched_)
              | ((_Self&&) __self).__closure_
              | transfer(__old_sched)
              | exec::write(exec::with(get_scheduler, __self.__sched_))
              ;
          }

        template <scheduler _OldSched>
          auto operator()(_OldSched __old_sched) && {
            return __call(std::move(*this), __old_sched);
          }
        template <scheduler _OldSched>
          auto operator()(_OldSched __old_sched) const & {
            return __call(*this, __old_sched);
          }
      };
    template <class _Sender, class _Scheduler, class _Closure>
      __continue_fn(_Sender, _Scheduler, _Closure)
        -> __continue_fn<_Sender, _Scheduler, _Closure>;

    template <class _SenderId, class _SchedulerId, class _ClosureId>
      struct __continue_on_sender {
        using _Sender = __t<_SenderId>;
        using _Scheduler = __t<_SchedulerId>;
        using _Closure = __t<_ClosureId>;

        _Sender __sndr_;
        _Scheduler __sched_;
        _Closure __closure_;

        template <class _Self>
          static auto __call(_Self&& __self) {
            return let_value(
              stdexec::read(get_scheduler),
              __continue_fn{
                ((_Self&&) __self).__sndr_,
                __self.__sched_,
                ((_Self&&) __self).__closure_});
          }

        template <class _Self>
          using __inner_t = decltype(__call(__declval<_Self>()));

        template <__decays_to<__continue_on_sender> _Self, receiver _Receiver>
            requires constructible_from<_Sender, __member_t<_Self, _Sender>> &&
              sender_to<__inner_t<_Self>, _Receiver>
          friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __receiver)
            -> connect_result_t<__inner_t<_Self>, _Receiver> {
            return connect(__call((_Self&&) __self), (_Receiver&&) __receiver);
          }

        template <__decays_to<__continue_on_sender> _Self, class _Env>
          friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
            -> completion_signatures_of_t<__inner_t<_Self>, _Env>;

        template <__decays_to<__continue_on_sender> _Self, __none_of<no_env> _Env>
            requires (!__callable<get_scheduler_t, _Env>)
          friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
            -> _ENVIRONMENT_HAS_NO_SCHEDULER_FOR_THE_ON_ADAPTOR_TO_TRANSITION_BACK_TO<_Env, _Sender>;

        // forward sender queries:
        template <tag_category<forwarding_sender_query> _Tag, class... _As _NVCXX_CAPTURE_PACK(_As)>
            requires __callable<_Tag, const _Sender&, _As...>
          friend auto tag_invoke(_Tag __tag, const __continue_on_sender& __self, _As&&... __as)
            noexcept(__nothrow_callable<_Tag, const _Sender&, _As...>)
            -> __call_result_if_t<tag_category<_Tag, forwarding_sender_query>, _Tag, const _Sender&, _As...> {
            _NVCXX_EXPAND_PACK_RETURN(_As, __as,
              return ((_Tag&&) __tag)(__self.__sndr_, (_As&&) __as...);
            )
          }
      };
    template <class _Sender, class _Scheduler, class _Closure>
      __continue_on_sender(_Sender, _Scheduler, _Closure)
        -> __continue_on_sender<__x<_Sender>, __x<_Scheduler>, __x<_Closure>>;

    template <>
      struct on_t<on_kind::continue_on> {
        template <sender _Sender, scheduler _Scheduler, __sender_adaptor_closure_for<_Sender> _Closure>
            requires constructible_from<decay_t<_Sender>, _Sender>
          auto operator()(_Sender&& __sndr, _Scheduler&& __sched, _Closure __closure) const {
            return __continue_on_sender{
              (_Sender&&) __sndr,
              (_Scheduler&&) __sched,
              (_Closure&&) __closure};
          }

        template <scheduler _Scheduler, __sender_adaptor_closure _Closure>
          auto operator()(_Scheduler&& __sched, _Closure __closure) const
            -> __binder_back<on_t, decay_t<_Scheduler>, _Closure> {
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
