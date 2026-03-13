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
#include "__sender_ref.hpp"
#include "__starts_on.hpp"
#include "__utility.hpp"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace STDEXEC
{
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.on]
  struct _CANNOT_RESTORE_EXECUTION_CONTEXT_AFTER_ON_;

  namespace __on
  {
    // If __is_root_env<_Env> is true, then this sender has no parent, so there is no need
    // to restore the execution context. We can use the inline scheduler as the scheduler
    // if __env does not have one.
    template <class _Child, class _Env>
    using __end_sched_t =
      __if_c<__is_root_env<_Env>,
             inline_scheduler,
             __not_a_scheduler<_WHAT_(_CANNOT_RESTORE_EXECUTION_CONTEXT_AFTER_ON_),
                               _WHY_(_THE_CURRENT_EXECUTION_ENVIRONMENT_DOESNT_HAVE_A_SCHEDULER_),
                               _WHERE_(_IN_ALGORITHM_, on_t),
                               _WITH_PRETTY_SENDER_<_Child>,
                               _WITH_ENVIRONMENT_(_Env)>>;

    // This transform_sender overload handles the case where `on` was called like `on(sch,
    // sndr)`. In this case, we find the old scheduler by looking in the receiver's
    // environment.
    template <class _Scheduler, class _Child, class _Env>
      requires scheduler<_Scheduler>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto __transform_sender(_Scheduler&& __new_sched,
                                      _Child&&     __child,
                                      _Env const & __env)
    {
      auto __default_sched = __end_sched_t<_Child, _Env>();
      auto __old_sched     = __with_default(get_scheduler, __default_sched)(__env);

      return continues_on(starts_on(static_cast<_Scheduler&&>(__new_sched),
                                    static_cast<_Child&&>(__child)),
                          std::move(__old_sched));
    }

    // This transform_sender overload handles the case where `on` was called like `sndr |
    // on(sch, clsur)` or `on(sndr, sch, clsur)`. In this case, __child is a predecessor
    // sender, so the scheduler we want to restore is the completion scheduler of __child.
    template <class _Data, class _Child, class _Env>
      requires(!scheduler<_Data>)
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto __transform_sender(_Data&& __data, _Child&& __child, _Env const & __env)
    {
      auto& [__new_sched, __clsur] = __data;
      auto __default_sched         = __end_sched_t<_Child, _Env>();
      auto __get_sched = __with_default(get_completion_scheduler<set_value_t>, __default_sched);
      auto __old_sched = __get_sched(get_env(__child), __env);

      return continues_on(STDEXEC::__forward_like<_Data>(__clsur)(
                            continues_on(static_cast<_Child&&>(__child),
                                         STDEXEC::__forward_like<_Data>(__new_sched))),
                          std::move(__old_sched));
    }

    template <class _Child>
    struct __attrs_base
    {
      template <__forwarding_query _Query, class... _Args>
        requires(!__completion_query<_Query>)
             && __queryable_with<env_of_t<_Child>, _Query, _Args...>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto query(_Query __query, _Args&&... __args) const
        noexcept(__nothrow_queryable_with<env_of_t<_Child>, _Query, _Args...>)
          -> __query_result_t<env_of_t<_Child>, _Query, _Args...>
      {
        return __query(STDEXEC::get_env(__child_), static_cast<_Args&&>(__args)...);
      }

      _Child const & __child_;
    };

    template <class _Child, class _Scheduler, class... _Closure>
    struct __attrs;

    template <class _Child, class _Scheduler, class _Closure>
    struct __attrs<_Child, _Scheduler, _Closure> : __attrs_base<_Child>
    {
      using __trnsfr_sndr_t  = __result_of<continues_on, __sender_proxy<_Child const>, _Scheduler>;
      using __clsur_result_t = __call_result_t<_Closure const &, __trnsfr_sndr_t>;
      template <class _Env>
      using __old_sched_t =
        __query_result_t<env_of_t<_Child>, get_completion_scheduler_t<set_value_t>, _Env>;
      template <class _Env>
      using __attrs_t = __trnsfr::__attrs<__old_sched_t<_Env>, __clsur_result_t>;
      using __attrs_base<_Child>::query;

      explicit constexpr __attrs(_Child const &   __child,
                                 _Scheduler       __sched,
                                 _Closure const & __clsur)
        : __attrs_base<_Child>{__child}
        , __clsur_result_(__clsur(continues_on(__sender_proxy{__child}, std::move(__sched))))
      {}

      template <class _Query, class _Env>
        requires __completion_query<_Query>  //
              && __queryable_with<env_of_t<_Child>, get_completion_scheduler_t<set_value_t>, _Env>
              && __queryable_with<__attrs_t<_Env>, _Query, _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto query(_Query __query, _Env&& __env) const noexcept
        -> __query_result_t<__attrs_t<_Env>, _Query, _Env>
      {
        auto&& __child_attrs = STDEXEC::get_env(this->__child_);
        auto   __old_sch     = get_completion_scheduler<set_value_t>(__child_attrs, __env);
        auto   __attrs       = __attrs_t<_Env>(__old_sch, STDEXEC::get_env(__clsur_result_));
        return __query(__attrs, static_cast<_Env&&>(__env));
      }

      __clsur_result_t __clsur_result_;
    };

    template <class _Child, class _Scheduler>
    struct __attrs<_Child, _Scheduler> : __attrs_base<_Child>
    {
      using __child_t       = __result_of<starts_on, _Scheduler, _Child>;
      using __child_attrs_t = __starts_on::__attrs<_Scheduler, _Child>;
      template <class _Env>
      using __attrs_t = __trnsfr::__attrs<__result_of<get_scheduler, _Env>, __child_t>;
      using __attrs_base<_Child>::query;

      template <class _Query, __queryable_with<get_scheduler_t> _Env>
        requires __completion_query<_Query> && __queryable_with<__attrs_t<_Env>, _Query, _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto query(_Query __query, _Env&& __env) const noexcept
        -> __query_result_t<__attrs_t<_Env>, _Query, _Env>
      {
        auto&& __child_attrs = STDEXEC::get_env(this->__child_);
        auto   __old_sch     = get_scheduler(__env);
        auto   __attrs       = __attrs_t<_Env>(__old_sch, __child_attrs_t(__sched_, __child_attrs));
        return __query(__attrs, static_cast<_Env&&>(__env));
      }

      _Scheduler __sched_;
    };
  }  // namespace __on

  ////////////////////////////////////////////////////////////////////////////////////////////////
  struct on_t
  {
    template <scheduler _Scheduler, sender _Sender>
    constexpr auto
    operator()(_Scheduler&& __sched, _Sender&& __sndr) const -> __well_formed_sender auto
    {
      return __make_sexpr<on_t>(static_cast<_Scheduler&&>(__sched), static_cast<_Sender&&>(__sndr));
    }

    template <sender _Sender, scheduler _Scheduler, __sender_adaptor_closure_for<_Sender> _Closure>
    constexpr auto operator()(_Sender&& __sndr, _Scheduler&& __sched, _Closure&& __clsur) const
      -> __well_formed_sender auto
    {
      return __make_sexpr<on_t>(__tuple{static_cast<_Scheduler&&>(__sched),
                                        static_cast<_Closure&&>(__clsur)},
                                static_cast<_Sender&&>(__sndr));
    }

    template <scheduler _Scheduler, __sender_adaptor_closure _Closure>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto operator()(_Scheduler&& __sched, _Closure&& __clsur) const
    {
      return __closure(*this, static_cast<_Scheduler&&>(__sched), static_cast<_Closure&&>(__clsur));
    }

    template <__decay_copyable _Sender, class _Env>
    STDEXEC_ATTRIBUTE(always_inline)
    static auto transform_sender(set_value_t, _Sender&& __sndr, _Env&& __env)
    {
      static_assert(__sender_for<_Sender, on_t>);
      auto& [__tag, __data, __child] = __sndr;
      return __on::__transform_sender(STDEXEC::__forward_like<_Sender>(__data),
                                      STDEXEC::__forward_like<_Sender>(__child),
                                      static_cast<_Env&&>(__env));
    }

    template <class _Sender, class _Env>
    static auto transform_sender(set_value_t, _Sender&&, _Env&&)
    {
      return __not_a_sender<_WHAT_(_SENDER_TYPE_IS_NOT_DECAY_COPYABLE_),
                            _WITH_PRETTY_SENDER_<_Sender>>{};
    }
  };

  inline constexpr on_t on{};

  template <>
  struct __sexpr_impl<on_t> : __sexpr_defaults
  {
    static constexpr auto __get_attrs =  //
      []<class _Data, class _Child>(__ignore, _Data const & __data, _Child const & __child) noexcept
    {
      if constexpr (scheduler<_Data>)
      {
        // This is the case where `on` was called like `on(sch, sndr)`, which is equivalent
        // to `continues_on(starts_on(sndr, sch), old_sch)`.
        using __attrs_t = __on::__attrs<_Child, _Data>;
        return __attrs_t{__child, __data};
      }
      else
      {
        // This is the case where `on` was called like `sndr | on(sch, clsur)` or
        // `on(sndr, sch, clsur)`, which is equivalent to
        // `continues_on(clsur(continues_on(sndr, sch)), old_sch)`.
        auto const& [__sched, __clsur] = __data;
        using __attrs_t = __on::__attrs<_Child, decltype(__sched), decltype(__clsur)>;
        return __attrs_t{__child, __sched, __clsur};
      }
    };

    template <class _Sender, class _Env>
    static constexpr auto __get_completion_signatures()
    {
      using __sndr_t = __detail::__transform_sender_result_t<on_t, set_value_t, _Sender, _Env>;
      return STDEXEC::get_completion_signatures<__sndr_t, _Env>();
    }
  };
}  // namespace STDEXEC

STDEXEC_PRAGMA_POP()
