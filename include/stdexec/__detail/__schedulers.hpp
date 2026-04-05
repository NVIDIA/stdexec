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
#include "__completion_signatures_of.hpp"  // IWYU pragma: keep for the sender concept
#include "__concepts.hpp"
#include "__config.hpp"
#include "__domain.hpp"
#include "__env.hpp"
#include "__query.hpp"
#include "__sender_concepts.hpp"
#include "__type_traits.hpp"
#include "__utility.hpp"

namespace STDEXEC
{
  // scheduler concept opt-in tag
  struct scheduler_t
  {};

  /////////////////////////////////////////////////////////////////////////////
  // [exec.schedule]
  template <class _Scheduler>
  concept __has_schedule_member = requires(_Scheduler &&__sched) {
    static_cast<_Scheduler &&>(__sched).schedule();
  };

  struct schedule_t
  {
    template <class _Scheduler>
      requires __has_schedule_member<_Scheduler>
    STDEXEC_ATTRIBUTE(host, device, always_inline)
    auto operator()(_Scheduler &&__sched) const
      noexcept(noexcept(static_cast<_Scheduler &&>(__sched).schedule()))
        -> decltype(static_cast<_Scheduler &&>(__sched).schedule())
    {
      static_assert(sender<decltype(static_cast<_Scheduler &&>(__sched).schedule())>,
                    "schedule() member functions must return a sender");
      return static_cast<_Scheduler &&>(__sched).schedule();
    }

    template <class _Scheduler>
      requires __has_schedule_member<_Scheduler> || __tag_invocable<schedule_t, _Scheduler>
    [[deprecated("the use of tag_invoke for schedule is deprecated")]]
    STDEXEC_ATTRIBUTE(host, device, always_inline)  //
      auto operator()(_Scheduler &&__sched) const
      noexcept(__nothrow_tag_invocable<schedule_t, _Scheduler>)
        -> __tag_invoke_result_t<schedule_t, _Scheduler>
    {
      static_assert(sender<__tag_invoke_result_t<schedule_t, _Scheduler>>);
      return __tag_invoke(*this, static_cast<_Scheduler &&>(__sched));
    }
  };

  inline constexpr schedule_t schedule{};

  /////////////////////////////////////////////////////////////////////////////
  // [exec.sched]
  template <class _Scheduler>
  concept scheduler = __callable<schedule_t, _Scheduler>  //
                   && __std::equality_comparable<__decay_t<_Scheduler>>
                   && __std::copy_constructible<__decay_t<_Scheduler>>
                   && __nothrow_move_constructible<__decay_t<_Scheduler>>;

  template <scheduler _Scheduler>
  using schedule_result_t = __call_result_t<schedule_t, _Scheduler>;

  template <class _SchedulerProvider>
  concept __start_scheduler_provider = requires(_SchedulerProvider const &__sp) {
    { get_start_scheduler(__sp) } -> scheduler;
  };

  struct get_start_scheduler_t : __query<get_start_scheduler_t>
  {
    using __query<get_start_scheduler_t>::operator();

    // defined in __read_env.hpp
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto operator()() const noexcept;

    template <class _Env>
    STDEXEC_ATTRIBUTE(always_inline, host, device)
    static constexpr void __validate() noexcept
    {
      static_assert(__nothrow_callable<get_start_scheduler_t, _Env const &>);
      static_assert(scheduler<__call_result_t<get_start_scheduler_t, _Env const &>>);
    }

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    static consteval auto query(forwarding_query_t) noexcept -> bool
    {
      return true;
    }
  };

  struct get_scheduler_t : __query<get_scheduler_t>
  {
    using __query<get_scheduler_t>::operator();

    // NOT TO SPEC: If _Env does not have a get_scheduler query but does have a
    // get_start_scheduler query, then we return the start scheduler.
    template <class _Env>
      requires(!__callable<__query<get_scheduler_t>, _Env const &>)
           && __callable<get_start_scheduler_t, _Env const &>
    STDEXEC_ATTRIBUTE(always_inline, host, device)
    constexpr auto operator()(_Env const &__env) const
      noexcept(__nothrow_callable<get_start_scheduler_t, _Env const &>)
        -> __call_result_t<get_start_scheduler_t, _Env const &>
    {
      return get_start_scheduler(__env);
    }

    // defined in __read_env.hpp
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto operator()() const noexcept;

    template <class _Env>
    STDEXEC_ATTRIBUTE(always_inline, host, device)
    static constexpr void __validate() noexcept
    {
      static_assert(__nothrow_callable<get_scheduler_t, _Env const &>);
      static_assert(scheduler<__call_result_t<get_scheduler_t, _Env const &>>);
    }

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    static consteval auto query(forwarding_query_t) noexcept -> bool
    {
      return true;
    }
  };

  //! The type for `get_delegation_scheduler` [exec.get.delegation.scheduler]
  //! A query object that asks for a scheduler that can be used to delegate
  //! work to for the purpose of forward progress delegation ([intro.progress]).
  struct get_delegation_scheduler_t : __query<get_delegation_scheduler_t>
  {
    using __query<get_delegation_scheduler_t>::operator();

    // defined in __read_env.hpp
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto operator()() const noexcept;

    template <class _Env>
    STDEXEC_ATTRIBUTE(always_inline, host, device)
    static constexpr void __validate() noexcept
    {
      static_assert(__nothrow_callable<get_delegation_scheduler_t, _Env const &>);
      static_assert(scheduler<__call_result_t<get_delegation_scheduler_t, _Env const &>>);
    }

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    static consteval auto query(forwarding_query_t) noexcept -> bool
    {
      return true;
    }
  };

  //! @brief A query type for asking a sender's attributes for the scheduler on which that
  //! sender will complete.
  //!
  //! @tparam _Tag one of set_value_t, set_error_t, or set_stopped_t
  template <__completion_tag _Tag>
  struct get_completion_scheduler_t
  {
    template <class Sig>
    inline static constexpr get_completion_scheduler_t<_Tag> (*signature)(Sig) = nullptr;

    // This function object reads the completion scheduler from an attribute object or a
    // scheduler, accounting for the fact that the query member function may or may not
    // accept an environment.
    struct __read_query_t
    {
      using __self_t = get_completion_scheduler_t;

      template <class _Attrs, class... _Env>
        requires(__queryable_with<_Attrs, __self_t, _Env const &> || ...)
             || __queryable_with<_Attrs, __self_t>
      constexpr auto operator()(_Attrs const &__attrs, _Env const &...__env) const noexcept
      {
        if constexpr ((__queryable_with<_Attrs, __self_t, _Env const &> || ...))
        {
          static_assert(noexcept(__attrs.query(__self_t{}, __env...)));
          static_assert(scheduler<__query_result_t<_Attrs, __self_t, _Env const &...>>);
          return __attrs.query(__self_t{}, __env...);
        }
        else
        {
          static_assert(noexcept(__attrs.query(__self_t{})));
          static_assert(scheduler<__query_result_t<_Attrs, __self_t>>);
          return __attrs.query(__self_t{});
        }
      }
    };

   private:
    // A scheduler might have a completion scheduler different from itself; for example, an
    // inline_scheduler completes wherever the scheduler's sender is started. So we
    // recursively ask the scheduler for its completion scheduler until we find one whose
    // completion scheduler is equal to itself (or it doesn't have one).
    struct __recurse_query_t
    {
      template <class _Self = __recurse_query_t, class _Sch, class... _Env>
      constexpr auto operator()([[maybe_unused]] _Sch __sch, _Env const &...__env) const noexcept
      {
        static_assert(scheduler<_Sch>);
        // When determining where the scheduler's operations will complete, we query
        // for the completion scheduler of the value channel:
        using __read_query_t = typename get_completion_scheduler_t<set_value_t>::__read_query_t;

        if constexpr (__callable<__read_query_t, _Sch, _Env const &...>)
        {
          using __sch2_t = __decay_t<__call_result_t<__read_query_t, _Sch, _Env const &...>>;
          if constexpr (__same_as<_Sch, __sch2_t>)
          {
            _Sch __prev = __sch;
            do
            {
              __prev = std::exchange(__sch, __read_query_t{}(__sch, __env...));
            }
            while (__prev != __sch);
            return __sch;
          }
          else
          {
            // New scheduler has different type. Recurse!
            return _Self{}(__read_query_t{}(__sch, __env...), __env...);
          }
        }
        else
        {
          if constexpr (__callable<__read_query_t,
                                   env_of_t<schedule_result_t<_Sch>>,
                                   _Env const &...>)
          {
            STDEXEC_ASSERT_FN(__sch == __read_query_t{}(get_env(__sch.schedule()), __env...));
          }
          return __sch;
        }
      }
    };

    template <class _Attrs, class... _Env, class _Sch>
    static constexpr auto __check_domain(_Sch __sch) noexcept -> _Sch
    {
      static_assert(scheduler<_Sch>);
      if constexpr (__callable<get_completion_domain_t<_Tag>, _Attrs const &, _Env const &...>)
      {
        using __domain_t =
          __call_result_t<get_completion_domain_t<_Tag>, _Attrs const &, _Env const &...>;
        static_assert(__same_as<__domain_t, __detail::__scheduler_domain_t<_Sch, _Env const &...>>,
                      "the sender claims to complete on a domain that is not the domain of its "
                      "completion "
                      "scheduler");
      }
      return __sch;
    }

    template <class _Attrs, class... _Env>
    static consteval auto __get_declfn() noexcept
    {
      // If __attrs has a completion scheduler, then return it (after checking the scheduler
      // for _its_ completion scheduler):
      if constexpr (__callable<__read_query_t, _Attrs const &, _Env const &...>)
      {
        using __result_t =
          __call_result_t<__recurse_query_t,
                          __call_result_t<__read_query_t, _Attrs const &, _Env const &...>,
                          _Env const &...>;
        return __declfn<__result_t>();
      }
      // Otherwise, if __attrs indicates that its sender completes inline, then we can ask
      // the environment for the current scheduler and return that (after checking the
      // scheduler for _its_ completion scheduler).
      else if constexpr (__completes_inline<_Tag, _Attrs, _Env...>
                         && (__callable<get_start_scheduler_t, _Env const &> || ...))
      {
        using __result_t = __call_result_t<__recurse_query_t,
                                           __call_result_t<get_start_scheduler_t, _Env const &>...,
                                           _Env const &...>;
        return __declfn<__result_t>();
      }
      // Otherwise, if we are asking a scheduler for a completion scheduler, return the
      // scheduler itself.
      else if constexpr (scheduler<_Attrs> && sizeof...(_Env) != 0)
      {
        return __declfn<__decay_t<_Attrs>>();
      }
      // Otherwise, no completion scheduler can be determined. Return void.
    }

   public:
    template <class _Attrs,
              class... _Env,
              auto _DeclFn = __get_declfn<_Attrs const &, _Env const &...>()>
    constexpr auto operator()(_Attrs const &__attrs, _Env const &...__env) const noexcept
      -> __unless_one_of_t<decltype(_DeclFn()), void>
    {
      // If __attrs has a completion scheduler, then return it (after checking the scheduler
      // for _its_ completion scheduler):
      if constexpr (__callable<__read_query_t, _Attrs const &, _Env const &...>)
      {
        return __check_domain<_Attrs, _Env...>(
          __recurse_query_t{}(__read_query_t{}(__attrs, __env...), __env...));
      }
      // Otherwise, if __attrs indicates that its sender completes inline, then we can ask
      // the environment for the current scheduler and return that (after checking the
      // scheduler for _its_ completion scheduler).
      else if constexpr (__completes_inline<_Tag, _Attrs, _Env...>
                         && __callable<get_start_scheduler_t, _Env const &...>)
      {
        return __check_domain<_Attrs, _Env...>(
          __recurse_query_t{}(get_start_scheduler(__env...), __hide_scheduler{__env}...));
      }
      // Otherwise, if we are asking a scheduler for a completion scheduler, return the
      // scheduler itself.
      else
      {
        return __attrs;
      }
    }

    static constexpr auto query(forwarding_query_t) noexcept -> bool
    {
      return true;
    }
  };

  template <class _Tag, class _Sender, class... _Env>
  concept __has_completion_scheduler_for =
    __sends<_Tag, _Sender, _Env...>
    && __callable<get_completion_scheduler_t<_Tag>, env_of_t<_Sender>, _Env const &...>;

  struct __execute_may_block_caller_t : __query<__execute_may_block_caller_t, true>
  {
    template <class _Attrs>
    STDEXEC_ATTRIBUTE(always_inline, host, device)
    static constexpr void __validate() noexcept
    {
      static_assert(
        __std::same_as<bool, __call_result_t<__execute_may_block_caller_t, _Attrs const &>>);
      static_assert(__nothrow_callable<__execute_may_block_caller_t, _Attrs const &>);
    }
  };

  struct get_forward_progress_guarantee_t
    : __query<get_forward_progress_guarantee_t,
              forward_progress_guarantee::weakly_parallel,
              __q1<__decay_t>>
  {
    template <class _Attrs>
    STDEXEC_ATTRIBUTE(always_inline, host, device)
    static constexpr void __validate() noexcept
    {
      using __result_t = __call_result_t<get_forward_progress_guarantee_t, _Attrs const &>;
      static_assert(__std::same_as<forward_progress_guarantee, __result_t>);
      static_assert(__nothrow_callable<get_forward_progress_guarantee_t, _Attrs const &>);
    }
  };

  inline constexpr __execute_may_block_caller_t     __execute_may_block_caller{};
  inline constexpr get_forward_progress_guarantee_t get_forward_progress_guarantee{};
  inline constexpr get_scheduler_t                  get_scheduler{};
  inline constexpr get_start_scheduler_t            get_start_scheduler{};
  inline constexpr get_delegation_scheduler_t       get_delegation_scheduler{};

#if !STDEXEC_GCC() || defined(__OPTIMIZE_SIZE__)
  template <__completion_tag _Query>
  inline constexpr get_completion_scheduler_t<_Query> get_completion_scheduler{};
#else
  template <>
  inline constexpr get_completion_scheduler_t<set_value_t> get_completion_scheduler<set_value_t>{};
  template <>
  inline constexpr get_completion_scheduler_t<set_error_t> get_completion_scheduler<set_error_t>{};
  template <>
  inline constexpr get_completion_scheduler_t<set_stopped_t>
    get_completion_scheduler<set_stopped_t>{};
#endif

  template <class _Tag, sender _Sender, class... _Env>
    requires __sends<_Tag, _Sender, _Env...>
  using __completion_scheduler_of_t =
    __call_result_t<get_completion_scheduler_t<_Tag>, env_of_t<_Sender>, _Env const &...>;

  // TODO(ericniebler): examine all uses of this struct.
  template <class _Scheduler>
  struct __sched_attrs
  {
    template <class... _Env>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto query(get_completion_scheduler_t<set_value_t>,
                         _Env const &...__env) const noexcept
      -> __call_result_t<get_completion_scheduler_t<set_value_t>, _Scheduler, _Env const &...>
    {
      return get_completion_scheduler<set_value_t>(__sched_, __env...);
    }

    template <class... _Env>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto query(get_completion_domain_t<set_value_t>, _Env const &...) const noexcept
      -> __call_result_t<get_completion_domain_t<set_value_t>, _Scheduler, _Env const &...>
    {
      return {};
    }

    _Scheduler __sched_;
  };

  template <class _Scheduler>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __sched_attrs(_Scheduler) -> __sched_attrs<_Scheduler>;

  //////////////////////////////////////////////////////////////////////////////
  // See SCHED-ENV from [exec.snd.expos]
  template <class _Scheduler, class... _Env>
  struct __sched_env
  {
    using __domain_t = __minvoke_or_q<__detail::__scheduler_domain_t, void, _Scheduler, _Env...>;

    constexpr explicit __sched_env(_Scheduler __sch, _Env const &...) noexcept
      : __sched_{std::move(__sch)}
    {}

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto query(get_start_scheduler_t) const noexcept
    {
      return __sched_;
    }

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto query(get_domain_t) const noexcept -> __domain_t
      requires __not_same_as<__domain_t, void>
    {
      return {};
    }

    _Scheduler __sched_;
  };

  template <class _Scheduler, class... _Env>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
  __sched_env(_Scheduler, _Env const &...) -> __sched_env<_Scheduler, _Env...>;

  template <class _Sch>
  constexpr auto __mk_sch_env(_Sch __sch) noexcept
  {
    return __sched_env{std::move(__sch)};
  }

  template <class _Sch, class _Env>
  constexpr auto __mk_sch_env([[maybe_unused]] _Sch __sch, _Env const &__env) noexcept
  {
    if constexpr (__completes_inline<set_value_t, env_of_t<schedule_result_t<_Sch>>, _Env>
                  && __callable<get_start_scheduler_t, _Env const &>)
    {
      auto __sch2 = get_completion_scheduler<set_value_t>(get_start_scheduler(__env),
                                                          __hide_scheduler{__env});
      return __sched_env{std::move(__sch2), __env};
    }
    else
    {
      return __sched_env{std::move(__sch), __env};
    }
  }

  namespace __detail
  {
    template <class... _SetTags>
    struct __mk_secondary_env_impl
    {
      template <class _CvFn, sender _Sender, class _Env>
      constexpr auto operator()(_CvFn, _Sender const &__sndr, _Env const &__env) const noexcept
      {
        using __domain_t = __make_domain_t<__completion_domain_of_t<_SetTags, _Sender, _Env>...>;
        // We can only know the scheduler that the secondary sender is started on if there
        // is exactly one kind of completion that starts the secondary sender.
        STDEXEC_CONSTEXPR_LOCAL bool __has_completion_scheduler =
          sizeof...(_SetTags) == 1
          && (__has_completion_scheduler_for<_SetTags, __mcall1<_CvFn, _Sender>, _Env> || ...);

        if constexpr (__has_completion_scheduler)
        {
          auto __sch = (get_completion_scheduler<_SetTags>(get_env(__sndr), __fwd_env(__env)), ...);
          return STDEXEC::__mk_sch_env(__sch, __fwd_env(__env));
        }
        else if constexpr (__not_same_as<__domain_t, indeterminate_domain<>>)
        {
          return __env::cprop<get_domain_t, __domain_t{}>();
        }
        else
        {
          return env{};
        }
      }
    };
  }  // namespace __detail

  //! This environment is for when one sender is started from the completion of another
  //! sender. In that case, the completion scheduler/domain for the first sender should
  //! be used as the scheduler/domain for the second sender.
  //!
  //! This environment is used by the \c let_[value|error|stopped] algorithms as well as
  //! the \c finally algorithm and \c sequence algorithms.
  //!
  //! \note This env assumes that the results of the first sender are decay-copied into
  //! the operation state of the composite sender.
  //!
  //! \tparam _CvSender The sender whose completion is starting the next sender.
  //! \tparam _Env The environment of the receiver connected to the primary sender.
  //! \tparam _SetTags The completions that cause the next sender to start. For example,
  //! for \c let_value, this would be \c set_value_t, and for \c finally, this would be
  //! \c set_value_t, \c set_error_t, and \c set_stopped_t.
  template <class... _SetTags>
  struct __mk_secondary_env_t
  {
    template <class _CvFn, class _Sender, class _Env>
    constexpr auto operator()(_CvFn __cv, _Sender const &__sndr, _Env const &__env) const noexcept
    {
      using namespace __detail;
      using __env_t          = __fwd_env_t<_Env const &>;
      using __never_sends_fn = __mbind_back_q<__never_sends_t, _Sender, __env_t>;
      using __make_env_fn    = __mremove_if<__never_sends_fn, __qq<__mk_secondary_env_impl>>;
      using __impl_t         = __minvoke<__make_env_fn, _SetTags...>;
      return __impl_t{}(__cv, __sndr, __env);
    }
  };

  template <class _CvSender, class _Env, class... _SetTags>
  using __secondary_env_t =
    __call_result_t<__mk_secondary_env_t<_SetTags...>, __copy_cvref_fn<_CvSender>, _CvSender, _Env>;

  //////////////////////////////////////////////////////////////////////////////////////////
  // __infallible_scheduler
  template <class _Env>
  using __unstoppable_env_t = env<prop<get_stop_token_t, never_stop_token>, _Env>;

  template <class _Sender, class... _Env>
  concept __infallible_sender = (!__sends<set_error_t, _Sender, _Env...>)
                             && (!__sends<set_stopped_t, _Sender, _Env...>);

  template <class _Scheduler, class... _Env>
  concept __infallible_scheduler = scheduler<_Scheduler>
                                && __infallible_sender<__result_of<schedule, _Scheduler>, _Env...>;

  // Deprecated interfaces
  using get_delegatee_scheduler_t
    [[deprecated("get_delegatee_scheduler_t has been renamed "
                 "get_delegation_scheduler_t")]] = get_delegation_scheduler_t;

  inline constexpr auto &get_delegatee_scheduler [[deprecated("get_delegatee_scheduler has been "
                                                              "renamed get_delegation_scheduler")]]
  = get_delegation_scheduler;
}  // namespace STDEXEC
