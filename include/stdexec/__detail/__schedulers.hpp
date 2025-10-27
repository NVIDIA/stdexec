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

    //! @brief A wrapper around an environment that hides a set of queries.
    template <class _Env, class... _Queries>
    struct __hide_query {
      explicit constexpr __hide_query(_Env&& __env, _Queries...) noexcept
        : __env_{static_cast<_Env&&>(__env)} {}

      template <class _Query, class... _As>
        requires __none_of<_Query, _Queries...> && __queryable_with<_Env, _Query, _As...>
      constexpr auto operator()(_Query __query, const _As&... __as) const
        noexcept(__nothrow_queryable_with<_Env, _Query, _As...>) -> __query_result_t<_Env, _Query, _As...> {
        return __env_.query(__query, __as...);
      }

    private:
      _Env __env_;
    };

    template <class _Env>
    struct __hide_scheduler : __hide_query<_Env, get_scheduler_t, get_domain_t> {
      explicit constexpr __hide_scheduler(_Env&& __env) noexcept
        : __hide_query<_Env, get_scheduler_t, get_domain_t>{static_cast<_Env&&>(__env), {}, {}} {}
    };

    template <class _Env>
    __hide_scheduler(_Env&&) -> __hide_scheduler<_Env>;

    //! @brief A query type for asking a sender's attributes for the scheduler on which that
    //! sender will complete.
    //!
    //! @tparam _Tag one of set_value_t, set_error_t, or set_stopped_t
    template <__completion_tag _Tag>
    struct get_completion_scheduler_t {
      template <class Sig>
      static inline constexpr get_completion_scheduler_t<_Tag> (*signature)(Sig) = nullptr;

      // This function object reads the completion scheduler from an attribute object or a
      // scheduler, accounting for the fact that the query member function may or may not
      // accept an environment.
      struct __read_query_t {
        template <class _Attrs, class _GetComplSch = get_completion_scheduler_t>
          requires __queryable_with<_Attrs, _GetComplSch>
        constexpr auto operator()(const _Attrs& __attrs, __ignore = {}) const noexcept
          -> __query_result_t<_Attrs, _GetComplSch> {
          static_assert(noexcept(__attrs.query(_GetComplSch{})));
          return __attrs.query(_GetComplSch{});
        }

        template <class _Attrs, class _Env, class _GetComplSch = get_completion_scheduler_t>
          requires __queryable_with<_Attrs, _GetComplSch, const _Env&>
        constexpr auto operator()(const _Attrs& __attrs, const _Env& __env) const noexcept
          -> __query_result_t<_Attrs, _GetComplSch, const _Env&> {
          static_assert(noexcept(__attrs.query(_GetComplSch{}, __env)));
          return __attrs.query(_GetComplSch{}, __env);
        }
      };

    private:
      // A scheduler might have a completion scheduler different from itself; for example, an
      // inline_scheduler completes wherever the scheduler's sender is started. So we
      // recursively ask the scheduler for its completion scheduler until we find one whose
      // completion scheduler is equal to itself (or it doesn't have one).
      struct __recurse_query_t {
        template <class _Self = __recurse_query_t, class _Sch, class... _Env>
        constexpr auto operator()([[maybe_unused]] _Sch __sch, const _Env&... __env) const noexcept {
          // When determining where the scheduler's operations will complete, we query
          // for the completion scheduler of the value channel:
          using __read_query_t = typename get_completion_scheduler_t<set_value_t>::__read_query_t;

          if constexpr (__callable<__read_query_t, _Sch, const _Env&...>) {
            using __sch2_t = __call_result_t<__read_query_t, _Sch, const _Env&...>;
            if constexpr (__same_as<_Sch, __sch2_t>) {
              _Sch __prev = __sch;
              do {
                __prev = std::exchange(__sch, __read_query_t{}(__sch, __env...));
              } while (__prev != __sch);
              return __sch;
            }
            else {
              // New scheduler has different type. Recurse!
              return _Self{}(__read_query_t{}(__sch, __env...), __env...);
            }
          }
          else {
            // TODO sfinae on schedulers being comparable first
            #if 0
            if constexpr (__callable<__read_query_t, env_of_t<__call_result_t<schedule_t, _Sch>>, const _Env&...>) {
              assert(__sch == __read_query_t{}(get_env(__sch.schedule()), __env...));
            }
            #endif
            return __sch;
          }
        }
      };

      template <class _Attrs, class... _Env, class _Sch>
      constexpr static auto __check_domain(_Sch __sch) noexcept -> _Sch {
        // Sanity check: if a completion domain can be determined, then it must match the
        // domain of the completion scheduler.
        if constexpr (__callable<get_completion_domain_t<_Tag>, const _Attrs&, const _Env&...>) {
          using __domain_t = __call_result_t<get_completion_domain_t<_Tag>, const _Attrs&, const _Env&...>;
          static_assert(__same_as<__domain_t, __detail::__scheduler_domain_t<_Sch, const _Env&...>>,
                        "the sender claims to complete on a domain that is not the domain of its completion scheduler");
        }
        return __sch;
      }

      template <class _Attrs, class... _Env>
      static consteval auto __get_declfn() noexcept
      {
        // If __attrs has a completion scheduler, then return it (after checking the scheduler
        // for _its_ completion scheduler):
        if constexpr (__callable<__read_query_t, const _Attrs&, const _Env&...>)
        {
          return __declfn<decltype(__recurse_query_t{}(
            __read_query_t{}(__declval<_Attrs>(), __declval<_Env>()...), __declval<_Env>()...))>{};
        }
        // Otherwise, if __attrs indicates that its sender completes inline, then we can ask
        // the environment for the current scheduler and return that (after checking the
        // scheduler for _its_ completion scheduler).
        else if constexpr (__completes_inline<_Attrs, _Env...> && __callable<get_scheduler_t, const _Env&...>)
        {
          return __declfn<decltype(__recurse_query_t{}(
            get_scheduler(__declval<_Env>()...), __hide_scheduler{__declval<_Env>()}...))>{};
        }
        // Otherwise, no completion scheduler can be determined. Return void.
      }

    public:
      template <class _Attrs, class... _Env, auto _DeclFn = __get_declfn<const _Attrs&, const _Env&...>()>
      constexpr auto operator()(const _Attrs& __attrs, const _Env&... __env) const noexcept
        -> __unless_one_of_t<decltype(_DeclFn()), void>
      {
        // If __attrs has a completion scheduler, then return it (after checking the scheduler
        // for _its_ completion scheduler):
        if constexpr (__callable<__read_query_t, const _Attrs&, const _Env&...>)
        {
          return __check_domain<_Attrs, _Env...>(__recurse_query_t{}(__read_query_t{}(__attrs, __env...), __env...));
        }
        // Otherwise, if __attrs indicates that its sender completes inline, then we can ask
        // the environment for the current scheduler and return that (after checking the
        // scheduler for _its_ completion scheduler).
        else
        {
          return __check_domain<_Attrs, _Env...>(__recurse_query_t{}(get_scheduler(__env...), __hide_scheduler{__env}...));
        }
      }

      static constexpr auto query(forwarding_query_t) noexcept -> bool
      {
        return true;
      }
    };
  } // namespace __queries

  template <class _Scheduler, class _Domain = __none_such>
  struct __sched_attrs {
    using __t = __sched_attrs;
    using __id = __sched_attrs;

    using __scheduler_t = __decay_t<_Scheduler>;
    using __sched_domain_t = __query_result_or_t<get_completion_domain_t<set_value_t>, __scheduler_t, default_domain>;

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto query(get_completion_scheduler_t<set_value_t>, __ignore = {}) const noexcept -> __scheduler_t {
      return __sched_;
    }

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto query(get_completion_domain_t<set_value_t>) const noexcept -> __sched_domain_t
    {
      return {};
    }

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto query(get_domain_override_t) const noexcept -> _Domain
      requires(!same_as<_Domain, __none_such>)
    {
      return {};
    }

    _Scheduler __sched_;
    STDEXEC_ATTRIBUTE(no_unique_address) _Domain __domain_ { };
  };

  template <class _Scheduler, class _LateDomain = __none_such>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __sched_attrs(_Scheduler, _LateDomain = {})
    -> __sched_attrs<std::unwrap_reference_t<_Scheduler>, _LateDomain>;

  template <class _Scheduler>
  struct __sched_env {
    using __t = __sched_env;
    using __id = __sched_env;

    using __scheduler_t = __decay_t<_Scheduler>;
    using __sched_domain_t = __query_result_or_t<get_domain_t, __scheduler_t, default_domain>;

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto query(get_scheduler_t) const noexcept -> __scheduler_t {
      return __sched_;
    }

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto query(get_domain_t) const noexcept -> __sched_domain_t {
      return {};
    }

    _Scheduler __sched_;
  };

  template <class _Scheduler>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
    __sched_env(_Scheduler) -> __sched_env<std::unwrap_reference_t<_Scheduler>>;

  struct __mk_sch_env_t {
    template <class _Sch, class... _Env>
    constexpr auto operator()([[maybe_unused]] _Sch __sch, const _Env&... __env) const noexcept {
      if constexpr (__completes_inline<env_of_t<schedule_result_t<_Sch>>, _Env...>
                    && (__callable<get_scheduler_t, const _Env&> || ...))
      {
        return __sched_env{get_scheduler(__env)...};
      }
      else
      {
        return __sched_env{__sch};
      }
    }
  };

  inline constexpr __mk_sch_env_t __mk_sch_env{};

} // namespace stdexec
