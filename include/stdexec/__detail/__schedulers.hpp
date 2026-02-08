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
#include "__completion_signatures_of.hpp" // IWYU pragma: keep for the sender concept
#include "__config.hpp"
#include "__domain.hpp"
#include "__query.hpp"
#include "__utility.hpp"

namespace STDEXEC {
  // scheduler concept opt-in tag
  struct scheduler_t { };

  /////////////////////////////////////////////////////////////////////////////
  // [exec.schedule]
  namespace __sched {
    template <class _Scheduler>
    concept __has_schedule_member = requires(_Scheduler&& __sched) {
      static_cast<_Scheduler&&>(__sched).schedule();
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
        requires __has_schedule_member<_Scheduler> || __tag_invocable<schedule_t, _Scheduler>
      [[deprecated("the use of tag_invoke for schedule is deprecated")]]
      STDEXEC_ATTRIBUTE(host, device, always_inline) //
        auto operator()(_Scheduler&& __sched) const
        noexcept(__nothrow_tag_invocable<schedule_t, _Scheduler>)
          -> __tag_invoke_result_t<schedule_t, _Scheduler> {
        static_assert(sender<__tag_invoke_result_t<schedule_t, _Scheduler>>);
        return __tag_invoke(*this, static_cast<_Scheduler&&>(__sched));
      }

      static constexpr auto query(forwarding_query_t) noexcept -> bool {
        return false;
      }
    };
  } // namespace __sched

  using __sched::schedule_t;
  inline constexpr schedule_t schedule{};

  /////////////////////////////////////////////////////////////////////////////
  // [exec.sched]
  template <class _Scheduler>
  concept scheduler = __callable<schedule_t, _Scheduler> //
                   && __std::equality_comparable<__decay_t<_Scheduler>>
                   && __std::copy_constructible<__decay_t<_Scheduler>>
                   && std::is_nothrow_move_constructible_v<__decay_t<_Scheduler>>;

  template <scheduler _Scheduler>
  using schedule_result_t = __call_result_t<schedule_t, _Scheduler>;

  template <class _SchedulerProvider>
  concept __scheduler_provider = requires(const _SchedulerProvider& __sp) {
    { get_scheduler(__sp) } -> scheduler;
  };

  namespace __queries {
    struct get_scheduler_t : __query<get_scheduler_t> {
      using __query<get_scheduler_t>::operator();

      // defined in __read_env.hpp
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()() const noexcept;

      template <class _Env>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr void __validate() noexcept {
        static_assert(__nothrow_callable<get_scheduler_t, const _Env&>);
        static_assert(scheduler<__call_result_t<get_scheduler_t, const _Env&>>);
      }

      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      static consteval auto query(forwarding_query_t) noexcept -> bool {
        return true;
      }
    };

    //! The type for `get_delegation_scheduler` [exec.get.delegation.scheduler]
    //! A query object that asks for a scheduler that can be used to delegate
    //! work to for the purpose of forward progress delegation ([intro.progress]).
    struct get_delegation_scheduler_t : __query<get_delegation_scheduler_t> {
      using __query<get_delegation_scheduler_t>::operator();

      // defined in __read_env.hpp
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()() const noexcept;

      template <class _Env>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr void __validate() noexcept {
        static_assert(__nothrow_callable<get_delegation_scheduler_t, const _Env&>);
        static_assert(scheduler<__call_result_t<get_delegation_scheduler_t, const _Env&>>);
      }

      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      static consteval auto query(forwarding_query_t) noexcept -> bool {
        return true;
      }
    };

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
          static_assert(scheduler<__query_result_t<_Attrs, _GetComplSch>>);
          return __attrs.query(_GetComplSch{});
        }

        template <class _Attrs, class _Env, class _GetComplSch = get_completion_scheduler_t>
          requires __queryable_with<_Attrs, _GetComplSch, const _Env&>
        constexpr auto operator()(const _Attrs& __attrs, const _Env& __env) const noexcept
          -> __query_result_t<_Attrs, _GetComplSch, const _Env&> {
          static_assert(noexcept(__attrs.query(_GetComplSch{}, __env)));
          static_assert(scheduler<__query_result_t<_Attrs, _GetComplSch, const _Env&>>);
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
        constexpr auto
          operator()([[maybe_unused]] _Sch __sch, const _Env&... __env) const noexcept {
          static_assert(scheduler<_Sch>);
          // When determining where the scheduler's operations will complete, we query
          // for the completion scheduler of the value channel:
          using __read_query_t = typename get_completion_scheduler_t<set_value_t>::__read_query_t;

          if constexpr (__callable<__read_query_t, _Sch, const _Env&...>) {
            using __sch2_t = __decay_t<__call_result_t<__read_query_t, _Sch, const _Env&...>>;
            if constexpr (__same_as<_Sch, __sch2_t>) {
              _Sch __prev = __sch;
              do {
                __prev = std::exchange(__sch, __read_query_t{}(__sch, __env...));
              } while (__prev != __sch);
              return __sch;
            } else {
              // New scheduler has different type. Recurse!
              return _Self{}(__read_query_t{}(__sch, __env...), __env...);
            }
          } else {
            if constexpr (
              __callable<__read_query_t, env_of_t<schedule_result_t<_Sch>>, const _Env&...>) {
              STDEXEC_ASSERT_FN(__sch == __read_query_t{}(get_env(__sch.schedule()), __env...));
            }
            return __sch;
          }
        }
      };

      template <class _Attrs, class... _Env, class _Sch>
      static constexpr auto __check_domain(_Sch __sch) noexcept -> _Sch {
        static_assert(scheduler<_Sch>);
        if constexpr (__callable<get_completion_domain_t<_Tag>, const _Attrs&, const _Env&...>) {
          using __domain_t =
            __call_result_t<get_completion_domain_t<_Tag>, const _Attrs&, const _Env&...>;
          static_assert(
            __same_as<__domain_t, __detail::__scheduler_domain_t<_Sch, const _Env&...>>,
            "the sender claims to complete on a domain that is not the domain of its completion "
            "scheduler");
        }
        return __sch;
      }

      template <class _Attrs, class... _Env>
      static consteval auto __get_declfn() noexcept {
        // If __attrs has a completion scheduler, then return it (after checking the scheduler
        // for _its_ completion scheduler):
        if constexpr (__callable<__read_query_t, const _Attrs&, const _Env&...>) {
          using __result_t = __call_result_t<
            __recurse_query_t,
            __call_result_t<__read_query_t, const _Attrs&, const _Env&...>,
            const _Env&...
          >;
          return __declfn<__result_t>();
        }
        // Otherwise, if __attrs indicates that its sender completes inline, then we can ask
        // the environment for the current scheduler and return that (after checking the
        // scheduler for _its_ completion scheduler).
        else if constexpr (
          __completes_inline<_Tag, _Attrs, _Env...>
          && (__callable<get_scheduler_t, const _Env&> || ...)) {
          using __result_t = __call_result_t<
            __recurse_query_t,
            __call_result_t<get_scheduler_t, const _Env&>...,
            const _Env&...
          >;
          return __declfn<__result_t>();
        }
        // Otherwise, if we are asking a scheduler for a completion scheduler, return the
        // scheduler itself.
        else if constexpr (scheduler<_Attrs> && sizeof...(_Env) != 0) {
          return __declfn<__decay_t<_Attrs>>();
        }
        // Otherwise, no completion scheduler can be determined. Return void.
      }

     public:
      template <
        class _Attrs,
        class... _Env,
        auto _DeclFn = __get_declfn<const _Attrs&, const _Env&...>()
      >
      constexpr auto operator()(const _Attrs& __attrs, const _Env&... __env) const noexcept
        -> __unless_one_of_t<decltype(_DeclFn()), void> {
        // If __attrs has a completion scheduler, then return it (after checking the scheduler
        // for _its_ completion scheduler):
        if constexpr (__callable<__read_query_t, const _Attrs&, const _Env&...>) {
          return __check_domain<_Attrs, _Env...>(
            __recurse_query_t{}(__read_query_t{}(__attrs, __env...), __env...));
        }
        // Otherwise, if __attrs indicates that its sender completes inline, then we can ask
        // the environment for the current scheduler and return that (after checking the
        // scheduler for _its_ completion scheduler).
        else if constexpr (
          __completes_inline<_Tag, _Attrs, _Env...>
          && __callable<get_scheduler_t, const _Env&...>) {
          return __check_domain<_Attrs, _Env...>(
            __recurse_query_t{}(get_scheduler(__env...), __hide_scheduler{__env}...));
        }
        // Otherwise, if we are asking a scheduler for a completion scheduler, return the
        // scheduler itself.
        else {
          return __attrs;
        }
      }

      static constexpr auto query(forwarding_query_t) noexcept -> bool {
        return true;
      }
    };

    struct execute_may_block_caller_t : __query<execute_may_block_caller_t, true> {
      template <class _Attrs>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr void __validate() noexcept {
        static_assert(
          __std::same_as<bool, __call_result_t<execute_may_block_caller_t, const _Attrs&>>);
        static_assert(__nothrow_callable<execute_may_block_caller_t, const _Attrs&>);
      }
    };

    struct get_forward_progress_guarantee_t
      : __query<
          get_forward_progress_guarantee_t,
          forward_progress_guarantee::weakly_parallel,
          __q1<__decay_t>
        > {
      template <class _Attrs>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr void __validate() noexcept {
        using __result_t = __call_result_t<get_forward_progress_guarantee_t, const _Attrs&>;
        static_assert(__std::same_as<forward_progress_guarantee, __result_t>);
        static_assert(__nothrow_callable<get_forward_progress_guarantee_t, const _Attrs&>);
      }
    };
  } // namespace __queries

  using __queries::execute_may_block_caller_t;
  using __queries::get_forward_progress_guarantee_t;
  using __queries::get_scheduler_t;
  using __queries::get_delegation_scheduler_t;
  using __queries::get_completion_scheduler_t;

  inline constexpr execute_may_block_caller_t execute_may_block_caller{};
  inline constexpr get_forward_progress_guarantee_t get_forward_progress_guarantee{};
  inline constexpr get_scheduler_t get_scheduler{};
  inline constexpr get_delegation_scheduler_t get_delegation_scheduler{};

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
    __call_result_t<get_completion_scheduler_t<_Tag>, env_of_t<_Sender>, const _Env&...>;

  // TODO(ericniebler): examine all uses of this struct.
  template <class _Scheduler>
  struct __sched_attrs {
    template <class... _Env>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto query(get_completion_scheduler_t<set_value_t>, const _Env&... __env)
      const noexcept
      -> __call_result_t<get_completion_scheduler_t<set_value_t>, _Scheduler, const _Env&...> {
      return get_completion_scheduler<set_value_t>(__sched_, __env...);
    }

    template <class... _Env>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto query(get_completion_domain_t<set_value_t>, const _Env&...) const noexcept
      -> __call_result_t<get_completion_domain_t<set_value_t>, _Scheduler, const _Env&...> {
      return {};
    }

    _Scheduler __sched_;
  };

  template <class _Scheduler>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
    __sched_attrs(_Scheduler) -> __sched_attrs<std::unwrap_reference_t<_Scheduler>>;

  //////////////////////////////////////////////////////////////////////////////
  // See SCHED-ENV from [exec.snd.expos]
  template <class _Scheduler>
  struct __sched_env {
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto query(get_scheduler_t) const noexcept {
      return __sched_;
    }

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto query(get_domain_t) const noexcept -> __call_result_t<get_domain_t, _Scheduler> {
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
      if constexpr (
        __completes_inline<set_value_t, env_of_t<schedule_result_t<_Sch>>, _Env...>
        && (__callable<get_scheduler_t, const _Env&> || ...)) {
        return (
          __sched_env{get_completion_scheduler<set_value_t>(
            get_scheduler(__env), __queries::__hide_scheduler{__env})},
          ...);
      } else {
        return __sched_env{__sch};
      }
    }
  };

  inline constexpr __mk_sch_env_t __mk_sch_env{};

  // Deprecated interfaces
  using get_delegatee_scheduler_t
    [[deprecated("get_delegatee_scheduler_t has been renamed get_delegation_scheduler_t")]] =
      get_delegation_scheduler_t;

  inline constexpr auto& get_delegatee_scheduler
    [[deprecated("get_delegatee_scheduler has been renamed get_delegation_scheduler")]]
    = get_delegation_scheduler;
} // namespace STDEXEC
