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

#include "__config.hpp"

#if STDEXEC_USE_MODULES() && !defined(STDEXEC_IN_MODULE_PURVIEW)

import stdexec;

#else

#  include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#  include "__basic_sender.hpp"
#  include "__completion_signatures_of.hpp"
#  include "__continues_on.hpp"
#  include "__diagnostics.hpp"
#  include "__env.hpp"
#  include "__inline_scheduler.hpp"
#  include "__meta.hpp"
#  include "__schedulers.hpp"
#  include "__sender_adaptor_closure.hpp"
#  include "__sender_introspection.hpp"
#  include "__sender_ref.hpp"
#  include "__starts_on.hpp"
#  include "__utility.hpp"

#  include "__prologue.hpp"

STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace STDEXEC
{
  /////////////////////////////////////////////////////////////////////////////
  // [exec.on]
  struct _CANNOT_RESTORE_EXECUTION_CONTEXT_AFTER_ON_;

  namespace __on
  {
    // If __is_root_env<_Env> is true, then this sender has no parent, so there is no need
    // to restore the execution context. We can use the inline scheduler as the scheduler
    // if __env does not have one.
    template <class _Child, class _Env>
    using __end_sched_t = __if_c<__is_root_env<_Env>, inline_scheduler, __not_a_scheduler<>>;

    // This transform_sender overload handles the case where `on` was called like `on(sch,
    // sndr)`. In this case, we find the old scheduler by looking in the receiver's
    // environment.
    template <scheduler _Scheduler, class _Child, class _Env>
    STDEXEC_ATTRIBUTE(nodiscard, host, device)
    constexpr auto __transform_sender(_Scheduler&& __new_sched,
                                      _Child&&     __child,
                                      _Env const & __env)
    {
      auto __default_sched = __end_sched_t<_Child, _Env>();
      auto __get_sched     = __with_default(get_start_scheduler, __default_sched);
      auto __old_sched     = __get_sched(__env);

      if constexpr (__same_as<decltype(__old_sched), __not_a_scheduler<>>)
      {
        return __not_a_sender<
          _WHAT_(_CANNOT_RESTORE_EXECUTION_CONTEXT_AFTER_ON_),
          _WHY_(_THE_CURRENT_EXECUTION_ENVIRONMENT_DOES_NOT_HAVE_A_START_SCHEDULER_),
          _WHERE_(_IN_ALGORITHM_, on_t),
          _WITH_PRETTY_SENDER_<_Child>,
          _WITH_ENVIRONMENT_(_Env)>();
      }
      else
      {
        return continues_on(starts_on(static_cast<_Scheduler&&>(__new_sched),
                                      static_cast<_Child&&>(__child)),
                            std::move(__old_sched));
      }
    }

    // This transform_sender overload handles the case where `on` was called like `sndr |
    // on(sch, clsur)` or `on(sndr, sch, clsur)`. In this case, __child is a predecessor
    // sender, so the scheduler we want to restore is the completion scheduler of __child.
    template <class _Data, class _Child, class _Env>
    STDEXEC_ATTRIBUTE(nodiscard, host, device)
    constexpr auto __transform_sender(_Data&& __data, _Child&& __child, _Env const & __env)
    {
      auto& [__new_sched, __clsur] = __data;
      auto __default_sched         = __end_sched_t<_Child, _Env>();
      auto __get_sched = __with_default(get_completion_scheduler<set_value_t>, __default_sched);
      auto __old_sched = __get_sched(STDEXEC::get_env(__child), __env);

      if constexpr (__same_as<decltype(__old_sched), __not_a_scheduler<>>)
      {
        return __not_a_sender<
          _WHAT_(_CANNOT_RESTORE_EXECUTION_CONTEXT_AFTER_ON_),
          _WHY_(_THE_PREDECESSOR_SENDER_DOES_NOT_KNOW_THE_SCHEDULER_ON_WHICH_IT_WILL_COMPLETE_),
          _WHERE_(_IN_ALGORITHM_, on_t),
          _WITH_PRETTY_SENDER_<_Child>,
          _WITH_ENVIRONMENT_(_Env)>();
      }
      else
      {
        return continues_on(STDEXEC::__forward_like<_Data>(__clsur)(
                              continues_on(static_cast<_Child&&>(__child),
                                           STDEXEC::__forward_like<_Data>(__new_sched))),
                            std::move(__old_sched));
      }
    }

    template <class _Data, class _Child, class _Env>
    using __transform_sender_t = decltype(__on::__transform_sender(__declval<_Data const &>(),
                                                                   __declval<_Child const &>(),
                                                                   __declval<_Env>()));

    template <class _Data, class _Child, class _Env>
    concept __has_known_scheduler =
      (scheduler<_Data> && __callable<get_start_scheduler_t, _Env const &>)
      || (!scheduler<_Data> && __has_completion_scheduler_for<set_value_t, _Child, _Env const &>);

    template <class _Child>
    struct __attrs_base
    {
      template <__forwarding_query _Query, class... _Args>
        requires(!__completion_query<_Query>)
             && __queryable_with<env_of_t<_Child>, _Query, _Args...>
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto query(_Query, _Args&&... __args) const
        noexcept(__nothrow_queryable_with<env_of_t<_Child>, _Query, _Args...>)
          -> __call_result_t<_Query, env_of_t<_Child>, _Args...>
      {
        return _Query()(STDEXEC::get_env(__child_), static_cast<_Args&&>(__args)...);
      }

      _Child const & __child_;
    };

    template <class _Child, class _Data>
    struct __attrs : __attrs_base<_Child>
    {
      template <class _Env>
      using __attrs_t = env_of_t<__transform_sender_t<_Data, __sender_proxy<_Child const>, _Env>>;
      using __attrs_base<_Child>::query;

      template <__completion_query _Query, class _Env>
        requires __has_known_scheduler<_Data, _Child const, _Env>
              && __queryable_with<__attrs_t<_Env>, _Query, _Env const &>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto query(_Query, _Env const & __env) const noexcept
        -> __call_result_t<_Query, __attrs_t<_Env>, _Env const &>
      {
        auto __tfx_sndr = __on::__transform_sender(__data_, __sender_proxy(this->__child_), __env);
        return _Query()(STDEXEC::get_env(__tfx_sndr), __env);
      }

      _Data const & __data_;
    };

    template <class _Child, class _Data>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
    __attrs(_Child const &, _Data const &) -> __attrs<_Child, _Data>;
  }  // namespace __on

  ////////////////////////////////////////////////////////////////////////////////////////////////
  //! @brief A sender adaptor that runs work on a different scheduler and then
  //!        transfers execution *back* to the original scheduler.
  //!
  //! @c on is the "go there, do work, come back" scheduling adaptor. It has
  //! two distinct shapes:
  //!
  //! 1. **Whole-sender form:** <tt>on(sched, sndr)</tt> — runs the entirety
  //!    of @c sndr on @c sched's execution resource. When @c sndr completes,
  //!    execution transfers back to the scheduler that started the operation
  //!    (the "start scheduler"), and the completion is delivered there.
  //!
  //!    This is the principal difference between @c on and
  //!    @ref starts_on_t — @c starts_on stays on @c sched; @c on returns home.
  //!
  //! 2. **Closure-insertion form:** <tt>on(sndr, sched, closure)</tt> — runs
  //!    @c sndr on its *current* scheduler, then transfers to @c sched,
  //!    applies @c closure (a sender-adaptor closure) to the result, runs
  //!    *that* on @c sched, and finally transfers back to the original
  //!    completion scheduler. Useful for inserting a CPU-bound transform
  //!    into an otherwise I/O-bound pipeline (or vice versa) without
  //!    permanently changing context.
  //!
  //!    This form also has a pipe shorthand:
  //!    <tt>sndr | on(sched, closure)</tt>.
  //!
  //! @code{.cpp}
  //! // Form 1: run sndr on sched, return to start scheduler.
  //! auto s1 = stdexec::on(sched, sndr);
  //!
  //! // Form 2: run sndr in place, hop to sched for closure, hop back.
  //! auto s2 = stdexec::on(sndr, sched, stdexec::then([](int x){ return x*2; }));
  //! auto s3 = sndr | stdexec::on(sched, stdexec::then([](int x){ return x*2; }));
  //! @endcode
  //!
  //! See [exec.on] in the C++26 working draft for the normative specification.
  //!
  //! **The round trip.**
  //!
  //! What distinguishes @c on from @c starts_on and @c continues_on is the
  //! restoration of the original scheduler:
  //!
  //! | Adaptor                | Where work runs | Where completion is delivered  |
  //! | ---------------------- | --------------- | ------------------------------ |
  //! | @c starts_on(sch,s)    | on @c sch       | on @c sch                      |
  //! | @c continues_on(s,sch) | on @c s's sched | on @c sch                      |
  //! | @c on(sch,s)           | on @c sch       | on the *start scheduler*       |
  //! | @c on(s,sch,closure)   | mixed (see above) | on the *start scheduler*     |
  //!
  //! The "start scheduler" is read from the receiver's environment via
  //! @c get_start_scheduler (form 1) or @c get_completion_scheduler<set_value_t>
  //! of @c sndr's environment (form 2).
  //!
  //! **Completion signatures.**
  //!
  //! Form 1 (<tt>on(sched, sndr)</tt>): essentially @c sndr's completion
  //! signatures, with possible additional @c set_error_t completions from
  //! the two scheduling hops.
  //!
  //! Form 2 (<tt>on(sndr, sched, closure)</tt>): the completion signatures
  //! of <tt>closure(continues_on(sndr, sched))</tt> after the final transfer
  //! back, again with possible additional @c set_error_t completions from
  //! the scheduling hops.
  //!
  //! If scheduling onto @c sched (or back) fails, an error completion is
  //! delivered on an unspecified execution agent.
  //!
  //! **Cancellation.**
  //!
  //! Cancellation flows through the scheduling hops normally; a stop
  //! request observed between hops typically results in @c set_stopped.
  //!
  //! **Example.**
  //!
  //! @code{.cpp}
  //! #include <stdexec/execution.hpp>
  //!
  //! int main() {
  //!   using namespace stdexec;
  //!
  //!   auto gpu = get_parallel_scheduler();   // pretend: GPU
  //!
  //!   // Compute on the GPU, but stay on the start scheduler afterwards:
  //!   auto sndr =
  //!     just(21)
  //!     | on(gpu, then([](int x) { return x * 2; }));
  //!
  //!   auto [v] = sync_wait(std::move(sndr)).value();
  //!   (void)v;  // == 42; sync_wait sees the result on its starting context
  //! }
  //! @endcode
  //!
  //! @see stdexec::schedule       — the primitive that produces a schedule-sender
  //! @see stdexec::starts_on      — begin on a scheduler and *stay* there
  //! @see stdexec::continues_on   — transfer to a scheduler *after* a sender completes
  struct on_t
  {
    //! @brief Form 1: run @c __sndr on @c __sched, then return to the start
    //!        scheduler.
    //!
    //! @tparam _Scheduler A type satisfying the @c stdexec::scheduler concept.
    //! @tparam _Sender    A type satisfying the @c stdexec::sender concept.
    //!
    //! @param __sched     The scheduler whose execution resource will host
    //!                    @c __sndr.
    //! @param __sndr      The sender to run on @c __sched.
    //!
    //! @returns A sender that, when connected and started, runs @c __sndr on
    //!          @c __sched then transfers execution back to the start
    //!          scheduler (taken from the receiver's environment via
    //!          @c get_start_scheduler) before forwarding @c __sndr's
    //!          completion to the receiver.
    template <scheduler _Scheduler, sender _Sender>
    constexpr auto
    operator()(_Scheduler&& __sched, _Sender&& __sndr) const -> __well_formed_sender auto
    {
      return __make_sexpr<on_t>(static_cast<_Scheduler&&>(__sched), static_cast<_Sender&&>(__sndr));
    }

    //! @brief Form 2: run @c __sndr in place, hop to @c __sched, apply
    //!        @c __clsur there, then hop back.
    //!
    //! @tparam _Sender    A type satisfying the @c stdexec::sender concept.
    //! @tparam _Scheduler A type satisfying the @c stdexec::scheduler concept.
    //! @tparam _Closure   A sender-adaptor closure suitable for chaining
    //!                    onto @c _Sender.
    //!
    //! @param __sndr      The predecessor sender (runs on its own scheduler).
    //! @param __sched     The scheduler to transition to before applying
    //!                    @c __clsur.
    //! @param __clsur     The adaptor closure (e.g. <tt>then(...)</tt>,
    //!                    <tt>bulk(...)</tt>) to apply on @c __sched.
    //!
    //! @returns A sender that completes on the *original* completion
    //!          scheduler of @c __sndr, with the result of
    //!          <tt>__clsur(continues_on(__sndr, __sched))</tt>.
    template <sender _Sender, scheduler _Scheduler, __sender_adaptor_closure_for<_Sender> _Closure>
    constexpr auto operator()(_Sender&& __sndr, _Scheduler&& __sched, _Closure&& __clsur) const
      -> __well_formed_sender auto
    {
      return __make_sexpr<on_t>(__tuple{static_cast<_Scheduler&&>(__sched),
                                        static_cast<_Closure&&>(__clsur)},
                                static_cast<_Sender&&>(__sndr));
    }

    //! @brief Pipe form of Form 2: construct a sender-adaptor closure that,
    //!        when applied to a sender, produces
    //!        <tt>on(sndr, __sched, __clsur)</tt>.
    //!
    //! @tparam _Scheduler A type satisfying the @c stdexec::scheduler concept.
    //! @tparam _Closure   A sender-adaptor closure.
    //!
    //! @param __sched     The scheduler to transition to.
    //! @param __clsur     The adaptor closure to apply on @c __sched.
    //!
    //! @returns A sender-adaptor closure capturing @c __sched and @c __clsur.
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

  //! @brief The customization point object for the @c on sender adaptor.
  //!
  //! @c on is an instance of @ref on_t. See @ref on_t for the full
  //! description, the distinction between @c on, @c starts_on, and
  //! @c continues_on, and usage examples.
  //!
  //! @hideinitializer
  inline constexpr on_t on{};

  template <>
  struct __sexpr_impl<on_t> : __sexpr_defaults
  {
    static constexpr auto __get_attrs =  //
      [](__ignore, auto const & __data, auto const & __child) noexcept
    {
      return __on::__attrs{__child, __data};
    };

    template <class _Sender, class _Env>
    static constexpr auto __get_completion_signatures()
    {
      using __sndr_t = __detail::__transform_sender_result_t<on_t, set_value_t, _Sender, _Env>;
      return STDEXEC::get_completion_signatures<__sndr_t, _Env>();
    }
  };
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
