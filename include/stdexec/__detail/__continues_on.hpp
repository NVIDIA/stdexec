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
#include "__concepts.hpp"
#include "__env.hpp"
#include "__meta.hpp"
#include "__operation_states.hpp"
#include "__schedule_from.hpp"
#include "__schedulers.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__senders.hpp"
#include "__storage.hpp"
#include "__transform_completion_signatures.hpp"
#include "__tuple.hpp"
#include "__utility.hpp"

#include "__prologue.hpp"

namespace STDEXEC
{
  /////////////////////////////////////////////////////////////////////////////
  // [exec.continues.on]
  namespace __trnsfr
  {
    template <class... _Values>
    using __decay_value_sig = set_value_t (*)(__decay_t<_Values>...);

    template <class _Error>
    using __decay_error_sig = set_error_t (*)(__decay_t<_Error>);

    template <class _Scheduler, class _Completions, class... _Env>
    using __completions_impl_t = __mtry_q<__concat_completion_signatures_t>::__f<
      __transform_reduce_completion_signatures_t<_Completions,
                                                 __decay_value_sig,
                                                 __decay_error_sig,
                                                 set_stopped_t (*)(),
                                                 __completion_signature_ptrs_t>,
      __transform_completion_signatures_t<
        __completion_signatures_of_t<schedule_result_t<_Scheduler>, _Env...>,
        __eptr_completion_unless_t<__nothrow_decay_copyable_results_t<_Completions>>,
        __mconst<completion_signatures<>>::__f>>;

    template <class _Scheduler, class _CvSender, class... _Env>
    using __completions_t =
      __completions_impl_t<_Scheduler, __completion_signatures_of_t<_CvSender, _Env...>, _Env...>;

    template <class _Sexpr, class _Receiver>
    struct __state_base
    {
      using __storage_t = __storage_for_t<__child_of<_Sexpr>, env_of_t<_Receiver>>;

      _Receiver   __rcvr_;
      __storage_t __data_;
    };

    // This receiver is to be completed on the execution context associated with the scheduler. When
    // the source sender completes, the completion information is saved off in the operation state
    // so that when this receiver completes, it can read the completion out of the operation state
    // and forward it to the output receiver after transitioning to the scheduler's context.
    template <class _Sexpr, class _Receiver>
    struct __receiver2
    {
      using receiver_concept = receiver_tag;

      constexpr void set_value() noexcept
      {
        std::move(__state_->__data_).__complete(__state_->__rcvr_);
      }

      template <class _Error>
      constexpr void set_error(_Error&& __err) noexcept
      {
        STDEXEC::set_error(static_cast<_Receiver&&>(__state_->__rcvr_),
                           static_cast<_Error&&>(__err));
      }

      constexpr void set_stopped() noexcept
      {
        STDEXEC::set_stopped(static_cast<_Receiver&&>(__state_->__rcvr_));
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> env_of_t<_Receiver>
      {
        return STDEXEC::get_env(__state_->__rcvr_);
      }

      __state_base<_Sexpr, _Receiver>* __state_;
    };

    template <class _Scheduler, class _Sexpr, class _Receiver>
    struct __state : __state_base<_Sexpr, _Receiver>
    {
      using __receiver2_t = __receiver2<_Sexpr, _Receiver>;

      constexpr explicit __state(_Scheduler __sched, _Receiver&& __rcvr)
        : __state::__state_base{static_cast<_Receiver&&>(__rcvr)}
        , __state2_(connect(schedule(__sched), __receiver2_t{this}))
      {}
      STDEXEC_IMMOVABLE(__state);

      connect_result_t<schedule_result_t<_Scheduler>, __receiver2_t> __state2_;
    };

    //! @brief The @c continues_on sender's attributes.
    template <class _Scheduler, class _Sender>
    struct __attrs
    {
     private:
      //! @brief Returns `true` when:
      //! - _SetTag is set_error_t, and
      //! - _Sender has value completions, and
      //! - at least one of the value completions is not nothrow decay-copyable.
      //! In that case, error completions can come from the sender's value completions.
      template <class _SetTag, class... _Env>
      static consteval bool __has_decay_copy_errors() noexcept
      {
        if constexpr (__same_as<_SetTag, set_error_t>)
        {
          if constexpr (__sends<set_value_t, _Sender, __fwd_env_t<_Env>...>)
          {
            return !__cmplsigs::__partitions_of_t<completion_signatures_of_t<
              _Sender,
              __fwd_env_t<_Env>...>>::__nothrow_decay_copyable::__values::value;
          }
        }
        return false;
      }

      _Scheduler        __sch_;
      env_of_t<_Sender> __attrs_;

     public:
      constexpr explicit __attrs(_Scheduler __sch, env_of_t<_Sender> __attrs) noexcept
        : __sch_(static_cast<_Scheduler&&>(__sch))
        , __attrs_(static_cast<env_of_t<_Sender>&&>(__attrs))
      {}

      //! @brief Queries the completion scheduler for a given @c _SetTag.
      //! @tparam _SetTag The completion tag to query for.
      //! @tparam _Env The environment to consider when querying for the completion
      //! scheduler.
      //!
      //! @note If @c _SetTag is @c set_value_t, then we are in the happy path: everything
      //! succeeded and execution continues on @c _Scheduler.
      //!
      //! Otherwise, if @c _Sender never completes with @c _SetTag, and either @c _SetTag is
      //! @c set_stopped_t or decay-copying @c _Sender's value results cannot throw, then a
      //! @c _SetTag completion can only come from the scheduler's sender. In this case, return
      //! the scheduler's completion scheduler if it has one.
      //!
      //! Otherwise, if the scheduler's sender never completes with @c _SetTag, then a
      //! @c _SetTag completion can only come from the original sender, so return the
      //! original sender's completion scheduler.
      template <class _SetTag, class... _Env>
        requires(__same_as<_SetTag, set_value_t>
                 || __never_sends<_SetTag, _Sender, __fwd_env_t<_Env>...>)
             && (!__has_decay_copy_errors<_SetTag, _Env...>())
      [[nodiscard]]
      constexpr auto
      query(get_completion_scheduler_t<_SetTag>, _Env const &... __env) const noexcept
        -> __call_result_t<get_completion_scheduler_t<_SetTag>, _Scheduler, __fwd_env_t<_Env>...>
      {
        return get_completion_scheduler<_SetTag>(__sch_, __fwd_env(__env)...);
      }

      //! @overload
      template <class _SetTag, class... _Env>
        requires __never_sends<_SetTag, schedule_result_t<_Scheduler>, __fwd_env_t<_Env>...>
      [[nodiscard]]
      constexpr auto
      query(get_completion_scheduler_t<_SetTag>, _Env const &... __env) const noexcept
        -> __call_result_t<get_completion_scheduler_t<_SetTag>,
                           env_of_t<_Sender>,
                           __fwd_env_t<_Env>...>
      {
        return get_completion_scheduler<_SetTag>(__attrs_, __fwd_env(__env)...);
      }

      //! @brief Queries the completion domain for a given @c _SetTag.
      //! @tparam _SetTag The completion tag to query for.
      //! @tparam _Env The environment to consider when querying for the completion domain.
      //!
      //! @note If @c _SetTag is @c set_value_t, then we are in the happy path: everything
      //! succeeded and execution continues on @c _Scheduler.
      //!
      //! Otherwise, if @c _SetTag is @c set_stopped_t or if decay-copying @c _Sender's value
      //! results cannot throw, then a @c _SetTag completion can happen on the sender's
      //! completion domain (if it has one) or the scheduler's completion domain (if it has
      //! one).
      //!
      //! @note Otherwise, @c _SetTag is @c set_error_t and decay-copying @c _Sender's value
      //! results can throw, so error completions can also come from the sender's value
      //! completions.
      template <__same_as<set_value_t> _SetTag, class... _Env>
      [[nodiscard]]
      constexpr auto
      query(get_completion_domain_t<_SetTag>, _Env const &...) const noexcept -> __unless_one_of_t<
        __completion_domain_of_t<_SetTag, schedule_result_t<_Scheduler>, __fwd_env_t<_Env>...>,
        indeterminate_domain<>>
      {
        return {};
      }

      //! @overload
      template <__one_of<set_error_t, set_stopped_t> _SetTag, class... _Env>
        requires(!__has_decay_copy_errors<_SetTag, _Env...>())
      [[nodiscard]]
      constexpr auto
      query(get_completion_domain_t<_SetTag>, _Env const &...) const noexcept -> __unless_one_of_t<
        __common_domain_t<
          __completion_domain_of_t<_SetTag, _Sender, __fwd_env_t<_Env>...>,
          __completion_domain_of_t<_SetTag, schedule_result_t<_Scheduler>, __fwd_env_t<_Env>...>>,
        indeterminate_domain<>>
      {
        return {};
      }

      //! @overload
      template <class _SetTag, class... _Env>
        requires(__has_decay_copy_errors<_SetTag, _Env...>())
      [[nodiscard]]
      constexpr auto
      query(get_completion_domain_t<_SetTag>, _Env const &...) const noexcept -> __unless_one_of_t<
        __common_domain_t<
          __completion_domain_of_t<_SetTag, _Sender, __fwd_env_t<_Env>...>,
          __completion_domain_of_t<_SetTag, schedule_result_t<_Scheduler>, __fwd_env_t<_Env>...>,
          __completion_domain_of_t<set_value_t, _Sender, __fwd_env_t<_Env>...>>,
        indeterminate_domain<>>
      {
        return {};
      }

      //! @brief Queries the completion behavior of the combined sender.
      //! @tparam _Env The environment to consider when querying for the completion behavior.
      //! @note The completion behavior is the minimum between the scheduler's sender and
      //! the original sender.
      template <class _Tag, class... _Env>
      [[nodiscard]]
      constexpr auto query(__get_completion_behavior_t<_Tag>, _Env const &...) const noexcept
      {
        using _SchSender = schedule_result_t<_Scheduler>;
        constexpr auto cb_sched =
          STDEXEC::__get_completion_behavior<_Tag, _SchSender, __fwd_env_t<_Env>...>();
        constexpr auto cb_sndr =
          STDEXEC::__get_completion_behavior<_Tag, _Sender, __fwd_env_t<_Env>...>();
        return cb_sched | cb_sndr;
      }

      //! @brief Forwards other queries to the underlying sender's environment.
      //! @pre @c _Query is a forwarding query but not a completion query.
      template <__forwarding_query _Query, class... _Args>
        requires(!__completion_query<_Query>)
             && __queryable_with<env_of_t<_Sender>, _Query, _Args...>
      [[nodiscard]]
      constexpr auto query(_Query, _Args&&... __args) const
        noexcept(__nothrow_queryable_with<env_of_t<_Sender>, _Query, _Args...>)
          -> __query_result_t<env_of_t<_Sender>, _Query, _Args...>
      {
        return __attrs_.query(_Query(), static_cast<_Args&&>(__args)...);
      }
    };

    struct __continues_on_impl : __sexpr_defaults
    {
     private:
      template <class _Sender, class... _Env>
      using __scheduler_t = __decay_t<__data_of<_Sender>>;

      template <class _Child, class... _Env>
      static consteval auto __get_child_completions()
      {
        auto __child_completions =
          STDEXEC::get_completion_signatures<_Child, __fwd_env_t<_Env>...>();
        STDEXEC_IF_OK(__child_completions)
        {
          // continues_on has the completions of the child sender, but with value and
          // error result types decayed.
          return __transform_completion_signatures(
            __child_completions,
            __decay_arguments<set_value_t, continues_on_t>(),
            __decay_arguments<set_error_t, continues_on_t>());
        }
      }

      template <class _Scheduler, class... _Env>
      static consteval auto __get_scheduler_completions()
      {
        using __sndr_t = schedule_result_t<_Scheduler>;
        auto __sched_completions =
          STDEXEC::get_completion_signatures<__sndr_t, __fwd_env_t<_Env>...>();
        STDEXEC_IF_OK(__sched_completions)
        {
          // The scheduler contributes only error and stopped completions; we ignore value
          // completions here
          return __transform_completion_signatures(__sched_completions, __ignore_completion());
        }
      }

      template <class _Sender, class _Receiver>
      using __state_for_t = __state<__decay_t<__data_of<_Sender>>, _Sender, _Receiver>;

     public:
      static constexpr auto __get_attrs =
        []<class _Scheduler, class _Child>(__ignore,
                                           _Scheduler const & __data,
                                           _Child const &     __child) noexcept
      {
        return __attrs<_Scheduler, _Child>{__data, STDEXEC::get_env(__child)};
      };

      template <class _Sender, class... _Env>
      static consteval auto __get_completion_signatures()
      {
        static_assert(__sender_for<_Sender, continues_on_t>);
        using __scheduler_t      = __decay_t<__data_of<_Sender>>;
        using __child_t          = __child_of<_Sender>;
        auto __child_completions = __get_child_completions<__child_t, _Env...>();
        return __concat_completion_signatures(
          __child_completions,
          __get_scheduler_completions<__scheduler_t, _Env...>(),
          __eptr_completion_unless_t<
            __nothrow_decay_copyable_results_t<decltype(__child_completions)>>());
      }

      static constexpr auto __get_state =
        []<class _Sender, class _Receiver>(_Sender&&   __sndr,
                                           _Receiver&& __rcvr) -> __state_for_t<_Sender, _Receiver>
        requires sender_in<__child_of<_Sender>, __fwd_env_t<env_of_t<_Receiver>>>
      {
        static_assert(__sender_for<_Sender, continues_on_t>);
        auto& [__tag, __sched, __child] = __sndr;
        return __state_for_t<_Sender, _Receiver>{__sched, static_cast<_Receiver&&>(__rcvr)};
      };

      static constexpr auto __complete =
        []<class _State, class _Tag, class... _Args>(__ignore,
                                                     _State& __state,
                                                     _Tag    __tag,
                                                     _Args&&... __args) noexcept -> void
      {
        // Write the tag and the args into the operation state so that we can forward the completion
        // from within the scheduler's execution context.
        if constexpr (__nothrow_callable<__mktuple_t, _Tag, _Args...>)
        {
          __state.__data_.__emplace_from(__mktuple, __tag, static_cast<_Args&&>(__args)...);
        }
        else
        {
          STDEXEC_TRY
          {
            __state.__data_.__emplace_from(__mktuple, __tag, static_cast<_Args&&>(__args)...);
          }
          STDEXEC_CATCH_ALL
          {
            STDEXEC::set_error(static_cast<_State&&>(__state).__rcvr_, std::current_exception());
            return;
          }
        }

        // Enqueue the schedule operation so the completion happens on the scheduler's execution
        // context.
        STDEXEC::start(__state.__state2_);
      };
    };
  }  // namespace __trnsfr

  //! @brief A pipeable sender adaptor that transfers a predecessor sender's
  //!        completion to a different scheduler's execution resource.
  //!
  //! @c continues_on lets a sender pipeline *change execution context* in the
  //! middle. Given a predecessor sender @c sndr and a scheduler @c sched,
  //! @c continues_on produces a sender that, when connected and started,
  //! runs @c sndr to completion on whatever context @c sndr ran on, then
  //! transfers execution to @c sched's resource, and only *then* forwards
  //! @c sndr's completion (value, error, or stopped) to the connected
  //! receiver. Anything chained after @c continues_on therefore runs on
  //! @c sched.
  //!
  //! Both call syntaxes are supported (the second is the *pipeable* form):
  //!
  //! @code{.cpp}
  //! auto s1 = stdexec::continues_on(sndr, sched);   // direct invocation
  //! auto s2 = sndr | stdexec::continues_on(sched);  // pipe syntax
  //! @endcode
  //!
  //! The two forms are expression-equivalent. See [exec.continues.on] in
  //! the C++26 working draft for the normative specification.
  //!
  //! **Completion signatures.**
  //!
  //! Given a predecessor sender @c sndr with completion signatures
  //!
  //! @code{.cpp}
  //! set_value_t(Vs...)    // forwarded — but delivered on `sched`'s resource
  //! set_error_t(Es)...    // forwarded — but delivered on `sched`'s resource
  //! set_stopped_t()       // forwarded — but delivered on `sched`'s resource
  //! @endcode
  //!
  //! the sender produced by <tt>continues_on(sndr, sched)</tt> has the same
  //! completion signatures as @c sndr, except that a
  //! @c set_error_t(std::exception_ptr) completion may be added if any of
  //! @c sndr's completion datums are not @c nothrow decay-copyable (the
  //! datums must be stored across the scheduling hop).
  //!
  //! @c continues_on does *not* alter @c sndr's values or errors; it only
  //! changes the execution context on which the completion is delivered.
  //!
  //! **Exception behavior.**
  //!
  //! If decay-copying a completion datum across the scheduling hop throws,
  //! the exception is delivered through @c set_error_t(std::exception_ptr).
  //! If scheduling onto @c sched fails, an error completion is delivered on
  //! an unspecified execution agent.
  //!
  //! **Cancellation.**
  //!
  //! @c continues_on respects the receiver's stop token while waiting to
  //! be scheduled onto @c sched: if cancellation is requested after @c sndr
  //! completes but before the hop finishes, the resulting sender typically
  //! completes via @c set_stopped.
  //!
  //! **Example.**
  //!
  //! @code{.cpp}
  //! #include <stdexec/execution.hpp>
  //!
  //! int main() {
  //!   using namespace stdexec;
  //!
  //!   auto io_sched   = get_parallel_scheduler();   // pretend: I/O
  //!   auto cpu_sched  = get_parallel_scheduler();   // pretend: compute
  //!
  //!   auto sndr =
  //!     starts_on(io_sched, just(42))               // produce on io_sched
  //!     | continues_on(cpu_sched)                   // hop to cpu_sched
  //!     | then([](int x) { return x * 2; });        // then() runs on cpu_sched
  //!
  //!   auto [v] = sync_wait(std::move(sndr)).value();
  //!   (void)v;  // == 84
  //! }
  //! @endcode
  //!
  //! @see stdexec::schedule     — the primitive that produces a schedule-sender
  //! @see stdexec::starts_on    — *begin* execution on a given scheduler
  //! @see stdexec::on           — run on a different scheduler, then transfer back
  struct continues_on_t
  {
    //! @brief Construct a sender that runs @c __sndr to completion, then
    //!        transfers execution to @c __sched before forwarding the
    //!        completion downstream.
    //!
    //! @tparam _Scheduler A type satisfying the @c stdexec::scheduler concept.
    //! @tparam _Sender    A type satisfying the @c stdexec::sender concept.
    //!
    //! @param __sndr      The predecessor sender. Forwarded into the
    //!                    resulting sender.
    //! @param __sched     The scheduler whose execution resource will host
    //!                    the delivery of @c __sndr's completion.
    //!
    //! @returns A sender with the same completion signatures as @c __sndr
    //!          (plus a possible @c set_error_t(std::exception_ptr) for
    //!          decay-copy failures during the hop). The completions are
    //!          delivered on @c __sched's execution resource.
    template <scheduler _Scheduler, sender _Sender>
    constexpr auto
    operator()(_Sender&& __sndr, _Scheduler __sched) const -> __well_formed_sender auto
    {
      return __make_sexpr<continues_on_t>(static_cast<_Scheduler&&>(__sched),
                                          schedule_from(static_cast<_Sender&&>(__sndr)));
    }

    //! @brief Construct a sender-adaptor closure that, when applied to a
    //!        sender, produces <tt>continues_on(sndr, __sched)</tt>.
    //!
    //! This overload enables the pipe syntax:
    //! <tt>sndr | continues_on(__sched)</tt> is equivalent to
    //! <tt>continues_on(sndr, __sched)</tt>.
    //!
    //! @tparam _Scheduler A type satisfying the @c stdexec::scheduler concept.
    //! @param __sched     The scheduler to transfer execution to when the
    //!                    closure is later applied to a sender.
    //!
    //! @returns A sender-adaptor closure object capturing @c __sched.
    template <scheduler _Scheduler>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto operator()(_Scheduler __sched) const noexcept
    {
      return __closure(*this, static_cast<_Scheduler&&>(__sched));
    }
  };

  //! @brief The customization point object for the @c continues_on sender adaptor.
  //!
  //! @c continues_on is an instance of @ref continues_on_t. See
  //! @ref continues_on_t for the full description, completion signatures,
  //! and a usage example.
  //!
  //! @hideinitializer
  inline constexpr continues_on_t continues_on{};

  template <>
  struct __sexpr_impl<continues_on_t> : __trnsfr::__continues_on_impl
  {};
}  // namespace STDEXEC

#include "__epilogue.hpp"
