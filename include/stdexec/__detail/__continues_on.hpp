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
#  include "__concepts.hpp"
#  include "__env.hpp"
#  include "__meta.hpp"
#  include "__schedule_from.hpp"
#  include "__schedulers.hpp"
#  include "__sender_adaptor_closure.hpp"
#  include "__sender_facade.hpp"
#  include "__sender_introspection.hpp"
#  include "__senders.hpp"
#  include "__storage.hpp"
#  include "__tuple.hpp"

#  include "__prologue.hpp"

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  //////////////////////////////////////////////////////////////////////////////
  // [exec.continues.on]
  namespace __trnsfr
  {
    // The key of continues_on's nested schedule operation
    struct __sched_op
    {};

    struct __algo;

    // Where the child completes, which is where the schedule operation starts (see
    // __mk_secondary_env_t): the child's completion scheduler if it completes in only one
    // way, or else the domain in which it completes. Or nothing if the child does not
    // type-check in the environment. (Such a sender cannot be connected, but the facade
    // still computes its attributes, and its state type when the walk has no environment.)
    template <class _Child, class _Env>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto __mk_env2(_Child const & __child, _Env const & __env) noexcept
    {
      using __mk_env2_t = __mk_secondary_env_t<set_value_t, set_error_t, set_stopped_t>;
      if constexpr (__callable<__mk_env2_t, _Child const &, _Env const &>)
        return __mk_env2_t()(__child, __env);
      else
        return env{};
    }

    // Does the child complete where it starts, however it completes? Then the schedule
    // operation starts where the operation started, which the receiver's environment says.
    // (The child sees the receiver's environment forwarded.)
    template <class _Child, class _Env>
    inline constexpr bool __child_stays_put_v = __stays_put<__decay_t<_Child>, __fwd_env_t<_Env>>;

    // Saves a completion in __storage, and then starts the nested operation __op, keyed by
    // _Key. If decay-copying the completion's arguments throws, saves the exception as an
    // error instead. If they cannot be decay-copied, reports a type error on behalf of
    // _AlgoTag.
    template <class _AlgoTag, class _Key, class _Storage, class _OpState, class _Tag, class... _Args>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto __save_and_start(_Storage& __storage,
                                    _OpState& __op,
                                    _Tag      __tag,
                                    _Args&&... __args) noexcept
    {
      if constexpr (!__decay_copyable<_Args...>)
      {
        return __facade::__decay_copy_error<_AlgoTag, _Args...>();
      }
      else
      {
        if constexpr (__nothrow_decay_copyable<_Args...>)
        {
          __storage.__emplace_from(__mktuple, __tag, static_cast<_Args&&>(__args)...);
        }
        else
        {
          STDEXEC_TRY
          {
            __storage.__emplace_from(__mktuple, __tag, static_cast<_Args&&>(__args)...);
          }
          STDEXEC_CATCH_ALL
          {
            __storage.__emplace_from(__mktuple, set_error, std::current_exception());
          }
        }
        return __facade::__start_nested<_Key>(__op);
      }
    }

    // Sends the completion that __save_and_start saved
    template <class _Storage, class _Receiver>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto __send_saved(_Storage& __storage, _Receiver& __rcvr) noexcept
    {
      return __facade::__visit(
        __storage,
        [&]<class _Tuple>(_Tuple& __tupl) noexcept
        {
          return STDEXEC::__apply(
            [&]<class _Tag, class... _Args>(_Tag __tag, _Args&... __args) noexcept
            { return __facade::__send(__rcvr, __tag, static_cast<_Args&&>(__args)...); },
            __tupl);
        });
    }

    template <class _Sender, class _Receiver>
    struct __state
    {
      using __env_t       = env_of_t<_Receiver>;
      using __child_t     = __child_of<_Sender>;
      using __scheduler_t = __decay_t<__data_of<_Sender>>;
      using __env2_t      = decltype(__trnsfr::__mk_env2(__declval<__decay_t<__child_t> const &>(),
                                                    __declval<__decay_t<__env_t> const &>()));
      using __storage_t =
        __mapply_q<__results_storage,
                   __facade::__completions_of_t<__algo, __facade::__child<0>, _Sender, _Receiver>>;
      using __sched_rcvr_t =
        __facade::__receiver_t<__algo, __sched_op, __state, _Sender, _Receiver>;
      using __sched_sndr_t                    = schedule_result_t<__scheduler_t const &>;
      static constexpr bool __child_stays_put = __trnsfr::__child_stays_put_v<__child_t, __env_t>;

      constexpr explicit __state(_Sender&& __sndr, _Receiver&& __rcvr)
        noexcept(__nothrow_decay_copyable<__data_of<_Sender>>
                 && __nothrow_callable<schedule_t, __scheduler_t const &>
                 && __nothrow_connectable<__sched_sndr_t, __sched_rcvr_t>)
        : __state(__facade::__get_data(static_cast<_Sender&&>(__sndr)),
                  __facade::__get_child<0>(__sndr),
                  static_cast<_Receiver&&>(__rcvr))
      {}

      STDEXEC_IMMOVABLE(__state);

      _Receiver                                        __rcvr_;
      __env2_t const                                   __env2_;
      __storage_t                                      __storage_{};
      connect_result_t<__sched_sndr_t, __sched_rcvr_t> __op_;

     private:
      template <class _Child>
      constexpr explicit __state(__scheduler_t __sch, _Child const & __child, _Receiver&& __rcvr)
        : __rcvr_(static_cast<_Receiver&&>(__rcvr))
        , __env2_(__trnsfr::__mk_env2(__child, STDEXEC::get_env(__rcvr_)))
        , __op_(STDEXEC::connect(STDEXEC::schedule(static_cast<__scheduler_t const &>(__sch)),
                                 __sched_rcvr_t{this}))
      {}
    };

    // continues_on(sndr, sch): when the child completes, its results are saved, and a
    // schedule operation on sch is started. When that completes with a value, the saved
    // results are sent.
    struct __algo : __facade::__defaults
    {
      // The value completion of the schedule operation, which (when the child completes with
      // a value) is started from the child's value completion
      using __sched_value_t =
        __facade::__via<__sched_op,
                        set_value_t,
                        __facade::__via<__facade::__child<0>, set_value_t, __facade::__from_start>>;

      // Customizations of continues_on are found in the domain where the schedule operation
      // completes. Value completions happen there, even if the child never completes with a
      // value (as with SCHED-ATTRS in the standard).
      using __dispatch_source = __sched_value_t;

      template <class _Tag>
      using __empty_channel_source = __if_c<__same_as<_Tag, set_value_t>, __sched_value_t, void>;

      // The child sees the forwarded environment of the receiver
      template <class _Sender, class _Env>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __env_for(__facade::__child<0>, _Sender const &, _Env&& __env) noexcept
        -> __fwd_env_t<_Env>
      {
        return __fwd_env(static_cast<_Env&&>(__env));
      }

      template <class _State>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __get_env(__facade::__child<0>, _State const & __state) noexcept
        -> decltype(auto)
      {
        // (The receiver's environment may be a reference to a forwarding environment,
        // which __fwd_env returns as it is.)
        return __fwd_env(STDEXEC::get_env(__state.__rcvr_));
      }

      // The schedule operation sees where the child completed (see __mk_env2). Unless the
      // child completes where it starts, that is not where the operation started, so the
      // receiver's start scheduler and domain are hidden from it. (Otherwise, when the child
      // does not say where it completes, a scheduler that completes where it starts would
      // claim to complete where the operation started.)
      //
      // NOT TO SPEC: the specification gives the schedule operation the forwarded
      // environment of the receiver.
      template <class _Sender, class _Env>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __env_for(__sched_op, _Sender const & __sndr, _Env&& __env) noexcept
      {
        if constexpr (__trnsfr::__child_stays_put_v<__facade::__nth_child_of_t<_Sender, 0>, _Env>)
          return __env::__join(__trnsfr::__mk_env2(__facade::__get_child<0>(__sndr), __env),
                               static_cast<_Env&&>(__env));
        else
          return __env::__join(__trnsfr::__mk_env2(__facade::__get_child<0>(__sndr), __env),
                               __hide_scheduler{static_cast<_Env&&>(__env)});
      }

      template <class _State>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __get_env(__sched_op, _State const & __state) noexcept
      {
        using __env2_t = __decay_t<decltype(__state.__env2_)>;
        if constexpr (_State::__child_stays_put)
          return __env::__join(__env2_t(__state.__env2_), STDEXEC::get_env(__state.__rcvr_));
        else
          return __env::__join(__env2_t(__state.__env2_),
                               __hide_scheduler{STDEXEC::get_env(__state.__rcvr_)});
      }

      using __facade::__defaults::__nested_sender;

      template <class _Sender, class... _Env>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __nested_attrs(__sched_op,
                                           _Sender const & __sndr,
                                           _Env const &...) noexcept
      {
        return __facade::__schedule_attrs<decltype(__facade::__get_data(__sndr))>{
          __facade::__get_data(__sndr)};
      }

      template <class _Sender, class... _Env>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __nested_sender(__sched_op, _Sender const & __sndr, _Env const &...)
      {
        return STDEXEC::schedule(__facade::__get_data(__sndr));
      }

      template <class _Sender, class _Receiver>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __get_state(_Sender&& __sndr, _Receiver __rcvr)
        noexcept(__nothrow_constructible_from<__state<_Sender, _Receiver>, _Sender, _Receiver>)
          -> __state<_Sender, _Receiver>
      {
        return __state<_Sender, _Receiver>(static_cast<_Sender&&>(__sndr),
                                           static_cast<_Receiver&&>(__rcvr));
      }

      // The child completed: save the results, then hop to the scheduler.
      template <class _State, class _Tag, class... _Args>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __complete(__facade::__child<0>,
                                       _State& __state,
                                       _Tag    __tag,
                                       _Args&&... __args) noexcept
      {
        return __trnsfr::__save_and_start<continues_on_t, __sched_op>(__state.__storage_,
                                                                      __state.__op_,
                                                                      __tag,
                                                                      static_cast<_Args&&>(
                                                                        __args)...);
      }

      // On the scheduler: send the saved results.
      template <class _State>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __complete(__sched_op, _State& __state, set_value_t) noexcept
      {
        return __trnsfr::__send_saved(__state.__storage_, __state.__rcvr_);
      }

      // The schedule operation failed or was stopped.
      template <class _State, class _Tag, class... _Args>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __complete(__sched_op,
                                       _State& __state,
                                       _Tag    __tag,
                                       _Args&&... __args) noexcept
      {
        return __facade::__send(__state.__rcvr_, __tag, static_cast<_Args&&>(__args)...);
      }
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
  //! the sender produced by `continues_on(sndr, sched)` has the same
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
  //!   auto io_sched  = get_parallel_scheduler();   // pretend: I/O
  //!   auto cpu_sched = get_parallel_scheduler();   // pretend: compute
  //!
  //!   auto sndr =
  //!     starts_on(io_sched, just(42))              // produce on io_sched
  //!     | continues_on(cpu_sched)                  // hop to cpu_sched
  //!     | then([](int x) { return x * 2; });       // then() runs on cpu_sched
  //!
  //!   auto [v] = sync_wait(std::move(sndr)).value();
  //!   assert(v == 84);
  //! }
  //! @endcode
  //!
  //! @see stdexec::schedule     — the primitive that produces a schedule-sender
  //! @see stdexec::starts_on    — *begin* execution on a given scheduler
  //! @see stdexec::on           — run on a different scheduler, then transfer
  //!                              back
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
      return __facade::__sender{continues_on_t(),
                                static_cast<_Scheduler&&>(__sched),
                                schedule_from(static_cast<_Sender&&>(__sndr))};
    }

    //! @brief Construct a sender-adaptor closure that, when applied to a
    //!        sender, produces `continues_on(sndr, __sched)`.
    //!
    //! This overload enables the pipe syntax:
    //! `sndr | continues_on(__sched)` is equivalent to
    //! `continues_on(sndr, __sched)`.
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

  //! @brief The customization point object for the @c continues_on sender
  //! adaptor.
  //!
  //! @c continues_on is an instance of @c continues_on_t. See
  //! @ref continues_on_t for the full description, completion signatures,
  //! and a usage example.
  //!
  //! @hideinitializer
  inline constexpr continues_on_t continues_on{};

  template <>
  struct __facade::__algorithm<continues_on_t> : __trnsfr::__algo
  {};
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
