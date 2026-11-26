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
#  include "__completion_signatures_of.hpp"
#  include "__concepts.hpp"
#  include "__continues_on.hpp"
#  include "__diagnostics.hpp"
#  include "__domain.hpp"
#  include "__env.hpp"
#  include "__just.hpp"
#  include "__schedulers.hpp"
#  include "__sender_facade.hpp"
#  include "__senders.hpp"
#  include "__sequence.hpp"
#  include "__subscribe.hpp"
#  include "__utility.hpp"

#  include "__prologue.hpp"

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  namespace __starts_on
  {
    // Is the child a sequence sender? Then so is starts_on, which is lowered to a
    // __sequence (see starts_on_t::transform_sender) that subscribes the child when the
    // schedule operation completes. Such a starts_on has the completions of the sender it is
    // lowered to, and the attributes below.
    // (This matches the sender's template arguments, so that it does not require the
    // sender to be complete.)
    template <class _Sender>
    inline constexpr bool __has_sequence_child_v = false;

    template <class _Tag, class _Data, class _Child>
    inline constexpr bool __has_sequence_child_v<__facade::__sender<_Tag, _Data, _Child>> =
      __sequence_sender_tagged<_Child>;

    template <class _Sexpr>
    concept __has_sequence_child = __has_sequence_child_v<__decay_t<_Sexpr>>;

    // The attributes of a starts_on whose child is a sequence sender
    template <class _Scheduler, class _Child>
    struct __attrs
    {
     private:
      template <class... _Env>
      static constexpr auto __mk_env2(_Scheduler __sch, _Env&&... __env)
      {
        return __env::__join(STDEXEC::__mk_sch_env(__sch, __env...), static_cast<_Env&&>(__env)...);
      }

      template <class... _Env>
      using __env2_t = decltype(__mk_env2(__declval<_Scheduler>(), __declval<_Env>()...));

      _Scheduler       __sched_;
      env_of_t<_Child> __attr_;

     public:
      constexpr explicit __attrs(_Scheduler __sch, env_of_t<_Child> __attr) noexcept
        : __sched_(static_cast<_Scheduler&&>(__sch))
        , __attr_(__attr)
      {}

      // Query for completion scheduler - delegates to child's env with
      // augmented environment
      template <class _SetTag, class... _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto query(get_completion_scheduler_t<_SetTag> __query,
                           _Env&&... __env) const noexcept
        -> __call_result_t<get_completion_scheduler_t<_SetTag>, env_of_t<_Child>, __env2_t<_Env...>>
      {
        // If child doesn't complete inline, delegate to child's completion
        // scheduler
        return __query(__attr_, __mk_env2(__sched_, __env...));
      }

      // Query for completion domain - calculate type from child's env with
      // augmented environment
      template <class _SetTag, class... _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto query(get_completion_domain_t<_SetTag>, _Env&&...) const noexcept
        -> __call_result_t<get_completion_domain_t<_SetTag>, env_of_t<_Child>, __env2_t<_Env...>>
      {
        return {};
      }

      template <__forwarding_query _Query, class... _Args>
        requires(!__completion_query<_Query>)
             && __queryable_with<env_of_t<_Child>, _Query, _Args...>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto query(_Query, _Args&&... __args) const
        noexcept(__nothrow_queryable_with<env_of_t<_Child>, _Query, _Args...>) -> decltype(auto)
      {
        return __attr_.query(_Query(), static_cast<_Args&&>(__args)...);
      }
    };

    // The key of starts_on's schedule operation
    struct __sched_op
    {};

    struct __algo;

    template <class _Sender, class _Receiver>
    struct __state
    {
      using __env_t        = env_of_t<_Receiver>;
      using __scheduler_t  = __decay_t<__data_of<_Sender>>;
      using __child_t      = __facade::__nth_child_of_t<_Sender, 0>;
      using __env2_t       = decltype(STDEXEC::__mk_sch_env(__declval<__scheduler_t>(),
                                                      __declval<__decay_t<__env_t> const &>()));
      using __sched_sndr_t = schedule_result_t<__scheduler_t const &>;
      using __sched_rcvr_t =
        __facade::__receiver_t<__algo, __sched_op, __state, _Sender, _Receiver>;
      using __child_rcvr_t =
        __facade::__receiver_t<__algo, __facade::__child<0>, __state, _Sender, _Receiver>;

      static constexpr bool __nothrow_v = __nothrow_callable<schedule_t, __scheduler_t const &>
                                       && __nothrow_connectable<__sched_sndr_t, __sched_rcvr_t>
                                       && __nothrow_connectable<__child_t, __child_rcvr_t>;

      // The scheduler is not moved from. (The child's environment holds a copy of it.)
      constexpr explicit __state(_Sender&& __sndr, _Receiver&& __rcvr) noexcept(__nothrow_v)
        : __rcvr_(static_cast<_Receiver&&>(__rcvr))
        , __env2_(STDEXEC::__mk_sch_env(__facade::__get_data(__sndr), STDEXEC::get_env(__rcvr_)))
        , __sched_op_(STDEXEC::connect(STDEXEC::schedule(static_cast<__scheduler_t const &>(
                                         __facade::__get_data(__sndr))),
                                       __sched_rcvr_t{this}))
        , __child_op_(STDEXEC::connect(__facade::__get_child<0>(static_cast<_Sender&&>(__sndr)),
                                       __child_rcvr_t{this}))
      {}

      STDEXEC_IMMOVABLE(__state);

      _Receiver                                        __rcvr_;
      __env2_t const                                   __env2_;
      connect_result_t<__sched_sndr_t, __sched_rcvr_t> __sched_op_;
      connect_result_t<__child_t, __child_rcvr_t>      __child_op_;
    };

    // starts_on(sch, sndr): starts a schedule operation on sch. When that completes with a
    // value, sndr is started, and sees sch as its start scheduler. The completions of sndr,
    // and the error and stopped completions of the schedule operation, are the operation's.
    //
    // NOT TO SPEC: the specification requires that starts_on be lowered to
    // let_value(schedule(sch), []{ return child; }), but that is inefficient on the GPU,
    // because it makes the child dependent on the completion of the schedule operation.
    // Here the child is connected up front.
    struct __algo : __facade::__defaults
    {
      // The child is connected with the state, and started by the schedule operation.
      static constexpr bool __connects_children = false;

      // The value completion of the schedule operation, where the child starts
      using __sched_value_t = __facade::__via<__sched_op, set_value_t, __facade::__from_start>;

      // Customizations of starts_on are found in the domain of the scheduler. If the child
      // never completes with a value, the value channel is reported as being there too.
      using __dispatch_source = __sched_value_t;

      template <class _Tag>
      using __empty_channel_source = __if_c<__same_as<_Tag, set_value_t>, __sched_value_t, void>;

      // A starts_on whose child is a sequence sender has the completions of the sender it
      // is lowered to, and the attributes above.
      template <class _Sender>
        requires __has_sequence_child<_Sender>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __get_attrs(_Sender const & __sndr) noexcept
      {
        auto const & __child = __facade::__get_child<0>(__sndr);
        return __attrs<__decay_t<decltype(__facade::__get_data(__sndr))>,
                       __decay_t<decltype(__child)>>{__facade::__get_data(__sndr),
                                                     STDEXEC::get_env(__child)};
      }

      template <class _Sender, class... _Env>
        requires __has_sequence_child<_Sender>
      static consteval auto __get_completion_signatures()
      {
        // (starts_on_t is incomplete here, so its transform_sender is named through the
        // sender's tag.)
        using __sndr_t =
          __detail::__transform_sender_result_t<tag_of_t<_Sender>, set_value_t, _Sender, env<>>;
        return STDEXEC::get_completion_signatures<__sndr_t, _Env...>();
      }

      // The child sees the scheduler as its start scheduler. (The schedule operation sees
      // the environment of the receiver: the default.)
      using __facade::__defaults::__env_for;
      using __facade::__defaults::__get_env;

      template <class _Sender, class _Env>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __env_for(__facade::__child<0>,
                                      _Sender const & __sndr,
                                      _Env&&          __env) noexcept
      {
        return __env::__join(STDEXEC::__mk_sch_env(__facade::__get_data(__sndr), __env),
                             static_cast<_Env&&>(__env));
      }

      template <class _State>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __get_env(__facade::__child<0>, _State const & __state) noexcept
      {
        using __env2_t = __decay_t<decltype(__state.__env2_)>;
        return __env::__join(__env2_t(__state.__env2_), STDEXEC::get_env(__state.__rcvr_));
      }

      using __facade::__defaults::__nested_sender;

      template <class _Sender, class... _Env>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __nested_sender(__sched_op, _Sender const & __sndr, _Env const &...)
      {
        return STDEXEC::schedule(__facade::__get_data(__sndr));
      }

      template <class _Sender, class... _Env>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __nested_attrs(__sched_op,
                                           _Sender const & __sndr,
                                           _Env const &...) noexcept
      {
        return __facade::__schedule_attrs<decltype(__facade::__get_data(__sndr))>{
          __facade::__get_data(__sndr)};
      }

      template <class _Sender, class _Receiver>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __get_state(_Sender&& __sndr, _Receiver __rcvr)
        noexcept(__state<_Sender, _Receiver>::__nothrow_v) -> __state<_Sender, _Receiver>
      {
        return __state<_Sender, _Receiver>(static_cast<_Sender&&>(__sndr),
                                           static_cast<_Receiver&&>(__rcvr));
      }

      template <class _State, class _Children>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __start(_State& __state, _Children&) noexcept
      {
        return __facade::__start_nested<__sched_op>(__state.__sched_op_);
      }

      // On the scheduler: start the child.
      template <class _State>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __complete(__sched_op, _State& __state, set_value_t) noexcept
      {
        return __facade::__start_nested<__facade::__child<0>>(__state.__child_op_);
      }

      // The child completed, or the schedule operation failed or was stopped.
      template <class _Key, class _State, class _Tag, class... _Args>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __complete(_Key,
                                       _State& __state,
                                       _Tag    __tag,
                                       _Args&&... __args) noexcept
      {
        return __facade::__send(__state.__rcvr_, __tag, static_cast<_Args&&>(__args)...);
      }
    };
  }  // namespace __starts_on

  //////////////////////////////////////////////////////////////////////////////
  // [exec.starts.on]

  //! @brief A sender adaptor that runs a sender starting on the execution
  //!        resource associated with a given scheduler.
  //!
  //! @c starts_on takes a scheduler @c sched and a sender @c sndr and produces
  //! a sender that, when connected and started, first hops onto @c sched's
  //! execution resource and then runs @c sndr there. The completions of the
  //! produced sender are delivered to the connected receiver from @c sched's
  //! resource — there is no "round trip" back to whatever scheduler started the
  //! operation (compare @ref on_t).
  //!
  //! Unlike most sender adaptors in stdexec, @c starts_on has no *pipe form*:
  //! it is always called as `starts_on(sched, sndr)`, never `sndr |
  //! starts_on(sched)`. This reflects the spec: @c starts_on takes the
  //! scheduler *first*, mirroring the order of operations (schedule, then run).
  //!
  //! @code{.cpp}
  //! auto s = stdexec::starts_on(some_sched, sndr);
  //! @endcode
  //!
  //! See [exec.starts.on] in the C++26 working draft for the normative
  //! specification.
  //!
  //! **Equivalence.**
  //!
  //! Semantically, `starts_on(sch, sndr)` is equivalent to
  //!
  //! @code{.cpp}
  //! schedule(sch) | let_value([sndr = std::forward<Sndr>(sndr)] {
  //!   return std::move(sndr);
  //! })
  //! @endcode
  //!
  //! stdexec's implementation is structured differently for efficiency on GPU
  //! contexts (it avoids making @c sndr dependent on the schedule-completion),
  //! but the observable semantics match.
  //!
  //! **Completion signatures.**
  //!
  //! The resulting sender's completion signatures are essentially those of
  //! @c sndr, augmented with any error completion that the scheduling step
  //! itself can produce:
  //!
  //! @code{.cpp}
  //! // (signatures of sndr) ...
  //! set_error_t(/* scheduler error type, if scheduling may fail */)
  //! set_stopped_t()                  // if not already present
  //! @endcode
  //!
  //! If scheduling onto @c sch fails, an error completion is delivered to the
  //! receiver on an *unspecified* execution agent.
  //!
  //! **Cancellation.**
  //!
  //! If the receiver requests stop before scheduling has produced its
  //! value-completion, the resulting sender typically completes via
  //! @c set_stopped rather than starting @c sndr at all — the precise
  //! behavior depends on the scheduler.
  //!
  //! **Example.**
  //!
  //! @code{.cpp}
  //! #include <stdexec/execution.hpp>
  //!
  //! int main() {
  //!   using namespace stdexec;
  //!
  //!   auto sched = get_parallel_scheduler();
  //!
  //!   // Run the entire chain on `sched`:
  //!   auto sndr =
  //!     starts_on(sched, just(21) | then([](int x) { return x * 2; }));
  //!
  //!   auto [v] = sync_wait(std::move(sndr)).value();
  //!   (void)v;  // == 42, computed on `sched`
  //! }
  //! @endcode
  //!
  //! @see stdexec::schedule       — the primitive that produces a
  //!                                schedule-sender
  //! @see stdexec::continues_on   — transfer to a scheduler *after* a sender
  //!                                completes
  //! @see stdexec::on             — run on a scheduler then transfer back to
  //!                                the original context
  struct starts_on_t
  {
    //! @brief Construct a sender that runs @c __sndr on @c __sched's
    //!        execution resource.
    //!
    //! @tparam _Scheduler A type satisfying the @c stdexec::scheduler concept.
    //! @tparam _Sender    A type satisfying the @c stdexec::sender concept.
    //!
    //! @param __sched     The scheduler whose execution resource will host
    //!                    @c __sndr.
    //! @param __sndr      The sender to run on @c __sched.
    //!
    //! @returns A sender that, when connected to a receiver and started,
    //!          first schedules onto @c __sched, then connects and starts
    //!          @c __sndr there, and forwards @c __sndr's completion to the
    //!          receiver.
    template <scheduler _Scheduler, sender _Sender>
    constexpr auto
    operator()(_Scheduler&& __sched, _Sender&& __sndr) const -> __well_formed_sender auto
    {
      return __facade::__sender{starts_on_t(),
                                static_cast<_Scheduler&&>(__sched),
                                static_cast<_Sender&&>(__sndr)};
    }

    // A starts_on whose child is a sequence sender is itself a sequence sender. It is
    // lowered to a __sequence, which can subscribe to its child.
    template <class _Sender>
      requires __starts_on::__has_sequence_child<_Sender>
    static constexpr auto transform_sender(set_value_t, _Sender&& __sndr, __ignore)
    {
      if constexpr (__decay_copyable<_Sender>)
      {
        auto& [__tag, __sched, __child] = __sndr;
        return __sequence(continues_on(just(), __sched), STDEXEC::__forward_like<_Sender>(__child));
      }
      else
      {
        return __not_a_sender<_SENDER_TYPE_IS_NOT_DECAY_COPYABLE_, _WITH_SENDER_<_Sender>>{};
      }
    }
  };

  //! @brief The customization point object for the @c starts_on sender adaptor.
  //!
  //! @c starts_on is an instance of @c starts_on_t. See @ref starts_on_t for
  //! the full description, completion signatures, and a usage example.
  //!
  //! @hideinitializer
  inline constexpr starts_on_t starts_on{};

  template <>
  struct __facade::__algorithm<starts_on_t> : __starts_on::__algo
  {};
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
