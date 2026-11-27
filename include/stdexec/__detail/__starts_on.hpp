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
#  include "__schedulers.hpp"
#  include "__sender_facade.hpp"
#  include "__sender_introspection.hpp"
#  include "__senders.hpp"
#  include "__utility.hpp"

#  include "__prologue.hpp"

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  namespace __starts_on
  {
    // The key of starts_on's schedule operation
    struct __sched_op
    {};

    struct __algo;

    // The state of a starts_on. If the starts_on is subscribed (_Subscribing is
    // __facade::__subscribing_t), so is its child, when that is a sequence sender (see
    // __facade::__attaches_child).
    template <class _Sender, class _Receiver, class... _Subscribing>
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
      using __child_op_t = __facade::
        __child_op_t<__algo, _Sender, 0, __state, __child_t, __child_rcvr_t, _Subscribing...>;

      static constexpr bool __nothrow_v = __nothrow_callable<schedule_t, __scheduler_t const &>
                                       && __nothrow_connectable<__sched_sndr_t, __sched_rcvr_t>
                                       && __facade::__nothrow_attachable_child<__algo,
                                                                               _Sender,
                                                                               0,
                                                                               __state,
                                                                               __child_t,
                                                                               __child_rcvr_t,
                                                                               _Subscribing...>;

      // The scheduler is not moved from. (The child's environment holds a copy of it.)
      constexpr explicit __state(_Sender&& __sndr, _Receiver&& __rcvr) noexcept(__nothrow_v)
        : __rcvr_(static_cast<_Receiver&&>(__rcvr))
        , __env2_(STDEXEC::__mk_sch_env(__facade::__get_data(__sndr), STDEXEC::get_env(__rcvr_)))
        , __sched_op_(STDEXEC::connect(STDEXEC::schedule(static_cast<__scheduler_t const &>(
                                         __facade::__get_data(__sndr))),
                                       __sched_rcvr_t{this}))
        , __child_op_(__attach_child(__facade::__get_child<0>(static_cast<_Sender&&>(__sndr))))
      {}

      STDEXEC_IMMOVABLE(__state);

      _Receiver                                        __rcvr_;
      __env2_t const                                   __env2_;
      connect_result_t<__sched_sndr_t, __sched_rcvr_t> __sched_op_;
      __child_op_t                                     __child_op_;

     private:
      // Connects the child, or attaches it (see __facade::__attaches_child)
      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto __attach_child(__child_t&& __child) noexcept(__nothrow_v) -> __child_op_t
      {
        if constexpr (__facade::__attaches_child<__algo, _Sender, 0>)
        {
          using __attach_t =
            __facade::__attach_child_t<__algo, _Sender, 0, __child_rcvr_t, _Subscribing...>;
          return __attach_t()(*this, static_cast<__child_t&&>(__child), __child_rcvr_t{this});
        }
        else
        {
          return STDEXEC::connect(static_cast<__child_t&&>(__child), __child_rcvr_t{this});
        }
      }
    };

    // starts_on(sch, sndr): starts a schedule operation on sch. When that completes with a
    // value, sndr is started, and sees sch as its start scheduler. The completions of sndr,
    // and the error and stopped completions of the schedule operation, are the operation's.
    // If sndr is a sequence sender, so is starts_on. Its items are sndr's, which it subscribes
    // when it is subscribed.
    //
    // NOT TO SPEC: the specification requires that starts_on be lowered to
    // let_value(schedule(sch), []{ return child; }), but that is inefficient on the GPU,
    // because it makes the child dependent on the completion of the schedule operation.
    // Here the child is connected, or subscribed, up front.
    struct __algo : __facade::__defaults
    {
      // The child is attached with the state, and started by the schedule operation.
      static constexpr bool __connects_children = false;

      template <class _Sender>
      using __item_source = __facade::__child<0>;

      // The value completion of the schedule operation, where the child starts
      using __sched_value_t = __facade::__via<__sched_op, set_value_t, __facade::__from_start>;

      // Customizations of starts_on are found in the domain of the scheduler. If the child
      // never completes with a value, the value channel is reported as being there too.
      using __dispatch_source = __sched_value_t;

      template <class _Tag>
      using __empty_channel_source = __if_c<__same_as<_Tag, set_value_t>, __sched_value_t, void>;

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

      // (The state of a subscribed starts_on takes __facade::__subscribing_t.)
      template <class _Sender, class _Receiver, class... _Subscribing>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __get_state(_Sender&& __sndr, _Receiver __rcvr, _Subscribing...)
        noexcept(__state<_Sender, _Receiver, _Subscribing...>::__nothrow_v)
          -> __state<_Sender, _Receiver, _Subscribing...>
      {
        return __state<_Sender, _Receiver, _Subscribing...>(static_cast<_Sender&&>(__sndr),
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

  // A starts_on's state is incomplete while the starts_on subscribes its child, since that
  // operation is part of the state. So the type of its receiver, to which the child's receiver
  // forwards items, is read from the state's type.
  template <class _Item, class _Sender, class _Receiver, class... _Subscribing>
  struct __facade::__state_receiver<__starts_on::__state<_Sender, _Receiver, _Subscribing...>,
                                    _Item>
  {
    using __t = _Receiver;
  };

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
    constexpr auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const -> auto
    {
      return STDEXEC::__check_well_formed(__facade::__sender{starts_on_t(),
                                                             static_cast<_Scheduler&&>(__sched),
                                                             static_cast<_Sender&&>(__sndr)});
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
