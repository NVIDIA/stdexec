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
#  include "__continues_on.hpp"
#  include "__diagnostics.hpp"
#  include "__env.hpp"
#  include "__inline_scheduler.hpp"
#  include "__meta.hpp"
#  include "__schedulers.hpp"
#  include "__sender_adaptor_closure.hpp"
#  include "__sender_facade.hpp"
#  include "__sender_introspection.hpp"
#  include "__starts_on.hpp"
#  include "__utility.hpp"

#  if !STDEXEC_USE_MODULES()
#    include <cstddef>
#    include <utility>  // for std::tuple_size and std::tuple_element
#  endif

#  include "__prologue.hpp"

STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  //////////////////////////////////////////////////////////////////////////////
  // [exec.on]
  struct _CANNOT_RESTORE_EXECUTION_CONTEXT_AFTER_ON_;
  struct _ON_NEEDS_A_SENDER_OR_A_SENDER_ADAPTOR_CLOSURE_;

  namespace __on
  {
    // The error for `sndr | on(sch)`: `on` has no form that takes only a
    // scheduler.
    template <class _Scheduler>
    using __scheduler_only_error_t =
      __mexception<_WHAT_(_INVALID_EXPRESSION_),
                   _WHERE_(_IN_ALGORITHM_, on_t),
                   _WHY_(_ON_NEEDS_A_SENDER_OR_A_SENDER_ADAPTOR_CLOSURE_),
                   _WITH_SCHEDULER_(_Scheduler)>;

    // Is the sender on(sndr, sch, clsur), rather than on(sch, sndr)? on(sch, sndr) keeps no
    // data (see __sndr). (This matches the sender's template arguments, so that it does not
    // need the sender to be complete.)
    template <class _Sender>
    inline constexpr bool __has_closure_v = false;

    template <class _Tag, class _Data, class _Child>
    inline constexpr bool __has_closure_v<__facade::__sender<_Tag, _Data, _Child>> =
      !__same_as<_Data, __>;

    // The sender that `on` runs on another scheduler: sndr, in both forms. (on(sch, sndr)
    // keeps it in starts_on(sch, sndr).)
    template <class _Sender>
    struct __user_child
    {
      using __t = __facade::__nth_child_of_t<_Sender, 0>;
    };

    template <class _Sender>
      requires(!__has_closure_v<__decay_t<_Sender>>)
    struct __user_child<_Sender>
    {
      using __t = __facade::__nth_child_of_t<__facade::__nth_child_of_t<_Sender, 0>, 0>;
    };

    template <class _Sender>
    using __user_child_t = __user_child<_Sender>::__t;

    // In a root environment, there is no need to restore the execution context, so if the
    // environment does not say where to return to, `on` completes inline.
    template <class _Env>
    using __end_sched_t = __if_c<__is_root_env<_Env>, inline_scheduler, __not_a_scheduler<>>;

    // The scheduler that `on` returns to: for on(sch, sndr), the start scheduler; for
    // on(sndr, sch, clsur), the scheduler on which sndr completes.
    template <class _Sender, class _Env>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto __return_sched(_Sender const & __sndr, _Env const & __env) noexcept
    {
      if constexpr (__has_closure_v<_Sender>)
      {
        auto __get_sched = __with_default(get_completion_scheduler<set_value_t>,
                                          __end_sched_t<_Env>());
        return __get_sched(STDEXEC::get_env(__facade::__get_child<0>(__sndr)), __env);
      }
      else
      {
        return __with_default(get_start_scheduler, __end_sched_t<_Env>())(__env);
      }
    }

    template <class _Sender, class _Env>
    using __return_sched_t = decltype(__on::__return_sched(__declval<_Sender const &>(),
                                                           __declval<_Env const &>()));

    template <class _Sender, class _Env>
    inline constexpr bool __can_return_v =
      __not_same_as<__return_sched_t<__decay_t<_Sender>, _Env>, __not_a_scheduler<>>;

    // The sender that `on` is lowered to when the environment does not say where to return to
    template <class _Sender, class _Env>
    using __cannot_return_t = __not_a_sender<
      _WHAT_(_CANNOT_RESTORE_EXECUTION_CONTEXT_AFTER_ON_),
      _WHY_(__if_c<__has_closure_v<__decay_t<_Sender>>,
                   _THE_PREDECESSOR_SENDER_DOES_NOT_KNOW_THE_SCHEDULER_ON_WHICH_IT_WILL_COMPLETE_,
                   _THE_CURRENT_EXECUTION_ENVIRONMENT_DOES_NOT_HAVE_A_START_SCHEDULER_>),
      _WHERE_(_IN_ALGORITHM_, on_t),
      _WITH_SENDER_<__decay_t<__user_child_t<_Sender>>>,
      _WITH_ENVIRONMENT_(_Env)>;

    // on(sch, sndr) keeps the sender that it runs, starts_on(sch, sndr), so that `on` does
    // not make it again when it is lowered, or when its attributes are asked for. Its
    // structured bindings are those of the standard's on(sch, sndr): the tag, sch, and sndr,
    // which are the parts of starts_on(sch, sndr) after its tag.
    // (It has the template parameters of the facade sender that it is, so that it is taken
    // apart the same way.)
    template <class _Tag, class _Data, class _StartsOn>
    struct __sndr : __facade::__sender<_Tag, _Data, _StartsOn>
    {
      template <std::size_t _Idx, class _Self>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __part(_Self&& __self) noexcept -> decltype(auto)
      {
        if constexpr (_Idx == 0)
          return _Tag();
        else
          return STDEXEC::__get<_Idx>(__facade::__get_child<0>(static_cast<_Self&&>(__self)));
      }

      template <std::size_t _Idx>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr auto get() & noexcept -> decltype(auto)
      {
        return __sndr::__part<_Idx>(*this);
      }

      template <std::size_t _Idx>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr auto get() const & noexcept -> decltype(auto)
      {
        return __sndr::__part<_Idx>(*this);
      }

      template <std::size_t _Idx>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr auto get() && noexcept -> decltype(auto)
      {
        return __sndr::__part<_Idx>(static_cast<__sndr&&>(*this));
      }
    };

    // `on` is lowered (see on_t::transform_sender) to the senders it is made of:
    //   on(sch, sndr) to continues_on(starts_on(sch, sndr), orig_sch), and
    //   on(sndr, sch, clsur) to continues_on(clsur(continues_on(sndr, sch)), orig_sch),
    // where orig_sch is the scheduler that it returns to (see __return_sched).
    template <class _StartsOn, class _Scheduler>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto __lower(_StartsOn&& __starts_on, _Scheduler __orig_sch)
    {
      return continues_on(static_cast<_StartsOn&&>(__starts_on),
                          static_cast<_Scheduler&&>(__orig_sch));
    }

    template <class _Data, class _Child, class _Scheduler>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto __lower(_Data&& __data, _Child&& __child, _Scheduler __orig_sch)
    {
      auto& [__sch, __clsur] = __data;
      return continues_on(STDEXEC::__forward_like<_Data>(__clsur)(
                            continues_on(static_cast<_Child&&>(__child),
                                         STDEXEC::__forward_like<_Data>(__sch))),
                          static_cast<_Scheduler&&>(__orig_sch));
    }

    // A stand-in for the sender that an `on` sender keeps, with that sender's completions
    // and attributes, so that the sender that `on` is lowered to can be made, and asked for
    // its attributes, without copying it (see __attrs). It is never connected. But the
    // facade computes the types of the operations of the senders made from it, so it says
    // what it would connect to.
    template <class _Child>
    struct __proxy
    {
      using sender_concept = sender_tag;

      template <class _Self, class... _Env>
      static consteval auto get_completion_signatures()
      {
        return STDEXEC::get_completion_signatures<_Child, _Env...>();
      }

      template <class _Receiver>
      [[noreturn]] STDEXEC_ATTRIBUTE(host, device) auto connect(_Receiver) && noexcept
        -> connect_result_t<_Child, _Receiver>
      {
        STDEXEC_TERMINATE();
      }

      [[nodiscard]]
      STDEXEC_ATTRIBUTE(always_inline, host, device) constexpr auto get_env() const noexcept
        -> env_of_t<_Child>
      {
        return STDEXEC::get_env(__child_);
      }

      _Child const & __child_;
    };

    template <class _Child>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __proxy(_Child const &) -> __proxy<_Child>;

    // An `on` sender lowered in an environment, with a proxy for the sender that it keeps
    template <class _Sender, class _Env>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto __lower_proxy(_Sender const & __sndr, _Env const & __env)
    {
      if constexpr (__has_closure_v<_Sender>)
        return __on::__lower(__facade::__get_data(__sndr),
                             __proxy{__facade::__get_child<0>(__sndr)},
                             __on::__return_sched(__sndr, __env));
      else
        return __on::__lower(__proxy{__facade::__get_child<0>(__sndr)},
                             __on::__return_sched(__sndr, __env));
    }

    template <class _Sender, class _Env>
    using __lowered_proxy_t = decltype(__on::__lower_proxy(__declval<_Sender const &>(),
                                                           __declval<_Env const &>()));

    // The attributes of an `on` sender: the completion attributes, in an environment in
    // which it can return, of the sender that it is lowered to there (made with a proxy, see
    // __proxy), and the other forwarding queries of the sender that it runs on another
    // scheduler
    template <class _Sender>
    struct __attrs
    {
      // (The result type is deduced. Compilers that substitute into the return type before
      // checking the constraints, as Clang does through version 19, would otherwise lower
      // the sender in an environment in which it cannot return.)
      template <class _Query, class _Env>
        requires __completion_query<_Query> && __can_return_v<_Sender, _Env>
              && __queryable_with<env_of_t<__lowered_proxy_t<_Sender, _Env>>, _Query, _Env const &>
      [[nodiscard]]
      STDEXEC_ATTRIBUTE(host, device) constexpr auto query(_Query,
                                                           _Env const & __env) const noexcept
        -> decltype(auto)
      {
        auto __lowered = __on::__lower_proxy(__sndr_, __env);
        return STDEXEC::get_env(__lowered).query(_Query(), __env);
      }

      template <__forwarding_query _Query, class... _Args>
        requires(!__completion_query<_Query>)
             && __queryable_with<env_of_t<__user_child_t<_Sender>>, _Query, _Args...>
      [[nodiscard]]
      STDEXEC_ATTRIBUTE(always_inline, host, device) constexpr auto query(_Query,
                                                                          _Args&&... __args) const
        noexcept(__nothrow_queryable_with<env_of_t<__user_child_t<_Sender>>, _Query, _Args...>)
          -> decltype(auto)
      {
        return STDEXEC::get_env(__user_child()).query(_Query(), static_cast<_Args&&>(__args)...);
      }

      _Sender const & __sndr_;

     private:
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr auto __user_child() const noexcept -> decltype(auto)
      {
        if constexpr (__has_closure_v<_Sender>)
          return __facade::__get_child<0>(__sndr_);
        else
          return __facade::__get_child<0>(__facade::__get_child<0>(__sndr_));
      }
    };

    // `on` is lowered, so it has the completions and the attributes of the sender that it is
    // lowered to. Both depend on the environment, which says where it returns to.
    struct __algo : __facade::__defaults
    {
      template <class _Sender>
      static constexpr bool __depends_on_env = true;

      template <class _Sender>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __get_attrs(_Sender const & __sndr) noexcept
      {
        return __attrs<_Sender>{__sndr};
      }

      template <class _Sender, class _Env>
      static consteval auto __get_completion_signatures()
      {
        // (on_t is incomplete here, so its transform_sender is named through the sender's
        // tag.)
        using __sndr_t =
          __detail::__transform_sender_result_t<tag_of_t<_Sender>, set_value_t, _Sender, _Env>;
        return STDEXEC::get_completion_signatures<__sndr_t, _Env>();
      }
    };
  }  // namespace __on

  //////////////////////////////////////////////////////////////////////////////
  //! @brief A sender adaptor that runs work on a different scheduler and then
  //!        transfers execution *back* to the original scheduler.
  //!
  //! @c on is the "go there, do work, come back" scheduling adaptor. It has
  //! two distinct shapes:
  //!
  //! 1. **Whole-sender form:** `on(sched, sndr)` — runs the entirety
  //!    of @c sndr on @c sched's execution resource. When @c sndr completes,
  //!    execution transfers back to the scheduler that started the operation
  //!    (the "start scheduler"), and the completion is delivered there.
  //!
  //!    This is the principal difference between @c on and
  //!    @ref starts_on_t — @c starts_on stays on @c sched; @c on returns home.
  //!
  //! 2. **Closure-insertion form:** `on(sndr, sched, closure)` — runs
  //!    @c sndr on its *current* scheduler, then transfers to @c sched,
  //!    applies @c closure (a sender-adaptor closure) to the result, runs
  //!    *that* on @c sched, and finally transfers back to the original
  //!    completion scheduler. Useful for inserting a CPU-bound transform
  //!    into an otherwise I/O-bound pipeline (or vice versa) without
  //!    permanently changing context.
  //!
  //!    This form also has a pipe shorthand:
  //!    `sndr | on(sched, closure)`.
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
  //! Form 1 (`on(sched, sndr)`): essentially @c sndr's completion
  //! signatures, with possible additional @c set_error_t completions from
  //! the two scheduling hops.
  //!
  //! Form 2 (`on(sndr, sched, closure)`): the completion signatures
  //! of `closure(continues_on(sndr, sched))` after the final transfer
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
  //! @see stdexec::schedule       — the primitive that produces a
  //!                                schedule-sender
  //! @see stdexec::starts_on      — begin on a scheduler and *stay* there
  //! @see stdexec::continues_on   — transfer to a scheduler *after* a sender
  //!                                completes
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
      // (on(sch, sndr) keeps the sender that it runs. See __on::__sndr.)
      using __starts_on_t = __call_result_t<starts_on_t, _Scheduler, _Sender>;
      return __on::__sndr<on_t, __, __starts_on_t>{
        __facade::__sender{on_t(),
                           __(),
                           starts_on(static_cast<_Scheduler&&>(__sched),
                           static_cast<_Sender&&>(__sndr))}
      };
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
    //! @param __clsur     The adaptor closure (e.g. `then(...)`,
    //!                    `bulk(...)`) to apply on @c __sched.
    //!
    //! @returns A sender that completes on the *original* completion
    //!          scheduler of @c __sndr, with the result of
    //!          `__clsur(continues_on(__sndr, __sched))`.
    template <sender _Sender, scheduler _Scheduler, __sender_adaptor_closure_for<_Sender> _Closure>
    constexpr auto operator()(_Sender&& __sndr, _Scheduler&& __sched, _Closure&& __clsur) const
      -> __well_formed_sender auto
    {
      return __facade::__sender{
        on_t(),
        __tuple{static_cast<_Scheduler&&>(__sched), static_cast<_Closure&&>(__clsur)},
        static_cast<_Sender&&>(__sndr)
      };
    }

    //! @brief Pipe form of Form 2: construct a sender-adaptor closure that,
    //!        when applied to a sender, produces
    //!        `on(sndr, __sched, __clsur)`.
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

    // For issuing a meaningful diagnostic for the erroneous `sndr | on(sch)`.
    // The constraint is never satisfied, so this overload is never selected.
    template <scheduler _Scheduler>
      requires __ok<__on::__scheduler_only_error_t<_Scheduler>>
    auto operator()(_Scheduler&&) const;

    // Lowers on(sch, sndr) to continues_on(starts_on(sch, sndr), orig_sch), and
    // on(sndr, sch, clsur) to continues_on(clsur(continues_on(sndr, sch)), orig_sch), where
    // orig_sch is the scheduler that `on` returns to, which the environment says (see
    // __on::__return_sched).
    template <__decay_copyable _Sender, class _Env>
    STDEXEC_ATTRIBUTE(always_inline)
    static constexpr auto transform_sender(set_value_t, _Sender&& __sndr, _Env const & __env)
    {
      static_assert(__sender_for<_Sender, on_t>);
      // (on(sch, sndr) is a type derived from a facade sender. See __on::__sndr.)
      using __sndr_t      = __facade::__facade_sender_t<_Sender>;
      auto const & __self = static_cast<__sndr_t const &>(__sndr);
      if constexpr (!__on::__can_return_v<__sndr_t, _Env>)
      {
        return __on::__cannot_return_t<__sndr_t, _Env>();
      }
      else if constexpr (!__on::__has_closure_v<__sndr_t>)
      {
        return __on::__lower(__facade::__get_child<0>(static_cast<_Sender&&>(__sndr)),
                             __on::__return_sched(__self, __env));
      }
      else
      {
        auto& [__tag, __data, __child] = __sndr;
        return __on::__lower(STDEXEC::__forward_like<_Sender>(__data),
                             STDEXEC::__forward_like<_Sender>(__child),
                             __on::__return_sched(__self, __env));
      }
    }

    template <class _Sender, class _Env>
    static constexpr auto transform_sender(set_value_t, _Sender&&, _Env const &)
    {
      return __not_a_sender<_WHAT_(_SENDER_TYPE_IS_NOT_DECAY_COPYABLE_), _WITH_SENDER_<_Sender>>{};
    }
  };

  //! @brief The customization point object for the @c on sender adaptor.
  //!
  //! @c on is an instance of @c on_t. See @ref on_t for the full
  //! description, the distinction between @c on, @c starts_on, and
  //! @c continues_on, and usage examples.
  //!
  //! @hideinitializer
  inline constexpr on_t on{};

  template <>
  struct __facade::__algorithm<on_t> : __on::__algo
  {};
}  // namespace STDEXEC

// The structured bindings of on(sch, sndr): the tag, sch, and sndr (see __on::__sndr)
template <class _Tag, class _Data, class _StartsOn>
struct std::tuple_size<STDEXEC::__on::__sndr<_Tag, _Data, _StartsOn>>
  : std::integral_constant<std::size_t, 3>
{};

template <std::size_t _Idx, class _Tag, class _Data, class _StartsOn>
struct std::tuple_element<_Idx, STDEXEC::__on::__sndr<_Tag, _Data, _StartsOn>>
{
  using __sndr_t = STDEXEC::__on::__sndr<_Tag, _Data, _StartsOn>;
  using type =
    STDEXEC::__decay_t<decltype(__sndr_t::template __part<_Idx>(STDEXEC::__declval<__sndr_t&>()))>;
};

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
