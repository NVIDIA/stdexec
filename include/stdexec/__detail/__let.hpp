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
#  include "__diagnostics.hpp"
#  include "__env.hpp"
#  include "__meta.hpp"
#  include "__schedulers.hpp"
#  include "__sender_adaptor_closure.hpp"
#  include "__sender_facade.hpp"
#  include "__sender_introspection.hpp"
#  include "__senders.hpp"
#  include "__utility.hpp"
#  include "__variant.hpp"

#  include "../functional.hpp"

#  if !STDEXEC_USE_MODULES()
#    include <exception>
#  endif

#  include "__prologue.hpp"

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  struct _FUNCTION_DOES_NOT_RETURN_A_SENDER_;
  struct _FUNCTION_RETURNS_A_SENDER_THAT_CANNOT_BE_CONNECTED_;

  //////////////////////////////////////////////////////////////////////////////
  // [exec.let]
  namespace __let
  {
    template <class _SetTag>
    struct __let_t;

    template <class _SetTag, class _Sender, class _Env>
    using __env2_t = __secondary_env_t<_Sender, _Env, _SetTag>;

    template <class _SetTag, class _Sender, class _Env>
    using __result_env_t = __join_env_t<__env2_t<_SetTag, _Sender, _Env>, _Env>;

    template <class _LetTag>
    struct __let_t
    {
      template <sender _Sender, __movable_value _Fun>
      constexpr auto operator()(_Sender&& __sndr, _Fun __fn) const -> auto
      {
        return STDEXEC::__check_well_formed(
          __facade::__sender{_LetTag(), static_cast<_Fun&&>(__fn), static_cast<_Sender&&>(__sndr)});
      }

      template <class _Fun>
      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()(_Fun __fn) const
      {
        return __closure(_LetTag(), static_cast<_Fun&&>(__fn));
      }

     private:
      friend _LetTag;
      __let_t() = default;
    };

    // The key of the operation of the sender that the function returns when the child
    // completes with arguments whose decayed types are _Ts...
    template <class... _Ts>
    struct __result_op
    {};

    template <class _LetTag, class _SetTag>
    struct __algo;

    template <class _SetTag, class _Sig>
    struct __args_of_sig
    {
      using __t = __mlist<>;
    };

    template <class _SetTag, class... _Args>
    struct __args_of_sig<_SetTag, _SetTag(_Args...)>
    {
      using __t = __mlist<__mlist<__decay_t<_Args>...>>;
    };

    // The decayed argument lists of the child's _SetTag completions
    template <class _SetTag, class _Sigs>
    struct __args_lists;

    template <class _SetTag, class... _Sigs>
    struct __args_lists<_SetTag, completion_signatures<_Sigs...>>
    {
      using __t = __mcall<__mconcat<__munique<>>, typename __args_of_sig<_SetTag, _Sigs>::__t...>;
    };

    // A stand-in for the operation of the sender that the function returns, when it cannot
    // be called with the child's results _Ts, or returns something that cannot be connected.
    // The algorithm reports that as a type error, and never makes one of these, but its state
    // must still be well-formed, so that the walk gets as far as reporting it.
    template <class... _Ts>
    struct __no_result_op
    {};

    template <class _Sender2, class _Receiver2, class... _Ts>
    struct __result_op_of
    {
      using __t = __no_result_op<_Ts...>;
    };

    template <class _Sender2, class _Receiver2, class... _Ts>
      requires sender_to<_Sender2, _Receiver2>
    struct __result_op_of<_Sender2, _Receiver2, _Ts...>
    {
      using __t = connect_result_t<_Sender2, _Receiver2>;
    };

    template <class _LetTag, class _Sender, class _Receiver>
    struct __state
    {
      using __set_tag_t = __t<_LetTag>;
      using __algo_t    = __algo<_LetTag, __set_tag_t>;
      using __fn_t      = __decay_t<__data_of<_Sender>>;
      using __env_t     = env_of_t<_Receiver>;
      using __env2_t    = __secondary_env_t<__child_of<_Sender>, __decay_t<__env_t>, __set_tag_t>;

      // The results of the child's __set_tag_t completions, decayed
      using __args_lists_t = __args_lists<
        __set_tag_t,
        __facade::__completions_of_t<__algo_t, __facade::__child<0>, _Sender, _Receiver>>::__t;

      template <class... _Ts>
      using __sender2_t = __invoke_result_t<__fn_t, _Ts&...>;

      template <class... _Ts>
      using __receiver2_t =
        __facade::__receiver_t<__algo_t, __result_op<_Ts...>, __state, _Sender, _Receiver>;

      template <class _List>
      struct __types;

      template <class... _Ts>
      struct __types<__mlist<_Ts...>>
      {
        using __tuple_t = __tuple<_Ts...>;
      };

      // The operation of the sender that the function returns when it is called with the
      // results in _List. (Kept apart from __types, which the results' storage names, so that
      // naming the storage does not name the operations: see __ops_lazy.)
      template <class _List>
      struct __result_op_for;

      template <class... _Ts>
      struct __result_op_for<__mlist<_Ts...>>
      {
        using __t = __result_op_of<__minvoke_or_q<__invoke_result_t, void, __fn_t, _Ts&...>,
                                   __receiver2_t<_Ts...>,
                                   _Ts...>::__t;
      };

      // The child is connected with the state. The operation of the sender that the
      // function returns replaces the child's, so that the child's is destroyed once the
      // function has been called.
      using __child_t = __facade::__nth_child_of_t<_Sender, 0>;
      using __child_rcvr_t =
        __facade::__receiver_t<__algo_t, __facade::__child<0>, __state, _Sender, _Receiver>;

      template <class... _Lists>
      struct __variants
      {
        using __args_t = __variant<typename __types<_Lists>::__tuple_t...>;
      };

      template <class... _Lists>
      struct __ops_variant
      {
        using __t = __variant<connect_result_t<__child_t, __child_rcvr_t>,
                              typename __result_op_for<_Lists>::__t...>;
      };

      using __variants_t = __mapply_q<__variants, __args_lists_t>;

      // (A walk needs only the state's type, so it does not name the operations of the child
      // and of the senders that the function returns: they would be connected to archetypes.)
      struct __ops_lazy
      {
        using __t = __mapply_q<__ops_variant, __args_lists_t>::__t;
      };
      using __ops_t =
        __facade::__lazy_state_ops_t<_Receiver, __ops_lazy, __variant<__facade::__walk_op>>;

      static consteval auto __walking() noexcept -> bool
      {
        return __facade::__walking_receiver<_Receiver>;
      }

      static constexpr bool __nothrow_v = __nothrow_decay_copyable<__data_of<_Sender>>
                                       && __nothrow_connectable<__child_t, __child_rcvr_t>;

      constexpr explicit __state(_Sender&& __sndr, _Receiver&& __rcvr) noexcept(__nothrow_v)
        : __rcvr_(static_cast<_Receiver&&>(__rcvr))
        , __fn_(
            STDEXEC::__allocator_aware_forward(__facade::__get_data(static_cast<_Sender&&>(__sndr)),
                                               __rcvr_))
        , __env2_(__mk_secondary_env_t<__set_tag_t>()(__facade::__get_child<0>(__sndr),
                                                      STDEXEC::get_env(__rcvr_)))
      {
        if constexpr (!__walking())
        {
          __ops_.__emplace_from(STDEXEC::connect,
                                __facade::__get_child<0>(static_cast<_Sender&&>(__sndr)),
                                __child_rcvr_t{this});
        }
      }

      STDEXEC_IMMOVABLE(__state);

      _Receiver      __rcvr_;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      __fn_t         __fn_;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      __env2_t const __env2_;
      // The results of the child, and the operation of the child, and then of the sender
      // that the function returned
      typename __variants_t::__args_t __args_{__no_init};
      __ops_t                         __ops_{__no_init};
    };

    // let_value(sndr, fn), let_error(sndr, fn), and let_stopped(sndr, fn): when the child
    // completes with _SetTag, its results are saved, and fn is called with them. The sender
    // that fn returns is connected and started, and its completions are the operation's.
    // Other completions of the child are passed through.
    template <class _LetTag, class _SetTag>
    struct __algo : __facade::__defaults
    {
      // Without an environment, it is not known whether connecting the senders that the
      // function returns can throw.
      template <class _Sender>
      static constexpr bool __depends_on_env = true;

      // The child is connected with the state (see __state).
      static constexpr bool __connects_children = false;

      // Customizations of let_ are found in the domain where the child completes with
      // _SetTag, which is where the function is called.
      using __dispatch_source =
        __facade::__via<__facade::__child<0>, _SetTag, __facade::__from_start>;

      // The sender that the function returns sees where the child completed.
      template <class... _Ts, class _Sender, class _Env>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __env_for(__result_op<_Ts...>,
                                      _Sender const & __sndr,
                                      _Env&&          __env) noexcept
      {
        return __env::__join(__mk_secondary_env_t<_SetTag>()(__facade::__get_child<0>(__sndr),
                                                             __env),
                             static_cast<_Env&&>(__env));
      }

      template <class... _Ts, class _State>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __get_env(__result_op<_Ts...>, _State const & __state) noexcept
      {
        using __env2_t = __decay_t<decltype(__state.__env2_)>;
        return __env::__join(__env2_t(__state.__env2_), STDEXEC::get_env(__state.__rcvr_));
      }

      using __facade::__defaults::__env_for;
      using __facade::__defaults::__get_env;

      // The sender that the function returns is created at run time, so it is known only by
      // its type.
      template <class... _Ts, class _Sender, class... _Env>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __nested_sender(__result_op<_Ts...>,
                                            _Sender const &,
                                            _Env const &...) noexcept
        -> __mtype<__invoke_result_t<__decay_t<__facade::__data_of_t<_Sender>>, _Ts&...>>
      {
        return {};
      }

      using __facade::__defaults::__nested_sender;

      // (See __facade::__state_type_t.)
      template <class _Sender, class _Receiver>
      using __state_type = __state<_LetTag, _Sender, _Receiver>;

      template <class _Sender, class _Receiver>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __get_state(_Sender&& __sndr, _Receiver __rcvr)
        noexcept(__state<_LetTag, _Sender, _Receiver>::__nothrow_v)
          -> __state<_LetTag, _Sender, _Receiver>
      {
        return __state<_LetTag, _Sender, _Receiver>(static_cast<_Sender&&>(__sndr),
                                                    static_cast<_Receiver&&>(__rcvr));
      }

      template <class _State, class _Children>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __start(_State& __state, _Children&) noexcept
      {
        return __facade::__start_nested<__facade::__child<0>>(__var::__get<0>(__state.__ops_));
      }

      // The child completed with _SetTag: save its results, call the function with them, and
      // connect and start the sender it returns.
      template <class _State, class... _Args>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __complete(__facade::__child<0>,
                                       _State& __state,
                                       _SetTag,
                                       _Args&&... __args) noexcept
      {
        using __fn_t = _State::__fn_t;
        if constexpr (!__decay_copyable<_Args...>)
        {
          return __facade::__type_error(
            __mexception<_WHAT_(_PREDECESSOR_RESULTS_ARE_NOT_DECAY_COPYABLE_),
                         _WHERE_(_IN_ALGORITHM_, _LetTag),
                         _WITH_ARGUMENTS_(_Args...)>());
        }
        else if constexpr (!__invocable<__fn_t, __decay_t<_Args>&...>)
        {
          return __facade::__type_error(
            __callable_error_t<_LetTag, __fn_t, __decay_t<_Args>&...>());
        }
        else if constexpr (!sender<__invoke_result_t<__fn_t, __decay_t<_Args>&...>>)
        {
          return __facade::__type_error(__mexception<_WHAT_(_FUNCTION_DOES_NOT_RETURN_A_SENDER_),
                                                     _WHERE_(_IN_ALGORITHM_, _LetTag),
                                                     _WITH_FUNCTION_(__fn_t),
                                                     _WITH_ARGUMENTS_(_Args...)>());
        }
        else if constexpr (!sender_to<__invoke_result_t<__fn_t, __decay_t<_Args>&...>,
                                      typename _State::template __receiver2_t<__decay_t<_Args>...>>)
        {
          using __sender2_t   = __invoke_result_t<__fn_t, __decay_t<_Args>&...>;
          using __receiver2_t = _State::template __receiver2_t<__decay_t<_Args>...>;
          // If the sender does not type-check in its environment, the walk reports why, when
          // it asks the sender for its completions there. (Then the algorithm's sender does
          // not type-check either, so it is never connected, and this never runs.)
          if constexpr (!sender_in<__sender2_t, env_of_t<__receiver2_t>>)
            return __facade::__effects<__facade::__starts<__result_op<__decay_t<_Args>...>>>();
          else
            return __facade::__type_error(
              __mexception<_WHAT_(_FUNCTION_RETURNS_A_SENDER_THAT_CANNOT_BE_CONNECTED_),
                           _WHERE_(_IN_ALGORITHM_, _LetTag),
                           _WITH_FUNCTION_(__fn_t),
                           _WITH_SENDER_<__sender2_t>>());
        }
        else
        {
          using __key_t              = __result_op<__decay_t<_Args>...>;
          using __sender2_t          = __invoke_result_t<__fn_t, __decay_t<_Args>&...>;
          using __receiver2_t        = _State::template __receiver2_t<__decay_t<_Args>...>;
          constexpr bool __nothrow_v = __nothrow_decay_copyable<_Args...>
                                    && __nothrow_invocable<__fn_t, __decay_t<_Args>&...>
                                    && __nothrow_connectable<__sender2_t, __receiver2_t>;
          using __started_t = __facade::__effects<__facade::__starts<__key_t>>;

          // In a walk, only the effects matter (and the state holds no operations to make).
          if constexpr (_State::__walking())
          {
            if constexpr (__nothrow_v)
              return __started_t();
            else
              return __facade::__union_t<__started_t,
                                         __facade::__effects<set_error_t(std::exception_ptr)>>(
                __started_t());
          }
          else
          {
            // The function is called while the child's operation is still alive. Connecting
            // the sender it returns then destroys the child's operation.
            auto __start_next = [&]()
            {
              auto&          __tupl  = __state.__args_.__emplace_from(__mktuple,
                                                            static_cast<_Args&&>(__args)...);
              decltype(auto) __sndr2 = STDEXEC::__apply(static_cast<__fn_t&&>(__state.__fn_),
                                                        __tupl);
              auto&          __op    = __state.__ops_.__emplace_from(STDEXEC::connect,
                                                         static_cast<__sender2_t&&>(__sndr2),
                                                         __receiver2_t{&__state});
              return __facade::__start_nested<__key_t>(__op);
            };

            if constexpr (__nothrow_v)
            {
              return __start_next();
            }
            else
            {
              using __result_t =
                __facade::__union_t<decltype(__start_next()),
                                    __facade::__effects<set_error_t(std::exception_ptr)>>;
              STDEXEC_TRY
              {
                return __result_t(__start_next());
              }
              STDEXEC_CATCH_ALL
              {
                return __result_t(
                  __facade::__send(__state.__rcvr_, set_error, std::current_exception()));
              }
            }
          }
        }
      }

      // The child's other completions, and those of the sender that the function returned,
      // are the operation's.
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
  }  // namespace __let

  //! @brief A pipeable sender adaptor that chains a *sender-returning*
  //!        function onto a predecessor's value completion.
  //!
  //! @c let_value is the way to launch another asynchronous operation based on
  //! the values produced by a predecessor sender. Where @ref then_t "then"
  //! takes a function returning a *value*, @c let_value takes a function
  //! returning a *sender*, which is then connected and started, becoming the
  //! tail of the pipeline.
  //!
  //! Both call syntaxes are supported (the second is the *pipeable* form):
  //!
  //! @code{.cpp}
  //! auto s1 = stdexec::let_value(sndr, f);   // direct invocation
  //! auto s2 = sndr | stdexec::let_value(f);  // pipe syntax
  //! @endcode
  //!
  //! The signature of the operator overloads (inherited from a detail base) is:
  //!
  //! @code{.cpp}
  //! template <sender Sender, movable-value Fun>
  //!   auto operator()(Sender&& sndr, Fun fun) const -> sender auto; // direct
  //!
  //! template <class Fun>
  //!   auto operator()(Fun fun) const;                               // closure
  //! @endcode
  //!
  //! The two forms are expression-equivalent. See [exec.let] in the
  //! C++26 working draft for the normative specification.
  //!
  //! **Completion signatures.**
  //!
  //! Given a predecessor sender @c sndr with completion signatures
  //!
  //! @code{.cpp}
  //! set_value_t(Vs...)               // one or more value completions
  //! set_error_t(Es)...               // forwarded unchanged
  //! set_stopped_t()                  // forwarded unchanged (if present)
  //! @endcode
  //!
  //! the sender produced by `let_value(sndr, f)` has completion
  //! signatures equal to the *union* of:
  //!
  //! - the completion signatures of every sender returned by an invocation
  //!   `std::invoke(f, vs...)` for each value-completion argument pack
  //!   @c (vs...) of @c sndr, *plus*
  //! - the @c set_error_t completions of @c sndr (forwarded unchanged),
  //! - the @c set_stopped_t completion of @c sndr (forwarded unchanged), and
  //! - @c set_error_t(std::exception_ptr) if invoking @c f or connecting its
  //!   returned sender may throw.
  //!
  //! @c f must be invocable with every value-completion argument pack of
  //! @c sndr, and every such invocation must return a type satisfying the
  //! @c sender concept. Otherwise the program is ill-formed at the point
  //! where the resulting sender is connected to a receiver.
  //!
  //! **Use** @ref then_t "then" **vs.** @c let_value: choose @c then when the
  //! callable returns a *value*; choose @c let_value when the callable returns
  //! a *sender* (i.e. when the next step is itself asynchronous). Passing a
  //! sender-returning function to @c then would forward the sender as a
  //! value — almost never what you want.
  //!
  //! **Exception behavior.**
  //!
  //! If invoking @c f throws, or if connecting the sender returned by @c f
  //! throws, the exception is delivered through
  //! @c set_error_t(std::exception_ptr) on the resulting sender. When both
  //! steps are @c noexcept, no @c std::exception_ptr error completion is added.
  //!
  //! **Cancellation.**
  //!
  //! @c let_value does not introduce stop-token interaction of its own. If
  //! @c sndr completes with @c set_stopped, @c f is not invoked and the
  //! stopped completion is forwarded.
  //!
  //! **Example.**
  //!
  //! @code{.cpp}
  //! #include <stdexec/execution.hpp>
  //! #include <cassert>
  //!
  //! int main() {
  //!   using namespace stdexec;
  //!
  //!   auto fetch_async = [](int id) {
  //!     return just(id * 10);  // pretend this is a non-trivial async op
  //!   };
  //!
  //!   auto sndr = just(7)
  //!             | let_value(fetch_async);
  //!
  //!   auto [v] = sync_wait(std::move(sndr)).value();
  //!   assert(v == 70);
  //! }
  //! @endcode
  //!
  //! @see stdexec::then          — adapt the value channel with a
  //!                               value-returning function
  //! @see stdexec::let_error     — adapt the error channel with a
  //!                               sender-returning function
  //! @see stdexec::let_stopped   — adapt the stopped channel with a
  //!                               sender-returning function
  struct let_value_t : __let::__let_t<let_value_t>
  {
    using __t     = set_value_t;
    let_value_t() = default;
  };

  //! @brief A pipeable sender adaptor that chains a *sender-returning*
  //!        function onto a predecessor's error completion.
  //!
  //! @c let_error is to @ref upon_error_t "upon_error" what @ref let_value_t
  //! "let_value" is to @ref then_t "then": it lets the recovery step be
  //! *another asynchronous operation*. When the predecessor completes with
  //! @c set_error_t(e), @c f is invoked with @c e and is expected to return a
  //! sender; that sender is connected and started, and its completions become
  //! the completions of the overall pipeline.
  //!
  //! Both call syntaxes are supported (the second is the *pipeable* form):
  //!
  //! @code{.cpp}
  //! auto s1 = stdexec::let_error(sndr, f);   // direct invocation
  //! auto s2 = sndr | stdexec::let_error(f);  // pipe syntax
  //! @endcode
  //!
  //! See [exec.let] in the C++26 working draft for the normative specification.
  //!
  //! **Completion signatures.**
  //!
  //! Given a predecessor sender @c sndr with completion signatures
  //!
  //! @code{.cpp}
  //! set_value_t(Vs...)               // forwarded unchanged
  //! set_error_t(Es)...               // one or more error completions
  //! set_stopped_t()                  // forwarded unchanged (if present)
  //! @endcode
  //!
  //! the sender produced by `let_error(sndr, f)` has completion
  //! signatures equal to the *union* of:
  //!
  //! - the @c set_value_t completions of @c sndr (forwarded unchanged),
  //! - the completion signatures of every sender returned by an invocation
  //!   `std::invoke(f, e)` for each error type @c e of @c sndr,
  //! - the @c set_stopped_t completion of @c sndr (forwarded unchanged), and
  //! - @c set_error_t(std::exception_ptr) if invoking @c f or connecting its
  //!   returned sender may throw.
  //!
  //! All original @c set_error_t completions are *consumed*: only errors
  //! produced by the senders that @c f returns (or thrown by @c f itself)
  //! survive on the error channel.
  //!
  //! **Use** @ref upon_error_t "upon_error" **vs.** @c let_error: choose
  //! @c upon_error when the recovery is synchronous (returns a value); choose
  //! @c let_error when the recovery is itself asynchronous (returns a sender).
  //!
  //! **Exception behavior.**
  //!
  //! If invoking @c f throws, or if connecting the sender returned by @c f
  //! throws, the exception is delivered through
  //! @c set_error_t(std::exception_ptr) on the resulting sender.
  //!
  //! **Cancellation.**
  //!
  //! @c let_error does not introduce stop-token interaction of its own. A
  //! @c set_stopped completion of @c sndr is forwarded without invoking @c f.
  //!
  //! **Example.**
  //!
  //! @code{.cpp}
  //! #include <stdexec/execution.hpp>
  //! #include <cassert>
  //! #include <system_error>
  //!
  //! int main() {
  //!   using namespace stdexec;
  //!
  //!   auto retry_async = [](std::error_code) { return just(7); };
  //!
  //!   auto sndr = just_error(std::error_code{ENOENT, std::system_category()})
  //!             | let_error(retry_async);
  //!
  //!   auto [v] = sync_wait(std::move(sndr)).value();
  //!   assert(v == 7);
  //! }
  //! @endcode
  //!
  //! @see stdexec::upon_error   — adapt the error channel with a
  //!                              value-returning function
  //! @see stdexec::let_value    — adapt the value channel with a
  //!                              sender-returning function
  //! @see stdexec::let_stopped  — adapt the stopped channel with a
  //!                              sender-returning function
  struct let_error_t : __let::__let_t<let_error_t>
  {
    using __t     = set_error_t;
    let_error_t() = default;
  };

  //! @brief A pipeable sender adaptor that chains a *sender-returning*
  //!        nullary function onto a predecessor's stopped completion.
  //!
  //! @c let_stopped is to @ref upon_stopped_t "upon_stopped" what
  //! @ref let_value_t "let_value" is to @ref then_t "then": it lets the
  //! recovery from cancellation be *another asynchronous operation*. When the
  //! predecessor completes with @c set_stopped, @c f is invoked with no
  //! arguments and is expected to return a sender; that sender is connected
  //! and started, and its completions become the completions of the overall
  //! pipeline.
  //!
  //! Both call syntaxes are supported (the second is the *pipeable* form):
  //!
  //! @code{.cpp}
  //! auto s1 = stdexec::let_stopped(sndr, f);   // direct invocation
  //! auto s2 = sndr | stdexec::let_stopped(f);  // pipe syntax
  //! @endcode
  //!
  //! See [exec.let] in the C++26 working draft for the normative specification.
  //!
  //! **Completion signatures.**
  //!
  //! Given a predecessor sender @c sndr with completion signatures
  //!
  //! @code{.cpp}
  //! set_value_t(Vs...)               // forwarded unchanged
  //! set_error_t(Es)...               // forwarded unchanged
  //! set_stopped_t()                  // (must be present)
  //! @endcode
  //!
  //! the sender produced by `let_stopped(sndr, f)` has completion
  //! signatures equal to the *union* of:
  //!
  //! - the @c set_value_t completions of @c sndr (forwarded unchanged),
  //! - the @c set_error_t completions of @c sndr (forwarded unchanged),
  //! - the completion signatures of the sender returned by
  //!   `std::invoke(f)`, and
  //! - @c set_error_t(std::exception_ptr) if invoking @c f or connecting its
  //!   returned sender may throw.
  //!
  //! The original @c set_stopped_t completion is *consumed*: it appears on
  //! the resulting sender only if the sender returned by @c f itself
  //! completes via @c set_stopped.
  //!
  //! **Use** @ref upon_stopped_t "upon_stopped" **vs.** @c let_stopped: choose
  //! @c upon_stopped when the recovery is synchronous (returns a value);
  //! choose @c let_stopped when the recovery is itself asynchronous
  //! (returns a sender).
  //!
  //! **Exception behavior.**
  //!
  //! If invoking @c f throws, or if connecting the sender returned by @c f
  //! throws, the exception is delivered through
  //! @c set_error_t(std::exception_ptr) on the resulting sender.
  //!
  //! **Cancellation.**
  //!
  //! @c let_stopped reacts to the predecessor's @c set_stopped; it does not
  //! initiate cancellation. The sender returned by @c f sees the receiver's
  //! stop token as normal.
  //!
  //! **Example.**
  //!
  //! @code{.cpp}
  //! #include <stdexec/execution.hpp>
  //! #include <cassert>
  //!
  //! int main() {
  //!   using namespace stdexec;
  //!
  //!   auto fallback_async = [] { return just(42); };
  //!
  //!   auto sndr = just_stopped()
  //!             | let_stopped(fallback_async);
  //!
  //!   auto [v] = sync_wait(std::move(sndr)).value();
  //!   assert(v == 42);
  //! }
  //! @endcode
  //!
  //! @see stdexec::upon_stopped — adapt the stopped channel with a
  //!                              value-returning function
  //! @see stdexec::let_value    — adapt the value channel with a
  //!                              sender-returning function
  //! @see stdexec::let_error    — adapt the error channel with a
  //!                              sender-returning function
  struct let_stopped_t : __let::__let_t<let_stopped_t>
  {
    using __t       = set_stopped_t;
    let_stopped_t() = default;
  };

  //! @brief The customization point object for the @c let_value sender adaptor.
  //!
  //! @c let_value is an instance of @c let_value_t. See @ref let_value_t
  //! for the full description, completion-signature transformation rules,
  //! exception and cancellation behavior, and a usage example.
  //!
  //! @hideinitializer
  inline constexpr let_value_t let_value{};

  //! @brief The customization point object for the @c let_error sender adaptor.
  //!
  //! @c let_error is an instance of @c let_error_t. See @ref let_error_t
  //! for the full description and example.
  //!
  //! @hideinitializer
  inline constexpr let_error_t let_error{};

  //! @brief The customization point object for the @c let_stopped sender
  //! adaptor.
  //!
  //! @c let_stopped is an instance of @c let_stopped_t. See
  //! @ref let_stopped_t for the full description and example.
  //!
  //! @hideinitializer
  inline constexpr let_stopped_t let_stopped{};

  template <>
  struct __facade::__algorithm<let_value_t> : __let::__algo<let_value_t, set_value_t>
  {};

  template <>
  struct __facade::__algorithm<let_error_t> : __let::__algo<let_error_t, set_error_t>
  {};

  template <>
  struct __facade::__algorithm<let_stopped_t> : __let::__algo<let_stopped_t, set_stopped_t>
  {};
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
