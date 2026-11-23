/*
 * Copyright (c) 2026 NVIDIA Corporation
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

#include "../../stdexec/__detail/__optional.hpp"
#include "../../stdexec/__detail/__sender_facade.hpp"
#include "../../stdexec/execution.hpp"

#include "../trampoline_scheduler.hpp"

#include <exception>
#include <type_traits>

// The repeating algorithms (repeat_n, repeat_until, and repeat) are facade algorithms (see
// __sender_facade.hpp) that share what is here. Each iteration bounces on a trampoline
// scheduler, so that iterations that complete synchronously do not overflow the stack,
// and then starts the body: the operation of the child sender, connected anew for each
// iteration.
namespace experimental::execution::__repeat
{
  using namespace STDEXEC;

  struct _THE_INPUT_SENDER_MUST_BE_COPYABLE_;
  struct _THE_SENDER_CANNOT_BE_CONNECTED_;

  // The key of the body. Each iteration connects a copy of the child.
  struct __body
  {};

  // The key of the schedule operation of the trampoline scheduler that starts each
  // iteration
  struct __bounce
  {};

  using __bounce_sender_t = schedule_result_t<trampoline_scheduler>;

  template <class _Sender>
  using __child_t = __decay_t<__facade::__nth_child_of_t<_Sender, 0>>;

  // An iteration starts where the one before it ended: where the body completed with a
  // value. (The trampoline completes where it starts.) So if the body completes with values
  // where it starts, or never does, every iteration starts where the operation started,
  // and the body and the trampoline see that place in the receiver's environment.
  // Otherwise the environment does not say where they start, because the receiver's
  // environment says where only the first iteration starts.
  template <class _Child, class _Env>
  concept __starts_in_place =
    __never_sends<set_value_t, _Child, __fwd_env_t<_Env>>
    || __completes_where_it_starts<set_value_t, env_of_t<_Child>, __fwd_env_t<_Env>>;

  template <class _Child, class _Env>
  STDEXEC_ATTRIBUTE(always_inline)
  constexpr auto __loop_env(_Env &&__env) noexcept
  {
    if constexpr (__starts_in_place<_Child, _Env>)
      return __fwd_env(static_cast<_Env &&>(__env));
    else
      return __hide_query(__fwd_env(static_cast<_Env &&>(__env)), get_start_scheduler, get_domain);
  }

  template <class _Algo, class _Sender, class _Receiver>
  struct __state;

  template <class _Algo, class _Sender, class _Receiver>
  using __body_receiver_t =
    __facade::__receiver_t<_Algo, __body, __state<_Algo, _Sender, _Receiver>, _Sender, _Receiver>;

  template <class _Algo, class _Sender, class _Receiver>
  concept __connectable =
    __std::copy_constructible<__child_t<_Sender>>
    && sender_to<__child_t<_Sender>, __body_receiver_t<_Algo, _Sender, _Receiver>>;

  // The state of a repeating operation. The algorithm's data says how many iterations are
  // left (repeat_n), or nothing.
  template <class _Algo, class _Sender, class _Receiver>
  struct __state
  {
    using __data_t        = __decay_t<__facade::__data_of_t<_Sender>>;
    using __child_t       = __repeat::__child_t<_Sender>;
    using __body_rcvr_t   = __body_receiver_t<_Algo, _Sender, _Receiver>;
    using __bounce_rcvr_t = __facade::__receiver_t<_Algo, __bounce, __state, _Sender, _Receiver>;
    using __body_op_t     = connect_result_t<__child_t, __body_rcvr_t>;
    using __bounce_op_t   = connect_result_t<__bounce_sender_t, __bounce_rcvr_t>;

    static constexpr bool __nothrow_connect = __nothrow_constructible_from<__child_t, __child_t &>
                                           && __nothrow_connectable<__child_t, __body_rcvr_t>;
    static constexpr bool __nothrow_v =
      __nothrow_decay_copyable<__facade::__data_of_t<_Sender>>
      && __nothrow_decay_copyable<__facade::__nth_child_of_t<_Sender, 0>> && __nothrow_connect;

    // The body of the first iteration, if there is one, is connected with the operation,
    // so that if connecting it throws, connecting the operation does.
    constexpr explicit __state(_Sender &&__sndr, _Receiver &&__rcvr) noexcept(__nothrow_v)
      : __rcvr_(static_cast<_Receiver &&>(__rcvr))
      , __data_(__facade::__get_data(static_cast<_Sender &&>(__sndr)))
      , __child_(__facade::__get_child<0>(static_cast<_Sender &&>(__sndr)))
    {
      if (_Algo::__iterates(__data_))
      {
        __connect_body();
      }
    }

    STDEXEC_IMMOVABLE(__state);

    // Connects the body of the next iteration, destroying the operation of the last one.
    // (This happens in the last one's completion. Operations that complete must not touch
    // themselves afterward.)
    constexpr auto __connect_body() noexcept(__nothrow_connect) -> __body_op_t &
    {
      return __body_op_.__emplace_from(STDEXEC::connect, __child_t(__child_), __body_rcvr_t{this});
    }

    // Destroys the operations, before the operation completes
    constexpr void __cleanup() noexcept
    {
      __body_op_.reset();
      __bounce_op_.reset();
    }

    _Receiver                 __rcvr_;
    __data_t                  __data_;
    __child_t                 __child_;
    __optional<__bounce_op_t> __bounce_op_{};
    __optional<__body_op_t>   __body_op_{};
  };

  template <class _Algo, class _Sender, class _Receiver>
  consteval auto __nothrow_state() noexcept -> bool
  {
    if constexpr (__connectable<_Algo, _Sender, _Receiver>)
      return __state<_Algo, _Sender, _Receiver>::__nothrow_v;
    else
      return true;
  }

  // The base of the repeating algorithms. _Derived is the algorithm, which says what the
  // body's value completions mean (with a __complete(__body, ...) handler), and how the
  // operation starts (with __start, which calls __start_iteration). Its static member
  // __iterates(data) says whether the operation has a first iteration, and
  // __decays_errors whether the body's errors are sent as values or with their value
  // categories.
  template <class _Derived>
  struct __algo_base : __facade::__defaults
  {
    // Each iteration connects the body (see __state).
    static constexpr bool __connects_children = false;

    // Customizations of the algorithm are found in the domain where the body of the first
    // iteration completes with a value.
    using __dispatch_source =
      __facade::__via<__body,
                      set_value_t,
                      __facade::__via<__bounce, set_value_t, __facade::__from_start>>;

    // The body and the trampoline (and the child, which the walk checks in the body's
    // environment) see the environment of the loop (see __loop_env).
    template <class _Key, class _Sender, class _Env>
    STDEXEC_ATTRIBUTE(always_inline)
    static constexpr auto __env_for(_Key, _Sender const &, _Env &&__env) noexcept
    {
      return __repeat::__loop_env<__child_t<_Sender>>(static_cast<_Env &&>(__env));
    }

    template <class _Key, class _State>
    STDEXEC_ATTRIBUTE(always_inline)
    static constexpr auto __get_env(_Key, _State const &__state) noexcept
    {
      return __repeat::__loop_env<typename _State::__child_t>(STDEXEC::get_env(__state.__rcvr_));
    }

    using __facade::__defaults::__nested_sender;

    // The body is a copy of the child, as an rvalue. (Its attributes are the child's.)
    template <class _Sender, class... _Env>
    STDEXEC_ATTRIBUTE(always_inline)
    static constexpr auto __nested_sender(__body, _Sender const &__sndr, _Env const &...) noexcept
      -> __child_t<_Sender> &&
    {
      using __child_t = __repeat::__child_t<_Sender>;
      return static_cast<__child_t &&>(const_cast<__child_t &>(__facade::__get_child<0>(__sndr)));
    }

    template <class _Sender, class... _Env>
    STDEXEC_ATTRIBUTE(always_inline)
    static constexpr auto __nested_sender(__bounce, _Sender const &, _Env const &...) noexcept
      -> __bounce_sender_t
    {
      return STDEXEC::schedule(trampoline_scheduler{});
    }

    template <class _Sender, class _Receiver>
    static constexpr auto __get_state(_Sender &&__sndr, _Receiver __rcvr)
      noexcept(__repeat::__nothrow_state<_Derived, _Sender, _Receiver>())
    {
      using __tag_t = __decay_t<_Sender>::__tag_t;
      if constexpr (!__std::copy_constructible<__child_t<_Sender>>)
      {
        return __mexception<_WHAT_(_INVALID_ARGUMENT_),
                            _WHERE_(_IN_ALGORITHM_, __tag_t),
                            _WHY_(_THE_INPUT_SENDER_MUST_BE_COPYABLE_),
                            _WITH_SENDER_<__child_t<_Sender>>>();
      }
      else if constexpr (!__connectable<_Derived, _Sender, _Receiver>)
      {
        return __mexception<_WHAT_(_INVALID_ARGUMENT_),
                            _WHERE_(_IN_ALGORITHM_, __tag_t),
                            _WHY_(_THE_SENDER_CANNOT_BE_CONNECTED_),
                            _WITH_SENDER_<__child_t<_Sender>>>();
      }
      else
      {
        return __state<_Derived, _Sender, _Receiver>(static_cast<_Sender &&>(__sndr),
                                                     static_cast<_Receiver &&>(__rcvr));
      }
    }

    // Starts an iteration: the trampoline, which starts the body
    template <class _State>
    static constexpr auto __start_iteration(_State &__state) noexcept
    {
      using __bounce_rcvr_t = _State::__bounce_rcvr_t;
      auto &__op            = __state.__bounce_op_.__emplace_from(STDEXEC::connect,
                                                       STDEXEC::schedule(trampoline_scheduler{}),
                                                       __bounce_rcvr_t{&__state});
      return __facade::__start_nested<__bounce>(__op);
    }

    // Starts the next iteration, whose body it connects. If that throws, the operation
    // completes with the exception.
    template <class _State>
    static constexpr auto __next(_State &__state) noexcept
    {
      if constexpr (_State::__nothrow_connect)
      {
        __state.__connect_body();
        return __algo_base::__start_iteration(__state);
      }
      else
      {
        using __result_t =
          __facade::__union_t<decltype(__algo_base::__start_iteration(__state)),
                              __facade::__effects<set_error_t(std::exception_ptr)>>;
        STDEXEC_TRY
        {
          __state.__connect_body();
          return __result_t(__algo_base::__start_iteration(__state));
        }
        STDEXEC_CATCH_ALL
        {
          __state.__cleanup();
          return __result_t(
            __facade::__send(__state.__rcvr_, set_error_t(), std::current_exception()));
        }
      }
    }

    // The loop is done.
    template <class _State>
    static constexpr auto __finish(_State &__state) noexcept
    {
      __state.__cleanup();
      return __facade::__send(__state.__rcvr_, set_value_t());
    }

    // The trampoline starts the body.
    template <class _State>
    static constexpr auto __complete(__bounce, _State &__state, set_value_t) noexcept
    {
      return __facade::__start_nested<__body>(*__state.__body_op_);
    }

    // The trampoline stops if the receiver asks it to.
    template <class _State>
    static constexpr auto __complete(__bounce, _State &__state, set_stopped_t) noexcept
    {
      __state.__cleanup();
      return __facade::__send(__state.__rcvr_, set_stopped_t());
    }

    // The body's errors are the operation's. An error is copied before the body's operation
    // is destroyed, because it may be in it.
    template <class _State, class _Error>
    static constexpr auto __complete(__body, _State &__state, set_error_t, _Error &&__err) noexcept
    {
      using __copy_t   = __decay_t<_Error>;
      using __sent_t   = std::conditional_t<_Derived::__decays_errors, __copy_t &&, _Error &&>;
      auto __send_copy = [&]()
      {
        __copy_t __copy(static_cast<_Error &&>(__err));
        __state.__cleanup();
        return __facade::__send(__state.__rcvr_, set_error_t(), static_cast<__sent_t>(__copy));
      };

      if constexpr (__nothrow_decay_copyable<_Error>)
      {
        return __send_copy();
      }
      else
      {
        using __result_t =
          __facade::__union_t<decltype(__send_copy()),
                              __facade::__effects<set_error_t(std::exception_ptr)>>;
        STDEXEC_TRY
        {
          return __result_t(__send_copy());
        }
        STDEXEC_CATCH_ALL
        {
          __state.__cleanup();
          return __result_t(
            __facade::__send(__state.__rcvr_, set_error_t(), std::current_exception()));
        }
      }
    }

    template <class _State>
    static constexpr auto __complete(__body, _State &__state, set_stopped_t) noexcept
    {
      __state.__cleanup();
      return __facade::__send(__state.__rcvr_, set_stopped_t());
    }
  };
}  // namespace experimental::execution::__repeat
