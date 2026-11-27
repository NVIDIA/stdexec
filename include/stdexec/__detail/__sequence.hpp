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

#include "__config.hpp"

#if STDEXEC_USE_MODULES() && !defined(STDEXEC_IN_MODULE_PURVIEW)

import stdexec;

#else

#  include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#  include "__schedulers.hpp"
#  include "__sender_facade.hpp"
#  include "__sender_introspection.hpp"
#  include "__senders.hpp"
#  include "__variant.hpp"

#  include "__prologue.hpp"

STDEXEC_PRAGMA_IGNORE_EDG(expr_has_no_effect)
STDEXEC_PRAGMA_IGNORE_EDG(type_qualifiers_ignored_on_reference)
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")
STDEXEC_PRAGMA_IGNORE_GNU("-Wunused-value")

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  STDEXEC_MODULE_EXPORT_AUTHORING
  struct __sequence_t;
  struct _ALL_SENDERS_BUT_THE_LAST_MUST_BE_SENDERS_OF_VOID_;

  namespace __seq
  {
    using __mk_env2_t = __mk_secondary_env_t<set_value_t>;

    // The first sender is connected from the sequence, with the sequence's value category.
    // The later ones are stored decayed, and connected as rvalues.
    template <std::size_t _Idx, class _Sender>
    struct __sender_of
    {
      using __t = __decay_t<__facade::__nth_child_of_t<_Sender, _Idx>>;
    };

    template <class _Sender>
    struct __sender_of<0, _Sender>
    {
      using __t = __facade::__nth_child_of_t<_Sender, 0>;
    };

    template <std::size_t _Idx, class _Sender>
    using __sender_of_t = __sender_of<_Idx, _Sender>::__t;

    // Whether connecting the later senders can throw depends on the receiver, so without an
    // environment the completions of a sequence of two or more senders are not known,
    // unless one of the senders can already complete with an exception_ptr.
    template <class _Sexpr, class _Indices>
    inline constexpr bool __may_throw_unknown = false;

    template <class _Sexpr, std::size_t... _Is>
    inline constexpr bool __may_throw_unknown<_Sexpr, __indices<_Is...>> =
      sizeof...(_Is) >= 2 && (sender_in<__facade::__nth_child_of_t<_Sexpr, _Is>> && ...)
      && !(__has_eptr_completion<__facade::__nth_child_of_t<_Sexpr, _Is>> || ...);

    template <std::size_t _Idx, class _Sender, class _Env>
    consteval auto __starts_at_start() noexcept -> bool;

    // The secondary environment of the _Idx-th sender (_Idx >= 1): where the sender before it
    // completed, in that sender's environment. The receiver's environment is passed by
    // reference, so that this has the same type at run time as in the walk.
    template <std::size_t _Idx, class _Sender, class _Env>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto __env2_for(_Sender const &__sndr, _Env const &__env) noexcept;

    // The environment of the _Idx-th sender. The first sees the receiver's, forwarded. Each
    // one after it sees where the one before it completed, and the receiver's environment,
    // whose start scheduler and domain are hidden unless the sender starts where the sequence
    // started (see __starts_at_start). (Otherwise, after a sender that does not say where it
    // completes, a sender that completes where it starts would claim to complete where the
    // sequence started.)
    template <std::size_t _Idx, class _Sender, class _Env>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto __env_of(_Sender const &__sndr, _Env &&__env) noexcept
    {
      if constexpr (_Idx == 0)
        return __fwd_env(static_cast<_Env &&>(__env));
      else if constexpr (__seq::__starts_at_start<_Idx, _Sender, __decay_t<_Env>>())
        return __env::__join(__seq::__env2_for<_Idx>(__sndr, __env), static_cast<_Env &&>(__env));
      else
        return __env::__join(__seq::__env2_for<_Idx>(__sndr, __env),
                             __hide_scheduler{static_cast<_Env &&>(__env)});
    }

    template <std::size_t _Idx, class _Sender, class _Env>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto __env2_for(_Sender const &__sndr, _Env const &__env) noexcept
    {
      static_assert(_Idx != 0);
      // A sender that does not type-check in its environment cannot be connected, but the
      // facade still computes its attributes.
      using __env_t = decltype(__seq::__env_of<_Idx - 1>(__sndr, __env));
      if constexpr (__callable<__mk_env2_t,
                               __facade::__nth_child_of_t<_Sender, _Idx - 1> const &,
                               __env_t const &>)
        return __mk_env2_t()(__facade::__get_child<_Idx - 1>(__sndr),
                             __seq::__env_of<_Idx - 1>(__sndr, __env));
      else
        return env{};
    }

    // Does the _Idx-th sender start where the sequence started? It does if the senders
    // before it all complete with a value where they start.
    template <std::size_t _Idx, class _Sender, class _Env>
    consteval auto __starts_at_start() noexcept -> bool
    {
      if constexpr (_Idx == 0)
      {
        return true;
      }
      else
      {
        using __env_t = decltype(__seq::__env_of<_Idx - 1>(__declval<_Sender const &>(),
                                                           __declval<_Env const &>()));
        return __seq::__starts_at_start<_Idx - 1, _Sender, _Env>()
            && __stays_put_for<set_value_t, __facade::__nth_child_of_t<_Sender, _Idx - 1>, __env_t>;
      }
    }

    struct __algo;

    // The state of a sequence. If the sequence is subscribed (_Subscribing is
    // __facade::__subscribing_t), so is its item source, the last sender, when that is a
    // sequence sender (see __facade::__attaches_child).
    template <class _Sender, class _Receiver, class _Indices, class... _Subscribing>
    struct __seq_state;

    template <class _Sender, class _Receiver, std::size_t... _Is, class... _Subscribing>
    struct __seq_state<_Sender, _Receiver, __indices<_Is...>, _Subscribing...>
    {
      static constexpr std::size_t __nbr = sizeof...(_Is);
      using __env_t                      = env_of_t<_Receiver>;

      // The later senders are connected and started when the one before them completes.
      template <std::size_t _Idx>
      using __sender_t = __sender_of_t<_Idx, _Sender>;

      template <std::size_t _Idx>
      using __receiver_t =
        __facade::__receiver_t<__algo, __facade::__child<_Idx>, __seq_state, _Sender, _Receiver>;

      // Is the _Idx-th sender the item source, attached with __attach_t? (See
      // __facade::__attaches_child.)
      template <std::size_t _Idx>
      static consteval auto __attaches() noexcept -> bool
      {
        return __facade::__attaches_child<__algo, _Sender, _Idx>;
      }

      template <std::size_t _Idx>
      using __attach_t =
        __facade::__attach_child_t<__algo, _Sender, _Idx, __receiver_t<_Idx>, _Subscribing...>;

      // The operation of the _Idx-th sender
      template <std::size_t _Idx>
      using __op_t = __facade::__child_op_t<__algo,
                                            _Sender,
                                            _Idx,
                                            __seq_state,
                                            __sender_t<_Idx>,
                                            __receiver_t<_Idx>,
                                            _Subscribing...>;

      // Can making the operation of the _Idx-th sender throw?
      template <std::size_t _Idx>
      static consteval auto __nothrow_attach() noexcept -> bool
      {
        return __facade::__nothrow_attachable_child<__algo,
                                                    _Sender,
                                                    _Idx,
                                                    __seq_state,
                                                    __sender_t<_Idx>,
                                                    __receiver_t<_Idx>,
                                                    _Subscribing...>;
      }

      template <std::size_t _Idx>
      using __env2_t = decltype(__seq::__env2_for<_Idx>(__declval<__decay_t<_Sender> const &>(),
                                                        __declval<__env_t const &>()));

      // Does the _Idx-th sender start where the sequence started? (See __env_of.)
      template <std::size_t _Idx>
      static constexpr bool __at_start_v =
        __seq::__starts_at_start<_Idx, __decay_t<_Sender>, __decay_t<__env_t>>();

      template <class _Indices>
      struct __later;

      template <std::size_t... _Js>
      struct __later<__indices<_Js...>>
      {
        using __senders_t = __tuple<__decay_t<__facade::__nth_child_of_t<_Sender, _Js + 1>>...>;
        using __env2s_t   = __tuple<__env2_t<_Js + 1>...>;

        static constexpr auto
        __mk_env2s(__decay_t<_Sender> const &__sndr, __env_t const &__env) noexcept -> __env2s_t
        {
          return __env2s_t{__seq::__env2_for<_Js + 1>(__sndr, __env)...};
        }

        static constexpr bool __nothrow_take =
          (__nothrow_decay_copyable<__facade::__nth_child_of_t<_Sender, _Js + 1>> && ...);

        static constexpr auto
        __take_senders(_Sender &&__sndr) noexcept(__nothrow_take) -> __senders_t
        {
          return __senders_t{__facade::__get_child<_Js + 1>(static_cast<_Sender &&>(__sndr))...};
        }
      };

      using __later_t = __later<__make_indices<(__nbr == 0 ? 0 : __nbr - 1)>>;

      static consteval auto __nothrow_connect_first() noexcept -> bool
      {
        if constexpr (__nbr == 0)
          return true;
        else
          return __nothrow_attach<0>();
      }

      static constexpr bool __nothrow_v = __later_t::__nothrow_take && __nothrow_connect_first();

      constexpr explicit __seq_state(_Sender &&__sndr, _Receiver &&__rcvr) noexcept(__nothrow_v)
        : __rcvr_(static_cast<_Receiver &&>(__rcvr))
        , __env2s_(__later_t::__mk_env2s(__sndr, STDEXEC::get_env(__rcvr_)))
        , __sndrs_(__later_t::__take_senders(static_cast<_Sender &&>(__sndr)))
      {
        if constexpr (__nbr != 0 && !__walking())
        {
          if constexpr (__attaches<0>())
          {
            __ops_.__emplace_from(__attach_t<0>(),
                                  *this,
                                  __facade::__get_child<0>(static_cast<_Sender &&>(__sndr)),
                                  __receiver_t<0>{this});
          }
          else
          {
            __ops_.__emplace_from(STDEXEC::connect,
                                  __facade::__get_child<0>(static_cast<_Sender &&>(__sndr)),
                                  __receiver_t<0>{this});
          }
        }
      }

      STDEXEC_IMMOVABLE(__seq_state);

      // (A walk needs only the state's type, so it does not name the senders' operations:
      // they would be connected to archetypes.)
      struct __ops_lazy
      {
        using __t = __variant<__op_t<_Is>...>;
      };
      static consteval auto __walking() noexcept -> bool
      {
        return __facade::__walking_receiver<_Receiver>;
      }

      using __ops_t =
        __facade::__lazy_state_ops_t<_Receiver, __ops_lazy, __variant<__facade::__walk_op>>;

      _Receiver                       __rcvr_;
      typename __later_t::__env2s_t   __env2s_;
      typename __later_t::__senders_t __sndrs_;
      __ops_t                         __ops_{__no_init};
    };

    // __sequence(sndrs...): starts the senders one after the other. Each but the last must
    // complete with set_value_t(), which starts the next. Other completions of any of them,
    // and every completion of the last, are the operation's. If the last is a sequence
    // sender, so is the sequence. Its items are the last sender's, which it subscribes when
    // the sequence is subscribed.
    struct __algo : __facade::__defaults
    {
      static constexpr bool __connects_children = false;

      template <class _Sender>
      using __item_source = __if_c<(__facade::__nbr_children_v<_Sender> != 0),
                                   __facade::__child<__facade::__nbr_children_v<_Sender> - 1>,
                                   void>;

      template <class _Sender>
      static constexpr bool __depends_on_env =
        __may_throw_unknown<_Sender, __make_indices<__nbr_children_of<_Sender>>>;

      // Each sender after the first sees where the one before it completed (see __env_of).
      template <std::size_t _Idx, class _Sender, class _Env>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __env_for(__facade::__child<_Idx>,
                                      _Sender const &__sndr,
                                      _Env         &&__env) noexcept
      {
        return __seq::__env_of<_Idx>(__sndr, static_cast<_Env &&>(__env));
      }

      template <std::size_t _Idx, class _State>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __get_env(__facade::__child<_Idx>, _State const &__state) noexcept
      {
        if constexpr (_Idx == 0)
        {
          return __fwd_env(STDEXEC::get_env(__state.__rcvr_));
        }
        else
        {
          using __env2_t = __decay_t<decltype(STDEXEC::__get<_Idx - 1>(__state.__env2s_))>;
          if constexpr (_State::template __at_start_v<_Idx>)
            return __env::__join(__env2_t(STDEXEC::__get<_Idx - 1>(__state.__env2s_)),
                                 STDEXEC::get_env(__state.__rcvr_));
          else
            return __env::__join(__env2_t(STDEXEC::__get<_Idx - 1>(__state.__env2s_)),
                                 __hide_scheduler{STDEXEC::get_env(__state.__rcvr_)});
        }
      }

      // Forwarding queries are answered by the last sender.
      template <class _Sender>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __fwd_sender(_Sender const &__sndr) noexcept -> decltype(auto)
      {
        if constexpr (__facade::__nbr_children_v<_Sender> != 0)
          return __facade::__get_child<__facade::__nbr_children_v<_Sender> - 1>(__sndr);
        else
          return __facade::__no_attrs();
      }

      template <class _Sender, class _Receiver, class... _Subscribing>
      using __state_t =
        __seq_state<_Sender, _Receiver, __make_indices<__nbr_children_of<_Sender>>, _Subscribing...>;

      // (See __facade::__state_type_t.)
      template <class _Sender, class _Receiver, class... _Subscribing>
      using __state_type = __state_t<_Sender, _Receiver, _Subscribing...>;

      template <class _Sender, class _Receiver>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __get_state(_Sender &&__sndr, _Receiver __rcvr)
        noexcept(__state_t<_Sender, _Receiver>::__nothrow_v) -> __state_t<_Sender, _Receiver>
      {
        return __state_t<_Sender, _Receiver>(static_cast<_Sender &&>(__sndr),
                                             static_cast<_Receiver &&>(__rcvr));
      }

      // The state of a subscribed sequence, which subscribes its last sender
      template <class _Sender, class _Receiver>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __get_state(_Sender &&__sndr,
                                        _Receiver __rcvr,
                                        __facade::__subscribing_t)
        noexcept(__state_t<_Sender, _Receiver, __facade::__subscribing_t>::__nothrow_v)
          -> __state_t<_Sender, _Receiver, __facade::__subscribing_t>
      {
        return __state_t<_Sender, _Receiver, __facade::__subscribing_t>(
          static_cast<_Sender &&>(__sndr),
          static_cast<_Receiver &&>(__rcvr));
      }

      template <class _State, class _Children>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __start(_State &__state, _Children &) noexcept
      {
        if constexpr (_State::__nbr == 0)
          return __facade::__send(__state.__rcvr_, set_value);
        else
          return __facade::__start_nested<__facade::__child<0>>(__var::__get<0>(__state.__ops_));
      }

      // A sender completed with a value: the next one starts, unless this was the last.
      template <std::size_t _Idx, class _State, class... _Args>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __complete(__facade::__child<_Idx>,
                                       _State &__state,
                                       set_value_t,
                                       _Args &&...__args) noexcept
      {
        if constexpr (_Idx + 1 == _State::__nbr)
        {
          return __facade::__send(__state.__rcvr_, set_value, static_cast<_Args &&>(__args)...);
        }
        else if constexpr (sizeof...(_Args) != 0)
        {
          return __facade::__type_error(
            __mexception<_WHAT_(_INVALID_ARGUMENT_),
                         _WHERE_(_IN_ALGORITHM_, __sequence_t),
                         _WHY_(_ALL_SENDERS_BUT_THE_LAST_MUST_BE_SENDERS_OF_VOID_),
                         _WITH_SENDER_<typename _State::template __sender_t<_Idx>>>());
        }
        else
        {
          return __algo::__start_next<_Idx + 1>(__state);
        }
      }

      template <std::size_t _Idx, class _State, class _Tag, class... _Args>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __complete(__facade::__child<_Idx>,
                                       _State &__state,
                                       _Tag    __tag,
                                       _Args &&...__args) noexcept
      {
        return __facade::__send(__state.__rcvr_, __tag, static_cast<_Args &&>(__args)...);
      }

     private:
      template <std::size_t _Idx, class _State>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __start_next(_State &__state) noexcept
      {
        using __sender_t   = _State::template __sender_t<_Idx>;
        using __receiver_t = _State::template __receiver_t<_Idx>;
        using __started_t  = __facade::__effects<__facade::__starts<__facade::__child<_Idx>>>;

        // In a walk, only the effects matter (and the state holds no operations to make).
        if constexpr (_State::__walking())
        {
          if constexpr (_State::template __nothrow_attach<_Idx>())
            return __started_t();
          else
            return __facade::__union_t<__started_t,
                                       __facade::__effects<set_error_t(std::exception_ptr)>>(
              __started_t());
        }
        else
        {
          // Connecting (or attaching) the next sender destroys the operation of the one that
          // just completed.
          auto __start = [&]()
          {
            if constexpr (_State::template __attaches<_Idx>())
            {
              using __attach_t = _State::template __attach_t<_Idx>;
              auto &__op       = __state.__ops_.__emplace_from(
                __attach_t(),
                __state,
                static_cast<__sender_t &&>(STDEXEC::__get<_Idx - 1>(__state.__sndrs_)),
                __receiver_t{&__state});
              return __facade::__start_nested<__facade::__child<_Idx>>(__op);
            }
            else
            {
              auto &__op = __state.__ops_.__emplace_from(
                STDEXEC::connect,
                static_cast<__sender_t &&>(STDEXEC::__get<_Idx - 1>(__state.__sndrs_)),
                __receiver_t{&__state});
              return __facade::__start_nested<__facade::__child<_Idx>>(__op);
            }
          };

          if constexpr (_State::template __nothrow_attach<_Idx>())
          {
            return __start();
          }
          else
          {
            using __result_t =
              __facade::__union_t<decltype(__start()),
                                  __facade::__effects<set_error_t(std::exception_ptr)>>;
            STDEXEC_TRY
            {
              return __result_t(__start());
            }
            STDEXEC_CATCH_ALL
            {
              return __result_t(
                __facade::__send(__state.__rcvr_, set_error, std::current_exception()));
            }
          }
        }
      }
    };
  }  // namespace __seq

  // A sequence's state is incomplete while the sequence subscribes its last sender, since that
  // operation is part of the state. So the type of its receiver, to which the receivers of its
  // senders forward items, is read from the state's type.
  template <class _Item, class _Sender, class _Receiver, class _Indices, class... _Subscribing>
  struct __facade::__state_receiver<
    __seq::__seq_state<_Sender, _Receiver, _Indices, _Subscribing...>,
    _Item>
  {
    using __t = _Receiver;
  };

  struct __sequence_t
  {
    template <class... _Senders>
    [[nodiscard]]
    constexpr auto operator()(_Senders &&...__sndrs) const  //
      noexcept(__nothrow_decay_copyable<_Senders...>) -> auto
    {
      return STDEXEC::__check_well_formed(
        __facade::__sender{__sequence_t(), __(), static_cast<_Senders &&>(__sndrs)...});
    }
  };

  inline constexpr __sequence_t __sequence{};

  template <>
  struct __facade::__algorithm<__sequence_t> : __seq::__algo
  {};
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
