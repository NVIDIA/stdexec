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
#  include "__connect.hpp"
#  include "__just.hpp"
#  include "__schedulers.hpp"
#  include "__sender_facade.hpp"
#  include "__sender_introspection.hpp"
#  include "__senders.hpp"
#  include "__subscribe.hpp"
#  include "__transform_completion_signatures.hpp"
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

    template <class _CvSender1, class _Env>
    using __env2_t = __secondary_env_t<_CvSender1, _Env, set_value_t>;

    template <class... _Senders>
    struct __sndr;

    ////////////////////////////////////////////////////////////////////////////
    // Attributes for __sequence. This is a bit more complicated than the
    // attributes for other algorithms because we need to be able to query the
    // completion scheduler (for example) of the second sender from the context
    // of the first sender's completions.
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class... _CvSenders>
    struct __attrs;

    template <>
    struct __attrs<> : __just::__attrs<set_value_t>
    {};

    template <class _Sender>
    struct __attrs<_Sender>
    {
      using _env_t = env_of_t<_Sender>;

      template <class _Query, class... _Args>
        requires __queryable_with<_env_t, _Query, _Args...>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto query(_Query, _Args &&...__args) const
        noexcept(__nothrow_queryable_with<_env_t, _Query, _Args...>) -> decltype(auto)
      {
        return STDEXEC::get_env(__sndr_).query(_Query(), static_cast<_Args &&>(__args)...);
      }

      _Sender __sndr_;
    };

    template <sender _Sender1, class _Sender2>
    struct __attrs<_Sender1, _Sender2>
    {
     private:
      using __joined_t = env<env_of_t<_Sender2>, env_of_t<_Sender1>>;

      // The env to use when querying _CvSender2: where the first sender completed, joined
      // with the outer environment, as __rcvr2 provides it.
      template <class _Env>
      using __env2_t = decltype(__env::__join(__mk_env2_t()(__declval<_Sender1 const &>(),
                                                            __declval<_Env &>()),
                                              __declval<_Env &>()));

      template <class _Env>
      constexpr auto __mk_env2(_Env &__env) const noexcept -> __env2_t<_Env>
      {
        return __env::__join(__mk_env2_t()(__sndr1_, __env), __env);
      }

      // Whether the first (second) sender can complete with _Tag. A sender that cannot has
      // no say in where the sequence completes that way.
      template <class _Tag, class... _Env>
      static constexpr bool __sends1 = !__never_sends<_Tag, _Sender1, __fwd_env_t<_Env>...>;

      template <class _Tag, class... _Env>
      static constexpr bool __sends2 = !__never_sends<_Tag, _Sender2, __env2_t<_Env>...>;

     public:
      template <class... _Env>
        requires __has_completion_scheduler_for<set_value_t, _Sender2, __env2_t<_Env>...>
      [[nodiscard]]
      constexpr auto query(get_completion_scheduler_t<set_value_t>, _Env &&...__env) const noexcept
      {
        return STDEXEC::get_completion_scheduler<set_value_t>(STDEXEC::get_env(__sndr2_),
                                                              __mk_env2(__env)...);
      }

      // We know the error or stopped completion scheduler only if just one of the two
      // senders can complete that way, and it knows its scheduler. (If both can, the
      // sequence completes on one of two schedulers, which can differ at run time.)
      template <__one_of<set_error_t, set_stopped_t> _Tag, class... _Env>
        requires(__sends1<_Tag, _Env...> != __sends2<_Tag, _Env...>)
             && (__sends1<_Tag, _Env...>
                   ? __has_completion_scheduler_for<_Tag, _Sender1, __fwd_env_t<_Env>...>
                   : __has_completion_scheduler_for<_Tag, _Sender2, __env2_t<_Env>...>)
      [[nodiscard]]
      constexpr auto query(get_completion_scheduler_t<_Tag>, _Env &&...__env) const noexcept
      {
        if constexpr (__sends2<_Tag, _Env...>)
        {
          return STDEXEC::get_completion_scheduler<_Tag>(STDEXEC::get_env(__sndr2_),
                                                         __mk_env2(__env)...);
        }
        else
        {
          return STDEXEC::get_completion_scheduler<_Tag>(STDEXEC::get_env(__sndr1_),
                                                         __fwd_env(__env)...);
        }
      }

      // The value completion domain is the domain of the second sender, started
      // from the completing context of the first sender.
      template <class... _Env>
      [[nodiscard]]
      constexpr auto query(get_completion_domain_t<set_value_t>, _Env &&...) const noexcept
      {
        using __domain_t =
          __completion_domain_t<set_value_t, env_of_t<_Sender2>, __env2_t<_Env>...>;
        return __domain_t();
      }

      // The set_error/set_stopped completion domain is the domain of the sender that can
      // complete that way, or the common domain of the two senders if both can (or neither
      // can).
      template <__one_of<set_error_t, set_stopped_t> _Tag, class... _Env>
      [[nodiscard]]
      constexpr auto query(get_completion_domain_t<_Tag>, _Env &&...) const noexcept
      {
        using __domain1_t = __completion_domain_t<_Tag, env_of_t<_Sender1>, __fwd_env_t<_Env>...>;
        using __domain2_t = __completion_domain_t<_Tag, env_of_t<_Sender2>, __env2_t<_Env>...>;
        if constexpr (__sends1<_Tag, _Env...> == __sends2<_Tag, _Env...>)
          return __common_domain_t<__domain1_t, __domain2_t>();
        else if constexpr (__sends1<_Tag, _Env...>)
          return __domain1_t();
        else
          return __domain2_t();
      }

      // The completion behavior of a pair of senders in sequence is the weakest
      // completion behavior of the two senders.
      template <class _Tag, class... _Env>
      [[nodiscard]]
      constexpr auto query(__get_completion_behavior_t<_Tag>, _Env &&...) const noexcept
      {
        return __completion_behavior::__common(
          STDEXEC::__get_completion_behavior<_Tag, _Sender1, __fwd_env_t<_Env>...>(),
          STDEXEC::__get_completion_behavior<_Tag, _Sender2, __env2_t<_Env>...>());
      }

      // For queries that are not related to completion schedulers, domains, or
      // behaviors, we can just check _CvSender2 and then _CvSender1.
      template <__forwarding_query _Query, class... _Args>
        requires(!__completion_query<_Query>) && __queryable_with<__joined_t, _Query, _Args...>
      [[nodiscard]]
      constexpr auto operator()(_Query, _Args &&...__args) const
        noexcept(__nothrow_queryable_with<__joined_t, _Query, _Args...>) -> decltype(auto)
      {
        return __joined_t{STDEXEC::get_env(__sndr2_), STDEXEC::get_env(__sndr1_)}
          .query(_Query(), static_cast<_Args &&>(__args)...);
      }

      _Sender1 __sndr1_;
      _Sender2 __sndr2_;
    };

    template <class _Child1, class _Child2, class... _Rest>
    struct __attrs<_Child1, _Child2, _Rest...> : __attrs<__sndr<_Child1, _Child2>, _Rest...>
    {};

    ////////////////////////////////////////////////////////////////////////////
    // __state: the part of the opstate that is referenced by __rcvr2.
    template <class _Receiver, class _Env2>
    struct __state
    {
      constexpr explicit __state(_Receiver &&__rcvr, _Env2 &&__env2) noexcept
        : __rcvr_(static_cast<_Receiver &&>(__rcvr))
        , __env_(static_cast<_Env2 &&>(__env2))
      {}

      virtual constexpr void __start_next() noexcept = 0;

      _Receiver   __rcvr_;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Env2 const __env_;
    };

    ////////////////////////////////////////////////////////////////////////////
    // __rcvr2: the receiver that is connected to the successor sender.
    template <class _Receiver, class _Env2>
    struct __rcvr2
    {
      using receiver_concept = receiver_tag;
      using __env_t          = __join_env_t<_Env2 const &, __fwd_env_t<env_of_t<_Receiver>>>;

      template <class... _As>
      constexpr explicit __rcvr2(__state<_Receiver, _Env2> *__self) noexcept
        : __self_(__self)
      {}

      template <class... _As>
      constexpr void set_value(_As &&...__as) noexcept
      {
        STDEXEC::set_value(static_cast<_Receiver &&>(__self_->__rcvr_),
                           static_cast<_As &&>(__as)...);
      }

      template <class _Error>
      constexpr void set_error(_Error &&__error) noexcept
      {
        STDEXEC::set_error(static_cast<_Receiver &&>(__self_->__rcvr_),
                           static_cast<_Error &&>(__error));
      }

      constexpr void set_stopped() noexcept
      {
        STDEXEC::set_stopped(static_cast<_Receiver &&>(__self_->__rcvr_));
      }

      // Forwards the items of a sequence sender to the receiver.
      template <class _Item, class _Rcvr = _Receiver>
      constexpr auto set_next(_Item &&__item) & noexcept(
        noexcept(__declval<_Rcvr &>().set_next(__declval<_Item>())))
        -> decltype(__declval<_Rcvr &>().set_next(__declval<_Item>()))
      {
        return __self_->__rcvr_.set_next(static_cast<_Item &&>(__item));
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> __env_t
      {
        return __env::__join(__self_->__env_, __fwd_env(STDEXEC::get_env(__self_->__rcvr_)));
      }

      __state<_Receiver, _Env2> *__self_;
    };

    template <class _Receiver, class _Env2>
    struct __rcvr1
    {
      using receiver_concept = receiver_tag;

      constexpr void set_value() noexcept
      {
        __state_->__start_next();
      }

      template <class _Error>
      constexpr void set_error(_Error &&__error) noexcept
      {
        STDEXEC::set_error(std::move(__state_->__rcvr_), static_cast<_Error &&>(__error));
      }

      constexpr void set_stopped() noexcept
      {
        STDEXEC::set_stopped(std::move(__state_->__rcvr_));
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Receiver>>
      {
        return __fwd_env(STDEXEC::get_env(__state_->__rcvr_));
      }

      __state<_Receiver, _Env2> *__state_;
    };

    // _Connect2 is connect_t, or __subscribe_t when _Sender2 is a sequence sender.
    template <class _CvSender1, class _Sender2, class _Receiver, class _Connect2 = connect_t>
    struct __opstate final : __state<_Receiver, __env2_t<_CvSender1, env_of_t<_Receiver>>>
    {
      constexpr explicit __opstate(_CvSender1 &&__sndr1, _Sender2 __sndr2, _Receiver __rcvr)
        noexcept(__nothrow_constructible)
        : __opstate(__sndr1, __sndr2, __rcvr, __mk_env2_t()(__sndr1, get_env(__rcvr)))
      {}

      STDEXEC_IMMOVABLE(__opstate);

      void start() & noexcept
      {
        STDEXEC::start(__var::__get<0>(__opstate_));
      }

      constexpr void __start_next() noexcept
      {
        // __start_next is virtual, so it is instantiated even when _CvSender1 never
        // completes with set_value. In that case the successor is never started, and
        // the completions of _Sender2 are not among the declared completions.
        if constexpr (!__sends<set_value_t, _CvSender1, __fwd_env_t<env_of_t<_Receiver>>>)
        {
          __std::unreachable();
        }
        else
        {
          __start_next_impl();
        }
      }

     private:
      constexpr void __start_next_impl() noexcept
      {
        STDEXEC_TRY
        {
          auto &__op =  //
            __opstate_.__emplace_from(_Connect2(), std::move(__sndr2_), __rcvr2_t{this});
          STDEXEC::start(__op);
        }
        STDEXEC_CATCH_ALL
        {
          if constexpr (!__nothrow_callable<_Connect2, _Sender2, __rcvr2_t>)
          {
            STDEXEC::set_error(std::move(this->__rcvr_), std::current_exception());
          }
        }
      }

      using __env2_t     = __seq::__env2_t<_CvSender1, env_of_t<_Receiver>>;
      using __rcvr1_t    = __rcvr1<_Receiver, __env2_t>;
      using __rcvr2_t    = __rcvr2<_Receiver, __env2_t>;
      using __opstate1_t = connect_result_t<_CvSender1, __rcvr1_t>;
      using __opstate2_t = __call_result_t<_Connect2, _Sender2, __rcvr2_t>;

      static constexpr bool __nothrow_constructible = __nothrow_move_constructible<_Sender2>
                                                   && __nothrow_connectable<_CvSender1, __rcvr1_t>;

      constexpr explicit __opstate(_CvSender1 &__sndr1,
                                   _Sender2   &__sndr2,
                                   _Receiver  &__rcvr,
                                   __env2_t    __env2) noexcept(__nothrow_constructible)
        : __state<_Receiver, __env2_t>{std::move(__rcvr), std::move(__env2)}
        , __sndr2_(std::move(__sndr2))
      {
        __opstate_.__emplace_from(STDEXEC::connect,
                                  static_cast<_CvSender1 &&>(__sndr1),
                                  __rcvr1_t{this});
      }

      _Sender2                              __sndr2_;
      __variant<__opstate1_t, __opstate2_t> __opstate_{__no_init};
    };

    template <class _Self>
    struct __eat_value_signatures
    {
      template <class _Arg>
      using __error_t = __mexception<_WHAT_(_INVALID_ARGUMENT_),
                                     _WHERE_(_IN_ALGORITHM_, __sequence_t),
                                     _WHY_(_ALL_SENDERS_BUT_THE_LAST_MUST_BE_SENDERS_OF_VOID_),
                                     _WITH_SENDER_<_Self>>;

      template <class... _Args>
      constexpr auto operator()() const noexcept
      {
        // This fold over the comma operator returns completion_signatures{} if
        // _Args is empty, and otherwise results in a compile-time error. The
        // predecessor sender should be a sender of void, so it's an error if
        // _Args is not empty.
        return (completion_signatures{},
                ...,
                STDEXEC::__throw_compile_time_error(__error_t<_Args>()));
      }
    };

    ////////////////////////////////////////////////////////////////////////////
    // Default implementation of the __sequence sender algorithm.
    template <class _Sender1, class _Sender2>
    struct __sndr<_Sender1, _Sender2>
    {
      // If the last sender is a sequence sender, then so is this one.
      using sender_concept =
        __if_c<__sequence_sender_tagged<_Sender2>, __sequence_sender_tag, sender_tag>;

      template <class _Self, class _Receiver>
      using __opstate_t = __opstate<__copy_cvref_t<_Self, _Sender1>, _Sender2, _Receiver>;

      template <class _Self, class _Receiver>
      using __subscribe_opstate_t =
        __opstate<__copy_cvref_t<_Self, _Sender1>, _Sender2, _Receiver, __subscribe_::__subscribe_t>;

      template <class _Self, class _Receiver>
      using __rcvr2_t =
        __rcvr2<_Receiver, __seq::__env2_t<__copy_cvref_t<_Self, _Sender1>, env_of_t<_Receiver>>>;

      template <class _Self, class _Env>
      using __env2_t = __join_env_t<__seq::__env2_t<__copy_cvref_t<_Self, _Sender1>, _Env> const &,
                                    __fwd_env_t<_Env>>;

      using __attrs_t = __attrs<_Sender1, _Sender2>;

      template <class _Self, class... _Env>
        requires(sizeof...(_Env) != 0)  //
             || __has_eptr_completion<__copy_cvref_t<_Self, _Sender1>>
             || __has_eptr_completion<_Sender2>
      static consteval auto get_completion_signatures()
      {
        using __cv_sender1_t = __copy_cvref_t<_Self, _Sender1>;

        if constexpr (!__decay_copyable<_Self>)
        {
          return STDEXEC::__throw_compile_time_error<_SENDER_TYPE_IS_NOT_DECAY_COPYABLE_,
                                                     _WITH_SENDER_<_Self>>();
        }
        else if constexpr (!__sends<set_value_t, __cv_sender1_t, __fwd_env_t<_Env>...>)
        {
          // If the first sender has no set_value completions, then the second
          // sender will never be started, so just return the (error and
          // stopped) completions of the first sender.
          return STDEXEC::get_completion_signatures<__cv_sender1_t, __fwd_env_t<_Env>...>();
        }
        else
        {
          constexpr bool __nothrow_connect2 =
            (__nothrow_connectable<_Sender2, __receiver_archetype<__env2_t<_Self, _Env>>> && ...);

          auto __completions1 =  //
            STDEXEC::__transform_completion_signatures(
              STDEXEC::get_completion_signatures<__cv_sender1_t, __fwd_env_t<_Env>...>(),
              __eat_value_signatures<_Self>{});

          auto __completions2 =
            STDEXEC::get_completion_signatures<_Sender2, __env2_t<_Self, _Env>...>();

          auto __eptr_completions =
            STDEXEC::__eptr_completion_unless_t<__mbool<__nothrow_connect2>>();

          return STDEXEC::__concat_completion_signatures(__completions1,
                                                         __completions2,
                                                         __eptr_completions);
        }
      }

      template <class _Self, class _Receiver>
      constexpr auto STDEXEC_EXPLICIT_THIS_BEGIN(connect)(this _Self &&__self, _Receiver __rcvr)
        noexcept(__nothrow_constructible_from<__opstate_t<_Self, _Receiver>,
                                              __copy_cvref_t<_Self, _Sender1>,
                                              __copy_cvref_t<_Self, _Sender2>,
                                              _Receiver>)
      {
        return __opstate_t<_Self, _Receiver>(static_cast<_Self &&>(__self).__sndr1_,
                                             static_cast<_Self &&>(__self).__sndr2_,
                                             static_cast<_Receiver &&>(__rcvr));
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      // Runs _Sender1, then subscribes _Sender2 when it is a sequence sender.
      template <class _Self, class _Receiver>
        requires __sequence_sender_tagged<_Sender2>
              && __subscribable<_Sender2, __rcvr2_t<_Self, _Receiver>>
      static constexpr auto __static_subscribe(_Self &&__self, _Receiver __rcvr)
        noexcept(__nothrow_constructible_from<__subscribe_opstate_t<_Self, _Receiver>,
                                              __copy_cvref_t<_Self, _Sender1>,
                                              __copy_cvref_t<_Self, _Sender2>,
                                              _Receiver>) -> __subscribe_opstate_t<_Self, _Receiver>
      {
        return __subscribe_opstate_t<_Self, _Receiver>(static_cast<_Self &&>(__self).__sndr1_,
                                                       static_cast<_Self &&>(__self).__sndr2_,
                                                       static_cast<_Receiver &&>(__rcvr));
      }

      // The items of this sequence are the items of _Sender2 in the
      // environment it is subscribed with.
      template <class _Self, class... _Env>
      using __item_types_from_t = __mlist<_Sender2, __env2_t<_Self, _Env>...>;

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> __attrs_t
      {
        return __attrs_t{__sndr1_, __sndr2_};
      }

      _Sender1 __sndr1_;
      _Sender2 __sndr2_;
    };

    template <class _Sender1, class _Sender2, class... _Senders>
    struct __sndr<_Sender1, _Sender2, _Senders...>  //
      : __sndr<__sndr<_Sender1, _Sender2>, _Senders...>
    {};

    // The secondary environment of the _Idx-th sender (_Idx >= 1): where the sender before it
    // completed, in that sender's environment. The receiver's environment is passed by
    // reference, so that this has the same type at run time as in the walk.
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
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto __env2_for(_Sender const &__sndr, _Env const &__env) noexcept
    {
      static_assert(_Idx != 0);
      if constexpr (_Idx == 1)
      {
        return __mk_env2_t()(__facade::__get_child<0>(__sndr), __fwd_env(__env));
      }
      else
      {
        return __mk_env2_t()(__facade::__get_child<_Idx - 1>(__sndr),
                             __env::__join(__seq::__env2_for<_Idx - 1>(__sndr, __env), __env));
      }
    }

    // Is the last sender a sequence sender? Then so is the sequence, which is lowered to a
    // __seq::__sndr (see __sequence_t::transform_sender), because the facade cannot connect
    // a sequence sender. (This matches the sender's template arguments, so that it does not
    // require the sender to be complete.)
    template <class _Sender>
    inline constexpr bool __ends_in_sequence_sender_v = false;

    template <class _Tag, class _Data, class _Child0, class... _Child>
    inline constexpr bool
      __ends_in_sequence_sender_v<__facade::__sender<_Tag, _Data, _Child0, _Child...>> =
        __sequence_sender_tagged<__m_at_c<sizeof...(_Child), _Child0, _Child...>>;

    template <class _Sender>
    concept __ends_in_sequence_sender = __ends_in_sequence_sender_v<__decay_t<_Sender>>;

    struct __algo;

    template <class _Sender, class _Receiver, class _Indices>
    struct __seq_state;

    template <class _Sender, class _Receiver, std::size_t... _Is>
    struct __seq_state<_Sender, _Receiver, __indices<_Is...>>
    {
      static constexpr std::size_t __nbr = sizeof...(_Is);
      using __env_t                      = env_of_t<_Receiver>;

      // The later senders are connected and started when the one before them completes.
      template <std::size_t _Idx>
      using __sender_t = __sender_of_t<_Idx, _Sender>;

      template <std::size_t _Idx>
      using __receiver_t =
        __facade::__receiver_t<__algo, __facade::__child<_Idx>, __seq_state, _Sender, _Receiver>;

      template <std::size_t _Idx>
      using __env2_t = decltype(__seq::__env2_for<_Idx>(__declval<__decay_t<_Sender> const &>(),
                                                        __declval<__env_t const &>()));

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
          return __nothrow_connectable<__sender_t<0>, __receiver_t<0>>;
      }

      static constexpr bool __nothrow_v = __later_t::__nothrow_take && __nothrow_connect_first();

      constexpr explicit __seq_state(_Sender &&__sndr, _Receiver &&__rcvr) noexcept(__nothrow_v)
        : __rcvr_(static_cast<_Receiver &&>(__rcvr))
        , __env2s_(__later_t::__mk_env2s(__sndr, STDEXEC::get_env(__rcvr_)))
        , __sndrs_(__later_t::__take_senders(static_cast<_Sender &&>(__sndr)))
      {
        if constexpr (__nbr != 0)
        {
          __ops_.__emplace_from(
            [&]
            {
              return STDEXEC::connect(__facade::__get_child<0>(static_cast<_Sender &&>(__sndr)),
                                      __receiver_t<0>{this});
            });
        }
      }

      STDEXEC_IMMOVABLE(__seq_state);

      _Receiver                                                          __rcvr_;
      typename __later_t::__env2s_t                                      __env2s_;
      typename __later_t::__senders_t                                    __sndrs_;
      __variant<connect_result_t<__sender_t<_Is>, __receiver_t<_Is>>...> __ops_{__no_init};
    };

    // __sequence(sndrs...): starts the senders one after the other. Each but the last must
    // complete with set_value_t(), which starts the next. Other completions of any of them,
    // and every completion of the last, are the operation's.
    struct __algo : __facade::__defaults
    {
      static constexpr bool __connects_children = false;

      // A sequence whose last sender is a sequence sender has the completions and the
      // attributes of the __seq::__sndr it is lowered to.
      template <class _Sender>
        requires __ends_in_sequence_sender<_Sender>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __get_attrs(_Sender const &__sndr) noexcept
      {
        return STDEXEC::__apply([]<class... _Child>(auto const &,
                                                    auto const &,
                                                    _Child const &...__child) noexcept
                                { return __attrs<_Child...>{__child...}; },
                                __sndr);
      }

      template <class _Sender, class... _Env>
        requires __ends_in_sequence_sender<_Sender>
      static consteval auto __get_completion_signatures()
      {
        using __sndr_t = __remove_rvalue_reference_t<transform_sender_result_t<_Sender, env<>>>;
        return STDEXEC::get_completion_signatures<__sndr_t, _Env...>();
      }

      template <class _Sender>
      static constexpr bool __depends_on_env =
        __may_throw_unknown<_Sender, __make_indices<__nbr_children_of<_Sender>>>;

      // Each sender after the first sees where the one before it completed.
      template <std::size_t _Idx, class _Sender, class _Env>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __env_for(__facade::__child<_Idx>,
                                      _Sender const &__sndr,
                                      _Env         &&__env) noexcept
      {
        if constexpr (_Idx == 0)
        {
          return __fwd_env(static_cast<_Env &&>(__env));
        }
        else
        {
          auto __env2 = __seq::__env2_for<_Idx>(__sndr, __env);
          return __env::__join(static_cast<decltype(__env2) &&>(__env2),
                               static_cast<_Env &&>(__env));
        }
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
          return __env::__join(__env2_t(STDEXEC::__get<_Idx - 1>(__state.__env2s_)),
                               STDEXEC::get_env(__state.__rcvr_));
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

      template <class _Sender, class _Receiver>
      using __state_t = __seq_state<_Sender, _Receiver, __make_indices<__nbr_children_of<_Sender>>>;

      template <class _Sender, class _Receiver>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __get_state(_Sender &&__sndr, _Receiver __rcvr)
        noexcept(__state_t<_Sender, _Receiver>::__nothrow_v) -> __state_t<_Sender, _Receiver>
      {
        return __state_t<_Sender, _Receiver>(static_cast<_Sender &&>(__sndr),
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

        // Connecting the next sender destroys the operation of the one that just completed.
        auto __start = [&]()
        {
          auto &__op = __state.__ops_.__emplace_from(
            [&]
            {
              return STDEXEC::connect(static_cast<__sender_t &&>(
                                        STDEXEC::__get<_Idx - 1>(__state.__sndrs_)),
                                      __receiver_t{&__state});
            });
          return __facade::__start_nested<__facade::__child<_Idx>>(__op);
        };

        if constexpr (__nothrow_connectable<__sender_t, __receiver_t>)
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
    };
  }  // namespace __seq

  struct __sequence_t
  {
    template <class... _Senders>
    [[nodiscard]]
    constexpr auto operator()(_Senders &&...__sndrs) const  //
      noexcept(__nothrow_decay_copyable<_Senders...>) -> __well_formed_sender auto
    {
      return __facade::__sender{__sequence_t(), __(), static_cast<_Senders &&>(__sndrs)...};
    }

    // A sequence whose last sender is a sequence sender is itself a sequence sender. Only
    // __seq::__sndr implements that, so such a sequence is lowered to one.
    template <class _Self>
      requires(__nbr_children_of<_Self> != 0)
           && __sequence_sender_tagged<
                __decay_t<__tuple_element_t<__nbr_children_of<_Self> + 1, __decay_t<_Self>>>>
    static constexpr auto transform_sender(set_value_t, _Self &&__self, __ignore)  //
      noexcept(__nothrow_transform<_Self>()) -> decltype(auto)
    {
      constexpr std::size_t __nbr_sndrs = __nbr_children_of<_Self>;
      if constexpr (__nbr_sndrs == 0)
      {
        return just();
      }
      else if constexpr (__nbr_sndrs == 1)
      {
        return static_cast<__tuple_element_t<2, _Self>>(
          STDEXEC::__get<2>(static_cast<_Self &&>(__self)));
      }
      else
      {
        return __apply(__mk_sndr, static_cast<_Self &&>(__self));
      }
    }

   private:
    static constexpr auto __mk_sndr =
      []<class... _Child>(__ignore, __ignore, _Child &&...__child)  //
      noexcept(__nothrow_decay_copyable<_Child...>)
    {
      return __seq::__sndr<__decay_t<_Child>...>{static_cast<_Child &&>(__child)...};
    };

    //! whether transform_sender, given a sequence of type _Self, can't throw
    template <class _Self>
    static consteval auto __nothrow_transform() noexcept -> bool
    {
      constexpr std::size_t __nbr_sndrs = __nbr_children_of<_Self>;
      if constexpr (__nbr_sndrs == 0)
      {
        return true;
      }
      else if constexpr (__nbr_sndrs == 1)
      {
        return noexcept(
          static_cast<__tuple_element_t<2, _Self>>(STDEXEC::__get<2>(__declval<_Self>())));
      }
      else
      {
        return __nothrow_applicable<decltype(__mk_sndr) const &, _Self>;
      }
    }
  };

  inline constexpr __sequence_t __sequence{};

  template <>
  struct __facade::__algorithm<__sequence_t> : __seq::__algo
  {};

  // __seq::__sndr is the result of a sender transform. It should not be
  // transformed further.
  template <class... _Senders>
  inline constexpr auto __structured_binding_size_v<__seq::__sndr<_Senders...>> = -1;
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
