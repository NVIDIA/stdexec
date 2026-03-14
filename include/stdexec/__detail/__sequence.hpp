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

#include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "__basic_sender.hpp"
#include "__connect.hpp"
#include "__just.hpp"
#include "__schedulers.hpp"
#include "__senders.hpp"
#include "__transform_completion_signatures.hpp"
#include "__variant.hpp"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(expr_has_no_effect)
STDEXEC_PRAGMA_IGNORE_EDG(type_qualifiers_ignored_on_reference)
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")
STDEXEC_PRAGMA_IGNORE_GNU("-Wunused-value")

namespace STDEXEC
{
  struct __sequence_t;
  struct _ALL_SENDERS_BUT_THE_LAST_MUST_BE_SENDERS_OF_VOID_;

  namespace __seq
  {
    using __mk_env2_t = __mk_secondary_env_t<set_value_t>;

    template <class _CvSender1, class _Env>
    using __env2_t = __secondary_env_t<_CvSender1, _Env, set_value_t>;

    template <class... _Senders>
    struct __sndr;

    //////////////////////////////////////////////////////////////////////////////////////
    // Attributes for __sequence. This is a bit more complicated than the attributes for
    // other algorithms because we need to be able to query the completion scheduler (for
    // example) of the second sender from the context of the first sender's completions.
    template <class... _CvSenders>
    struct __attrs;

    template <>
    struct __attrs<> : __just::__attrs<set_value_t>
    {};

    template <class _CvSender>
    struct __attrs<_CvSender>
    {
      template <class _Query, class... _Args>
        requires __queryable_with<env_of_t<_CvSender>, _Query, _Args...>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()(_Query, _Args &&...__args) const
        noexcept(__nothrow_queryable_with<env_of_t<_CvSender>, _Query, _Args...>)
          -> __query_result_t<env_of_t<_CvSender>, _Query, _Args...>
      {
        return __query<_Query>()(STDEXEC::get_env(__sndr_), static_cast<_Args &&>(__args)...);
      }

      _CvSender __sndr_;
    };

    template <sender _CvSender1, class _CvSender2>
    struct __attrs<_CvSender1, _CvSender2>
    {
     private:
      using __joined_t = env<env_of_t<_CvSender2>, env_of_t<_CvSender1>>;

      // The env to use when querying _CvSender2:
      template <class _Env>
      using __env2_t = __secondary_env_t<_CvSender1 const &, _Env, set_value_t>;

      template <class _Env>
      constexpr auto __mk_env2(_Env &__env) const noexcept -> __env2_t<_Env>
      {
        return __mk_env2_t()(__cp{}, __sndr1_, __env);
      }

     public:
      template <class... _Env>
        requires __has_completion_scheduler_for<set_value_t, _CvSender2, __env2_t<_Env>...>
      [[nodiscard]]
      constexpr auto query(get_completion_scheduler_t<set_value_t>, _Env &&...__env) const noexcept
      {
        return STDEXEC::get_completion_scheduler<set_value_t>(STDEXEC::get_env(__sndr2_),
                                                              __mk_env2(__env)...);
      }

      // We only know the error or stopped completion scheduler if exactly one of the two
      // senders knows its error/stopped completion scheduler.
      template <__one_of<set_error_t, set_stopped_t> _Tag, class... _Env>
        requires(__has_completion_scheduler_for<_Tag, _CvSender1, __fwd_env_t<_Env>...>
                 != __has_completion_scheduler_for<_Tag, _CvSender2, __env2_t<_Env>...>)
      [[nodiscard]]
      constexpr auto query(get_completion_scheduler_t<_Tag>, _Env &&...__env) const noexcept
      {
        if constexpr (__has_completion_scheduler_for<_Tag, _CvSender2, __env2_t<_Env>...>)
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

      // The value completion domain is the domain of the second sender, started from the
      // completing context of the first sender.
      template <class... _Env>
      [[nodiscard]]
      constexpr auto query(get_completion_domain_t<set_value_t>, _Env &&...) const noexcept
      {
        using __domain_t =
          __completion_domain_t<set_value_t, env_of_t<_CvSender2>, __env2_t<_Env>...>;
        return __domain_t();
      }

      // The set_error/set_stopped completion domains are the common domain of the two
      // senders.
      template <__one_of<set_error_t, set_stopped_t> _Tag, class... _Env>
      [[nodiscard]]
      constexpr auto query(get_completion_domain_t<_Tag>, _Env &&...) const noexcept
      {
        using __domain_t =
          __common_domain_t<__completion_domain_t<_Tag, env_of_t<_CvSender1>, __fwd_env_t<_Env>...>,
                            __completion_domain_t<_Tag, env_of_t<_CvSender2>, __env2_t<_Env>...>>;
        return __domain_t();
      }

      // The completion behavior of a pair of senders in sequence is the weakest
      // completion behavior of the two senders.
      template <class _Tag, class... _Env>
      [[nodiscard]]
      constexpr auto query(__get_completion_behavior_t<_Tag>, _Env &&...) const noexcept
      {
        return __completion_behavior::__common(
          STDEXEC::__get_completion_behavior<_Tag, _CvSender1, __fwd_env_t<_Env>...>(),
          STDEXEC::__get_completion_behavior<_Tag, _CvSender2, __env2_t<_Env>...>());
      }

      // For queries that are not related to completion schedulers, domains, or behaviors,
      // we can just check _CvSender2 and then _CvSender1.
      template <__forwarding_query _Query, class... _Args>
        requires(!__completion_query<_Query>) && __queryable_with<__joined_t, _Query, _Args...>
      [[nodiscard]]
      constexpr auto operator()(_Query, _Args &&...__args) const
        noexcept(__nothrow_queryable_with<__joined_t, _Query, _Args...>)
          -> __query_result_t<__joined_t, _Query, _Args...>
      {
        return __query<_Query>()(__joined_t{STDEXEC::get_env(__sndr2_), STDEXEC::get_env(__sndr1_)},
                                 static_cast<_Args &&>(__args)...);
      }

      _CvSender1 __sndr1_;
      _CvSender2 __sndr2_;
    };

    template <class _Child1, class _Child2, class... _Rest>
    struct __attrs<_Child1, _Child2, _Rest...> : __attrs<__sndr<_Child1, _Child2>, _Rest...>
    {};

    //////////////////////////////////////////////////////////////////////////////////////
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

    //////////////////////////////////////////////////////////////////////////////////////
    // __rcvr2: the receiver that is connected to the successor sender.
    template <class _Receiver, class _Env2>
    struct __rcvr2
    {
      using receiver_concept = receiver_t;
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
      using receiver_concept = receiver_t;

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

    template <class _CvSender1, class _Sender2, class _Receiver>
    struct __opstate final : __state<_Receiver, __env2_t<_CvSender1, env_of_t<_Receiver>>>
    {
      using __cv_fn = __copy_cvref_fn<_CvSender1>;

      constexpr explicit __opstate(_CvSender1 &&__sndr1, _Sender2 __sndr2, _Receiver __rcvr)
        noexcept(__nothrow_constructible)
        : __opstate(__sndr1, __sndr2, __rcvr, __mk_env2_t()(__cv_fn{}, __sndr1, get_env(__rcvr)))
      {}

      STDEXEC_IMMOVABLE(__opstate);

      void start() & noexcept
      {
        STDEXEC::start(__var::__get<0>(__opstate_));
      }

      constexpr void __start_next() noexcept
      {
        STDEXEC_TRY
        {
          auto &__op =  //
            __opstate_.__emplace_from(STDEXEC::connect, std::move(__sndr2_), __rcvr2_t{this});
          STDEXEC::start(__op);
        }
        STDEXEC_CATCH_ALL
        {
          if constexpr (!__nothrow_connectable<_Sender2, __rcvr2_t>)
          {
            STDEXEC::set_error(std::move(this->__rcvr_), std::current_exception());
          }
        }
      }

     private:
      using __env2_t     = __seq::__env2_t<_CvSender1, env_of_t<_Receiver>>;
      using __rcvr1_t    = __rcvr1<_Receiver, __env2_t>;
      using __rcvr2_t    = __rcvr2<_Receiver, __env2_t>;
      using __opstate1_t = connect_result_t<_CvSender1, __rcvr1_t>;
      using __opstate2_t = connect_result_t<_Sender2, __rcvr2_t>;

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
                                     _WITH_PRETTY_SENDER_<_Self>>;

      template <class... _Args>
      constexpr auto operator()() const noexcept
      {
        // This fold over the comma operator returns completion_signatures{} if _Args is
        // empty, and otherwise results in a compile-time error. The predecessor sender
        // should be a sender of void, so it's an error if _Args is not empty.
        return (completion_signatures{},
                ...,
                STDEXEC::__throw_compile_time_error(__error_t<_Args>()));
      }
    };

    //////////////////////////////////////////////////////////////////////////////////////
    // Default implementation of the __sequence sender algorithm.
    template <class _Sender1, class _Sender2>
    struct __sndr<_Sender1, _Sender2>
    {
      using sender_concept = sender_t;

      template <class _Self, class _Receiver>
      using __opstate_t = __opstate<__copy_cvref_t<_Self, _Sender1>, _Sender2, _Receiver>;

      template <class _Self, class _Env>
      using __env2_t = __join_env_t<__seq::__env2_t<__copy_cvref_t<_Self, _Sender1>, _Env> const &,
                                    __fwd_env_t<_Env>>;

      using __attrs_t = __attrs<_Sender1 const &, _Sender2 const &>;

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
                                                     _WITH_PRETTY_SENDER_<_Self>>();
        }
        else if constexpr (!__sends<set_value_t, __cv_sender1_t, __fwd_env_t<_Env>...>)
        {
          // If the first sender has no set_value completions, then the second sender will
          // never be started, so just return the (error and stopped) completions of the
          // first sender.
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
      constexpr STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this _Self &&__self, _Receiver __rcvr)
        noexcept(__nothrow_constructible_from<__opstate_t<_Self, _Receiver>,
                                              __copy_cvref_t<_Self, _Sender1>,
                                              _Sender2,
                                              _Receiver>)
      {
        return __opstate_t<_Self, _Receiver>(static_cast<_Self &&>(__self).__sndr1_,
                                             static_cast<_Self &&>(__self).__sndr2_,
                                             static_cast<_Receiver &&>(__rcvr));
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

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

    struct __impls : __sexpr_defaults
    {
      static constexpr auto __get_attrs =                                   //
        []<class... _Child>(auto, auto, _Child const &...__child) noexcept  //
        -> __attrs<_Child const &...>
      {
        return __attrs<_Child const &...>{__child...};
      };

      template <class _Self>
      static consteval auto __get_completion_signatures()
      {
        using __sndr_t = transform_sender_result_t<_Self, env<>>;
        return STDEXEC::get_completion_signatures<__sndr_t>();
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
      return __make_sexpr<__sequence_t>({}, static_cast<_Senders &&>(__sndrs)...);
    }

    template <class _Self>
    static constexpr auto transform_sender(set_value_t, _Self &&__self, __ignore)  //
      -> decltype(auto)
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
    static constexpr auto __mk_sndr = []<class... _Child>(__ignore, __ignore, _Child &&...__child)
    {
      return __seq::__sndr<__decay_t<_Child>...>{static_cast<_Child &&>(__child)...};
    };
  };

  inline constexpr __sequence_t __sequence{};

  template <>
  struct __sexpr_impl<__sequence_t> : __seq::__impls
  {};

  // __seq::__sndr is the result of a sender transform. It should not be transformed
  // further.
  template <class... _Senders>
  inline constexpr auto __structured_binding_size_v<__seq::__sndr<_Senders...>> = -1;

  namespace __detail
  {
    template <class... _Senders>
    extern __declfn_t<__seq::__sndr<__demangle_t<_Senders>...>>
      __demangle_v<__seq::__sndr<_Senders...>>;
  }  // namespace __detail
}  // namespace STDEXEC

STDEXEC_PRAGMA_POP()
