/* Copyright (c) 2023 Maikel Nadolski
 * Copyright (c) 2023 NVIDIA Corporation
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

#include "../stdexec/__detail/__any.hpp"
#include "../stdexec/__detail/__receiver_ref.hpp"
#include "../stdexec/__detail/__receivers.hpp"

#include "env.hpp"

#include <utility>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Woverloaded-virtual")

namespace experimental::execution
{
  template <class... Sigs>
  struct queries;

  template <class Sigs, class Queries = queries<>>
  struct any_receiver;

  template <class AnyReceiver, class SenderQueries = queries<>>
  struct any_sender;

  template <class AnySender, class SchedulerQueries = queries<>>
  struct any_scheduler;

  struct CANNOT_TYPE_ERASE_THE_GIVEN_RECEIVER;
  struct THE_RECEIVERS_ENVIRONMENT_LACKS_A_VALUE_FOR_A_QUERY;
  struct THE_RESULT_TYPE_OF_THE_RECEIVERS_QUERY_IS_NOT_CONVERTIBLE_TO_THE_RESULT_OF_THE_TYPE_ERASED_QUERY;
  struct EXPECTED_QUERY_RESULT_TYPE;
  struct ACTUAL_QUERY_RESULT_TYPE;
  struct THE_RECEIVERS_QUERY_IS_NOT_NOEXCEPT_BUT_THE_TYPE_ERASED_QUERY_IS;

  namespace _any
  {
    using namespace STDEXEC;

    template <class QueryFn, bool Noexcept, class... Env>
    inline constexpr auto _check_query_impl_v = __msuccess{};

    template <class Result, class Query, class... Args, bool Noexcept, class Env>
      requires(!__std::convertible_to<__call_result_t<Query, Env, Args...>, Result>)
    inline constexpr auto _check_query_impl_v<Result(Query, Args...), Noexcept, Env> =  //
      _ERROR_<
        _WHAT_(CANNOT_TYPE_ERASE_THE_GIVEN_RECEIVER),
        _WHY_(
          THE_RESULT_TYPE_OF_THE_RECEIVERS_QUERY_IS_NOT_CONVERTIBLE_TO_THE_RESULT_OF_THE_TYPE_ERASED_QUERY),
        _WITH_QUERY_(Query),
        EXPECTED_QUERY_RESULT_TYPE(Result),
        ACTUAL_QUERY_RESULT_TYPE(__call_result_t<Query, Env, Args...>)>{};

    template <class Result, class Query, class... Args, class Env>
      requires(!__nothrow_queryable_with<Env, Query, Args...>)
    inline constexpr auto _check_query_impl_v<Result(Query, Args...), true, Env> =
      _ERROR_<_WHAT_(CANNOT_TYPE_ERASE_THE_GIVEN_RECEIVER),
              _WHY_(THE_RECEIVERS_QUERY_IS_NOT_NOEXCEPT_BUT_THE_TYPE_ERASED_QUERY_IS),
              _WITH_QUERY_(Query),
              _WITH_ENVIRONMENT_(Env)>{};

    template <class QueryFn, class... Env>
    using _no_query_error_t = _ERROR_<_WHAT_(CANNOT_TYPE_ERASE_THE_GIVEN_RECEIVER),
                                      _WHY_(THE_RECEIVERS_ENVIRONMENT_LACKS_A_VALUE_FOR_A_QUERY),
                                      _WITH_QUERY_(QueryFn),
                                      _WITH_ENVIRONMENT_(Env...)>;

    template <class QueryFn, class... Env>
    inline constexpr auto _check_query_v = std::conditional_t<sizeof...(Env) == 0,
                                                              __mexception<dependent_sender_error>,
                                                              _no_query_error_t<QueryFn, Env...>>{};

    template <class Result, class Query, class... Args, class... Env>
      requires(__queryable_with<Env, Query, Args...> || ...)
    inline constexpr auto _check_query_v<Result(Query, Args...), Env...> =
      _check_query_impl_v<Result(Query, Args...), false, Env...>;

    template <class Result, class Query, class... Args, class... Env>
      requires(__queryable_with<Env, Query, Args...> || ...)
    inline constexpr auto _check_query_v<Result(Query, Args...) noexcept, Env...> =
      _check_query_impl_v<Result(Query, Args...), true, Env...>;

    template <class... Env>
    inline constexpr auto _check_query_v<never_stop_token(get_stop_token_t) noexcept, Env...> =
      __msuccess{};

    template <class Env>
      requires __same_as<stop_token_of_t<Env>, STDEXEC::never_stop_token>
    inline constexpr auto _check_query_v<inplace_stop_token(get_stop_token_t) noexcept, Env> =
      __msuccess{};

    template <class QueryFn, class... Env>
    using _check_query_t = __decay_t<decltype(_check_query_v<QueryFn, Env...>)>;

    template <class QueryFn, class... Env>
    concept _valid_query_for = __ok<_check_query_t<QueryFn, Env...>>;

    template <class Base>
    struct _inject_query_memfn : Base
    {
      using _has_query_memfn_t = void;
      using Base::Base;
      auto query(__none_such) const noexcept -> __none_such = delete;
    };

    template <class Base>
    concept _has_query_memfn = requires { typename Base::_has_query_memfn_t; };

    template <class Base>
    using _inject_query_memfn_t = __if_c<_has_query_memfn<Base>, Base, _inject_query_memfn<Base>>;

    template <class Base>
    struct _inject_receiver_memfns : Base
    {
      using _has_receiver_memfns_t = void;
      using Base::Base;
      void set_value(__none_such) const noexcept = delete;
      void set_error(__none_such) const noexcept = delete;
    };

    template <class Base>
    concept _has_receiver_memfns = requires { typename Base::_has_receiver_memfns_t; };

    template <class Base>
    using _inject_receiver_memfns_t =
      __if_c<_has_receiver_memfns<Base>, Base, _inject_receiver_memfns<Base>>;

    template <class QuerySig, bool Indirect = true, bool Noexcept = false>
    struct _iquery_memfn;

    template <class Value, class Query, class... Args, bool Indirect, bool Noexcept>
    struct _iquery_memfn<Value(Query, Args...), Indirect, Noexcept>
    {
      template <class Base>
      struct _interface_
        : STDEXEC::__any::__interface_base<_interface_, _inject_query_memfn_t<Base>>
      {
       private:
        using _base_t = STDEXEC::__any::__interface_base<_interface_, _inject_query_memfn_t<Base>>;
       public:
        using _base_t::_base_t;
        using _base_t::query;

        virtual constexpr auto query(Query, Args... _args) const noexcept(Noexcept) -> Value
        {
          if constexpr (Indirect)
            // This branch is used for any_sender and any_receiver, which put their
            // queries behind a call to get_env.
            return Query()(STDEXEC::get_env(STDEXEC::__any::__value(*this)),
                           static_cast<Args &&>(_args)...);
          else
            // This branch is used for any_scheduler, which puts its queries directly on
            // the type-erased scheduler.
            return Query()(STDEXEC::__any::__value(*this), static_cast<Args &&>(_args)...);
        }

        [[nodiscard]]
        constexpr auto get_env() const noexcept -> _interface_ const &
          requires Indirect
        {
          return *this;
        }
      };
    };

    template <class Value, class Query, class... Args, bool Indirect>
    struct _iquery_memfn<Value(Query, Args...) noexcept, Indirect>
      : _iquery_memfn<Value(Query, Args...), Indirect, true>
    {};

    template <class Sig>
    struct _ireceiver_memfn;

    template <class... As>
    struct _ireceiver_memfn<set_value_t(As...)>
    {
      template <class Base>
      struct _interface_
        : STDEXEC::__any::__interface_base<_interface_, _inject_receiver_memfns_t<Base>>
      {
       private:
        using _base_t =
          STDEXEC::__any::__interface_base<_interface_, _inject_receiver_memfns_t<Base>>;
       public:
        using _base_t::_base_t;
        using _base_t::set_value;

        virtual constexpr void set_value(As... _as) && noexcept
        {
          STDEXEC::__any::__value(std::move(*this)).set_value(static_cast<As &&>(_as)...);
        }
      };
    };

    template <class Error>
    struct _ireceiver_memfn<set_error_t(Error)>
    {
      template <class Base>
      struct _interface_
        : STDEXEC::__any::__interface_base<_interface_, _inject_receiver_memfns_t<Base>>
      {
       private:
        using _base_t =
          STDEXEC::__any::__interface_base<_interface_, _inject_receiver_memfns_t<Base>>;
       public:
        using _base_t::_base_t;
        using _base_t::set_error;

        virtual constexpr void set_error(Error _err) && noexcept
        {
          STDEXEC::__any::__value(std::move(*this)).set_error(static_cast<Error &&>(_err));
        }
      };
    };

    template <>
    struct _ireceiver_memfn<set_stopped_t()>
    {
      template <class Base>
      struct _interface_ : STDEXEC::__any::__interface_base<_interface_, Base>
      {
        using STDEXEC::__any::__interface_base<_interface_, Base>::__interface_base;

        virtual constexpr void set_stopped() && noexcept
        {
          STDEXEC::__any::__value(std::move(*this)).set_stopped();
        }
      };
    };

    template <class Sigs, class ReceiverQueries>
    struct _ireceiver;

    template <class... Sigs, class... Queries>
    struct _ireceiver<completion_signatures<Sigs...>, queries<Queries...>>
    {
      using _extends_base_t =
        STDEXEC::__any::__extends<_ireceiver_memfn<Sigs>::template _interface_...,
                                  _iquery_memfn<Queries>::template _interface_...>;

      // Used for type-erased receiver pointers, which do not require the underlying
      // receiver to be movable.
      template <class Base>
      struct _isemi_receiver
        : STDEXEC::__any::__interface_base<_isemi_receiver, Base, _extends_base_t>
      {
       private:
        using _base_t = STDEXEC::__any::__interface_base<_isemi_receiver, Base, _extends_base_t>;
       public:
        using receiver_concept = STDEXEC::receiver_tag;
        using _base_t::_base_t;
      };

      using _pointer_t = STDEXEC::__any::__any_ptr<_isemi_receiver>;
      using _extends_t = STDEXEC::__any::__extends<_isemi_receiver, STDEXEC::__any::__imovable>;

      template <class Base>
      struct _interface_ : STDEXEC::__any::__interface_base<_interface_, Base, _extends_t>
      {
       private:
        using _base_t = STDEXEC::__any::__interface_base<_interface_, Base, _extends_t>;
       public:
        using _base_t::_base_t;
      };
    };

    //////////////////////////////////////////////////////////////////////////////////////
    // _state_base
    template <class Receiver, class TargetStopToken>
    struct _state;

    template <class Receiver, class TargetStopToken>
    struct _state_base : Receiver
    {
      constexpr explicit _state_base(Receiver _rcvr) noexcept
        : Receiver(static_cast<Receiver &&>(_rcvr))
      {}

      [[nodiscard]]
      constexpr auto get_env() const noexcept
      {
        using _state_t = _state<Receiver, TargetStopToken>;
        auto _query_fn = [this](get_stop_token_t) noexcept -> TargetStopToken
        {
          return static_cast<_state_t const &>(*this)._get_token();
        };
        Receiver const &_rcvr = *this;
        return STDEXEC::__env::__join(env_from{_query_fn}, STDEXEC::get_env(_rcvr));
      }
    };

    //////////////////////////////////////////////////////////////////////////////////////
    // _state

    // A specialization of _state for when the receiver's stop token is not compatible
    // with the one required by the type-erased operation state. In this case, we use an
    // inplace_stop_source to create a stop token that is compatible with the type-erased
    // operation state.
    template <class Receiver>
    struct _state<Receiver, inplace_stop_token> : _state_base<Receiver, inplace_stop_token>
    {
      using _state_base<Receiver, inplace_stop_token>::_state_base;

      template <class... As>
      constexpr void set_value(As &&..._as) noexcept
      {
        _callback_.__destroy();
        Receiver::set_value(static_cast<As &&>(_as)...);
      }

      template <class Error>
      constexpr void set_error(Error &&_err) noexcept
      {
        _callback_.__destroy();
        Receiver::set_error(static_cast<Error &&>(_err));
      }

      constexpr void set_stopped() noexcept
      {
        _callback_.__destroy();
        Receiver::set_stopped();
      }

      [[nodiscard]]
      constexpr auto _get_token() const noexcept -> inplace_stop_token
      {
        return _stop_source_.get_token();
      }

      constexpr void _register_callback() noexcept
      {
        _callback_.__construct(  //
          get_stop_token(STDEXEC::get_env(static_cast<Receiver const &>(*this))),
          __forward_stop_request{_stop_source_});
      }

     private:
      using _stop_token_t    = stop_token_of_t<env_of_t<Receiver>>;
      using _stop_callback_t = stop_callback_for_t<_stop_token_t, __forward_stop_request<>>;
      inplace_stop_source                 _stop_source_;
      __manual_lifetime<_stop_callback_t> _callback_;
    };

    // A specialization of _state for when the receiver's stop token is compatible with
    // the one required by the type-erased operation state.
    template <class Receiver, class TargetStopToken>
      requires __std::convertible_to<stop_token_of_t<env_of_t<Receiver>>, TargetStopToken>
    struct _state<Receiver, TargetStopToken> : Receiver
    {
      constexpr _state(Receiver _rcvr) noexcept
        : Receiver(static_cast<Receiver &&>(_rcvr))
      {}

      constexpr void _register_callback() noexcept
      {
        // no-op
      }
    };

    template <class Receiver, class TargetStopToken>
      requires __std::convertible_to<stop_token_of_t<env_of_t<Receiver>>, TargetStopToken>
            || _valid_query_for<TargetStopToken(get_stop_token_t) noexcept, env_of_t<Receiver>>
    struct _state<Receiver, TargetStopToken> : _state_base<Receiver, TargetStopToken>
    {
      using _state_base<Receiver, TargetStopToken>::_state_base;

      [[nodiscard]]
      static constexpr auto _get_token() noexcept -> TargetStopToken
      {
        return TargetStopToken();
      }

      constexpr void _register_callback() noexcept
      {
        // no-op
      }
    };

    template <class Base>
    struct _iopstate;

    template <class Base>
    using _iopstate_base_t =
      STDEXEC::__any::__interface_base<_iopstate, Base, STDEXEC::__any::__extends<>, 64>;

    //////////////////////////////////////////////////////////////////////////////////////
    // _iopstate
    template <class Base>
    struct _iopstate : _iopstate_base_t<Base>
    {
      using operation_state_concept = STDEXEC::operation_state_tag;
      using _iopstate_base_t<Base>::_iopstate_base_t;

      virtual constexpr void start() & noexcept
      {
        STDEXEC::__any::__value(*this).start();
      }
    };

    // This type is the result of connecting a type-erased sender to a type-erased
    // receiver ref. _any_opstate derives from this type and stores the concrete receiver
    // so it can pass a reference to it when connecting the type-erased sender.
    struct _any_opstate_base final : STDEXEC::__any::__any<_iopstate>
    {
      using STDEXEC::__any::__any<_any::_iopstate>::__any;
      STDEXEC_IMMOVABLE(_any_opstate_base);
    };

    //////////////////////////////////////////////////////////////////////////////////////
    // _any_opstate
    template <class Receiver, class TargetStopToken>
    struct _any_opstate
    {
      using operation_state_concept = STDEXEC::operation_state_tag;

      template <class AnySender>
      constexpr explicit _any_opstate(AnySender &&_sndr, Receiver _rcvr)
        : _rcvr_{static_cast<Receiver &&>(_rcvr)}
        , _opstate_(static_cast<AnySender &&>(_sndr).connect(
            typename AnySender::_any_receiver_ref_t(std::addressof(_rcvr_))))
      {}

      constexpr void start() & noexcept
      {
        _rcvr_._register_callback();
        _opstate_.start();
      }

     private:
      _state<Receiver, TargetStopToken> _rcvr_;
      _any_opstate_base                 _opstate_;
    };

    template <class, class>
    struct _any_schedule_sender;

    //////////////////////////////////////////////////////////////////////////////////////
    // _isender
    template <class AnyReceiver, class SenderQueries>
    struct _isender;

    template <class... Sigs, class... Queries, class... SenderQueries>
    struct _isender<any_receiver<completion_signatures<Sigs...>, queries<Queries...>>,
                    queries<SenderQueries...>>
    {
      using _extends_t =
        STDEXEC::__any::__extends<STDEXEC::__any::__imovable,
                                  _iquery_memfn<SenderQueries>::template _interface_...>;

      template <class Base>
      struct _interface_ : STDEXEC::__any::__interface_base<_interface_, Base, _extends_t>
      {
       private:
        template <class, class>
        friend struct _any_opstate;
        template <class, class>
        friend struct _any_schedule_sender;
        using _ireceiver_t        = _ireceiver<completion_signatures<Sigs...>, queries<Queries...>>;
        using _any_receiver_ref_t = __pointer_receiver<typename _ireceiver_t::_pointer_t>;
        using _base_t             = STDEXEC::__any::__interface_base<_interface_, Base, _extends_t>;
       public:
        using sender_concept = STDEXEC::sender_tag;
        using _base_t::_base_t;

        template <__std::derived_from<_interface_> Self, class... Env>
        static consteval auto get_completion_signatures()
        {
          // throw if Env does not contain the queries needed to type-erase the receiver:
          using _check_queries_t = __mfind_error<_check_query_t<Queries, Env...>...>;
          if constexpr (__merror<_check_queries_t>)
            return STDEXEC::__throw_compile_time_error(_check_queries_t{});
          else
            return completion_signatures<Sigs...>{};
        }

        [[nodiscard]]
        virtual constexpr auto connect(_any_receiver_ref_t _rcvr) && -> _any_opstate_base
        {
          STDEXEC_ASSERT(Base::__box_kind != STDEXEC::__any::__box_kind::__abstract);

          if constexpr (Base::__box_kind == STDEXEC::__any::__box_kind::__abstract)
            __std::unreachable();
          else if constexpr (Base::__box_kind == STDEXEC::__any::__box_kind::__proxy)
            // The result of the call to __value(*this) below is a reference to a
            // polymophic sender. If we pass that to STDEXEC::connect, it will attempt to
            // transform that sender, which will cause it to be sliced. Instead, we call
            // .connect(_rcvr) directly on the contained value. transform_sender gets
            // called when the next branch is taken, which will happen as a result of the
            // call to .connect(_rcvr) in this branch.
            return STDEXEC::__any::__value(std::move(*this)).connect(std::move(_rcvr));
          else
            return _any_opstate_base{__in_place_from,
                                     STDEXEC::connect,
                                     STDEXEC::__any::__value(std::move(*this)),
                                     std::move(_rcvr)};
        }

        [[nodiscard]]
        constexpr auto get_env() const noexcept -> _interface_ const &
        {
          return *this;
        }
      };
    };

    //////////////////////////////////////////////////////////////////////////////////////
    // _ischeduler
    template <class AnySender, class SchedulerQueries>
    struct _ischeduler;

    template <class... Sigs, class Queries, class SenderQueries, class... SchedulerQueries>
    struct _ischeduler<
      any_sender<any_receiver<completion_signatures<Sigs...>, Queries>, SenderQueries>,
      queries<SchedulerQueries...>>
    {
     private:
      static_assert(STDEXEC::__one_of<set_value_t(), Sigs...>,
                    "any_scheduler requires set_value_t() in the completion signatures");

      using _any_receiver_t = any_receiver<completion_signatures<Sigs...>, Queries>;
      using _isender_t      = _any::_isender<_any_receiver_t, SenderQueries>;
      using _any_sender_t   = STDEXEC::__any::__any<_isender_t::template _interface_>;
      using _extends_t =
        STDEXEC::__any::__extends<STDEXEC::__any::__isemiregular,
                                  _iquery_memfn<SchedulerQueries, false>::template _interface_...>;
     public:
      template <class Base>
      struct _interface_
        : STDEXEC::__any::__interface_base<_interface_, _inject_query_memfn_t<Base>, _extends_t>
      {
       private:
        using _base_t =
          STDEXEC::__any::__interface_base<_interface_, _inject_query_memfn_t<Base>, _extends_t>;
       public:
        using scheduler_concept = STDEXEC::scheduler_tag;
        using _base_t::_base_t;

        [[nodiscard]]
        virtual constexpr auto schedule() const -> _any_sender_t
        {
          return _any_sender_t{STDEXEC::schedule(STDEXEC::__any::__value(*this))};
        }
      };
    };

    // Adds the get_completion_scheduler_t<set_value_t> query to the type-erased sender's
    // attributes. This type is used for the return of any_scheduler::schedule.
    template <class AnyScheduler, class AnySender>
    struct _any_schedule_sender final : AnySender
    {
     private:
      using _any_receiver_ref_t = AnySender::_any_receiver_ref_t;
      using _stop_token_t       = STDEXEC::stop_token_of_t<STDEXEC::env_of_t<_any_receiver_ref_t>>;

     public:
      constexpr explicit _any_schedule_sender(AnyScheduler _sched)
        : AnySender(STDEXEC::schedule(STDEXEC::__any::__value(_sched)))
        , _sched_env_{static_cast<AnyScheduler &&>(_sched)}
      {}

      template <class Receiver>
      [[nodiscard]]
      constexpr auto connect(Receiver _rcvr) && -> _any_opstate<Receiver, _stop_token_t>
      {
        using _opstate_t = _any_opstate<Receiver, _stop_token_t>;
        return _opstate_t{static_cast<AnySender &&>(*this), static_cast<Receiver &&>(_rcvr)};
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept
      {
        return STDEXEC::__env::__join(_sched_env_, AnySender::get_env());
      }

     private:
      struct _sched_env_t
      {
        [[nodiscard]]
        constexpr auto
        query(STDEXEC::get_completion_scheduler_t<STDEXEC::set_value_t>) const noexcept
          -> AnyScheduler
        {
          return _sched;
        }

        AnyScheduler _sched;
      };
      _sched_env_t _sched_env_;
    };

    template <class Sigs, class Queries = queries<>>
    struct _any_receiver_ref
      : __pointer_receiver<typename _any::_ireceiver<Sigs, Queries>::_pointer_t>
    {
      template <__not_decays_to<_any_receiver_ref> Receiver>
        requires STDEXEC::receiver_of<Receiver, Sigs>
      constexpr _any_receiver_ref(Receiver &_rcvr) noexcept
        : _any_receiver_ref::__pointer_receiver(std::addressof(_rcvr))
      {}
    };
  }  // namespace _any

  ////////////////////////////////////////////////////////////////////////////////////////
  // any_receiver
  template <class Sigs, class Queries>
  struct any_receiver final
    : STDEXEC::__any::__any<_any::_ireceiver<Sigs, Queries>::template _interface_>
  {
    template <STDEXEC::__not_same_as<any_receiver> Receiver>
      requires STDEXEC::receiver_of<Receiver, Sigs>
    constexpr any_receiver(Receiver _rcvr)
      noexcept(STDEXEC::__nothrow_constructible_from<typename any_receiver::__any, Receiver>)
      : any_receiver::__any{static_cast<Receiver &&>(_rcvr)}
    {}

   private:
    template <class, class>
    friend struct any_sender;
    using _completions_t = Sigs;
    using _queries_t     = Queries;
  };

  //////////////////////////////////////////////////////////////////////////////////////
  // any_sender
  template <class AnyReceiver, class SenderQueries>
  struct any_sender final
    : STDEXEC::__any::__any<_any::_isender<AnyReceiver, SenderQueries>::template _interface_>
  {
   private:
    using _completions_t = AnyReceiver::_completions_t;
    using _stop_token_t  = STDEXEC::stop_token_of_t<STDEXEC::env_of_t<AnyReceiver>>;
    using _isender_t     = _any::_isender<AnyReceiver, SenderQueries>;
    using _base_t        = STDEXEC::__any::__any<_isender_t::template _interface_>;

   public:
    template <STDEXEC::__not_same_as<any_sender> Sender>
      requires STDEXEC::sender_to<Sender, AnyReceiver>
    constexpr any_sender(Sender _sndr)
      noexcept(STDEXEC::__nothrow_constructible_from<_base_t, Sender>)
      : _base_t{static_cast<Sender &&>(_sndr)}
    {}

    template <class Receiver>
    constexpr auto connect(Receiver _rcvr) && -> _any::_any_opstate<Receiver, _stop_token_t>
    {
      static_assert(STDEXEC::receiver_of<Receiver, _completions_t>);
      using _opstate_t = _any::_any_opstate<Receiver, _stop_token_t>;
      return _opstate_t{static_cast<_base_t &&>(*this), static_cast<Receiver &&>(_rcvr)};
    }
  };

  //////////////////////////////////////////////////////////////////////////////////////
  // any_scheduler
  template <class AnyReceiver, class SenderQueries, class SchedulerQueries>
  struct any_scheduler<any_sender<AnyReceiver, SenderQueries>, SchedulerQueries> final
    : STDEXEC::__any::__any<_any::_ischeduler<any_sender<AnyReceiver, SenderQueries>,
                                              SchedulerQueries>::template _interface_>
  {
   private:
    using _any_sender_t      = any_sender<AnyReceiver, SenderQueries>;
    using _isender_t         = _any::_isender<AnyReceiver, SenderQueries>;
    using _any_sender_base_t = STDEXEC::__any::__any<_isender_t::template _interface_>;
    using _ischeduler_t      = _any::_ischeduler<_any_sender_t, SchedulerQueries>;
    using _schedule_sender_t = _any::_any_schedule_sender<any_scheduler, _any_sender_base_t>;
    using _base_t            = STDEXEC::__any::__any<_ischeduler_t::template _interface_>;

   public:
    using _base_t::_base_t;
    using _base_t::query;

    template <std::same_as<STDEXEC::__none_such> = STDEXEC::__none_such>
    [[nodiscard]]
    constexpr auto schedule() const -> _schedule_sender_t
    {
      return _schedule_sender_t(*this);
    }

    [[nodiscard]]
    constexpr auto
    query(STDEXEC::get_completion_scheduler_t<STDEXEC::set_value_t>) const noexcept -> any_scheduler
    {
      return *this;
    }
  };

  ////////////////////////////////////////////////////////////////////////////////////////
  // Legacy interfaces for type-erased senders and receivers.

  namespace _any
  {
    template <class QueryPtr>
    extern STDEXEC::__undefined<QueryPtr> _query_sig_v;

    template <class Query, class Result, class... Args>
    inline constexpr STDEXEC::__midentity<Result (*)(Query, Args...)>
      _query_sig_v<Query (*)(Result (*)(Args...))> = nullptr;

    template <class Query, class Result, class... Args>
    inline constexpr STDEXEC::__midentity<Result (*)(Query, Args...) noexcept>
      _query_sig_v<Query (*)(Result (*)(Args...) noexcept)> = nullptr;

    template <auto QueryPtr>
    using _query_sig_t = std::remove_pointer_t<decltype(_query_sig_v<decltype(QueryPtr)>)>;

    template <auto... Sigs>
    using _queries_t = queries<_query_sig_t<Sigs>...>;
  }  // namespace _any

  template <class Completions, auto... Queries>
  struct any_receiver_ref;

  template <class... Sigs, auto... Queries>
  struct [[deprecated("use exec::any_receiver<Sigs, ReceiverQueries> instead")]]
  any_receiver_ref<STDEXEC::completion_signatures<Sigs...>, Queries...>
    : _any::_any_receiver_ref<STDEXEC::completion_signatures<Sigs...>, _any::_queries_t<Queries...>>
  {
   private:
    using _any_receiver_t =
      any_receiver<STDEXEC::completion_signatures<Sigs...>, _any::_queries_t<Queries...>>;
    using _base_t = _any::_any_receiver_ref<STDEXEC::completion_signatures<Sigs...>,
                                            _any::_queries_t<Queries...>>;
   public:
    template <STDEXEC::__not_decays_to<any_receiver_ref> Receiver>
      requires STDEXEC::receiver_of<Receiver, STDEXEC::completion_signatures<Sigs...>>
    constexpr any_receiver_ref(Receiver &_rcvr) noexcept
      : _base_t{static_cast<Receiver &>(_rcvr)}
    {}

    template <auto... SenderQueries>
    struct [[deprecated("use exec::any_sender<exec::any_receiver<Sigs, ReceiverQueries>, "
                        "SenderQueries> instead")]] any_sender final
      : STDEXEC::__any::__any<
          _any::_isender<_any_receiver_t, _any::_queries_t<SenderQueries...>>::template _interface_>
    {
      using _stop_token_t = STDEXEC::stop_token_of_t<STDEXEC::env_of_t<_any_receiver_t>>;
      using _isender_t    = _any::_isender<_any_receiver_t, _any::_queries_t<SenderQueries...>>;
      using _any_sender_t = exec::any_sender<_any_receiver_t, _any::_queries_t<SenderQueries...>>;
      using _base_t       = STDEXEC::__any::__any<_isender_t::template _interface_>;
      using _base_t::_base_t;

      template <STDEXEC::receiver_of<STDEXEC::completion_signatures<Sigs...>> Receiver>
      constexpr auto connect(Receiver _rcvr) && -> _any::_any_opstate<Receiver, _stop_token_t>
      {
        using _opstate_t = _any::_any_opstate<Receiver, _stop_token_t>;
        return _opstate_t{static_cast<_base_t &&>(*this), static_cast<Receiver &&>(_rcvr)};
      }

      template <auto... SchedulerQueries>
      struct [[deprecated("use exec::any_scheduler<exec::any_sender<exec::any_receiver<Sigs, "
                          "ReceiverQueries>, SenderQueries>, SchedulerQueries> instead")]]
      any_scheduler final
        : STDEXEC::__any::__any<
            _any::_ischeduler<_any_sender_t,
                              _any::_queries_t<SchedulerQueries...>>::template _interface_>
      {
        using _ischeduler_t =
          _any::_ischeduler<_any_sender_t, _any::_queries_t<SchedulerQueries...>>;
        using _any_sender_base_t = STDEXEC::__any::__any<_isender_t::template _interface_>;
        using _schedule_sender_t = _any::_any_schedule_sender<any_scheduler, _any_sender_base_t>;
        using _base_t            = STDEXEC::__any::__any<_ischeduler_t::template _interface_>;
        using _base_t::_base_t;
        using _base_t::query;

        template <std::same_as<STDEXEC::__none_such> = STDEXEC::__none_such>
        [[nodiscard]]
        constexpr auto schedule() const -> _schedule_sender_t
        {
          return _schedule_sender_t(*this);
        }

        [[nodiscard]]
        constexpr auto
        query(STDEXEC::get_completion_scheduler_t<STDEXEC::set_value_t>) const noexcept
          -> any_scheduler
        {
          return *this;
        }
      };
    };
  };
}  // namespace experimental::execution

STDEXEC_PRAGMA_POP()

namespace exec = experimental::execution;
