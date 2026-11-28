/*
 * Copyright (c) 2025 NVIDIA Corporation
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

#include "../stdexec/__detail/__storage.hpp"
#include "../stdexec/execution.hpp"

#include <exception>

namespace experimental::execution
{
  struct fork_join_t;

  struct INVALID_ARGUMENTS_TO_FORK_JOIN;

  namespace _fork_join
  {
    struct _mk_when_all_fn
    {
      template <class CacheSndr, class... Closures>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr auto operator()(CacheSndr sndr, Closures&&... closures) const
        noexcept(noexcept(STDEXEC::when_all(static_cast<Closures&&>(closures)(sndr)...)))
      {
        return STDEXEC::when_all(static_cast<Closures&&>(closures)(sndr)...);
      }
    };

    template <class Completions,
              bool = STDEXEC::__nothrow_decay_copyable_results_t<Completions>::value>
    struct _cache_completions
    {
      using type = Completions;
    };

    template <class Completions>
    struct _cache_completions<Completions, false>
    {
      using type =
        STDEXEC::__concat_completion_signatures_t<Completions, STDEXEC::__eptr_completion_t>;
    };

    // Account that decay-copying into the cache may throw.
    template <class Completions>
    using _variant_t = STDEXEC::__mapply_q<STDEXEC::__results_storage,
                                           typename _cache_completions<Completions>::type>;

    // The attributes of the cache sender, which completes inline, in start, with the cached
    // results
    struct _inline_attrs
    {
      template <class Tag>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto query(STDEXEC::__get_completion_behavior_t<Tag>) noexcept
      {
        return STDEXEC::__completion_behavior::__inline_completion;
      }
    };

    template <class Domain>
    struct _env_t : _inline_attrs
    {
      using _inline_attrs::query;

      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto query(STDEXEC::get_domain_t) noexcept -> Domain
      {
        return {};
      }
    };

    template <>
    struct _env_t<STDEXEC::indeterminate_domain<>> : _inline_attrs
    {};

    template <class Tag, class... Args>
    using _cref_sig_t = Tag(STDEXEC::__decay_t<Args> const &...);

    // Given a set of async results, each of the form `tuple<Tag, Args...>`,
    // compute the corresponding completion signatures, where each signature is
    // of the form `Tag(const Args&...)`.
    template <class... AsyncResults>
    using _cache_sndr_completions_t =
      STDEXEC::completion_signatures<STDEXEC::__mapply<STDEXEC::__q<_cref_sig_t>, AsyncResults>...>;

    template <class Rcvr, class Variant>
    struct _cache_opstate
    {
      using operation_state_concept = STDEXEC::operation_state_tag;

      STDEXEC_ATTRIBUTE(host, device) void start() noexcept
      {
        std::as_const(*_results_).__complete(_rcvr_);
      }

      Rcvr            _rcvr_;
      Variant const * _results_;
    };

    template <class Variant, class Domain>
    struct _cache_sndr
    {
      using sender_concept = STDEXEC::sender_tag;

      template <class _Self, class... _Env>
      STDEXEC_ATTRIBUTE(host, device)
      static consteval auto get_completion_signatures()
      {
        return STDEXEC::__mapply<STDEXEC::__qq<_cache_sndr_completions_t>, Variant>{};
      }

      template <class Rcvr>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr auto connect(Rcvr rcvr) const noexcept -> _cache_opstate<Rcvr, Variant>
      {
        return _cache_opstate<Rcvr, Variant>{static_cast<Rcvr&&>(rcvr), _results_};
      }

      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto get_env() noexcept -> _env_t<Domain>
      {
        return {};
      }

      Variant const * _results_;
    };

    template <class Completions, class Closures, class Domain>
    using _when_all_sndr_t =
      STDEXEC::__apply_result_t<_mk_when_all_fn,
                                Closures,
                                _cache_sndr<_variant_t<Completions>, Domain>>;

    // The key of the operation of when_all(closures(cache)...)
    struct _fork_op
    {};

    struct _algo;

    template <class Sndr, class... Env>
    using _child_domain_t =
      STDEXEC::__completion_domain_of_t<STDEXEC::set_value_t,
                                        STDEXEC::__facade::__nth_child_of_t<Sndr, 0>,
                                        STDEXEC::__fwd_env_t<Env>...>;

    // The operation of a sender that cannot be connected to the receiver it is given. (The
    // walk asks the sender for its completions, and reports its error.)
    struct _no_op
    {};

    // The type of an operation that the state holds: the operation, or _no_op, or, if the
    // state is for a walk (its receiver is an archetype), a placeholder, since the walk needs
    // only the state's type. Naming the operation there would instantiate the operations
    // below it, connected to archetypes. (A class keyed by bools only, so that the debug
    // information, which names the class when it names the operation's type, does not repeat
    // the sender's and the receiver's types.)
    template <bool Walking, bool Connectable>
    struct _op_of
    {
      template <class Sender, class Receiver>
      using __f = STDEXEC::connect_result_t<Sender, Receiver>;
    };

    template <bool Connectable>
    struct _op_of<true, Connectable>
    {
      template <class, class>
      using __f = STDEXEC::__facade::__walk_op;
    };

    template <>
    struct _op_of<false, false>
    {
      template <class, class>
      using __f = _no_op;
    };

    // Can the operations be connected? A walk does not ask: its state holds placeholders.
    template <bool Walking>
    struct _connectable_of
    {
      template <class Child, class ChildRcvr, class Fork, class ForkRcvr>
      static consteval auto __f() noexcept -> bool
      {
        return STDEXEC::sender_to<Child, ChildRcvr> && STDEXEC::sender_to<Fork, ForkRcvr>;
      }
    };

    template <>
    struct _connectable_of<true>
    {
      template <class, class, class, class>
      static consteval auto __f() noexcept -> bool
      {
        return true;
      }
    };

    template <class Sndr, class Rcvr>
    using __child_completions_of_t =
      STDEXEC::__facade::__completions_of_t<_algo, STDEXEC::__facade::__child<0>, Sndr, Rcvr>;

    template <class Sndr, class Rcvr>
    struct _state
    {
      using _child_t             = STDEXEC::__facade::__nth_child_of_t<Sndr, 0>;
      using _closures_t          = STDEXEC::__facade::__data_of_t<Sndr>;
      using _child_completions_t = __child_completions_of_t<Sndr, Rcvr>;
      using _domain_t            = _child_domain_t<Sndr, STDEXEC::env_of_t<Rcvr>>;
      using _cache_t             = _variant_t<_child_completions_t>;
      using _cache_sndr_t        = _fork_join::_cache_sndr<_cache_t, _domain_t>;
      using _when_all_sndr_t =
        STDEXEC::__apply_result_t<_mk_when_all_fn, _closures_t, _cache_sndr_t>;
      using _child_rcvr_t =
        STDEXEC::__facade::__receiver_t<_algo, STDEXEC::__facade::__child<0>, _state, Sndr, Rcvr>;
      using _fork_rcvr_t = STDEXEC::__facade::__receiver_t<_algo, _fork_op, _state, Sndr, Rcvr>;
      using _env2_t      = decltype(STDEXEC::__trnsfr::__mk_env2(
        STDEXEC::__declval<STDEXEC::__decay_t<_child_t> const &>(),
        STDEXEC::__declval<STDEXEC::__decay_t<STDEXEC::env_of_t<Rcvr>> const &>()));
      static constexpr bool _child_stays_put =
        STDEXEC::__trnsfr::__child_stays_put_v<_child_t, STDEXEC::env_of_t<Rcvr>>;

      // (A walk's state holds placeholders: see _op_of.)
      using _connectable_of_t = _connectable_of<STDEXEC::__facade::__walking_receiver<Rcvr>>;
      static constexpr bool _connectable =
        _connectable_of_t::template __f<_child_t, _child_rcvr_t, _when_all_sndr_t, _fork_rcvr_t>();
      using _op_of_t    = _op_of<STDEXEC::__facade::__walking_receiver<Rcvr>, _connectable>;
      using _child_op_t = _op_of_t::template __f<_child_t, _child_rcvr_t>;
      using _fork_op_t  = _op_of_t::template __f<_when_all_sndr_t, _fork_rcvr_t>;

      static constexpr bool _nothrow_v =
        STDEXEC::__nothrow_applicable<_mk_when_all_fn, _closures_t, _cache_sndr_t>
        && STDEXEC::__nothrow_connectable<_when_all_sndr_t, _fork_rcvr_t>
        && STDEXEC::__nothrow_connectable<_child_t, _child_rcvr_t>;

      // The forked operations are connected first, to the cache. The child's operation is
      // destroyed when it completes.
      STDEXEC_ATTRIBUTE(host, device)
      constexpr explicit _state(Sndr&& sndr, Rcvr&& rcvr) noexcept(_nothrow_v)
        : __rcvr_(static_cast<Rcvr&&>(rcvr))
        , __env2_(STDEXEC::__trnsfr::__mk_env2(STDEXEC::__facade::__get_child<0>(sndr),
                                               STDEXEC::get_env(__rcvr_)))
        , _fork_op_(_state::_connect_fork(static_cast<Sndr&&>(sndr), this))
      {
        if constexpr (_connectable && !STDEXEC::__facade::__walking_receiver<Rcvr>)
        {
          _child_op_.__emplace_from(STDEXEC::connect,
                                    STDEXEC::__facade::__get_child<0>(static_cast<Sndr&&>(sndr)),
                                    _child_rcvr_t{this});
        }
      }

      STDEXEC_IMMOVABLE(_state);

      Rcvr                             __rcvr_;
      _env2_t const                    __env2_;
      _cache_t                         _cache_{};
      STDEXEC::__optional<_child_op_t> _child_op_{};
      _fork_op_t                       _fork_op_;

     private:
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto
      _connect_fork(Sndr&& sndr, _state* self) noexcept(_nothrow_v) -> _fork_op_t
      {
        if constexpr (_connectable && !STDEXEC::__facade::__walking_receiver<Rcvr>)
        {
          return STDEXEC::connect(STDEXEC::__apply(_mk_when_all_fn{},
                                                   STDEXEC::__facade::__get_data(
                                                     static_cast<Sndr&&>(sndr)),
                                                   _cache_sndr_t{&self->_cache_}),
                                  _fork_rcvr_t{self});
        }
        else
        {
          return {};
        }
      }
    };

    // fork_join(sndr, closures...): when the child completes, its results are cached, its
    // operation is destroyed, and when_all(closures(cache)...) is started, where cache is a
    // sender of the cached results (by const reference). Its completions are the operation's.
    struct _algo : STDEXEC::__facade::__defaults
    {
      static constexpr bool __connects_children = false;

      // The domain of the child's value completions, in which the closures' senders look for
      // customizations, cannot always be known without an environment.
      template <class Sndr>
      static constexpr bool __depends_on_env =
        !STDEXEC::__minvocable_q<_child_domain_t, STDEXEC::__decay_t<Sndr>>;

      using STDEXEC::__facade::__defaults::__env_for;
      using STDEXEC::__facade::__defaults::__get_env;

      // The child sees the forwarded environment of the receiver.
      template <class Sndr, class Env>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __env_for(STDEXEC::__facade::__child<0>,
                                      Sndr const &,
                                      Env&& env) noexcept -> STDEXEC::__fwd_env_t<Env>
      {
        return STDEXEC::__fwd_env(static_cast<Env&&>(env));
      }

      template <class State>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __get_env(STDEXEC::__facade::__child<0>, State const & state) noexcept
        -> decltype(auto)
      {
        return STDEXEC::__fwd_env(STDEXEC::get_env(state.__rcvr_));
      }

      // The forked operations start where the child completed, which they see (see
      // __trnsfr::__mk_env2). Unless the child completes where it starts, that is not where the
      // operation started, so the receiver's start scheduler and domain are hidden from them.
      // (The cache sender completes inline, where it starts.)
      template <class Sndr, class Env>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __env_for(_fork_op, Sndr const & sndr, Env&& env) noexcept
      {
        auto env2 = STDEXEC::__trnsfr::__mk_env2(STDEXEC::__facade::__get_child<0>(sndr), env);
        if constexpr (STDEXEC::__trnsfr::__child_stays_put_v<
                        STDEXEC::__facade::__nth_child_of_t<Sndr, 0>,
                        Env>)
          return STDEXEC::__env::__join(std::move(env2), static_cast<Env&&>(env));
        else
          return STDEXEC::__env::__join(std::move(env2),
                                        STDEXEC::__hide_scheduler{static_cast<Env&&>(env)});
      }

      template <class State>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __get_env(_fork_op, State const & state) noexcept
      {
        using env2_t = STDEXEC::__decay_t<decltype(state.__env2_)>;
        if constexpr (State::_child_stays_put)
          return STDEXEC::__env::__join(env2_t(state.__env2_), STDEXEC::get_env(state.__rcvr_));
        else
          return STDEXEC::__env::__join(env2_t(state.__env2_),
                                        STDEXEC::__hide_scheduler{STDEXEC::get_env(state.__rcvr_)});
      }

      // Customizations are found in the domain where the child completes with a value, in
      // which the closures' senders look for theirs.
      using __dispatch_source = STDEXEC::__facade::__via<STDEXEC::__facade::__child<0>,
                                                         STDEXEC::set_value_t,
                                                         STDEXEC::__facade::__from_start>;

      using STDEXEC::__facade::__defaults::__nested_sender;

      // when_all(closures(cache)...). The operation's cache sender points into its state. The
      // one here has no cache: this sender is only asked for its attributes. (If the closures
      // cannot be applied to the sender's data without throwing, or give another sender that
      // way, only the type is known.)
      template <class Sndr, class... Env>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __nested_sender(_fork_op, Sndr const & sndr, Env const &...) noexcept
      {
        using completions_t =
          STDEXEC::__completion_signatures_of_t<STDEXEC::__facade::__nth_child_of_t<Sndr, 0>,
                                                STDEXEC::__fwd_env_t<Env>...>;
        using cache_sndr_t =
          _fork_join::_cache_sndr<_variant_t<completions_t>, _child_domain_t<Sndr, Env...>>;
        using when_all_t = _when_all_sndr_t<completions_t,
                                            STDEXEC::__facade::__data_of_t<Sndr>,
                                            _child_domain_t<Sndr, Env...>>;
        using data_t     = decltype(STDEXEC::__facade::__get_data(sndr));
        if constexpr (STDEXEC::__nothrow_applicable<_mk_when_all_fn, data_t, cache_sndr_t>)
        {
          if constexpr (STDEXEC::__same_as<
                          STDEXEC::__apply_result_t<_mk_when_all_fn, data_t, cache_sndr_t>,
                          when_all_t>)
            return STDEXEC::__apply(_mk_when_all_fn{},
                                    STDEXEC::__facade::__get_data(sndr),
                                    cache_sndr_t{nullptr});
          else
            return STDEXEC::__mtype<when_all_t>();
        }
        else
        {
          return STDEXEC::__mtype<when_all_t>();
        }
      }

      // The _StateT template-template parameter exists so that the __state_type
      // alias below can compute the return type of __get_state without needing
      // to know if it throws.
      template <template <class...> class _StateT = _state, class Sndr, class Rcvr>
        requires STDEXEC::__minvocable_q<_child_domain_t, Sndr, STDEXEC::env_of_t<Rcvr>>
              && STDEXEC::__decay_copyable_results_t<__child_completions_of_t<Sndr, Rcvr>>::value
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __get_state(Sndr&& sndr, Rcvr rcvr)
        noexcept(STDEXEC::__nothrow_constructible_from<_StateT<Sndr, Rcvr>, Sndr, Rcvr>)
          -> _state<Sndr, Rcvr>
      {
        return _state<Sndr, Rcvr>(static_cast<Sndr&&>(sndr), static_cast<Rcvr&&>(rcvr));
      }

      template <template <class...> class _StateT = _state, class Sndr, class Rcvr>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __get_state(Sndr&&, Rcvr) noexcept
      {
        using namespace STDEXEC;
        if constexpr (__minvocable_q<_child_domain_t, Sndr, env_of_t<Rcvr>>)
        {
          return __mexception<_WHAT_(_PREDECESSOR_RESULTS_ARE_NOT_DECAY_COPYABLE_),
                              _IN_ALGORITHM_(fork_join_t)>();
        }
        else
        {
          return __mexception<INVALID_ARGUMENTS_TO_FORK_JOIN,
                              _WITH_SENDERS_<__facade::__nth_child_of_t<Sndr, 0>>,
                              _WITH_ENVIRONMENT_(env_of_t<Rcvr>)>();
        }
      }

      // The type that __get_state returns, named without asking whether it can
      // throw (see __facade::__state_type_t).
      template <class Sndr, class Rcvr>
      using __state_type = decltype(_algo::__get_state<STDEXEC::__mconst<STDEXEC::__ignore>::__f>(
        STDEXEC::__declval<Sndr>(),
        STDEXEC::__declval<Rcvr>()));

      template <class State, class Children>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __start(State& state, Children&) noexcept
      {
        if constexpr (State::_connectable)
          return STDEXEC::__facade::__start_nested<STDEXEC::__facade::__child<0>>(
            *state._child_op_);
        else
          return STDEXEC::__facade::__effects<
            STDEXEC::__facade::__starts<STDEXEC::__facade::__child<0>>>();
      }

      using STDEXEC::__facade::__defaults::__complete;

      // The child completed: cache its results (or, if decay-copying them throws, the
      // exception, as an error), destroy its operation, and start the forked operations.
      template <class State, class Tag, class... Args>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __complete(STDEXEC::__facade::__child<0>,
                                       State& state,
                                       Tag    tag,
                                       Args&&... args) noexcept
      {
        if constexpr (STDEXEC::__nothrow_decay_copyable<Args...>)
        {
          state._cache_.__emplace_from(STDEXEC::__mktuple, tag, static_cast<Args&&>(args)...);
        }
        else
        {
          STDEXEC_TRY
          {
            state._cache_.__emplace_from(STDEXEC::__mktuple, tag, static_cast<Args&&>(args)...);
          }
          STDEXEC_CATCH_ALL
          {
            state._cache_.__emplace_from(STDEXEC::__mktuple,
                                         STDEXEC::set_error,
                                         ::std::current_exception());
          }
        }
        state._child_op_.reset();
        if constexpr (State::_connectable)
          return STDEXEC::__facade::__start_nested<_fork_op>(state._fork_op_);
        else
          return STDEXEC::__facade::__effects<STDEXEC::__facade::__starts<_fork_op>>();
      }
    };
  }  // namespace _fork_join

  struct fork_join_t
  {
    /// No closure given.
    template <STDEXEC::sender Sndr>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto operator()(Sndr&& sndr) const noexcept(STDEXEC::__nothrow_decay_copyable<Sndr>)
    {
      return static_cast<Sndr&&>(sndr);
    }

    /// Unary closure.
    template <STDEXEC::sender Sndr, class Closure>
      requires(!STDEXEC::sender<Closure>)
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto operator()(Sndr&& sndr, Closure&& clsr) const
      noexcept(STDEXEC::__nothrow_callable<Closure, Sndr>)
    {
      return static_cast<Closure&&>(clsr)(static_cast<Sndr&&>(sndr));
    }

    /// One sender and multiple closures.
    template <STDEXEC::sender Sndr, class... Closures>
      requires(sizeof...(Closures) > 1)
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto operator()(Sndr&& sndr, Closures&&... closures) const
      noexcept(STDEXEC::__nothrow_decay_copyable<Sndr, Closures...>) -> auto
    {
      return STDEXEC::__check_well_formed(
        STDEXEC::__facade::__sender{fork_join_t(),
                                    STDEXEC::__tuple{static_cast<Closures&&>(closures)...},
                                    static_cast<Sndr&&>(sndr)});
    }

    /// One or more closures.
    template <class... Closures>
      requires((!STDEXEC::sender<Closures>) && ...)
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto operator()(Closures&&... closures) const
    {
      return STDEXEC::__closure{*this, static_cast<Closures&&>(closures)...};
    }
  };

  inline constexpr fork_join_t fork_join{};

}  // namespace experimental::execution

namespace exec = experimental::execution;

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  template <>
  struct __facade::__algorithm<exec::fork_join_t> : exec::_fork_join::_algo
  {};
}  // namespace STDEXEC
