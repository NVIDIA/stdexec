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

#include "../../stdexec/__detail/__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "../../stdexec/__detail/__atomic.hpp"
#include "../../stdexec/__detail/__env.hpp"
#include "../../stdexec/__detail/__memory.hpp"
#include "../../stdexec/__detail/__meta.hpp"
#include "../../stdexec/__detail/__queries.hpp"
#include "../../stdexec/__detail/__receivers.hpp"
#include "../../stdexec/__detail/__sender_facade.hpp"
#include "../../stdexec/__detail/__storage.hpp"
#include "../../stdexec/__detail/__tuple.hpp"
#include "../../stdexec/__detail/__variant.hpp"  // IWYU pragma: keep
#include "../../stdexec/stop_token.hpp"

#include "../completion_signatures.hpp"
#include "../stoppable_base.hpp"
#include "intrusive_slist.hpp"

#include <exception>
#include <mutex>
#include <type_traits>  // IWYU pragma: keep
#include <utility>

////////////////////////////////////////////////////////////////////////////////
// shared components of split and ensure_started
//
// The split and ensure_started algorithms are very similar in implementation.
// The salient differences are:
//
// split: the input async operation is always connected. It is only
//   started when one of the split senders is connected and started.
//   split senders are copyable, so there are multiple operation states
//   to be notified on completion. These are stored in an intrusive
//   linked list.
//
// ensure_started: the input async operation is always started, so
//   the internal receiver will always be completed. The ensure_started
//   sender is move-only and single-shot, so there will only ever be one
//   operation state to be notified on completion.
//
// The shared state should add-ref itself when the input async
// operation is started and release itself when its completion
// is notified.
//
// The senders that split and ensure_started return are facade senders (see
// __sender_facade.hpp). Their operations receive the shared operation's results
// through the nested operation __shared_op, or from start, if the shared operation
// has already completed; and a stop request through the nested operation
// __stop_request. So their completion attributes are honest: an operation that starts
// after the shared operation completes completes where it starts, and one that waits
// completes where the shared operation does.
namespace experimental::execution
{
  struct split_t;
  struct ensure_started_t;
}  // namespace experimental::execution

namespace exec = experimental::execution;

namespace experimental::execution::__shared
{
  using namespace STDEXEC;

  template <class _Env, class _Variant>
  struct __shared_state_base;

  template <class _CvSender, class _Env>
  struct __shared_state;

  template <class _Env>
  using __env_t = __join_env_t<prop<get_stop_token_t, inplace_stop_token>, _Env>;

  //////////////////////////////////////////////////////////////////////////////
  template <class _Env, class _Variant>
  struct __receiver
  {
    using receiver_concept   = receiver_tag;
    using __child_receiver_t = void;  // see STDEXEC::__child_receiver
    template <class... _As>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr void set_value(_As &&...__as) noexcept
    {
      __sh_state_->__complete(set_value_t(), static_cast<_As &&>(__as)...);
    }

    template <class _Error>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr void set_error(_Error &&__err) noexcept
    {
      __sh_state_->__complete(set_error_t(), static_cast<_Error &&>(__err));
    }

    STDEXEC_ATTRIBUTE(always_inline)
    constexpr void set_stopped() noexcept
    {
      __sh_state_->__complete(set_stopped_t());
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> __env_t<_Env> const &
    {
      return __sh_state_->__env_;
    }

    // The receiver does not hold a reference to the shared state.
    __shared_state_base<_Env, _Variant> *__sh_state_;
  };

  //////////////////////////////////////////////////////////////////////////////
  template <class _CvSender, class _Env>
  using __result_variant_t =
    __mapply<__mbind_front_q<__results_storage, set_stopped_t(), set_error_t(std::exception_ptr)>,
             __completion_signatures_of_t<_CvSender, _Env>>;

  //////////////////////////////////////////////////////////////////////////////
  template <class _CvChild, class _Env>
  [[nodiscard]]
  constexpr auto
  __mk_sh_state(_CvChild &&__child, _Env __env) -> std::shared_ptr<__shared_state<_CvChild, _Env>>
  {
    using __sh_state_t = __shared_state<_CvChild, _Env>;
    auto __tmp         = __with_default(get_allocator, std::allocator<void>{})(__env);
    auto __alloc       = STDEXEC::__rebind_allocator<__sh_state_t>(__tmp);
    return std::allocate_shared<__sh_state_t>(__alloc,
                                              static_cast<_CvChild &&>(__child),
                                              static_cast<_Env &&>(__env));
  }

  //////////////////////////////////////////////////////////////////////////////
  struct __local_state_base
  {
    using __notify_fn_t = void (*)(__local_state_base &) noexcept;

    constexpr void __notify() noexcept
    {
      __notify_fn_(*this);
    }

    __notify_fn_t       __notify_fn_ = nullptr;
    __local_state_base *__next_      = nullptr;
  };

  //////////////////////////////////////////////////////////////////////////////
  // The attributes of the shared operation, as the operations that wait for it see them:
  // those of the child, in its environment _Env, except that the shared operation
  // completes asynchronously to an operation that waits for it. (A domain whose operations
  // receive results in other ways can say so with _Behavior.) They are computed before the
  // child is connected (and so consumed).
  // (_Known is false when the scheduler is not the only place where the completions happen.)
  template <class _SetTag, class _Attrs, class _Env, bool _Known = true>
  struct __maybe_sched
  {
    static constexpr bool __has_sched = false;

    constexpr explicit __maybe_sched(_Attrs const &, _Env const &) noexcept {}
  };

  template <class _SetTag, class _Attrs, class _Env>
    requires __callable<get_completion_scheduler_t<_SetTag>, _Attrs const &, _Env const &>
  struct __maybe_sched<_SetTag, _Attrs, _Env, true>
  {
    static constexpr bool __has_sched = true;
    using __sched_t =
      __call_result_t<get_completion_scheduler_t<_SetTag>, _Attrs const &, _Env const &>;

    constexpr explicit __maybe_sched(_Attrs const &__attrs, _Env const &__env) noexcept
      : __sch_(get_completion_scheduler<_SetTag>(__attrs, __env))
    {}

    __sched_t __sch_;
  };

  template <class _Child, class _Env, class _Behavior = __completion_behavior::__asynchronous_t>
  struct __shared_attrs
  {
    using __child_attrs_t = env_of_t<_Child>;
    using __child_env_t   = _Env;  // the environment of the child

    // The operations receive the child's errors where the child sends them, and an
    // exception from copying the child's values (see __shared_state_base::__complete) where
    // the child sends those. So their errors complete on the child's error scheduler only if
    // copying its values cannot throw.
    static constexpr bool __nothrow_value_copies =
      __value_types_t<__completion_signatures_of_t<_Child, __child_env_t>,
                      __q<__nothrow_decay_copyable_t>,
                      __qq<__mand>>::value;

    template <class _SetTag>
    using __sched_of_t = __maybe_sched<_SetTag,
                                       __child_attrs_t,
                                       __child_env_t,
                                       !__same_as<_SetTag, set_error_t> || __nothrow_value_copies>;

    // The domain of the child's _SetTag completions, if it says
    template <class _SetTag>
    using __child_domain_t = __call_result_t<get_completion_domain_t<_SetTag>,
                                             __child_attrs_t const &,
                                             __child_env_t const &>;

    constexpr explicit __shared_attrs(__child_attrs_t const &__attrs,
                                      __child_env_t const   &__env) noexcept
      : __value_(__attrs, __env)
      , __error_(__attrs, __env)
      , __stopped_(__attrs, __env)
    {}

    // The attributes answer queries with an environment and without one (as a facade
    // sender's do). The answers do not depend on it: they are about the child, in its own
    // environment.
    template <__completion_tag _SetTag, class... _QEnv>
      requires(sizeof...(_QEnv) <= 1) && __sched_of_t<_SetTag>::__has_sched
    [[nodiscard]]
    constexpr auto query(get_completion_scheduler_t<_SetTag>, _QEnv const &...) const noexcept
    {
      if constexpr (__same_as<_SetTag, set_value_t>)
        return __value_.__sch_;
      else if constexpr (__same_as<_SetTag, set_error_t>)
        return __error_.__sch_;
      else
        return __stopped_.__sch_;
    }

    // A completion that has a scheduler completes in the scheduler's domain, which
    // get_completion_domain asks it for. Others complete in the domain that the child says,
    // if it says one. (If it does not, the domain that implements the child is the one in
    // which it completes; see get_completion_domain<>.)
    template <class _SetTag>
    using __read_domain_t = get_completion_domain_t<_SetTag>::__read_query_t;

    template <__completion_tag _SetTag, class... _QEnv>
      requires(sizeof...(_QEnv) <= 1) && (!__sched_of_t<_SetTag>::__has_sched)
           && (!__same_as<_SetTag, set_error_t> || __nothrow_value_copies)
           && __callable<__read_domain_t<_SetTag>, __child_attrs_t const &, __child_env_t const &>
    [[nodiscard]]
    constexpr auto
    query(get_completion_domain_t<_SetTag>, _QEnv const &...) const noexcept -> __decay_t<
      __call_result_t<__read_domain_t<_SetTag>, __child_attrs_t const &, __child_env_t const &>>
    {
      return {};
    }

    // Errors happen in the domains of the child's errors and of its values, if copying its
    // values can throw (see __nothrow_value_copies).
    template <class... _QEnv>
      requires(sizeof...(_QEnv) <= 1) && (!__nothrow_value_copies)
           && __callable<get_completion_domain_t<set_error_t>,
                         __child_attrs_t const &,
                         __child_env_t const &>
           && __callable<get_completion_domain_t<set_value_t>,
                         __child_attrs_t const &,
                         __child_env_t const &>
    [[nodiscard]]
    constexpr auto query(get_completion_domain_t<set_error_t>, _QEnv const &...) const noexcept
      -> __facade::__domain_join_t<__child_domain_t<set_error_t>, __child_domain_t<set_value_t>>
    {
      return {};
    }

    template <class... _QEnv>
      requires(sizeof...(_QEnv) <= 1)
           && __callable<get_completion_domain_t<>, __child_attrs_t const &, __child_env_t const &>
    [[nodiscard]]
    constexpr auto query(get_completion_domain_t<>, _QEnv const &...) const noexcept
      -> __call_result_t<get_completion_domain_t<>, __child_attrs_t const &, __child_env_t const &>
    {
      return {};
    }

    template <__completion_tag _SetTag, class... _QEnv>
      requires(sizeof...(_QEnv) <= 1)
    [[nodiscard]]
    constexpr auto query(__get_completion_behavior_t<_SetTag>, _QEnv const &...) const noexcept
    {
      return _Behavior();
    }

    STDEXEC_ATTRIBUTE(no_unique_address) __sched_of_t<set_value_t> __value_;
    STDEXEC_ATTRIBUTE(no_unique_address) __sched_of_t<set_error_t> __error_;
    STDEXEC_ATTRIBUTE(no_unique_address) __sched_of_t<set_stopped_t> __stopped_;
  };

  //////////////////////////////////////////////////////////////////////////////
  //! Base class for heap-allocatable shared state for `split` and
  //! `ensure_started`.
  template <class _Env, class _Variant>
  struct __shared_state_base : __local_state_base
  {
    using __waiters_list_t = detail::intrusive_slist<&__local_state_base::__next_>;

    constexpr explicit __shared_state_base(_Env __env)
      : __env_(__env::__join(prop{get_stop_token, __stop_source_.get_token()},
                             static_cast<_Env &&>(__env)))
    {
      __results_.template emplace<__tuple<set_stopped_t>>();
    }

    virtual ~__shared_state_base() = 0;

    //! @brief This is called when the shared async operation completes.
    //! @post __waiters_ is set to a known "tombstone" value.
    template <class _Tag, class... _As>
    void __complete(_Tag, _As &&...__as) noexcept
    {
      STDEXEC_TRY
      {
        using __tuple_t = __decayed_tuple<_Tag, _As...>;
        __results_.template emplace<__tuple_t>(_Tag(), static_cast<_As &&>(__as)...);
      }
      STDEXEC_CATCH_ALL
      {
        if constexpr (!__nothrow_decay_copyable<_As...>)
        {
          using __tuple_t = __decayed_tuple<set_error_t, std::exception_ptr>;
          __results_.template emplace<__tuple_t>(set_error, std::current_exception());
        }
      }

      __notify_waiters();
    }

    //! @brief This is called when the shared async operation completes.
    //! @post __waiters_ is set to a known "tombstone" value.
    void __notify_waiters() noexcept
    {
      __waiters_list_t __waiters_copy{this};

      // Set the waiters list to a known "tombstone" value that we can check
      // later.
      {
        std::lock_guard __lock{this->__mutex_};
        this->__waiters_.swap(__waiters_copy);
      }

      STDEXEC_ASSERT(__waiters_copy.front() != this);
      for (auto __itr = __waiters_copy.begin(); __itr != __waiters_copy.end();)
      {
        __local_state_base *__item = *__itr;

        // We must increment the iterator before calling notify, since notify
        // may end up triggering *__item to be destructed on another thread, and
        // the intrusive slist's iterator increment relies on __item.
        ++__itr;
        __item->__notify();
      }

      // Set the "is running" bit in the ref count to zero. Delete the shared
      // state if the ref-count is now zero.
      __set_completed();
    }

    constexpr virtual void __set_completed() noexcept = 0;

    std::mutex          __mutex_{};  // This mutex guards access to __waiters_.
    __waiters_list_t    __waiters_{};
    inplace_stop_source __stop_source_{};
    __env_t<_Env>       __env_;
    _Variant            __results_;  // Initialized to the "set_stopped" state
                                     // in the ctor.
  };

  template <class _Env, class _Variant>
  __shared_state_base<_Env, _Variant>::~__shared_state_base() = default;

  // What happened when an operation tried to wait for the shared operation
  enum class __waiting : unsigned char
  {
    __added,    // the operation waits for the shared operation
    __done,     // the shared operation has completed
    __stopped,  // stop was requested
  };

  //////////////////////////////////////////////////////////////////////////////
  //! Heap-allocatable shared state for `stdexec::split` and
  //! `stdexec::ensure_started`.
  template <class _CvSender, class _Env>
  struct STDEXEC_ATTRIBUTE(empty_bases) __shared_state final
    : std::enable_shared_from_this<__shared_state<_CvSender, _Env>>
    , __shared_state_base<_Env, __result_variant_t<_CvSender, _Env>>
  {
    using __receiver_t     = __receiver<_Env, __result_variant_t<_CvSender, _Env>>;
    using __waiters_list_t = __shared_state::__shared_state_base::__waiters_list_t;
    using __attrs_t        = __shared_attrs<__decay_t<_CvSender>, __env_t<_Env>>;

    constexpr explicit __shared_state(_CvSender &&__sndr, _Env __env)
      : __shared_state::__shared_state_base(static_cast<_Env &&>(__env))
      , __attrs_(STDEXEC::get_env(__sndr), this->__env_)
      , __shared_op_(STDEXEC::connect(static_cast<_CvSender &&>(__sndr), __receiver_t{this}))
    {}

    bool __set_started() noexcept
    {
      if (!__started_.test_and_set(__std::memory_order_acq_rel))
      {
        // we were the first to start the operation
        std::shared_ptr<__shared_state> __expected{};
        return __started_ref_.compare_exchange_strong(__expected,
                                                      this->shared_from_this(),
                                                      __std::memory_order_acq_rel,
                                                      __std::memory_order_acquire);
      }
      return false;  // already started
    }

    //! @post The `__started_` atomic flag is set in the shared state's ref
    //! count, OR the __waiters_ list is set to the known "tombstone" value
    //! indicating completion.
    void __try_start() noexcept
    {
      // With the split algorithm, multiple split senders can be started
      // simultaneously, but only one should start the shared async operation.
      // If __set_started() reports that the operation has already been started,
      // do nothing.
      if (this->__set_started())
      {
        // we are the first to start the underlying operation
        if (this->__stop_source_.stop_requested())
        {
          // Stop has already been requested. Rather than starting the
          // operation, complete with set_stopped immediately.
          // 1. Sets __waiters_ to a known "tombstone" value.
          // 2. Notifies all the waiters that the operation has stopped.
          // 3. Sets the "is running" bit in the ref count to 0.
          this->__notify_waiters();
        }
        else
        {
          STDEXEC::start(__shared_op_);
        }
      }
    }

    template <class _StopToken>
    auto __try_add_waiter(__local_state_base *__waiter, _StopToken __stok) noexcept -> __waiting
    {
      std::unique_lock __lock{this->__mutex_};
      if (this->__waiters_.front() == this)
      {
        // The work has already completed.
        return __waiting::__done;
      }
      else if (__stok.stop_requested())
      {
        // Stop has been requested. Do not add the waiter.
        return __waiting::__stopped;
      }
      else
      {
        // Add the waiter to the list.
        this->__waiters_.push_front(__waiter);
        return __waiting::__added;
      }
    }

    constexpr void __detach() noexcept
    {
      // increments the use count:
      auto __started_ptr = __started_ref_.load();
      // If the use count is 3 (one for *this, one for __started_ref_, and one
      // for __started_ptr), then we are the final "consumer". Ask the operation
      // to stop early.
      if (3 == __started_ptr.use_count())
      {
        this->__stop_source_.request_stop();
      }
    }

    constexpr virtual void __set_completed() noexcept final
    {
      __started_ref_.store({}, __std::memory_order_release);
    }

    __std::atomic_flag __started_{};
    // this shared_ptr is non-null when the operation is running:
    __std::__atomic_shared_ptr<__shared_state> __started_ref_;
    __attrs_t                                  __attrs_;
    connect_result_t<_CvSender, __receiver_t>  __shared_op_;
  };

  //////////////////////////////////////////////////////////////////////////////
  // The keys of the nested operations of the operation of a split or ensure_started
  // sender: the shared operation, whose results the operation sends; and a stop request
  // from the receiver, which stops the operation first.
  struct __shared_op
  {};

  struct __stop_request
  {};

  // split completes with const T&. ensure_started completes with T&&.
  template <class _Tag>
  using __cvref_results_t =
    __mcompose<__if_c<__same_as<_Tag, split_t>, __cpclr, __cp>, __q1<__decay_t>>;

  // The shared operation, as the operations that wait for it see it: it completes with its
  // results (see __cvref_results_t), and it has the attributes of __shared_attrs. It is
  // never connected.
  template <class _Tag, class _Child, class _Env>
  struct __results_sender
  {
    using sender_concept = sender_tag;
    using __attrs_t      = __shared_attrs<_Child, __env_t<_Env>>;

    template <class _SetTag>
    static constexpr auto __transform_sig = []<class... _Ts>() noexcept
    {
      return completion_signatures<_SetTag(__mcall1<__cvref_results_t<_Tag>, _Ts>...)>();
    };

    template <class _Self, class... _OtherEnv>
    static consteval auto get_completion_signatures()
    {
      // (_Child is the decayed child, which the shared state holds and connects as an
      // rvalue. Making the shared state copies or moves it there, so a child that cannot be
      // copied or moved never gets this far.)
      using __cvref_eptr_t = __mcall1<__cvref_results_t<_Tag>, std::exception_ptr>;
      return ::exec::transform_completion_signatures(
        ::STDEXEC::get_completion_signatures<_Child, __env_t<_Env>>(),
        __transform_sig<set_value_t>,
        __transform_sig<set_error_t>,
        {},
        completion_signatures<set_error_t(__cvref_eptr_t), set_stopped_t()>());
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> __attrs_t
    {
      return __attrs_;
    }

    __attrs_t __attrs_;
  };

  // A stop request, which completes the operation with set_stopped_t() on the thread that
  // requests it, so nothing is known about where
  struct __stop_request_sender
  {
    using sender_concept        = sender_tag;
    using completion_signatures = STDEXEC::completion_signatures<set_stopped_t()>;
  };

  // Calls __fn(tag, args...) with the results of the shared operation, as the operation
  // sends them (see __cvref_results_t), and returns what it returns
  template <class _Tag, class _Results, class _Fn>
  constexpr auto __with_results(_Results &__results, _Fn __fn) noexcept
  {
    return __facade::__visit(__results,
                             [&](auto &__tupl) noexcept
                             {
                               return STDEXEC::__apply(
                                 [&]<class _SetTag, class... _Args>(_SetTag __tag,
                                                                    _Args &...__args) noexcept
                                 {
                                   if constexpr (__same_as<_Tag, split_t>)
                                     return __fn(__tag, static_cast<_Args const &>(__args)...);
                                   else
                                     return __fn(__tag, static_cast<_Args &&>(__args)...);
                                 },
                                 __tupl);
                             });
  }

  // The tokens of sending the results that a variant of results holds (see __with_results)
  template <class... _Ts>
  auto __alternatives_of(__variant<_Ts...> const &) -> __mlist<_Ts...>;

  template <class _Tag, class _Tuple>
  struct __send_token;

  template <class _Tag, class _SetTag, class... _Ts>
  struct __send_token<_Tag, __tuple<_SetTag, _Ts...>>
  {
    using __t = __facade::__effects<_SetTag(__mcall1<__cvref_results_t<_Tag>, _Ts>...)>;
  };

  template <class _Tag, class _Alternatives>
  struct __results_token;

  template <class _Tag, class... _Tuples>
  struct __results_token<_Tag, __mlist<_Tuples...>>
  {
    using __t =
      __facade::__union_t<__facade::__effects<>, typename __send_token<_Tag, _Tuples>::__t...>;
  };

  template <class _Tag, class _Variant>
  using __results_token_t =
    __results_token<_Tag, decltype(__shared::__alternatives_of(__declval<_Variant &>()))>::__t;

  // The token of the operation's start: it waits for the shared operation (and, if the
  // receiver's stop token is stoppable, for a stop request), or sends the results, or (if
  // stoppable) completes with set_stopped_t().
  template <class _Tag, class _Variant, class _Receiver>
  using __start_token_t =
    std::conditional_t<unstoppable_token<stop_token_of_t<env_of_t<_Receiver>>>,
                       __facade::__union_t<__facade::__effects<__facade::__starts<__shared_op>>,
                                           __results_token_t<_Tag, _Variant>>,
                       __facade::__union_t<__facade::__effects<__facade::__starts<__shared_op>,
                                                               __facade::__starts<__stop_request>>,
                                           __results_token_t<_Tag, _Variant>,
                                           __facade::__effects<set_stopped_t()>>>;

  //////////////////////////////////////////////////////////////////////////////
  // The state of the operation of a split or ensure_started sender. There is one for each
  // receiver that is connected to such a sender, and 0 or more of them for each shared
  // operation. It holds a reference to the shared state.
  template <class _Algo, class _Tag, class _CvChild, class _Env, class _Receiver>
  struct __local_state
    : __local_state_base
    , stoppable_base_for_t<stop_token_of_t<env_of_t<_Receiver>>,
                           __local_state<_Algo, _Tag, _CvChild, _Env, _Receiver>>
  {
    using __sh_state_t     = __shared_state<_CvChild, _Env>;
    using __sh_state_ptr_t = std::shared_ptr<__sh_state_t>;
    using __variant_t      = __result_variant_t<_CvChild, _Env>;
    using __receiver_t     = _Receiver;

    static_assert(__one_of<_Tag, split_t, ensure_started_t>);

    constexpr explicit __local_state(__sh_state_ptr_t __sh_state, _Receiver __rcvr) noexcept
      : __local_state::__local_state_base{&__local_state::__notify_impl}
      , __rcvr_(static_cast<_Receiver &&>(__rcvr))
      , __sh_state_(std::move(__sh_state))
    {}

    STDEXEC_IMMOVABLE(__local_state);

    constexpr ~__local_state()
    {
      if (__sh_state_)
      {
        __sh_state_->__detach();
      }
    }

    // The algorithm registers the stop callback in __start (see __algo).
    template <class _StopToken>
    void __register_stop_callback(_StopToken const &__stok) noexcept
    {
      this->register_stoppable(__stok);
    }

    void __deregister_stop_callback() noexcept
    {
      this->deregister_stoppable();
    }

    // Stop request callback:
    void on_stop_requested() noexcept
    {
      // We reach here when a split/ensure_started sender has received a stop
      // request from the receiver to which it is connected.
      if (std::unique_lock __lock{__sh_state_->__mutex_})
      {
        // Remove this operation from the waiters list. Removal can fail if:
        //   1. It was already removed by another thread, or
        //   2. It hasn't been added yet (see `__start`), or
        //   3. The underlying operation has already completed.

        // In each case, the right thing to do is nothing. If (1) then we raced
        // with another thread and lost. In that case, the other thread will
        // take care of it. If (2) then `__start` will take care of it. If (3)
        // then this stop request is safe to ignore.
        if (!__sh_state_->__waiters_.remove(this))
          return;
      }

      // The following code and the __notify function cannot both execute. This
      // is because the __notify function is called from the shared state's
      // __notify_waiters function, which first sets __waiters_ to the completed
      // state. As a result, the attempt to remove `this` from the waiters list
      // above will fail and this stop request is ignored.
      std::exchange(__sh_state_, {})->__detach();
      (void) _Algo::__complete(__stop_request(), *this, set_stopped_t());
    }

    // This is called from __shared_state::__notify_waiters when the shared operation
    // completes, if this operation waits for it. It cannot race with
    // on_stop_requested(). See the comment there.
    static constexpr void __notify_impl(__local_state_base &__base) noexcept
    {
      auto &__self = static_cast<__local_state &>(__base);
      __self.deregister_stoppable();
      (void) __shared::__with_results<_Tag>(__self.__sh_state_->__results_,
                                            [&](auto __tag, auto &&...__args) noexcept
                                            {
                                              return _Algo::__complete(
                                                __shared_op(),
                                                __self,
                                                __tag,
                                                static_cast<decltype(__args) &&>(__args)...);
                                            });
    }

    _Receiver        __rcvr_;
    __sh_state_ptr_t __sh_state_;
  };

  //////////////////////////////////////////////////////////////////////////////
  // The data of a split or ensure_started sender: a reference to the shared state, and the
  // shared operation's attributes. (They are kept here, so that a sender whose operation
  // has taken its reference still has them.) A split sender is copyable; an
  // ensure_started sender is move-only.
  template <class _Tag, class _Child, class _Env>
  struct __handle
  {
    using __child_type = _Child;
    using __env_type   = _Env;
    using __sh_state_t = __shared_state<_Child, _Env>;
    using __attrs_t    = __shared_attrs<_Child, __env_t<_Env>>;

    constexpr explicit __handle(std::shared_ptr<__sh_state_t> __sh_state) noexcept
      : __sh_state_(std::move(__sh_state))
      , __attrs_(__sh_state_->__attrs_)
    {}

    __handle(__handle &&) noexcept = default;

    __handle(__handle const &)
      requires __same_as<_Tag, split_t>
    = default;

    // Assigning over a handle detaches it from its shared state, as destroying it does.
    constexpr auto operator=(__handle &&__other) noexcept -> __handle &
    {
      if (this != &__other)
      {
        __detach();
        __sh_state_ = std::move(__other.__sh_state_);
        __attrs_    = std::move(__other.__attrs_);
      }
      return *this;
    }

    constexpr auto operator=(__handle const &__other) -> __handle &
      requires __same_as<_Tag, split_t>
    {
      if (this != &__other)
      {
        __handle __copy(__other);
        *this = std::move(__copy);
      }
      return *this;
    }

    constexpr ~__handle()
    {
      __detach();
    }

    // Tells the shared state that this handle no longer refers to it. (The handle still
    // does, so that the shared state can count it.)
    constexpr void __detach() noexcept
    {
      if (__sh_state_)
      {
        __sh_state_->__detach();
      }
    }

    std::shared_ptr<__sh_state_t> __sh_state_;
    __attrs_t                     __attrs_;
  };

  // The environment that split and ensure_started give the sender that describes them: a
  // copy, if the environment can be copied, because transform_sender gets the environment
  // too, and must not get one that has been moved from
  template <class _Env>
  constexpr auto __env_copy(_Env &&__env) -> __decay_t<_Env>
  {
    if constexpr (__std::copy_constructible<__decay_t<_Env>>)
      return static_cast<__decay_t<_Env> const &>(__env);
    else
      return static_cast<_Env &&>(__env);
  }

  // The tag of the senders that split and ensure_started return
  template <class _Tag>
  struct __sender_tag
  {};

  template <class _Tag, class _CvChild, class _Env>
  constexpr auto __make_sender(_Tag, _CvChild &&__child, _Env &&__env)
  {
    using __child_t    = __decay_t<_CvChild>;
    using __data_env_t = __decay_t<_Env>;
    auto __sh_state    = __shared::__mk_sh_state(__child_t(static_cast<_CvChild &&>(__child)),
                                              __data_env_t(static_cast<_Env &&>(__env)));
    return __facade::__sender{__sender_tag<_Tag>(),
                              __handle<_Tag, __child_t, __data_env_t>(std::move(__sh_state))};
  }

  struct _AN_ENSURE_STARTED_SENDER_CAN_ONLY_BE_CONNECTED_AS_AN_RVALUE_;

  //////////////////////////////////////////////////////////////////////////////
  // The algorithm of the senders that split and ensure_started return: when the operation
  // starts, it starts the shared operation if no other operation has, and then either
  // waits for its results, or, if it has already completed, sends them from start.
  template <class _Tag>
  struct __algo : __facade::__defaults
  {
    // Customizations are found where the shared operation completes with a value (as they
    // are for the senders that the domain customizes; see __describe_algo).
    using __dispatch_source = __facade::__via<__shared_op, set_value_t, __facade::__from_start>;

    template <class _Sender>
    using __handle_t = __decay_t<__facade::__data_of_t<_Sender>>;

    using __facade::__defaults::__nested_sender;

    template <class _Sender, class... _Env>
    STDEXEC_ATTRIBUTE(always_inline)
    static constexpr auto __nested_sender(__shared_op,
                                          _Sender const &__sndr,
                                          _Env const &...) noexcept
    {
      using __handle_t = __algo::__handle_t<_Sender>;
      return __results_sender<_Tag,
                              typename __handle_t::__child_type,
                              typename __handle_t::__env_type>{
        __facade::__get_data(__sndr).__attrs_};
    }

    template <class _Sender, class... _Env>
    STDEXEC_ATTRIBUTE(always_inline)
    static constexpr auto __nested_sender(__stop_request, _Sender const &, _Env const &...) noexcept
      -> __mtype<__stop_request_sender>
    {
      return {};
    }

    template <class _Sender, class _Receiver>
    using __local_state_t = __local_state<__algo,
                                          _Tag,
                                          typename __algo::__handle_t<_Sender>::__child_type,
                                          typename __algo::__handle_t<_Sender>::__env_type,
                                          _Receiver>;

    template <class _Sender, class _Receiver>
    static constexpr auto __get_state(_Sender &&__sndr, _Receiver __rcvr) noexcept  //
      -> __local_state_t<_Sender, _Receiver>
    {
      using __state_t = __local_state_t<_Sender, _Receiver>;
      // An rvalue sender gives the operation its reference to the shared state. A split
      // sender's operation can share it.
      return __state_t(__facade::__get_data(static_cast<_Sender &&>(__sndr)).__sh_state_,
                       static_cast<_Receiver &&>(__rcvr));
    }

    template <class _Sender, class _Receiver>
      requires __same_as<_Tag, ensure_started_t> && std::is_lvalue_reference_v<_Sender>
    static constexpr auto __get_state(_Sender &&, __ignore) noexcept
    {
      return __mexception<_WHAT_(_INVALID_ARGUMENT_),
                          _WHERE_(_IN_ALGORITHM_, ensure_started_t),
                          _WHY_(_AN_ENSURE_STARTED_SENDER_CAN_ONLY_BE_CONNECTED_AS_AN_RVALUE_)>();
    }

    template <class _State, class _Children>
    static constexpr auto __start(_State &__state, _Children &) noexcept
      -> __start_token_t<_Tag, typename _State::__variant_t, typename _State::__receiver_t>
    {
      using __stop_token_t = stop_token_of_t<env_of_t<typename _State::__receiver_t>>;
      using __result_t =
        __start_token_t<_Tag, typename _State::__variant_t, typename _State::__receiver_t>;
      auto __send_results = [&]() noexcept
      {
        return __shared::__with_results<_Tag>(__state.__sh_state_->__results_,
                                              [&](auto __tag, auto &&...__args) noexcept
                                              {
                                                return __facade::__send(
                                                  __state.__rcvr_,
                                                  __tag,
                                                  static_cast<decltype(__args) &&>(__args)...);
                                              });
      };

      if constexpr (unstoppable_token<__stop_token_t>)
      {
        __state.__sh_state_->__try_start();
        if (__state.__sh_state_->__try_add_waiter(&__state, never_stop_token())
            == __waiting::__added)
        {
          return __result_t(__facade::__effects<__facade::__starts<__shared_op>>());
        }
        else
        {
          // The shared operation has completed: send its results now.
          return __result_t(__send_results());
        }
      }
      else
      {
        // Scenario: there are no more split senders, this is the only operation
        // state, the underlying operation has not yet been started, and the
        // receiver's stop token is already in the "stop requested" state. Then
        // registering the stop callback will call on_stop_requested() on the state
        // synchronously. It may also be called asynchronously at any point after
        // the callback is registered. Beware. We are guaranteed, however, that
        // on_stop_requested() will not complete the operation or decrement the
        // shared state's ref count until after the state has been added to the
        // waiters list.
        auto const __stok = STDEXEC::get_stop_token(STDEXEC::get_env(__state.__rcvr_));
        __state.__register_stop_callback(__stok);

        // We haven't put the state in the waiters list yet and we are holding a ref
        // count to the shared state, so nothing can happen to it here.

        // Start the shared op. As an optimization, skip it if the receiver's stop
        // token has already been signaled.
        if (!__stok.stop_requested())
        {
          __state.__sh_state_->__try_start();
          switch (__state.__sh_state_->__try_add_waiter(&__state, __stok))
          {
          case __waiting::__added:
            return __result_t(__facade::__effects<__facade::__starts<__shared_op>,
                                                  __facade::__starts<__stop_request>>());
          case __waiting::__done:
            // The shared operation has completed: send its results now.
            __state.__deregister_stop_callback();
            return __result_t(__send_results());
          case __waiting::__stopped:
            break;
          }
        }

        // Otherwise, failed to add the waiter because of a stop-request.
        // Complete synchronously with set_stopped().
        __state.__deregister_stop_callback();
        std::exchange(__state.__sh_state_, {})->__detach();
        return __result_t(__facade::__send(__state.__rcvr_, set_stopped_t()));
      }
    }
  };

  // The sender that split(sndr, env) or ensure_started(sndr, env) makes for its domain to
  // customize: a description of the sender that it returns by default (see
  // split_t::transform_sender). Its data is the environment, and its child is sndr. It has
  // the completions and attributes of the sender it describes. (Its __shared_op is known
  // only by type, because the child is not connected.) It is never connected.
  template <class _Tag>
  struct __describe_algo : __algo<_Tag>
  {
    template <class _Sender>
    using __child_t = __decay_t<__facade::__nth_child_of_t<_Sender, 0>>;

    template <class _Sender>
    using __data_env_t = __decay_t<__facade::__data_of_t<_Sender>>;

    // (The algorithm's overload for __shared_op is not brought in: an overload of a
    // function template with another return type does not hide it.)
    using __facade::__defaults::__nested_sender;

    template <class _Sender, class... _Env>
    STDEXEC_ATTRIBUTE(always_inline)
    static constexpr auto __nested_sender(__shared_op, _Sender const &, _Env const &...) noexcept
      -> __mtype<__results_sender<_Tag, __child_t<_Sender>, __data_env_t<_Sender>>>
    {
      return {};
    }

    template <class _Sender, class... _Env>
    STDEXEC_ATTRIBUTE(always_inline)
    static constexpr auto __nested_sender(__stop_request, _Sender const &, _Env const &...) noexcept
      -> __mtype<__stop_request_sender>
    {
      return {};
    }

    // The state of the operation, as the walk sees it. (The state of the operation of the
    // sender that is described would connect the child to the shared state, and the walk
    // would instantiate the code that completes the child's receiver, which can be device
    // code: a CUDA stream operation completes a receiver that is not a stream receiver from
    // a kernel. And the child's domain customizes the described sender anyway.)
    template <class _Child, class _Env, class _Receiver>
    struct __state_type
    {
      using __variant_t  = __result_variant_t<_Child, _Env>;
      using __receiver_t = _Receiver;

      _Receiver __rcvr_;
    };

    template <class _Sender, class _Receiver>
    static auto __get_state(_Sender &&, _Receiver) noexcept
      -> __state_type<__child_t<_Sender>, __data_env_t<_Sender>, _Receiver>
    {
      static_assert(sizeof(_Sender) == 0,
                    "The sender that split or ensure_started makes for its domain to customize "
                    "cannot be connected.");
      STDEXEC_UNREACHABLE();
    }

    template <class _State, class _Children>
    static constexpr auto __start(_State &, _Children &) noexcept
      -> __start_token_t<_Tag, typename _State::__variant_t, typename _State::__receiver_t>
    {
      return {};
    }
  };
}  // namespace experimental::execution::__shared

namespace exec = experimental::execution;

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  template <class _Tag>
  struct __facade::__algorithm<exec::__shared::__sender_tag<_Tag>> : exec::__shared::__algo<_Tag>
  {};
}  // namespace STDEXEC
