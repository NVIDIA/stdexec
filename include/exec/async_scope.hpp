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

#include "../stdexec/__detail/__intrusive_queue.hpp"
#include "../stdexec/__detail/__optional.hpp"
#include "../stdexec/execution.hpp"
#include "../stdexec/stop_token.hpp"
#include "env.hpp"

#include "../stdexec/__detail/__atomic.hpp"
#include <mutex>

namespace exec {
  /////////////////////////////////////////////////////////////////////////////
  // async_scope
  namespace __scope {
    using namespace STDEXEC;

    struct __impl;
    struct async_scope;

    template <class _A>
    concept __async_scope = requires(_A& __a) {
      { __a.nest(STDEXEC::just()) } -> sender_of<STDEXEC::set_value_t()>;
    };

    struct __task : __immovable {
      const __impl* __scope_;
      void (*__notify_waiter)(__task*) noexcept;
      __task* __next_ = nullptr;
    };

    template <class _BaseEnv>
    using __env_t = make_env_t<_BaseEnv, prop<get_stop_token_t, inplace_stop_token>>;

    struct __impl {
      ~__impl() {
        std::unique_lock __guard{__lock_};
        STDEXEC_ASSERT(__active_ == 0);
        STDEXEC_ASSERT(__waiters_.empty());
      }

      inplace_stop_source __stop_source_{};
      mutable std::mutex __lock_{};
      mutable __std::atomic_ptrdiff_t __active_ = 0;
      mutable __intrusive_queue<&__task::__next_> __waiters_{};
    };

    ////////////////////////////////////////////////////////////////////////////
    // async_scope::when_empty implementation
    template <class _Constrained, class _Receiver>
    struct __when_empty_opstate : __task {
      constexpr explicit __when_empty_opstate(
        const __impl* __scope,
        _Constrained&& __sndr,
        _Receiver __rcvr)
        : __task{{}, __scope, __notify_waiter}
        , __op_(
            STDEXEC::connect(
              static_cast<_Constrained&&>(__sndr),
              static_cast<_Receiver&&>(__rcvr))) {
      }

      void start() & noexcept {
        // must get lock before checking __active, or if the __active is drained before
        // the waiter is queued but after __active is checked, the waiter will never be notified
        std::unique_lock __guard{this->__scope_->__lock_};
        auto& __active = this->__scope_->__active_;
        auto& __waiters = this->__scope_->__waiters_;
        if (__active.load(__std::memory_order_acquire) != 0) {
          __waiters.push_back(this);
          return;
        }
        __guard.unlock();
        STDEXEC::start(this->__op_);
      }

     private:
      static constexpr void __notify_waiter(__task* __self) noexcept {
        STDEXEC::start(static_cast<__when_empty_opstate*>(__self)->__op_);
      }

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      connect_result_t<_Constrained, _Receiver> __op_;
    };

    template <class _Constrained>
    struct __when_empty_sender {
      using sender_concept = STDEXEC::sender_t;

      template <class _Self, class _Receiver>
      using __when_empty_opstate_t =
        __when_empty_opstate<__copy_cvref_t<_Self, _Constrained>, _Receiver>;

      template <__decays_to<__when_empty_sender> _Self, receiver _Receiver>
        requires sender_to<__copy_cvref_t<_Self, _Constrained>, _Receiver>
      [[nodiscard]]
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this _Self&& __self, _Receiver __rcvr)
        -> __when_empty_opstate_t<_Self, _Receiver> {
        return __when_empty_opstate_t<_Self, _Receiver>{
          __self.__scope_, static_cast<_Self&&>(__self).__c_, static_cast<_Receiver&&>(__rcvr)};
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      template <__decays_to<__when_empty_sender> _Self, class... _Env>
      static consteval auto get_completion_signatures()
        -> __completion_signatures_of_t<__copy_cvref_t<_Self, _Constrained>, __env_t<_Env>...> {
        return {};
      }

      const __impl* __scope_;
      STDEXEC_ATTRIBUTE(no_unique_address)
      _Constrained __c_;
    };

    ////////////////////////////////////////////////////////////////////////////
    // async_scope::nest implementation
    template <class _Receiver>
    struct __nest_opstate_base : __immovable {
      const __impl* __scope_;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Receiver __rcvr_;
    };

    template <class _Receiver>
    struct __nest_receiver {
      using receiver_concept = STDEXEC::receiver_t;

      static void __complete(const __impl* __scope) noexcept {
        auto& __active = __scope->__active_;
        std::unique_lock __guard{__scope->__lock_};
        if (__active.fetch_sub(1, __std::memory_order_acq_rel) == 1) {
          auto __local_waiters = std::move(__scope->__waiters_);
          __guard.unlock();
          __scope = nullptr;
          // do not access __scope
          while (!__local_waiters.empty()) {
            auto* __next = __local_waiters.pop_front();
            __next->__notify_waiter(__next);
            // __scope must be considered deleted
          }
        }
      }

      template <class... _As>
      void set_value(_As&&... __as) noexcept {
        auto __scope = __opstate_->__scope_;
        STDEXEC::set_value(std::move(__opstate_->__rcvr_), static_cast<_As&&>(__as)...);
        // do not access __op_
        // do not access this
        __complete(__scope);
      }

      template <class _Error>
      void set_error(_Error&& __err) noexcept {
        auto __scope = __opstate_->__scope_;
        STDEXEC::set_error(std::move(__opstate_->__rcvr_), static_cast<_Error&&>(__err));
        // do not access __op_
        // do not access this
        __complete(__scope);
      }

      constexpr void set_stopped() noexcept {
        auto __scope = __opstate_->__scope_;
        STDEXEC::set_stopped(std::move(__opstate_->__rcvr_));
        // do not access __op_
        // do not access this
        __complete(__scope);
      }

      constexpr auto get_env() const noexcept -> __env_t<env_of_t<_Receiver>> {
        return make_env(
          STDEXEC::get_env(__opstate_->__rcvr_),
          STDEXEC::prop{get_stop_token, __opstate_->__scope_->__stop_source_.get_token()});
      }

      __nest_opstate_base<_Receiver>* __opstate_;
    };

    template <class _Constrained, class _Receiver>
    struct __nest_opstate : __nest_opstate_base<_Receiver> {
      using __nest_rcvr_t = __nest_receiver<_Receiver>;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      connect_result_t<_Constrained, __nest_rcvr_t> __op_;

      template <__decays_to<_Constrained> _Sender, __decays_to<_Receiver> _Rcvr>
      constexpr explicit __nest_opstate(const __impl* __scope, _Sender&& __c, _Rcvr&& __rcvr)
        : __nest_opstate_base<_Receiver>{{}, __scope, static_cast<_Rcvr&&>(__rcvr)}
        , __op_(STDEXEC::connect(static_cast<_Sender&&>(__c), __nest_rcvr_t{this})) {
      }

      constexpr void start() & noexcept {
        STDEXEC_ASSERT(this->__scope_);
        auto& __active = this->__scope_->__active_;
        __active.fetch_add(1, __std::memory_order_relaxed);
        STDEXEC::start(__op_);
      }
    };

    template <class _Constrained>
    struct __nest_sender {
      using sender_concept = STDEXEC::sender_t;

      template <__decays_to<__nest_sender> _Self, receiver _Receiver>
        requires sender_to<__copy_cvref_t<_Self, _Constrained>, __nest_receiver<_Receiver>>
      [[nodiscard]]
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this _Self&& __self, _Receiver __rcvr)
        -> __nest_opstate<_Constrained, _Receiver> {
        return __nest_opstate<_Constrained, _Receiver>{
          __self.__scope_, static_cast<_Self&&>(__self).__c_, static_cast<_Receiver&&>(__rcvr)};
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      template <__decays_to<__nest_sender> _Self, class... _Env>
      static consteval auto get_completion_signatures()
        -> __completion_signatures_of_t<__copy_cvref_t<_Self, _Constrained>, __env_t<_Env>...> {
        return {};
      }

      const __impl* __scope_;
      STDEXEC_ATTRIBUTE(no_unique_address) _Constrained __c_;
    };

    ////////////////////////////////////////////////////////////////////////////
    // async_scope::spawn_future implementation
    enum class __future_step {
      __invalid = 0,
      __created,
      __future,
      __no_future,
      __deleted
    };

    template <class _Sender, class _Env>
    struct __future_state;

    struct __forward_stopped {
      inplace_stop_source* __stop_source_;

      void operator()() noexcept {
        __stop_source_->request_stop();
      }
    };

    struct __subscription : __immovable {
      constexpr void __complete() noexcept {
        __complete_(this);
      }

      void (*__complete_)(__subscription*) noexcept = nullptr;
      __subscription* __next_ = nullptr;
    };

    template <class _Sender, class _Env, class _Receiver>
    struct __future_opstate : __subscription {
     private:
      using __forward_consumer_t =
        stop_callback_for_t<stop_token_of_t<env_of_t<_Receiver>>, __forward_stopped>;

      constexpr void __complete_() noexcept {
        STDEXEC_TRY {
          __forward_consumer_.reset();
          auto __state = std::move(__state_);
          STDEXEC_ASSERT(__state != nullptr);
          std::unique_lock __guard{__state->__mutex_};
          // either the future is still in use or it has passed ownership to __state->__no_future_
          if (__state->__no_future_.get() != nullptr || __state->__step_ != __future_step::__future) {
            // invalid state - there is a code bug in the state machine
            std::terminate();
          } else if (get_stop_token(get_env(__rcvr_)).stop_requested()) {

            __guard.unlock();
            STDEXEC::set_stopped(static_cast<_Receiver&&>(__rcvr_));
            __guard.lock();
          } else {
            std::visit(
              [this, &__guard]<class _Tup>(_Tup& __tup) {
                if constexpr (__std::same_as<_Tup, std::monostate>) {
                  std::terminate();
                } else {
                  std::apply(
                    [this, &__guard]<class... _As>(auto tag, _As&... __as) {
                      __guard.unlock();
                      tag(static_cast<_Receiver&&>(__rcvr_), static_cast<_As&&>(__as)...);
                      __guard.lock();
                    },
                    __tup);
                }
              },
              __state->__data_);
          }
        }
        STDEXEC_CATCH_ALL {

          STDEXEC::set_error(static_cast<_Receiver&&>(__rcvr_), std::current_exception());
        }
      }

      STDEXEC_ATTRIBUTE(no_unique_address) _Receiver __rcvr_;
      std::unique_ptr<__future_state<_Sender, _Env>> __state_;
      STDEXEC_ATTRIBUTE(no_unique_address)
      STDEXEC::__optional<__forward_consumer_t> __forward_consumer_;

     public:
      template <class _Receiver2>
      constexpr explicit __future_opstate(
        _Receiver2&& __rcvr,
        std::unique_ptr<__future_state<_Sender, _Env>> __state)
        : __subscription{
            {},
            [](__subscription* __self) noexcept -> void {
              static_cast<__future_opstate*>(__self)->__complete_();
            }}
        , __rcvr_(static_cast<_Receiver2&&>(__rcvr))
        , __state_(std::move(__state))
        , __forward_consumer_(
            std::in_place,
            get_stop_token(get_env(__rcvr_)),
            __forward_stopped{&__state_->__stop_source_}) {
      }

      constexpr ~__future_opstate() noexcept {
        if (__state_ != nullptr) {
          auto __raw_state = __state_.get();
          std::unique_lock __guard{__raw_state->__mutex_};
          if (__raw_state->__data_.index() > 0) {
            // completed given sender
            // state is no longer needed
            return;
          }
          __raw_state->__no_future_ = std::move(__state_);
          __raw_state
            ->__step_from_to_(__guard, __future_step::__future, __future_step::__no_future);
        }
      }

      constexpr void start() & noexcept {
        STDEXEC_TRY {
          if (!!__state_) {
            std::unique_lock __guard{__state_->__mutex_};
            if (__state_->__data_.index() != 0) {
              __guard.unlock();
              __complete_();
            } else {
              __state_->__subscribers_.push_back(this);
            }
          }
        }
        STDEXEC_CATCH_ALL {
          STDEXEC::set_error(static_cast<_Receiver&&>(__rcvr_), std::current_exception());
        }
      }
    };

#if STDEXEC_EDG()
    template <class _Fn>
    struct __completion_as_tuple2_;

    template <class _Tag, class... _Ts>
    struct __completion_as_tuple2_<_Tag(_Ts...)> {
      using __t = std::tuple<_Tag, _Ts...>;
    };
    template <class _Fn>
    using __completion_as_tuple_t = STDEXEC::__t<__completion_as_tuple2_<_Fn>>;

#else

    template <class _Tag, class... _Ts>
    constexpr auto __completion_as_tuple_(_Tag (*)(_Ts...)) -> std::tuple<_Tag, _Ts...>;

    template <class _Fn>
    using __completion_as_tuple_t = decltype(__scope::__completion_as_tuple_(
      static_cast<_Fn*>(nullptr)));
#endif

    template <class... _Ts>
    using __decay_values_t = completion_signatures<set_value_t(__decay_t<_Ts>...)>;

    template <class _Ty>
    using __decay_error_t = completion_signatures<set_error_t(__decay_t<_Ty>)>;

    template <class _Sender, class _Env>
    using __future_completions_t = transform_completion_signatures_of<
      _Sender,
      __env_t<_Env>,
      completion_signatures<set_stopped_t(), set_error_t(std::exception_ptr)>,
      __decay_values_t,
      __decay_error_t
    >;

    template <class _Completions>
    using __completions_as_variant = __mapply<
      __mtransform<__q<__completion_as_tuple_t>, __mbind_front_q<std::variant, std::monostate>>,
      _Completions
    >;

    template <class _Ty>
    struct __dynamic_delete {
      constexpr __dynamic_delete()
        : __delete_([](_Ty* __p) { delete __p; }) {
      }

      template <class _Uy>
        requires __std::convertible_to<_Uy*, _Ty*>
      __dynamic_delete(std::default_delete<_Uy>)
        : __delete_([](_Ty* __p) { delete static_cast<_Uy*>(__p); }) {
      }

      template <class _Uy>
        requires __std::convertible_to<_Uy*, _Ty*>
      auto operator=(std::default_delete<_Uy> __d) -> __dynamic_delete& {
        __delete_ = __dynamic_delete{__d}.__delete_;
        return *this;
      }

      constexpr void operator()(_Ty* __p) {
        __delete_(__p);
      }

      void (*__delete_)(_Ty*);
    };

    template <class _Completions, class _Env>
    struct __future_state_base {
      constexpr __future_state_base(_Env __env, const __impl* __scope)
        : __forward_scope_{
            std::in_place,
            __scope->__stop_source_.get_token(),
            __forward_stopped{&__stop_source_}}
        , __env_(make_env(
            static_cast<_Env&&>(__env),
            STDEXEC::prop{get_stop_token, __scope->__stop_source_.get_token()})) {
      }

      ~__future_state_base() {
        std::unique_lock __guard{__mutex_};
        if (__step_ == __future_step::__created) {
          // exception during connect() will end up here
          __step_from_to_(__guard, __future_step::__created, __future_step::__deleted);
        } else if (__step_ != __future_step::__deleted) {
          // completing the given sender before the future is dropped will end here
          __step_from_to_(__guard, __future_step::__future, __future_step::__deleted);
        }
      }

      constexpr void __step_from_to_(
        std::unique_lock<std::mutex>& __guard,
        __future_step __from,
        __future_step __to) {
        STDEXEC_ASSERT(__guard.owns_lock());
        auto actual = std::exchange(__step_, __to);
        STDEXEC_ASSERT(actual == __from);
      }

      inplace_stop_source __stop_source_;
      STDEXEC::__optional<inplace_stop_callback<__forward_stopped>> __forward_scope_;
      std::mutex __mutex_;
      __future_step __step_ = __future_step::__created;
      std::unique_ptr<__future_state_base, __dynamic_delete<__future_state_base>> __no_future_;
      __completions_as_variant<_Completions> __data_;
      __intrusive_queue<&__subscription::__next_> __subscribers_;
      __env_t<_Env> __env_;
    };

    template <class _Completions, class _Env>
    struct __future_receiver {
      using receiver_concept = STDEXEC::receiver_t;

      constexpr void __dispatch_result_(std::unique_lock<std::mutex>& __guard) noexcept {
        auto& __state = *__state_;
        auto __local_subscribers = std::move(__state.__subscribers_);
        __state.__forward_scope_.reset();
        if (__state.__no_future_.get() != nullptr) {
          // nobody is waiting for the results
          // delete this and return
          __state.__step_from_to_(__guard, __future_step::__no_future, __future_step::__deleted);
          __guard.unlock();
          __state.__no_future_.reset();
          return;
        }
        __guard.unlock();
        while (!__local_subscribers.empty()) {
          auto* __sub = __local_subscribers.pop_front();
          __sub->__complete();
        }
      }

      template <class _Tag, class... _As>
      constexpr void __save_completion(_Tag, _As&&... __as) noexcept {
        auto& __state = *__state_;
        STDEXEC_TRY {
          using _Tuple = __decayed_std_tuple<_Tag, _As...>;
          __state.__data_.template emplace<_Tuple>(_Tag(), static_cast<_As&&>(__as)...);
        }
        STDEXEC_CATCH_ALL {
          using _Tuple = std::tuple<set_error_t, std::exception_ptr>;
          __state.__data_.template emplace<_Tuple>(set_error_t(), std::current_exception());
        }
      }

      template <__movable_value... _As>
      constexpr void set_value(_As&&... __as) noexcept {
        auto& __state = *__state_;
        std::unique_lock __guard{__state.__mutex_};
        __save_completion(set_value_t(), static_cast<_As&&>(__as)...);
        __dispatch_result_(__guard);
      }

      template <__movable_value _Error>
      constexpr void set_error(_Error&& __err) noexcept {
        auto& __state = *__state_;
        std::unique_lock __guard{__state.__mutex_};
        __save_completion(set_error_t(), static_cast<_Error&&>(__err));
        __dispatch_result_(__guard);
      }

      constexpr void set_stopped() noexcept {
        auto& __state = *__state_;
        std::unique_lock __guard{__state.__mutex_};
        __save_completion(set_stopped_t());
        __dispatch_result_(__guard);
      }

      constexpr auto get_env() const noexcept -> const __env_t<_Env>& {
        return __state_->__env_;
      }

      __future_state_base<_Completions, _Env>* __state_;
      const __impl* __scope_;
    };

    template <class _Sender, class _Env>
    using __future_receiver_t = __future_receiver<__future_completions_t<_Sender, _Env>, _Env>;

    template <class _Sender, class _Env>
    struct __future_state : __future_state_base<__future_completions_t<_Sender, _Env>, _Env> {
      using __completions_t = __future_completions_t<_Sender, _Env>;

      constexpr explicit __future_state(
        connect_t,
        _Sender&& __sndr,
        _Env __env,
        const __impl* __scope)
        : __future_state_base<__completions_t, _Env>(static_cast<_Env&&>(__env), __scope)
        , __op_(static_cast<_Sender&&>(__sndr), __future_receiver_t<_Sender, _Env>{this, __scope}) {
      }

      constexpr explicit __future_state(_Sender __sndr, _Env __env, const __impl* __scope)
        : __future_state(
            STDEXEC::connect,
            static_cast<_Sender&&>(__sndr),
            static_cast<_Env&&>(__env),
            __scope) {
        // If the operation completes synchronously, then the following line will cause
        // the destruction of *this, which is not a problem because we used a delegating
        // constructor, so *this is considered fully constructed.
        __op_.submit(
          static_cast<_Sender&&>(__sndr), __future_receiver_t<_Sender, _Env>{this, __scope});
      }

      STDEXEC_ATTRIBUTE(no_unique_address)
      submit_result<_Sender, __future_receiver_t<_Sender, _Env>> __op_{};
    };

    template <class _Sender, class _Env>
    struct __future {
     private:
      template <class _Self>
      using __completions_t = __future_completions_t<__mfront<_Sender, _Self>, _Env>;

      template <class _Receiver>
      using __future_opstate_t = __future_opstate<_Sender, _Env, _Receiver>;

     public:
      using sender_concept = STDEXEC::sender_t;

      __future(__future&&) = default;
      auto operator=(__future&&) -> __future& = default;

      ~__future() noexcept {
        if (__state_ != nullptr) {
          auto __raw_state = __state_.get();
          std::unique_lock __guard{__raw_state->__mutex_};
          if (__raw_state->__data_.index() != 0) {
            // completed given sender
            // state is no longer needed
            return;
          }
          __raw_state->__no_future_ = std::move(__state_);
          __raw_state
            ->__step_from_to_(__guard, __future_step::__future, __future_step::__no_future);
        }
      }

      template <__decays_to<__future> _Self, receiver _Receiver>
        requires receiver_of<_Receiver, __completions_t<_Self>>
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this _Self&& __self, _Receiver __rcvr)
        -> __future_opstate_t<_Receiver> {
        return __future_opstate_t<_Receiver>{
          static_cast<_Receiver&&>(__rcvr), static_cast<_Self&&>(__self).__state_};
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      template <__decays_to<__future> _Self, class... _OtherEnv>
      static consteval auto get_completion_signatures() -> __completions_t<_Self> {
        return {};
      }

     private:
      friend struct async_scope;

      constexpr explicit __future(std::unique_ptr<__future_state<_Sender, _Env>> __state) noexcept
        : __state_(std::move(__state)) {
        std::unique_lock __guard{__state_->__mutex_};
        __state_->__step_from_to_(__guard, __future_step::__created, __future_step::__future);
      }

      std::unique_ptr<__future_state<_Sender, _Env>> __state_;
    };

    template <class _Sender, class _Env>
    using __future_t = __future<__nest_sender<__decay_t<_Sender>>, __decay_t<_Env>>;

    ////////////////////////////////////////////////////////////////////////////
    // async_scope::spawn implementation
    struct __spawn_env {
      [[nodiscard]]
      constexpr auto query(get_stop_token_t) const noexcept -> inplace_stop_token {
        return __token_;
      }

      [[nodiscard]]
      constexpr auto query(get_scheduler_t) const noexcept -> STDEXEC::inline_scheduler {
        return {};
      }

      inplace_stop_token __token_;
    };

    template <class _Env>
    using __spawn_env_t = __join_env_t<_Env, __spawn_env>;

    template <class _Env>
    struct __spawn_opstate_base {
      __spawn_env_t<_Env> __env_;
      void (*__delete_)(__spawn_opstate_base*);
    };

    template <class _Env>
    struct __spawn_receiver {
      using receiver_concept = STDEXEC::receiver_t;

      constexpr void set_value() noexcept {
        __op_->__delete_(__op_);
      }

      // BUGBUG NOT TO SPEC spawn shouldn't accept senders that can fail.
      [[noreturn]]
      void set_error(std::exception_ptr __eptr) noexcept {
        std::rethrow_exception(std::move(__eptr));
      }

      constexpr void set_stopped() noexcept {
        __op_->__delete_(__op_);
      }

      constexpr auto get_env() const noexcept -> const __spawn_env_t<_Env>& {
        return __op_->__env_;
      }

      __spawn_opstate_base<_Env>* __op_;
    };

    template <class _Sender, class _Env>
    struct __spawn_opstate : __spawn_opstate_base<_Env> {
      constexpr explicit __spawn_opstate(
        connect_t,
        _Sender&& __sndr,
        _Env __env,
        const __impl* __scope)
        : __spawn_opstate_base<_Env>{
            __env::__join(
              static_cast<_Env&&>(__env),
              __spawn_env{__scope->__stop_source_.get_token()}),
            [](__spawn_opstate_base<_Env>* __op) { delete static_cast<__spawn_opstate*>(__op); }}
        , __data_(static_cast<_Sender&&>(__sndr), __spawn_receiver<_Env>{this}) {
      }

      constexpr explicit __spawn_opstate(_Sender __sndr, _Env __env, const __impl* __scope)
        : __spawn_opstate(
            STDEXEC::connect,
            static_cast<_Sender&&>(__sndr),
            static_cast<_Env&&>(__env),
            __scope) {
        // If the operation completes synchronously, then the following line will cause
        // the destruction of *this, which is not a problem because we used a delegating
        // constructor, so *this is considered fully constructed.
        __data_.submit(static_cast<_Sender&&>(__sndr), __spawn_receiver<_Env>{this});
      }

      STDEXEC_ATTRIBUTE(no_unique_address)
      submit_result<_Sender, __spawn_receiver<_Env>> __data_;
    };

    ////////////////////////////////////////////////////////////////////////////
    // async_scope
    struct async_scope : __immovable {
      async_scope() = default;

      template <sender _Constrained>
      [[nodiscard]]
      constexpr auto when_empty(_Constrained&& __c) const //
        -> __when_empty_sender<__decay_t<_Constrained>> {
        return __when_empty_sender<__decay_t<_Constrained>>{
          &__impl_, static_cast<_Constrained&&>(__c)};
      }

      [[nodiscard]]
      constexpr auto on_empty() const {
        return when_empty(just());
      }

      template <sender _Constrained>
      [[nodiscard]]
      constexpr auto nest(_Constrained&& __c) -> __nest_sender<__decay_t<_Constrained>> {
        return __nest_sender<__decay_t<_Constrained>>{&__impl_, static_cast<_Constrained&&>(__c)};
      }

      template <__movable_value _Env = env<>, sender_in<__spawn_env_t<_Env>> _Sender>
        requires sender_to<__nest_sender<__decay_t<_Sender>>, __spawn_receiver<_Env>>
      void spawn(_Sender&& __sndr, _Env __env = {}) {
        using __opstate_t = __spawn_opstate<__nest_sender<__decay_t<_Sender>>, _Env>;
        // this will connect and start the operation, after which the operation state is
        // responsible for deleting itself after it completes.
        [[maybe_unused]]
        auto* __opstate = new __opstate_t{
          nest(static_cast<_Sender&&>(__sndr)), static_cast<_Env&&>(__env), &__impl_};
      }

      template <__movable_value _Env = env<>, sender_in<__env_t<_Env>> _Sender>
      [[nodiscard]]
      auto spawn_future(_Sender&& __sndr, _Env __env = {}) -> __future_t<_Sender, _Env> {
        using __state_t = __future_state<__nest_sender<__decay_t<_Sender>>, _Env>;
        auto __state = std::make_unique<__state_t>(
          nest(static_cast<_Sender&&>(__sndr)), static_cast<_Env&&>(__env), &__impl_);
        return __future_t<_Sender, _Env>{std::move(__state)};
      }

      [[nodiscard]]
      constexpr auto get_stop_source() noexcept -> inplace_stop_source& {
        return __impl_.__stop_source_;
      }

      [[nodiscard]]
      constexpr auto get_stop_token() const noexcept -> inplace_stop_token {
        return __impl_.__stop_source_.get_token();
      }

      auto request_stop() noexcept -> bool {
        return __impl_.__stop_source_.request_stop();
      }

     private:
      __impl __impl_;
    };
  } // namespace __scope

  using __scope::async_scope;

  template <class _AsyncScope, class _Sender>
  using nest_result_t = decltype(STDEXEC::__declval<_AsyncScope&>()
                                   .nest(STDEXEC::__declval<_Sender&&>()));
} // namespace exec
