/*
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#include <stdexec/execution.hpp>
#include <stdexec/__detail/__intrusive_queue.hpp>
#include <exec/env.hpp>

namespace exec {
  /////////////////////////////////////////////////////////////////////////////
  // async_scope
  namespace __scope {
    using namespace stdexec;

    struct __impl;

    struct __task : __immovable {
      const __impl* __scope_;
      void (*__notify_waiter)(__task*) noexcept;
      friend void __empty_sender_notify_waiter(__task& __self) {
        __self.__notify_waiter(&__self);
      }
      __task* __next_ = nullptr;
    };

    using __env_t =
      make_env_t<
        with_t<get_stop_token_t, in_place_stop_token>>;

    struct __impl {
      in_place_stop_source __stop_source_{};
      mutable std::mutex __lock_{};
      mutable std::ptrdiff_t __active_ = 0;
      mutable __intrusive_queue<&__task::__next_> __waiters_{};

      ~__impl() {
        std::unique_lock __guard{__lock_};
        STDEXEC_ASSERT(__active_ == 0);
        STDEXEC_ASSERT(__waiters_.empty());
      }

      __env_t __make_env_() const noexcept {
        return make_env(
          with(stdexec::get_stop_token, __stop_source_.get_token()));
      }
    };

    ////////////////////////////////////////////////////////////////////////////
    // async_scope::when_empty implementation
    template <class _ReceiverId>
      struct __when_empty_op_base : __task {
        using _Receiver = __t<_ReceiverId>;
        [[no_unique_address]] _Receiver __rcvr_;
      };

    template <class _ReceiverId>
      struct __when_empty_rcvr {
        __when_empty_op_base<_ReceiverId>* __op_;

        template <
            __one_of<set_value_t, set_error_t, set_stopped_t> _Tag,
            class... _As _NVCXX_CAPTURE_PACK(_As)>
          friend void tag_invoke(_Tag, __when_empty_rcvr&& __self, _As&&... __as) noexcept {
            _NVCXX_EXPAND_PACK(_As, __as,
              _Tag{}(std::move(__self.__op_->__rcvr_), (_As&&) __as...);
            )
        }
        friend __env_t tag_invoke(get_env_t, const __when_empty_rcvr& __self) noexcept {
          return __self.__op_->__scope_->__make_env_();
        }
      };

    template <class _ConstrainedId, class _ReceiverId>
      struct __when_empty_op : __when_empty_op_base<_ReceiverId> {
        using _Constrained = __t<_ConstrainedId>;
        using _Receiver = __t<_ReceiverId>;
        STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
          connect_result_t<_Constrained, __when_empty_rcvr<_ReceiverId>> __op_;
        template <__decays_to<_Constrained> _Sender, class _Receiver>
          explicit __when_empty_op(const __impl* __scope, _Sender&& __sndr, _Receiver&& __rcvr)
            : __when_empty_op_base<_ReceiverId>{
                {{}, __scope, __notify_waiter},
                (_Receiver&&) __rcvr}
            , __op_(connect((_Sender&&) __sndr, __when_empty_rcvr<_ReceiverId>{this}))
          {}
      private:
        static void __notify_waiter(__task* __self) noexcept {
          start(static_cast<__when_empty_op*>(__self)->__op_);
        }

        void __start_() noexcept {
          std::unique_lock __guard{this->__scope_->__lock_};
          auto& __active = this->__scope_->__active_;
          auto& __waiters = this->__scope_->__waiters_;
          if (__active != 0) {
            __waiters.push_back(this);
            return;
          }
          __guard.unlock();
          start(this->__op_);
        }
        friend void tag_invoke(start_t, __when_empty_op& __self) noexcept {
          return __self.__start_();
        }
      };

    template <class _ConstrainedId>
      struct __when_empty_sender {
        using _Constrained = __t<_ConstrainedId>;

        const __impl* __scope_;
        [[no_unique_address]] _Constrained __c_;
      private:
        template <class _Receiver>
          using __when_empty_op_t =
            __when_empty_op<_ConstrainedId, __x<remove_cvref_t<_Receiver>>>;
        template <__decays_to<__when_empty_sender> _Self, class _Receiver>
            requires receiver_of<_Receiver, completion_signatures<set_value_t()>>
          [[nodiscard]] friend __when_empty_op_t<_Receiver>
          tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr) {
            return __when_empty_op_t<_Receiver>{
              __self.__scope_,
              ((_Self&&) __self).__c_,
              (_Receiver&&)__rcvr};
          }
        template <__decays_to<__when_empty_sender> _Self, class _Env>
          friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
            -> completion_signatures_of_t<__member_t<_Self, _Constrained>, _Env>;
      };

    template <class _Constrained>
      using __when_empty_sender_t =
        __when_empty_sender<__x<remove_cvref_t<_Constrained>>>;

    ////////////////////////////////////////////////////////////////////////////
    // async_scope::nest implementation
    template <class _ReceiverId>
      struct __nest_op_base : __immovable {
        using _Receiver = __t<_ReceiverId>;
        const __impl* __scope_;
        [[no_unique_address]] _Receiver __rcvr_;
      };

    template <class _ReceiverId>
      struct __nest_rcvr {
        using _Receiver = __t<_ReceiverId>;
        __nest_op_base<_ReceiverId>* __op_;

        static void __complete(const __impl* __scope) noexcept {
          std::unique_lock __guard{__scope->__lock_};
          auto& __active = __scope->__active_;
          if (--__active == 0) {
            auto __local = std::move(__scope->__waiters_);
            __guard.unlock();
            __scope = nullptr;
            // do not access __scope
            while (!__local.empty()) {
              auto __next = __local.pop_front();
              __empty_sender_notify_waiter(*__next);
              // __scope must be considered deleted
            }
          }
        }

        template <
            __one_of<set_value_t, set_error_t, set_stopped_t> _Tag,
            class... _As _NVCXX_CAPTURE_PACK(_As)>
            requires __callable<_Tag, _Receiver, _As...>
          friend void tag_invoke(_Tag, __nest_rcvr&& __self, _As&&... __as) noexcept {
            auto __scope = __self.__op_->__scope_;
            _NVCXX_EXPAND_PACK(_As, __as,
              _Tag{}(std::move(__self.__op_->__rcvr_), (_As&&) __as...);
            )
            // do not access __op_
            // do not access this
            __complete(__scope);
          }

        friend __env_t tag_invoke(get_env_t, const __nest_rcvr& __self) noexcept {
          return __self.__op_->__scope_->__make_env_();
        }
      };

    template <class _ConstrainedId, class _ReceiverId>
      struct __nest_op : __nest_op_base<_ReceiverId> {
        using _Constrained = __t<_ConstrainedId>;
        using _Receiver = __t<_ReceiverId>;
        STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
          connect_result_t<_Constrained, __nest_rcvr<_ReceiverId>> __op_;
        template <class _Constrained, class _Receiver>
          explicit __nest_op(const __impl* __scope, _Constrained&& __c, _Receiver&& __rcvr)
            : __nest_op_base<_ReceiverId>{{}, __scope, (_Receiver&&) __rcvr}
            , __op_(connect((_Constrained&&)__c, __nest_rcvr<_ReceiverId>{this})) {}
      private:
        void __start_() noexcept {
          STDEXEC_ASSERT(this->__scope_);
          std::unique_lock __guard{this->__scope_->__lock_};
          auto& __active = this->__scope_->__active_;
          ++__active;
          __guard.unlock();
          start(__op_);
        }
        friend void tag_invoke(start_t, __nest_op& __self) noexcept {
          return __self.__start_();
        }
      };

    template <class _ConstrainedId>
      struct __nest_sender {
        using _Constrained = __t<_ConstrainedId>;
        const __impl* __scope_;
        [[no_unique_address]] _Constrained __c_;
      private:
        template <class _Receiver>
          using __nest_operation_t = __nest_op<_ConstrainedId, __x<remove_cvref_t<_Receiver>>>;
        template <class _Receiver>
          using __nest_receiver_t = __nest_rcvr<__x<_Receiver>>;

        template <__decays_to<__nest_sender> _Self, class _Receiver>
            requires sender_to<__member_t<_Self, _Constrained>, __nest_receiver_t<_Receiver>>
          [[nodiscard]] friend __nest_operation_t<_Receiver>
          tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr) {
            return __nest_operation_t<_Receiver>{
              __self.__scope_,
              ((_Self&&) __self).__c_,
              (_Receiver&&)__rcvr};
          }
        template <__decays_to<__nest_sender> _Self, class _Env>
          friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
            -> completion_signatures_of_t<__member_t<_Self, _Constrained>, _Env>;
      };

    template <class _Constrained>
      using __nest_sender_t =
        __nest_sender<__x<remove_cvref_t<_Constrained>>>;

    ////////////////////////////////////////////////////////////////////////////
    // async_scope::spawn_future implementation
    enum class __future_step {
      __invalid = 0,
      __created,
      __future,
      __no_future,
      __deleted
    };

    struct __future_state_base {
      in_place_stop_source __stop_source_;
    };

    template <class _Sender>
      struct __future_state;

    struct __forward_stopped {
      __future_state_base* __state_;
      void operator()() noexcept {
        __state_->__stop_source_.request_stop();
      }
    };

    struct __subscription : __immovable {
      void (*__complete_)(__subscription*) noexcept = nullptr;
      void __complete() noexcept {
        __complete_(this);
      }
      __subscription* __next_ = nullptr;
    };

    template <class _SenderId, class _ReceiverId>
      class __future_op : __subscription {
        using _Sender = __t<_SenderId>;
        using _Receiver = __t<_ReceiverId>;

        using __forward_consumer =
          typename stop_token_of_t<env_of_t<_Receiver>>::template
            callback_type<__forward_stopped>;

        friend void tag_invoke(start_t, __future_op& __self) noexcept {
          __self.__start_();
        }

        void __complete_() noexcept try {
          static_assert(sender_to<_Sender, _Receiver>);
          auto __state = std::move(__state_);
          STDEXEC_ASSERT(__state != nullptr);
          std::unique_lock __guard{__state->__mutex_};
          // either the future is still in use or it has passed ownership to __state->__no_future_
          if (__state->__no_future_ != nullptr || __state->__step_ != __future_step::__future) {
            // invalid state - there is a code bug in the state machine
            std::terminate();
          } else if (get_stop_token(get_env(__rcvr_)).stop_requested()) {
            __guard.unlock();
            set_stopped((_Receiver&&) __rcvr_);
            __guard.lock();
          } else {
            std::visit(
              [this, &__guard]<class _Tup>(_Tup& __tup) {
                if constexpr (same_as<_Tup, std::monostate>) {
                  std::terminate();
                } else {
                  std::apply(
                    [this, &__guard]<class... _As>(auto tag, _As&... __as) {
                      __guard.unlock();
                      tag((_Receiver&&) __rcvr_, (_As&&) __as...);
                      __guard.lock();
                    },
                    __tup);
                }
              },
              __state->__data_);
          }
        } catch(...) {
          set_error((_Receiver&&) __rcvr_, std::current_exception());
        }

        void __start_() noexcept try {
          if (!!__state_) {
            std::unique_lock __guard{__state_->__mutex_};
            if (__state_->__data_.index() != 0) {
              __guard.unlock();
              __complete_();
            } else {
              __state_->__subscribers_.push_back(this);
            }
          }
        } catch(...) {
          set_error((_Receiver&&) __rcvr_, std::current_exception());
        }

        [[no_unique_address]] _Receiver __rcvr_;
        std::unique_ptr<__future_state<_Sender>> __state_;
        [[no_unique_address]] __forward_consumer __forward_consumer_;

      public:
        ~__future_op() noexcept {
          if (__state_ != nullptr) {
            auto __raw_state = __state_.get();
            std::unique_lock __guard{__raw_state->__mutex_};
            if (__raw_state->__data_.index() > 0) {
              // completed given sender
              // state is no longer needed
              return;
            }
            __raw_state->__no_future_ = std::move(__state_);
            __raw_state->__step_from_to_(__guard, __future_step::__future, __future_step::__no_future);
          }
        }
        template <class _Receiver2>
          explicit __future_op(_Receiver2&& __rcvr, std::unique_ptr<__future_state<_Sender>> __state)
            : __subscription{{}, [](__subscription* __self) noexcept -> void {
                static_cast<__future_op*>(__self)->__complete_();
              }}
            , __rcvr_((_Receiver2 &&) __rcvr)
            , __state_(std::move(__state))
            , __forward_consumer_(
                get_stop_token(get_env(__rcvr_))
              , __forward_stopped{__state_.get()})
          {}
      };

    template <class _Sender>
      using __future_result_t =
        __value_types_of_t<
          _Sender,
          __env_t,
          __mbind_front_q<__decayed_tuple, set_value_t>,
          __q<__single_t>>;

    template <class _SenderId>
      struct __future_rcvr {
        using _Sender = __t<_SenderId>;
        __future_state<_Sender>* __state_;
        const __impl* __scope_;

        void __dispatch_result_() noexcept {
          auto& __state = *__state_;
          std::unique_lock __guard{__state.__mutex_};
          auto __local = std::move(__state.__subscribers_);
          __state.__forward_scope_ = std::nullopt;
          if (!!__state.__no_future_) {
            // nobody is waiting for the results
            // delete this and return
            __state.__step_from_to_(__guard, __future_step::__no_future, __future_step::__deleted);
            __guard.unlock();
            __state.__no_future_.reset();
            return;
          }
          __guard.unlock();
          while(!__local.empty()) {
            auto* __sub = __local.pop_front();
            __sub->__complete();
          }
        }

        template <
            __one_of<set_value_t, set_stopped_t> _Tag,
            __movable_value... _As _NVCXX_CAPTURE_PACK(_As)>
          friend void tag_invoke(_Tag, __future_rcvr&& __self, _As&&... __as) noexcept {
            auto& __state = *__self.__state_;
            std::unique_lock __guard{__state.__mutex_};
            _NVCXX_EXPAND_PACK(_As, __as,
              using _Tuple = __decayed_tuple<_Tag, _As...>;
              __state.__data_.template emplace<_Tuple>(_Tag{}, (_As&&) __as...);
            )
            __guard.unlock();
            __self.__dispatch_result_();
          }

        [[noreturn]] friend void tag_invoke(set_error_t, __future_rcvr&&, std::exception_ptr) noexcept {
          std::terminate();
        }

        friend __env_t tag_invoke(get_env_t, const __future_rcvr& __self) {
          return __self.__scope_->__make_env_();
        }
      };

    template <class _Sender>
      using __future_receiver_t = __future_rcvr<__x<_Sender>>;

    template <class _Sender>
      struct __future_state : __future_state_base {
        __future_state(_Sender __sndr, const __impl* __scope)
          : __forward_scope_{
              std::in_place,
              __scope->__stop_source_.get_token(),
              __forward_stopped{this}}
          , __op_(connect(
              (_Sender&&) __sndr,
              __future_receiver_t<_Sender>{this, __scope}))
        {}

        ~__future_state() {
          std::unique_lock __guard{__mutex_};
          if (__step_ == __future_step::__created) {
            // exception during connect() will end up here
            __step_from_to_(__guard, __future_step::__created, __future_step::__deleted);
          } else if (__step_ != __future_step::__deleted) {
            // completing the given sender before the future is dropped will end here
            __step_from_to_(__guard, __future_step::__future, __future_step::__deleted);
          }
        }

        void __step_from_to_(std::unique_lock<std::mutex>& __guard, __future_step __from, __future_step __to) {
          STDEXEC_ASSERT(__guard.owns_lock());
          auto actual = std::exchange(__step_, __to);
          STDEXEC_ASSERT(actual == __from);
        }

        std::optional<in_place_stop_callback<__forward_stopped>> __forward_scope_;
        connect_result_t<_Sender, __future_rcvr<__x<_Sender>>> __op_;

        std::mutex __mutex_;
        __future_step __step_ = __future_step::__created;
        std::unique_ptr<__future_state> __no_future_;
        std::variant<std::monostate, __future_result_t<_Sender>, std::tuple<set_stopped_t>> __data_;
        __intrusive_queue<&__subscription::__next_> __subscribers_;
      };

    template <class _SenderId>
      class __future {
        using _Sender = __t<_SenderId>;
        friend struct async_scope;
      public:
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
            __raw_state->__step_from_to_(__guard, __future_step::__future, __future_step::__no_future);
          }
        }
        __future(__future&&) = default;
        __future& operator=(__future&&) = default;
      private:
        explicit __future(std::unique_ptr<__future_state<_Sender>> __state) noexcept
          : __state_(std::move(__state)) {
            std::unique_lock __guard{__state_->__mutex_};
            __state_->__step_from_to_(__guard, __future_step::__created, __future_step::__future);
        }

        template <__decays_to<__future> _Self, class _Receiver>
            requires receiver_of<_Receiver, completion_signatures_of_t<_Sender, __env_t>>
          friend __future_op<_SenderId, __x<decay_t<_Receiver>>>
          tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr) {
            return __future_op<_SenderId, __x<decay_t<_Receiver>>>{
                (_Receiver &&) __rcvr,
                std::move(__self.__state_)};
          }

        template <__decays_to<__future> _Self, class _Env>
          friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
            -> completion_signatures_of_t<__member_t<_Self, _Sender>, _Env>;

        std::unique_ptr<__future_state<_Sender>> __state_;
      };

    template <class _Sender>
      using __future_t = __future<__x<__nest_sender_t<_Sender>>>;

    ////////////////////////////////////////////////////////////////////////////
    // async_scope::spawn implementation
    template <class _SenderId>
      struct __spawn_op;

    template <class _SenderId>
      struct __spawn_rcvr {
        using _Sender = __t<_SenderId>;
        __spawn_op<_SenderId>* __op_;
        const __impl* __scope_;

        template <__one_of<set_value_t, set_stopped_t> _Tag>
          friend void tag_invoke(_Tag, __spawn_rcvr&& __self) noexcept {
            delete __self.__op_;
          }

        [[noreturn]] friend void tag_invoke(set_error_t, __spawn_rcvr&&, std::exception_ptr) noexcept {
          std::terminate();
        }

        friend __env_t tag_invoke(get_env_t, const __spawn_rcvr&__self) noexcept {
          return __self.__scope_->__make_env_();
        }
      };

    template <class _Sender>
      using __spawn_receiver_t =
        __spawn_rcvr<__x<_Sender>>;

    template <class _SenderId>
      struct __spawn_op {
        using _Sender = __t<_SenderId>;

        template <__decays_to<_Sender> _Sndr>
          __spawn_op(_Sndr&& __sndr, const __impl* __scope)
            : __op_(connect((_Sndr&&) __sndr, __spawn_rcvr<_SenderId>{this, __scope}))
          {}

        void __start_() noexcept {
          start(__op_);
        }

        friend void tag_invoke(start_t, __spawn_op& __self) noexcept {
          return __self.__start_();
        }

        connect_result_t<_Sender, __spawn_rcvr<_SenderId>> __op_;
      };

    template <class _Sender>
      using __spawn_operation_t = __spawn_op<__x<_Sender>>;

    ////////////////////////////////////////////////////////////////////////////
    // async_scope
    struct async_scope : __immovable {
      async_scope() = default;

      template <sender _Constrained>
      [[nodiscard]] __when_empty_sender_t<_Constrained>
        when_empty(_Constrained&& __c) const {
          return __when_empty_sender_t<_Constrained>{&__impl_, __c};
        }
      [[nodiscard]] auto on_empty() const {
        return when_empty(just());
      }

      template <sender _Constrained>
        using nest_result_t = __nest_sender_t<_Constrained>;

      template <sender _Constrained>
        [[nodiscard]] nest_result_t<_Constrained>
        nest(_Constrained&& __c) {
          return nest_result_t<_Constrained>{&__impl_, (_Constrained&&) __c};
        }

      template <sender _Sender>
          requires sender_to<
            nest_result_t<_Sender>,
            __spawn_receiver_t<nest_result_t<_Sender>>>
        void spawn(_Sender&& __sndr) {
          using __op_t = __spawn_operation_t<nest_result_t<_Sender>>;
          // start is noexcept so we can assume that the operation will complete
          // after this, which means we can rely on its self-ownership to ensure
          // that it is eventually deleted
          stdexec::start(*new __op_t{nest((_Sender&&) __sndr), &__impl_});
        }

      template <sender _Sender>
          requires sender_to<
            nest_result_t<_Sender>,
            __future_receiver_t<nest_result_t<_Sender>>>
        __future_t<_Sender> spawn_future(_Sender&& __sndr) {
          using __state_t = __future_state<nest_result_t<_Sender>>;
          auto __state =
            std::make_unique<__state_t>(nest((_Sender&&) __sndr), &__impl_);
          stdexec::start(__state->__op_);
          return __future_t<_Sender>{std::move(__state)};
        }

      in_place_stop_source& get_stop_source() noexcept {
        return __impl_.__stop_source_;
      }

      in_place_stop_token get_stop_token() const noexcept {
        return __impl_.__stop_source_.get_token();
      }

      bool request_stop() noexcept {
        return __impl_.__stop_source_.request_stop();
      }

    private:
      __impl __impl_;
    };
  } // namespace __scope

  using __scope::async_scope;
} // namespace exec
