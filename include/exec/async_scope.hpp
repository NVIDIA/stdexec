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

    struct async_scope;

    struct __receiver_base {
      void* __op_;
      async_scope* __scope_;
      __receiver_base* __next_;
    };

    template <class _SenderId>
      struct __receiver;

    template <class _Sender>
      using __operation_t = connect_result_t<_Sender, __receiver<__x<_Sender>>>;

    template <class _SenderId>
      struct __future_receiver;

    template <class _Sender>
      using __future_operation_t =
        connect_result_t<_Sender, __future_receiver<__x<_Sender>>>;

    in_place_stop_token __get_stop_token_(async_scope*) noexcept;
    template<class T>
    async_scope* __get_scope_(T* t) noexcept;

    template <class _SenderId>
      struct __receiver
        : private receiver_adaptor<__receiver<_SenderId>>
        , __receiver_base {
        using _Sender = stdexec::__t<_SenderId>;

        template <class _Op>
          explicit __receiver(_Op* __op, async_scope* __scope) noexcept
            : receiver_adaptor<__receiver>{}, __receiver_base{__op, __scope} {
            static_assert(same_as<_Op, std::optional<__operation_t<_Sender>>>);
          }

        // receivers uniquely own themselves; we don't need any special move-
        // construction behaviour, but we do need to ensure no copies are made
        __receiver(__receiver&&) noexcept = default;

        ~__receiver() = default;

        // it's just simpler to skip this
        __receiver& operator=(__receiver&&) = delete;

      private:
        friend receiver_adaptor<__receiver>;

        void set_value() noexcept {
          set_stopped();
        }

        [[noreturn]] void set_error(std::exception_ptr) noexcept {
              std::terminate();
        }

        void set_stopped() noexcept {
          delete static_cast<std::optional<__operation_t<_Sender>>*>(__op_);
        }

        auto get_env() const& {
          return make_env(with(std::execution::get_stop_token, __get_stop_token_(__scope_)));
        }
      };

    enum class __future_state_steps_t {
      __invalid = 0,
      __created,
      __future,
      __no_future,
      __deleted
    };

    template <sender _Sender>
      struct __future_state;

    namespace __impl {
      struct __subscription : __immovable {
        virtual void __complete_() noexcept = 0;
        __subscription* __next_ = nullptr;
      };

      template <class _SenderId, class _ReceiverId>
        class __operation : __subscription {
          using _Sender = __t<_SenderId>;
          using _Receiver = __t<_ReceiverId>;

          struct __forward_stopped {
            __operation* __op_;
            void operator()() noexcept {
              __op_->__state_->__stop_source_.request_stop();
            }
          };

          using __forward_consumer =
            typename stop_token_of_t<env_of_t<_Receiver>&>::template
              callback_type<__forward_stopped>;

          friend void tag_invoke(start_t, __operation& __op_state) noexcept {
            __op_state.__start_();
          }

          void __complete_() noexcept final override try {
            static_assert(sender_to<_Sender, _Receiver>);
            auto __state = std::move(__state_);
            STDEXEC_ASSERT(__state != nullptr);
            std::unique_lock __guard{__state->__mutex_};
            // either the future is still in use or it has passed ownership to __state->__no_future_
            if (__state->__no_future_ != nullptr || __state->__step_ != __future_state_steps_t::__future) {
              // invalid state - there is a code bug in the state machine
              std::terminate();
            } else if (__state->__data_.index() == 2 || get_stop_token(get_env(__rcvr_)).stop_requested()) {
              __guard.unlock();
              set_stopped((_Receiver&&) __rcvr_);
              __guard.lock();
            } else if (__state->__data_.index() == 1) {
              std::apply(
                [this, &__guard](auto&&... __as) {
                  __guard.unlock();
                  set_value((_Receiver&&) __rcvr_, ((decltype(__as)&&) __as)...);
                  __guard.lock();
                },
                std::move(std::get<1>(__state->__data_)));
            } else {
              std::terminate();
            }
          } catch(...) {
            set_error((_Receiver&&) __rcvr_, std::current_exception());
          }

          void __start_() noexcept;

          [[no_unique_address]] _Receiver __rcvr_;
          std::unique_ptr<__future_state<_Sender>> __state_;
          [[no_unique_address]] __forward_consumer __forward_consumer_;

        public:
          ~__operation() noexcept {
            if (__state_ != nullptr) {
              auto __raw_state = __state_.get();
              std::unique_lock __guard{__raw_state->__mutex_};
              if (__raw_state->__data_.index() > 0) {
                // completed given sender
                // state is no longer needed
                return;
              }
              __raw_state->__no_future_ = std::move(__state_);
              __raw_state->__step_from_to_(__guard, __future_state_steps_t::__future, __future_state_steps_t::__no_future);
            }
          }
          template <class _Receiver2>
            explicit __operation(_Receiver2&& __rcvr, std::unique_ptr<__future_state<_Sender>> __state)
              : __rcvr_((_Receiver2 &&) __rcvr)
              , __state_(std::move(__state))
              , __forward_consumer_(get_stop_token(get_env(__rcvr_)), __forward_stopped{this})
            {}
        };

      template <sender _Sender>
        using __future_result_t =
          execution::value_types_of_t<
            _Sender,
            __empty_env,
            execution::__decayed_tuple,
            __single_t>;
    } // namespace __impl

    template <class _SenderId>
      struct __future_receiver
        : private receiver_adaptor<__future_receiver<_SenderId>>
        , __receiver_base {
        using _Sender = stdexec::__t<_SenderId>;

        template <class _State>
          explicit __future_receiver(_State* __state, async_scope* __scope) noexcept
            : receiver_adaptor<__future_receiver<_SenderId>>{}
            , __receiver_base{__state, __scope} {
            static_assert(same_as<_State, __future_state<_Sender>>);
          }

        // receivers uniquely own themselves; we don't need any special move-
        // construction behaviour, but we do need to ensure no copies are made
        __future_receiver(__future_receiver&&) noexcept = default;

        ~__future_receiver() = default;

        // it's just simpler to skip this
        __future_receiver& operator=(__future_receiver&&) = delete;

      private:
        friend receiver_adaptor<__future_receiver>;

        template <class _Sender2 = _Sender, class... _As>
            requires constructible_from<__impl::__future_result_t<_Sender2>, _As...>
          void set_value(_As&&... __as) noexcept {
            auto& __state = *static_cast<__future_state<_Sender>*>(__op_);
            std::unique_lock __guard{__state.__mutex_};
            __state.__data_.template emplace<1>((_As&&) __as...);
            __guard.unlock();
            __dispatch_result_();
          }

        [[noreturn]] void set_error(std::exception_ptr) noexcept {
              std::terminate();
        }

        void set_stopped() noexcept {
          auto& __state = *static_cast<__future_state<_Sender>*>(__op_);
          std::unique_lock __guard{__state.__mutex_};
          __state.__data_.template emplace<2>(execution::set_stopped);
          __guard.unlock();
          __dispatch_result_();
        }

        void __dispatch_result_() noexcept {
          auto& __state = *static_cast<__future_state<_Sender>*>(__op_);
          std::unique_lock __guard{__state.__mutex_};
          auto __local = std::move(__state.__subscribers_);
          __state.__forward_scope_ = std::nullopt;
          if (!!__state.__no_future_) {
            // nobody is waiting for the results
            // delete this and return
            __state.__step_from_to_(__guard, __future_state_steps_t::__no_future, __future_state_steps_t::__deleted);
            __guard.unlock();
            __state.__no_future_.reset();
            return;
          }
          __guard.unlock();
          while(!__local.empty()) {
            auto* __sub = __local.pop_front();
            __sub->__complete_();
          }
        }

        auto get_env() const& {
          return make_env(with(std::execution::get_stop_token, __get_stop_token_(__scope_)));
        }
      };

    template <class _SenderId>
      class __future;

    template <sender _Sender>
      struct __future_state {
        template <class, class>
          friend class __impl::__operation;
        template <class>
          friend struct __future_receiver;
        template <class>
          friend class __future;
        friend struct async_scope;

        ~__future_state() {
          std::unique_lock __guard{__mutex_};
          if (__step_ == __future_state_steps_t::__created) {
            // exception during connect() will end up here
            __step_from_to_(__guard, __future_state_steps_t::__created, __future_state_steps_t::__deleted);
          } else if (__step_ != __future_state_steps_t::__deleted) {
            // completing the given sender before the future is dropped will end here
            __step_from_to_(__guard, __future_state_steps_t::__future, __future_state_steps_t::__deleted);
          }
        }

        __future_state() = default;

        struct __forward_stopped {
          __future_state* __op_;
          void operator()() noexcept {
            __op_->__stop_source_.request_stop();
          }
        };

        using __op_t = __future_operation_t<_Sender>;

        void __step_from_to_(std::unique_lock<std::mutex>& __guard, __future_state_steps_t __from, __future_state_steps_t __to) {
          STDEXEC_ASSERT(__guard.owns_lock());
          auto actual = std::exchange(__step_, __to);
          STDEXEC_ASSERT(actual == __from);
        }
      private:
        std::optional<__op_t> __op_;
        in_place_stop_source __stop_source_;
        std::optional<in_place_stop_callback<__forward_stopped>> __forward_scope_;

        std::mutex __mutex_;
        __future_state_steps_t __step_ = __future_state_steps_t::__created;
        std::unique_ptr<__future_state> __no_future_;
        std::variant<std::monostate, __impl::__future_result_t<_Sender>, execution::set_stopped_t> __data_;
        __intrusive_queue<&__impl::__subscription::__next_> __subscribers_;
      };

    namespace __impl {
      template <class _SenderId, class _ReceiverId>
        inline void __operation<_SenderId, _ReceiverId>::__start_() noexcept try {
          if (!!__state_) {
            std::unique_lock __guard{__state_->__mutex_};
            if (__state_->__data_.index() > 0) {
              __guard.unlock();
              __complete_();
            } else {
              __state_->__subscribers_.push_back(this);
            }
          }
        } catch(...) {
          set_error((_Receiver&&) __rcvr_, std::current_exception());
        }
    } // namespace __impl

    template <class _SenderId>
      class __future {
        using _Sender = __t<_SenderId>;
        friend struct async_scope;
      public:
        ~__future() noexcept {
          if (__state_ != nullptr) {
            auto __raw_state = __state_.get();
            std::unique_lock __guard{__raw_state->__mutex_};
            if (__raw_state->__data_.index() > 0) {
              // completed given sender
              // state is no longer needed
              return;
            }
            __raw_state->__no_future_ = std::move(__state_);
            __raw_state->__step_from_to_(__guard, __future_state_steps_t::__future, __future_state_steps_t::__no_future);
          }
        }
        __future(__future&&) = default;
        __future& operator=(__future&&) = default;
      private:
        explicit __future(std::unique_ptr<__future_state<_Sender>> __state) noexcept
          : __state_(std::move(__state))
        {
            std::unique_lock __guard{__state_->__mutex_};
            __state_->__step_from_to_(__guard, __future_state_steps_t::__created, __future_state_steps_t::__future);
        }

        template <__decays_to<__future> _Self, class _Receiver>
            requires receiver_of<_Receiver, completion_signatures_of_t<_Sender, __empty_env>>
          friend __impl::__operation<_SenderId, __x<decay_t<_Receiver>>>
          tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr) {
            return __impl::__operation<_SenderId, __x<decay_t<_Receiver>>>{
                (_Receiver &&) __rcvr,
                std::move(__self.__state_)};
          }

        template <__decays_to<__future> _Self, class _Env>
          friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
            -> completion_signatures_of_t<__member_t<_Self, _Sender>, _Env>;

        std::unique_ptr<__future_state<_Sender>> __state_;
      };

    struct async_scope : __immovable {
        ~async_scope() noexcept {
            std::unique_lock __guard{__lock_};
            STDEXEC_ASSERT(__active_ == 0);
            STDEXEC_ASSERT(__waiters_.empty());
        }
        async_scope()
          : __active_(0) {}

        struct __op_base : __immovable {
            explicit __op_base(async_scope* __scope) : __scope_(__scope), __next_(nullptr) {}
            async_scope* __scope_;
            __op_base* __next_;
        protected:
            virtual void notify_waiter() noexcept =0;
            friend void __empty_sender_notify_waiter(__op_base& __self) {
                __self.notify_waiter();
            }
        };
        // a constraint that starts a supplied sender when there are no active senders
        struct __when_empty {
          template<class _ConstrainedId>
            struct __sender {
              using _Constrained = __t<_ConstrainedId>;

              template<class _ReceiverId>
                struct __operation : __op_base {
                  using __Receiver = __t<_ReceiverId>;
                  struct __receiver : private receiver_adaptor<__receiver> {
                      __operation* __op_;
                      template <class _Op>
                        explicit __receiver(_Op* __op) noexcept
                          : receiver_adaptor<__receiver>{}, __op_(__op) {
                          static_assert(same_as<_Op, __operation>);
                        }
                  private:
                      friend receiver_adaptor<__receiver>;
                      template<class... _An>
                      void set_value(_An&&... __an) noexcept {
                          std::execution::set_value(std::move(__op_->__rcvr_), (_An&&)__an...);
                      }
                      template<class _Err>
                      void set_error(_Err&& __e) noexcept {
                          std::execution::set_error(std::move(__op_->__rcvr_), __e);
                      }
                      void set_stopped() noexcept {
                          std::execution::set_stopped(std::move(__op_->__rcvr_));
                      }
                      auto get_env() const& {
                        return make_env(with(std::execution::get_stop_token, __get_stop_token_(__get_scope_(__op_))));
                      }
                  };
                  [[no_unique_address]] __Receiver __rcvr_;
                  STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS connect_result_t<_Constrained, __receiver> __op_;
                  template<class _Constrained, class _Receiver>
                    explicit __operation(async_scope* __scope, _Constrained&& __c, _Receiver&& __r)
                      : __op_base(__scope)
                      , __rcvr_((_Receiver&&)__r)
                      , __op_(connect((_Constrained&&)__c, __receiver{this})) {}
                private:
                  void notify_waiter() noexcept override {
                      start(__op_);
                  }

                  void __start_impl() noexcept {
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
                  friend void tag_invoke(start_t, __operation& __self) noexcept {
                      return __self.__start_impl();
                  }
                };
                async_scope* __scope_;
                [[no_unique_address]] _Constrained __c_;
            private:
                template <__decays_to<__sender> _Self, class _Receiver>
                    requires receiver_of<_Receiver, completion_signatures<set_value_t()>>
                  [[nodiscard]] friend __operation<__x<remove_cvref_t<_Receiver>>>
                  tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr) {
                    return __operation<__x<remove_cvref_t<_Receiver>>>{__self.__scope_, ((_Self&&) __self).__c_, (_Receiver&&)__rcvr};
                  }
                template <__decays_to<__sender> _Self, class _Env>
                  friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
                    -> completion_signatures_of_t<__member_t<_Self, _Constrained>, _Env>;
            };
        };
        template<sender _Constrained>
        [[nodiscard]] __when_empty::__sender<__x<remove_cvref_t<_Constrained>>>
          when_empty(_Constrained&& __c) const {
            return __when_empty::__sender<__x<remove_cvref_t<_Constrained>>>{const_cast<async_scope*>(this), __c};
          }
        [[nodiscard]] auto on_empty() const {
          return when_empty(just());
        }
        template<class _ConstrainedId>
          struct __sender {
            using _Constrained = __t<_ConstrainedId>;

            template<class _ReceiverId>
              struct __operation : __immovable {
                using __Receiver = __t<_ReceiverId>;
                struct __receiver : private receiver_adaptor<__receiver> {
                    __operation* __op_;
                    template <class _Op>
                      explicit __receiver(_Op* __op) noexcept
                        : receiver_adaptor<__receiver>{}, __op_(__op) {
                        static_assert(same_as<_Op, __operation>);
                      }
                private:
                    static void __complete(async_scope* __scope) noexcept {
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
                    friend receiver_adaptor<__receiver>;
                    template<class... _An>
                    void set_value(_An&&... an) noexcept {
                        auto __scope = __op_->__scope_;
                        std::execution::set_value(std::move(__op_->__rcvr_), (_An&&)an...);
                        // do not access __op_
                        // do not access this
                        __complete(__scope);
                    }
                    template<class _Err>
                    void set_error(_Err&& __e) noexcept {
                        auto __scope = __op_->__scope_;
                        std::execution::set_error(std::move(__op_->__rcvr_), (_Err&&)__e);
                        // do not access __op_
                        // do not access this
                        __complete(__scope);
                    }
                    void set_stopped() noexcept {
                        auto __scope = __op_->__scope_;
                        std::execution::set_stopped(std::move(__op_->__rcvr_));
                        // do not access __op_
                        // do not access this
                        __complete(__scope);
                    }
                    auto get_env() const& {
                      return make_env(with(std::execution::get_stop_token, __get_stop_token_(__get_scope_(__op_))));
                    }
                };
                async_scope* __scope_;
                [[no_unique_address]] __Receiver __rcvr_;
                STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS connect_result_t<_Constrained, __receiver> __op_;
                template<class _Constrained, class _Receiver>
                  explicit __operation(async_scope* __scope, _Constrained&& __c, _Receiver&& __r)
                    : __scope_(__scope)
                    , __rcvr_((_Receiver&&)__r)
                    , __op_(connect((_Constrained&&)__c, __receiver{this})) {}
            private:
                void __start_impl() noexcept {
                    std::unique_lock __guard{this->__scope_->__lock_};
                    auto& __active = this->__scope_->__active_;
                    ++__active;
                    __guard.unlock();
                    start(this->__op_);
                }
                friend void tag_invoke(start_t, __operation& __self) noexcept {
                  return __self.__start_impl();
                }
            };
            async_scope* __scope_;
            [[no_unique_address]] _Constrained __c_;
        private:
            template <__decays_to<__sender> _Self, class _Receiver>
              requires receiver_of<_Receiver, completion_signatures_of_t<_Constrained, __empty_env>>
              [[nodiscard]] friend __operation<__x<remove_cvref_t<_Receiver>>>
              tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr) {
                return __operation<__x<remove_cvref_t<_Receiver>>>{__self.__scope_, ((_Self&&) __self).__c_, (_Receiver&&)__rcvr};
              }
            template <__decays_to<__sender> _Self, class _Env>
              friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
                -> completion_signatures_of_t<__member_t<_Self, _Constrained>, _Env>;
        };
        template<sender _Constrained>
          using nest_result_t = __sender<__x<remove_cvref_t<_Constrained>>>;
        template<sender _Constrained>
          [[nodiscard]] nest_result_t<_Constrained>
          nest(_Constrained&& __c) {
            return nest_result_t<_Constrained>{this, (_Constrained&&)__c};
          }
        template <sender _Sender>
            requires sender_to<
              nest_result_t<_Sender>,
              __receiver<__x<nest_result_t<_Sender>>>>
          void spawn(_Sender&& __sndr) {
            using __op_t = connect_result_t<
              nest_result_t<_Sender>,
              __receiver<__x<nest_result_t<_Sender>>>>;

            // this could throw; if it does, there's nothing to clean up
            auto __op_to_start = std::make_unique<std::optional<__op_t>>();

            // this could throw; if it does, the only clean-up we need is to
            // deallocate the optional, which is handled by opToStart's
            // destructor so we're good
            __op_to_start->emplace(__conv{[&] {
              return connect(
                  this->nest((_Sender&&) __sndr),
                  __receiver<__x<nest_result_t<_Sender>>>{__op_to_start.get(), this});
            }});

            // start is noexcept so we can assume that the operation will complete
            // after this, which means we can rely on its self-ownership to ensure
            // that it is eventually deleted
            execution::start(**__op_to_start.release());
          }

        template <sender _Sender>
            requires sender_to<nest_result_t<_Sender>, __future_receiver<__x<nest_result_t<_Sender>>>>
          __future<__x<nest_result_t<_Sender>>>
          spawn_future(_Sender&& __sndr) {
            using __state_t = __future_state<nest_result_t<_Sender>>;
            // this could throw; if it does, there's nothing to clean up
            auto __state = std::make_unique<__state_t>();

            // if this throws, there's nothing to clean up
            __state->__forward_scope_.emplace(
                __stop_source_.get_token(),
                typename __state_t::__forward_stopped{__state.get()});

            // this could throw; if it does, the only clean-up we need is to
            // deallocate the optional, which is handled by __op_'s
            // destructor so we're good
            __state->__op_.emplace(__conv{[&] {
              return connect(
                  this->nest((_Sender&&) __sndr),
                  __future_receiver<__x<nest_result_t<_Sender>>>{__state.get(), this});
            }});

            // start is noexcept so we can assume that the operation will complete
            // after this, which means we can rely on its self-ownership to ensure
            // that it is eventually deleted
            execution::start(*__state->__op_);
            return __future<__x<nest_result_t<_Sender>>>{std::move(__state)};
          }

        in_place_stop_source& get_stop_source() noexcept {
          return __stop_source_;
        }

        in_place_stop_token get_stop_token() const noexcept {
          return __stop_source_.get_token();
        }

        bool request_stop() noexcept {
          return __stop_source_.request_stop();
        }
        friend in_place_stop_token __get_stop_token_(async_scope* __scope) noexcept {
          return __scope->__stop_source_.get_token();
        }
    private:
      in_place_stop_source __stop_source_;
      std::mutex __lock_;
      std::ptrdiff_t __active_;
      __intrusive_queue<&__op_base::__next_> __waiters_;
    };

    template<class _T>
    async_scope* __get_scope_(_T* __t) noexcept {
      return __t->__scope_;
    }
  } // namespace __scope

  using __scope::async_scope;
} // namespace exec
