/*
 * Copyright (c) NVIDIA
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

#include <async_manual_reset_event.hpp>
#include <execution.hpp>

namespace std::execution {
  /////////////////////////////////////////////////////////////////////////////
  // async_scope
  namespace __scope {
    struct async_scope;

    struct __receiver_base {
      void* __op_;
      async_scope* __scope_;
    };

    template <class _SenderId, class _ReceiverId>
      struct __nest_receiver;

    template <class _Sender, class _Receiver>
      using __nest_operation_t = connect_result_t<_Sender, __nest_receiver<__x<_Sender>, __x<_Receiver>>>;

    template <class _SenderId>
      struct __receiver;

    template <class _Sender>
      using __operation_t = connect_result_t<_Sender, __receiver<__x<_Sender>>>;

    template <class _SenderId>
      struct __future_receiver;

    template <class _Sender>
      using __future_operation_t = connect_result_t<_Sender, __future_receiver<__x<_Sender>>>;

    bool __try_record_start_(async_scope*) noexcept;
    void __record_done_(async_scope*) noexcept;

   namespace __impl {
      template <class _SenderId, class _ReceiverId>
        class __nest_operation : __immovable {
          using _Sender = __t<_SenderId>;
          using _Receiver = __t<_ReceiverId>;

          friend void tag_invoke(start_t, __nest_operation& __op_state) noexcept {
            if (!std::exchange(__op_state.__nested_, false)) { terminate(); }
            __op_state.__start_();
          }

          void __start_() noexcept;

          friend __nest_receiver<_SenderId, _ReceiverId>;

          [[no_unique_address]] _Receiver __rcvr_;
          async_scope* __scope_;
          __nest_operation_t<_Sender, _Receiver> __op_;
          bool __nested_;

        public:
          ~__nest_operation(){
            if(__nested_) {
              __record_done_(__scope_);
            }
          }
          template <class _Receiver2>
            explicit __nest_operation(_Receiver2&& __rcvr, _Sender&& __sndr, async_scope* __scope)
              : __rcvr_((_Receiver2 &&) __rcvr) 
              , __scope_(__scope) 
              , __op_(connect(std::move(__sndr), __nest_receiver<_SenderId, _ReceiverId>{this, __scope}))
              , __nested_(true) {}
        };
    } // namespace __impl

      template <class _SenderId, class _ReceiverId>
      struct __nest_receiver
        : private receiver_adaptor<__nest_receiver<_SenderId, _ReceiverId>>, __receiver_base {
        using _Sender = __t<_SenderId>;

        template<class _State>
          explicit __nest_receiver(_State* __state, async_scope* __scope) noexcept
            : receiver_adaptor<__nest_receiver<_SenderId, _ReceiverId>>{}
            , __receiver_base{__state, __scope} {
            static_assert(same_as<_State, __impl::__nest_operation<_SenderId, _ReceiverId>>);
          }

        // receivers uniquely own themselves; we don't need any special move-
        // construction behaviour, but we do need to ensure no copies are made
        __nest_receiver(__nest_receiver&&) noexcept = default;

        ~__nest_receiver() = default;

        // it's just simpler to skip this
        __nest_receiver& operator=(__nest_receiver&&) = delete;

      private:
        friend receiver_adaptor<__nest_receiver>;

        __impl::__nest_operation<_SenderId, _ReceiverId>& get_state() {
          return *reinterpret_cast<__impl::__nest_operation<_SenderId, _ReceiverId>*>(__op_);
        }

        template <class... _As>
          void set_value(_As&&... __as) noexcept {
            execution::set_value(std::move(get_state().__rcvr_), (_As &&) __as...);
            __record_done_(__scope_);
          }
        template<class _Error>
        [[noreturn]] void set_error(_Error&& e) noexcept {
          execution::set_error(std::move(get_state().__rcvr_), (_Error &&) e);
          __record_done_(__scope_);
        }
        void set_stopped() noexcept {
          execution::set_stopped(std::move(get_state().__rcvr_));
          __record_done_(__scope_);
        }

        make_env_t<get_stop_token_t, never_stop_token> get_env() const& {
          return make_env<get_stop_token_t>(never_stop_token{});
        }
      };

    namespace __impl {
      template <class _SenderId, class _ReceiverId>
        inline void __nest_operation<_SenderId, _ReceiverId>::__start_() noexcept {
          start(__op_);
        }
    } // namespace __impl

    template <class _SenderId>
      class __nest {
        using _Sender = __t<_SenderId>;
        friend struct async_scope;
        template <class _Receiver>
          using __completions = completion_signatures_of_t<_Sender, env_of_t<_Receiver>>;

        public:

        ~__nest() {
          if (__nested_) {
            __record_done_(__scope_);
          }
        }

        private:

        template<class _Sender2>
        explicit __nest(_Sender2&& __sndr, async_scope* __scope) noexcept
          : __sndr_((_Sender2 &&) __sndr)
          , __scope_(__scope)
          , __nested_(__try_record_start_(this->__scope_)) {
            if (!__nested_) { terminate(); } // code bug
          }

        __nest(const __nest&) noexcept = delete;
        __nest& operator=(const __nest&) noexcept = delete;

        public:

        __nest(__nest&& o) noexcept 
          : __sndr_(std::move(o.__sndr_))
          , __scope_(std::exchange(o.__scope_, nullptr))
          , __nested_(std::exchange(o.__nested_, false)) {
            if (!__nested_) { terminate(); } // code bug
          }
        __nest& operator=(__nest&& o) noexcept {
            __nest expired = std::move(*this);
            this->~__nest();
            new(this) __nest{std::move(o)};
            return *this;
        }

        private:

        template <class _Receiver>
            requires receiver_of<_Receiver, __completions<_Receiver>>
          friend __impl::__nest_operation<_SenderId, __x<decay_t<_Receiver>>>
          tag_invoke(connect_t, __nest&& __self, _Receiver&& __rcvr) {
            if (!std::exchange(__self.__nested_, false)) { terminate(); } // code bug - moved-from
            try {
              return __impl::__nest_operation<_SenderId, __x<decay_t<_Receiver>>>{
                  (_Receiver &&) __rcvr, std::move(__self.__sndr_), __self.__scope_};
            } catch(...) {
              __record_done_(__self.__scope_);
              throw;
            }
          }

        template <__decays_to<__nest> _Self, class _Env>
          friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
            -> completion_signatures_of_t<__member_t<_Self, _Sender>, _Env>;

        [[no_unique_address]] _Sender __sndr_;
        async_scope* __scope_;
        bool __nested_;
      };

    template <class _SenderId>
      struct __receiver
        : private receiver_adaptor<__receiver<_SenderId>>, __receiver_base {
        using _Sender = __t<_SenderId>;

        template <class Op>
          explicit __receiver(Op* __op, async_scope* __scope) noexcept
            : receiver_adaptor<__receiver>{}, __receiver_base{__op, __scope} {
            static_assert(same_as<Op, optional<__operation_t<_Sender>>>);
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

        [[noreturn]] void set_error(exception_ptr) noexcept {
          terminate();
        }

        void set_stopped() noexcept {
          // we're about to delete this, so save the __scope for later
          auto __scope = __scope_;
          delete static_cast<optional<__operation_t<_Sender>>*>(__op_);
          __record_done_(__scope);
        }

        make_env_t<get_stop_token_t, never_stop_token> get_env() const& {
          return make_env<get_stop_token_t>(never_stop_token{});
        }
      };

    template <sender _Sender>
      class __future_state;

    namespace __impl {
      struct __subscription : __immovable {
        virtual void __complete_() noexcept = 0;
        __subscription* __next_ = nullptr;
      };

      template <class _SenderId, class _ReceiverId>
        class __operation : __subscription {
          using _Sender = __t<_SenderId>;
          using _Receiver = __t<_ReceiverId>;

          friend void tag_invoke(start_t, __operation& __op_state) noexcept {
            __op_state.__start_();
          }

          void __complete_() noexcept final override try {
            static_assert(sender_to<_Sender, _Receiver>);
            if (__state_->__data_.index() == 2 || get_stop_token(get_env(__rcvr_)).stop_requested()) {
              set_stopped((_Receiver&&) __rcvr_);
            } else if (__state_->__data_.index() == 1) {
              std::apply(
                [this](auto&&... as) {
                  set_value((_Receiver&&) __rcvr_, ((decltype(as)&&)as)...);
                },
                std::move(std::get<1>(__state_->__data_)));
            } else {
              terminate();
            }
          } catch(...) {
            set_error((_Receiver&&) __rcvr_, current_exception());
          }

          void __start_() noexcept;

          [[no_unique_address]] _Receiver __rcvr_;
          unique_ptr<__future_state<_Sender>> __state_;

        public:
          template <class _Receiver2>
            explicit __operation(_Receiver2&& __rcvr, unique_ptr<__future_state<_Sender>> __state)
              : __rcvr_((_Receiver2 &&) __rcvr)
              , __state_(std::move(__state)) {}
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
        : private receiver_adaptor<__future_receiver<_SenderId>>, __receiver_base {
        using _Sender = __t<_SenderId>;

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
          void set_value(_As&&... __as) noexcept try {
            auto& state = *reinterpret_cast<__future_state<_Sender>*>(__op_);
            state.__data_.template emplace<1>((_As&&) __as...);
            __dispatch_result_();
          } catch(...) {
            terminate();
          }
        [[noreturn]] void set_error(exception_ptr) noexcept {
          terminate();
        }
        void set_stopped() noexcept {
          auto& state = *reinterpret_cast<__future_state<_Sender>*>(__op_);
          state.__data_.template emplace<2>(execution::set_stopped);
          __dispatch_result_();
        }

        void __dispatch_result_() {
          auto& state = *reinterpret_cast<__future_state<_Sender>*>(__op_);
          while(auto* __sub = state.__pop_front_()) {
            __sub->__complete_();
          }
          __record_done_(__scope_);
        }

        make_env_t<get_stop_token_t, never_stop_token> get_env() const& {
          return make_env<get_stop_token_t>(never_stop_token{});
        }
      };

    template <class _SenderId>
      class __future;

    template <sender _Sender>
      class __future_state : std::enable_shared_from_this<__future_state<_Sender>> {
        template <class, class>
          friend class __impl::__operation;
        template <class>
          friend struct __future_receiver;
        template <class>
          friend class __future;
        friend struct async_scope;

        using __op_t = __future_operation_t<_Sender>;

        optional<__op_t> __op_;
        variant<monostate, __impl::__future_result_t<_Sender>, execution::set_stopped_t> __data_;

        void __push_back_(__impl::__subscription* __task);
        __impl::__subscription* __pop_front_();

        mutex __mutex_;
        condition_variable __cv_;
        __impl::__subscription* __head_ = nullptr;
        __impl::__subscription* __tail_ = nullptr;
      };

    namespace __impl {
      template <class _SenderId, class _ReceiverId>
        inline void __operation<_SenderId, _ReceiverId>::__start_() noexcept try {
          if (!!__state_) {
            if (__state_->__data_.index() > 0) {
              __complete_();
            } else {
              __state_->__push_back_(this);
            }
          } else {
            set_stopped((_Receiver&&) __rcvr_);
          }
        } catch(...) {
          set_error((_Receiver&&) __rcvr_, current_exception());
        }
    } // namespace __impl

    template <sender _Sender>
      inline void __future_state<_Sender>::__push_back_(__impl::__subscription* __subscription) {
        unique_lock __lock{__mutex_};
        if (__head_ == nullptr) {
          __head_ = __subscription;
        } else {
          __tail_->__next_ = __subscription;
        }
        __tail_ = __subscription;
        __subscription->__next_ = nullptr;
        __cv_.notify_one();
      }

    template <sender _Sender>
      inline __impl::__subscription* __future_state<_Sender>::__pop_front_() {
        unique_lock __lock{__mutex_};
        if (__head_ == nullptr) {
          return nullptr;
        }
        auto* __subscription = __head_;
        __head_ = __subscription->__next_;
        if (__head_ == nullptr)
          __tail_ = nullptr;
        return __subscription;
      }

    template <class _SenderId>
      class __future {
        using _Sender = __t<_SenderId>;
        friend struct async_scope;
        template <class _Receiver>
          using __completions = completion_signatures_of_t<_Sender, env_of_t<_Receiver>>;

        explicit __future(unique_ptr<__future_state<_Sender>> __state) noexcept
          : __state_(std::move(__state))
        {}

        template <class _Receiver>
            requires receiver_of<_Receiver, __completions<_Receiver>>
          friend __impl::__operation<_SenderId, __x<decay_t<_Receiver>>>
          tag_invoke(connect_t, __future&& __self, _Receiver&& __rcvr) {
            return __impl::__operation<_SenderId, __x<decay_t<_Receiver>>>{
                (_Receiver &&) __rcvr,
                std::move(__self.__state_)};
          }

        template <__decays_to<__future> _Self, class _Env>
          friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
            -> completion_signatures_of_t<__member_t<_Self, _Sender>, _Env>;

        unique_ptr<__future_state<_Sender>> __state_;
      };

    struct async_scope {
    private:
      struct end_of_scope_callback {
          async_scope* __async_scope_;
          void operator()() {
            __async_scope_->__end_of_scope_(); // enter stop state
          }
      };
      in_place_stop_source __stop_source_;
      in_place_stop_callback<end_of_scope_callback> __end_of_scope_callback_;
      // (__op_state_ & 1) is 1 until we've been stopped
      // (__op_state_ >> 1) is the number of outstanding operations
      atomic<size_t> __op_state_{1};
      async_manual_reset_event __evt_;

      [[nodiscard]] auto __await_and_sync_() const noexcept {
        return then(__evt_.async_wait(),
        [this]() noexcept {
          // make sure to synchronize with all the fetch_subs being done while
          // operations complete
          (void)__op_state_.load(std::memory_order_acquire);
        });
      }

    public:
      async_scope() noexcept : __end_of_scope_callback_(__stop_source_.get_token(), end_of_scope_callback{this}) {};

      ~async_scope() {
        __end_of_scope_(); // enter stop state

        [[maybe_unused]] auto __state = __op_state_.load(std::memory_order_relaxed);

        if (!__is_stopping_(__state) || __op_count_(__state) != 0) {
          terminate();
        }
      }

      template <class _Sender>
        __nest<__x<remove_cvref_t<_Sender>>> nest(_Sender&& __sndr) {
          return __nest<__x<remove_cvref_t<_Sender>>>{(_Sender &&) __sndr, this};
        }

      template <class _Sender>
          requires sender_to<_Sender, __receiver<__x<remove_cvref_t<_Sender>>>>
        void spawn(_Sender&& __sndr) {
          using __op_t = __operation_t<remove_cvref_t<_Sender>>;

          // this could throw; if it does, there's nothing to clean up
          auto __op_to_start = make_unique<optional<__op_t>>();

          // this could throw; if it does, the only clean-up we need is to
          // deallocate the optional, which is handled by opToStart's
          // destructor so we're good
          __op_to_start->emplace(__conv{[&] {
            return connect(
                (_Sender&&) __sndr,
                __receiver<__x<remove_cvref_t<_Sender>>>{__op_to_start.get(), this});
          }});

          // At this point, the rest of the function is noexcept, but __op_to_start's
          // destructor is no longer enough to properly clean up because it won't
          // invoke destruct(). We need to ensure that we either call destruct()
          // ourselves or complete the operation so *it* can call destruct().

          if (__try_record_start_(this)) {
            // start is noexcept so we can assume that the operation will complete
            // after this, which means we can rely on its self-ownership to ensure
            // that it is eventually deleted
            execution::start(**__op_to_start.release());
          }
        }

      template <class _Sender>
          requires sender_to<_Sender, __future_receiver<__x<remove_cvref_t<_Sender>>>>
        __future<__x<remove_cvref_t<_Sender>>> spawn_future(_Sender&& __sndr) {
          // this could throw; if it does, there's nothing to clean up
          auto __state = make_unique<__future_state<remove_cvref_t<_Sender>>>();

          // this could throw; if it does, the only clean-up we need is to
          // deallocate the optional, which is handled by __op_to_start's
          // destructor so we're good
          __state->__op_.emplace(__conv{[&] {
            return connect(
                (_Sender&&) __sndr,
                __future_receiver<__x<remove_cvref_t<_Sender>>>{__state.get(), this});
          }});

          // At this point, the rest of the function is noexcept, but __op_to_start's
          // destructor is no longer enough to properly clean up because it won't
          // invoke destruct().  We need to ensure that we either call destruct()
          // ourselves or complete the operation so *it* can call destruct().

          if (__try_record_start_(this)) {
            // start is noexcept so we can assume that the operation will complete
            // after this, which means we can rely on its self-ownership to ensure
            // that it is eventually deleted
            execution::start(*__state->__op_);
            return __future<__x<remove_cvref_t<_Sender>>>{std::move(__state)};
          }
          // __future will complete with set_stopped
          return __future<__x<remove_cvref_t<_Sender>>>{nullptr};
        }

      [[nodiscard]] auto empty() const noexcept {
        return __await_and_sync_();
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

    private:
      static constexpr size_t __stopped_bit_{1};

      static bool __is_stopping_(size_t __state) noexcept {
        return (__state & __stopped_bit_) == 0;
      }

      static size_t __op_count_(size_t __state) noexcept {
        return __state >> 1;
      }

      [[nodiscard]] friend bool __try_record_start_(async_scope* __scope) noexcept {
        auto __op_state = __scope->__op_state_.load(std::memory_order_relaxed);

        do {
          if (__is_stopping_(__op_state)) {
            return false;
          }

          assert(__op_state + 2 > __op_state);
        } while (!__scope->__op_state_.compare_exchange_weak(
            __op_state,
            __op_state + 2,
            std::memory_order_relaxed));

        __scope->__evt_.reset();
        return true;
      }

      friend void __record_done_(async_scope* __scope) noexcept {
        auto __old_state = __scope->__op_state_.fetch_sub(2, std::memory_order_release);

        if (__op_count_(__old_state) == 1) {
          // the last op to finish
          __scope->__evt_.set();
        }
      }

      void __end_of_scope_() noexcept {
        // stop adding work
        auto __old_state = __op_state_.fetch_and(~__stopped_bit_, std::memory_order_release);

        if (__op_count_(__old_state) == 0) {
          // there are no outstanding operations to wait for
          __evt_.set();
        }
      }
    };
  } // namespace __scope

  using __scope::async_scope;
} // namespace execution
