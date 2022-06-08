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

#include <execution.hpp>

namespace std::execution {
  /////////////////////////////////////////////////////////////////////////////
  // async_manual_reset_event
  namespace __event {
    struct __op_base;

    template <class _ReceiverId>
    struct __operation;

    struct async_manual_reset_event;

    struct __sender {
      using completion_signatures =
        execution::completion_signatures<
          execution::set_value_t(),
          execution::set_error_t(exception_ptr)>;

      template <__decays_to<__sender> _Self, receiver _Receiver>
          requires __scheduler_provider<env_of_t<_Receiver>> &&
            receiver_of<_Receiver, completion_signatures>
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr)
            -> __operation<__x<remove_cvref_t<_Receiver>>> {
          return __operation<__x<remove_cvref_t<_Receiver>>>{
            *__self.__evt_,
            (_Receiver&&) __rcvr};
        }

      const async_manual_reset_event* __evt_;
    };

    struct async_manual_reset_event {
      async_manual_reset_event() noexcept
        : async_manual_reset_event(false) {}

      explicit async_manual_reset_event(bool __start_signalled) noexcept
        : __state_(__start_signalled ? this : nullptr) {}

      void set() noexcept;

      bool ready() const noexcept {
        return __state_.load(std::memory_order_acquire) ==
            static_cast<const void*>(this);
      }

      void reset() noexcept {
        // transition from signalled (i.__e. __state_ == this) to not-signalled
        // (i.__e. __state_ == nullptr).
        void* __old_state = this;

        // We can ignore the the result.  We're using _strong so it won't fail
        // spuriously; if it fails, it means it wasn't previously in the signalled
        // state so resetting is a no-op.
        (void)__state_.compare_exchange_strong(
            __old_state, nullptr, std::memory_order_acq_rel);
      }

      [[nodiscard]] __sender async_wait() const noexcept {
        return __sender{this};
      }

    private:
      atomic<void*> __state_{};

      friend struct __op_base;

      // note: this is a static method that takes __evt *second* because the caller
      //       a member function on __op_base and so will already have op in first
      //       argument position; making this function a member would require some
      //       register-juggling code, which would increase binary size
      static void __start_or_wait_(__op_base& __op, const async_manual_reset_event& __evt) noexcept;
    };

    struct __op_base : __immovable {
      // note: __next_ is intentionally left indeterminate until the operation is
      //       pushed on the event's stack of waiting operations
      //
      // note: __next_ and __set_value_ are the first two members because this ordering
      //       leads to smaller code than others; on ARM, the first two members can
      //       be loaded into a pair of registers in one instruction, which turns
      //       out to be important in both async_manual_reset_event::set() and
      //       __start_or_wait_().
      __op_base* __next_;
      void (*__set_value_)(__op_base*) noexcept;
      const async_manual_reset_event* __evt_;

      // This intentionally leaves __next_ uninitialized
      explicit __op_base(
          const async_manual_reset_event* __evt,
          void (*__set_value)(__op_base*) noexcept)
        : __set_value_(__set_value)
        , __evt_(__evt)
      {}

      void __set_value() noexcept {
        __set_value_(this);
      }

      void __start() noexcept {
        async_manual_reset_event::__start_or_wait_(*this, *__evt_);
      }
    };

    template <class _ReceiverId>
      class __receiver
        : private receiver_adaptor<__receiver<_ReceiverId>, __t<_ReceiverId>> {
        using _Receiver = __t<_ReceiverId>;
        friend receiver_adaptor<__receiver, _Receiver>;

        auto get_env() const&
          -> make_env_t<get_stop_token_t, never_stop_token, env_of_t<_Receiver>> {
          return make_env<get_stop_token_t>(
            never_stop_token{},
            execution::get_env(this->base()));
        }

      public:
        using receiver_adaptor<__receiver, _Receiver>::receiver_adaptor;
      };

    template <class _Receiver>
      using __scheduler_from_env_of =
        __call_result_t<get_scheduler_t, env_of_t<_Receiver>>;

    template <class _Receiver>
      auto __connect_as_unstoppable(_Receiver&& __rcvr)
        noexcept(
          __has_nothrow_connect<
            schedule_result_t<__scheduler_from_env_of<_Receiver>>,
            __receiver<__x<_Receiver>>>)
        -> connect_result_t<
            schedule_result_t<__scheduler_from_env_of<_Receiver>>,
            __receiver<__x<_Receiver>>> {
        return connect(
            schedule(get_scheduler(get_env(__rcvr))),
            __receiver<__x<_Receiver>>{(_Receiver&&) __rcvr});
      }

    template <class _ReceiverId>
      struct __operation : private __op_base {
        using _Receiver = __t<_ReceiverId>;

        explicit __operation(const async_manual_reset_event& __evt, _Receiver __rcvr)
            noexcept(noexcept(__connect_as_unstoppable(std::move(__rcvr))))
          : __op_base{&__evt, &__set_value_}
          , __op_(__connect_as_unstoppable(std::move(__rcvr)))
        {}

      private:
        friend void tag_invoke(start_t, __operation& __self) noexcept {
          __self.__start();
        }

        using __op_t = decltype(__connect_as_unstoppable(__declval<_Receiver>()));
        __op_t __op_;

        static void __set_value_(__op_base* __base) noexcept {
          auto* __self = static_cast<__operation*>(__base);
          execution::start(__self->__op_);
        }
      };

    inline void async_manual_reset_event::set() noexcept {
      void* const __signalled_state = this;

      // replace the stack of waiting operations with a sentinel indicating we've
      // been signalled
      void* __top = __state_.exchange(__signalled_state, std::memory_order_acq_rel);

      if (__top == __signalled_state) {
        // we were already signalled so there are no waiting operations
        return;
      }

      // We are the first thread to set the state to signalled; iteratively pop
      // the stack and complete each operation.
      auto* __op = static_cast<__op_base*>(__top);
      while (__op != nullptr) {
        std::exchange(__op, __op->__next_)->__set_value();
      }
    }

    inline void async_manual_reset_event::__start_or_wait_(__op_base& __op, const async_manual_reset_event& __evt) noexcept {
      async_manual_reset_event& __e = const_cast<async_manual_reset_event&>(__evt);
      // Try to push op onto the stack of waiting ops.
      void* const __signalled_state = &__e;

      void* __top = __e.__state_.load(std::memory_order_acquire);

      do {
        if (__top == __signalled_state) {
          // Already in the signalled state; don't push it.
          __op.__set_value();
          return;
        }

        // note: on the first iteration, this line transitions __op.__next_ from
        //       indeterminate to a well-defined value
        __op.__next_ = static_cast<__op_base*>(__top);
      } while (!__e.__state_.compare_exchange_weak(
          __top,
          static_cast<void*>(&__op),
          std::memory_order_release,
          std::memory_order_acquire));
    }
  } // namespace __event

  using __event::async_manual_reset_event;
} // namespace execution

