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

#include "__execution_fwd.hpp"

#include "__awaitable.hpp"
#include "__concepts.hpp"
#include "__config.hpp"
#include "__meta.hpp"
#include "__receivers.hpp"
#include "__senders.hpp"
#include "__tag_invoke.hpp"
#include "__transform_completion_signatures.hpp"
#include "__type_traits.hpp"

#include <exception>
#include <system_error>
#include <variant>

namespace stdexec {
#if !STDEXEC_STD_NO_COROUTINES()
  /////////////////////////////////////////////////////////////////////////////
  // stdexec::as_awaitable [execution.coro_utils.as_awaitable]
  namespace __as_awaitable {
    struct __void { };

    template <class _Value>
    using __value_or_void_t = __if_c<__same_as<_Value, void>, __void, _Value>;

    template <class _Value>
    using __expected_t =
      std::variant<std::monostate, __value_or_void_t<_Value>, std::exception_ptr>;

    template <class _Value>
    struct __receiver_base {
      using receiver_concept = receiver_t;

      template <class... _Us>
        requires constructible_from<__value_or_void_t<_Value>, _Us...>
      void set_value(_Us&&... __us) noexcept {
        STDEXEC_TRY {
          __result_->template emplace<1>(static_cast<_Us&&>(__us)...);
          __continuation_.resume();
        }
        STDEXEC_CATCH_ALL {
          stdexec::set_error(static_cast<__receiver_base&&>(*this), std::current_exception());
        }
      }

      template <class _Error>
      void set_error(_Error&& __err) noexcept {
        if constexpr (__decays_to<_Error, std::exception_ptr>)
          __result_->template emplace<2>(static_cast<_Error&&>(__err));
        else if constexpr (__decays_to<_Error, std::error_code>)
          __result_->template emplace<2>(std::make_exception_ptr(std::system_error(__err)));
        else
          __result_->template emplace<2>(std::make_exception_ptr(static_cast<_Error&&>(__err)));
        __continuation_.resume();
      }

      __expected_t<_Value>* __result_;
      __coro::coroutine_handle<> __continuation_;
    };

    template <class _PromiseId, class _Value>
    struct __receiver {
      using _Promise = stdexec::__t<_PromiseId>;

      struct __t : __receiver_base<_Value> {
        using __id = __receiver;

        void set_stopped() noexcept {
          auto __continuation = __coro::coroutine_handle<_Promise>::from_address(
            this->__continuation_.address());
          __coro::coroutine_handle<> __stopped_continuation = __continuation.promise()
                                                                .unhandled_stopped();
          __stopped_continuation.resume();
        }

        // Forward get_env query to the coroutine promise
        auto get_env() const noexcept -> env_of_t<_Promise&> {
          auto __continuation = __coro::coroutine_handle<_Promise>::from_address(
            this->__continuation_.address());
          return stdexec::get_env(__continuation.promise());
        }
      };
    };

    // BUGBUG NOT TO SPEC: make senders of more-than-one-value awaitable
    // by packaging the values into a tuple.
    // See: https://github.com/cplusplus/sender-receiver/issues/182
    template <std::size_t _Count>
    extern const __q<__decayed_std_tuple> __as_single;

    template <>
    inline const __q<__midentity> __as_single<1>;

    template <>
    inline const __mconst<void> __as_single<0>;

    template <class... _Values>
    using __single_value = __minvoke<decltype(__as_single<sizeof...(_Values)>), _Values...>;

    template <class _Sender, class _Promise>
    using __value_t = __decay_t<
      __value_types_of_t<_Sender, env_of_t<_Promise&>, __q<__single_value>, __msingle_or<void>>
    >;

    template <class _Sender, class _Promise>
    using __receiver_t = __t<__receiver<__id<_Promise>, __value_t<_Sender, _Promise>>>;

    template <class _Value>
    struct __sender_awaitable_base {
      [[nodiscard]]
      auto await_ready() const noexcept -> bool {
        return false;
      }

      auto await_resume() -> _Value {
        switch (__result_.index()) {
        case 0: // receiver contract not satisfied
          STDEXEC_ASSERT(false && +"_Should never get here" == nullptr);
          break;
        case 1: // set_value
          if constexpr (!__same_as<_Value, void>)
            return static_cast<_Value&&>(std::get<1>(__result_));
          else
            return;
        case 2: // set_error
          std::rethrow_exception(std::get<2>(__result_));
        }
        std::terminate();
      }

     protected:
      __expected_t<_Value> __result_;
    };

    template <class _PromiseId, class _SenderId>
    struct __sender_awaitable {
      using _Promise = stdexec::__t<_PromiseId>;
      using _Sender = stdexec::__t<_SenderId>;
      using __value = __value_t<_Sender, _Promise>;

      struct __t : __sender_awaitable_base<__value> {
        __t(_Sender&& sndr, __coro::coroutine_handle<_Promise> __hcoro)
          noexcept(__nothrow_connectable<_Sender, __receiver>)
          : __op_state_(connect(
              static_cast<_Sender&&>(sndr),
              __receiver{
                {&this->__result_, __hcoro}
        })) {
        }

        void await_suspend(__coro::coroutine_handle<_Promise>) noexcept {
          stdexec::start(__op_state_);
        }

       private:
        using __receiver = __receiver_t<_Sender, _Promise>;
        connect_result_t<_Sender, __receiver> __op_state_;
      };
    };

    template <class _Promise, class _Sender>
    using __sender_awaitable_t = __t<__sender_awaitable<__id<_Promise>, __id<_Sender>>>;

    template <class _Sender, class _Promise>
    concept __awaitable_sender = sender_in<_Sender, env_of_t<_Promise&>>
                              && __mvalid<__value_t, _Sender, _Promise>
                              && sender_to<_Sender, __receiver_t<_Sender, _Promise>>
                              && requires(_Promise& __promise) {
                                   {
                                     __promise.unhandled_stopped()
                                   } -> convertible_to<__coro::coroutine_handle<>>;
                                 };

    struct __unspecified {
      auto get_return_object() noexcept -> __unspecified;
      auto initial_suspend() noexcept -> __unspecified;
      auto final_suspend() noexcept -> __unspecified;
      void unhandled_exception() noexcept;
      void return_void() noexcept;
      auto unhandled_stopped() noexcept -> __coro::coroutine_handle<>;
    };

    template <class _Tp, class _Promise>
    concept __has_as_awaitable_member = requires(_Tp&& __t, _Promise& __promise) {
      static_cast<_Tp &&>(__t).as_awaitable(__promise);
    };

    struct as_awaitable_t {
      template <class _Tp, class _Promise>
      static constexpr auto __select_impl_() noexcept {
        if constexpr (__has_as_awaitable_member<_Tp, _Promise>) {
          using _Result = decltype(__declval<_Tp>().as_awaitable(__declval<_Promise&>()));
          constexpr bool _Nothrow = noexcept(__declval<_Tp>().as_awaitable(__declval<_Promise&>()));
          return static_cast<_Result (*)() noexcept(_Nothrow)>(nullptr);
        } else if constexpr (tag_invocable<as_awaitable_t, _Tp, _Promise&>) {
          using _Result = tag_invoke_result_t<as_awaitable_t, _Tp, _Promise&>;
          constexpr bool _Nothrow = nothrow_tag_invocable<as_awaitable_t, _Tp, _Promise&>;
          return static_cast<_Result (*)() noexcept(_Nothrow)>(nullptr);
          // NOLINTNEXTLINE(bugprone-branch-clone)
        } else if constexpr (__awaitable<_Tp, __unspecified>) { // NOT __awaitable<_Tp, _Promise> !!
          using _Result = _Tp&&;
          return static_cast<_Result (*)() noexcept>(nullptr);
        } else if constexpr (__awaitable_sender<_Tp, _Promise>) {
          using _Result = __sender_awaitable_t<_Promise, _Tp>;
          constexpr bool _Nothrow =
            __nothrow_constructible_from<_Result, _Tp, __coro::coroutine_handle<_Promise>>;
          return static_cast<_Result (*)() noexcept(_Nothrow)>(nullptr);
        } else {
          using _Result = _Tp&&;
          return static_cast<_Result (*)() noexcept>(nullptr);
        }
      }

      template <class _Tp, class _Promise>
      using __select_impl_t = decltype(__select_impl_<_Tp, _Promise>());

      template <class _Tp, class _Promise>
      auto operator()(_Tp&& __t, _Promise& __promise) const
        noexcept(__nothrow_callable<__select_impl_t<_Tp, _Promise>>)
          -> __call_result_t<__select_impl_t<_Tp, _Promise>> {
        if constexpr (__has_as_awaitable_member<_Tp, _Promise>) {
          using _Result = decltype(static_cast<_Tp&&>(__t).as_awaitable(__promise));
          static_assert(__awaitable<_Result, _Promise>);
          return static_cast<_Tp&&>(__t).as_awaitable(__promise);
        } else if constexpr (tag_invocable<as_awaitable_t, _Tp, _Promise&>) {
          using _Result = tag_invoke_result_t<as_awaitable_t, _Tp, _Promise&>;
          static_assert(__awaitable<_Result, _Promise>);
          return tag_invoke(*this, static_cast<_Tp&&>(__t), __promise);
          // NOLINTNEXTLINE(bugprone-branch-clone)
        } else if constexpr (__awaitable<_Tp, __unspecified>) { // NOT __awaitable<_Tp, _Promise> !!
          return static_cast<_Tp&&>(__t);
        } else if constexpr (__awaitable_sender<_Tp, _Promise>) {
          auto __hcoro = __coro::coroutine_handle<_Promise>::from_promise(__promise);
          return __sender_awaitable_t<_Promise, _Tp>{static_cast<_Tp&&>(__t), __hcoro};
        } else {
          return static_cast<_Tp&&>(__t);
        }
      }
    };
  } // namespace __as_awaitable

  using __as_awaitable::as_awaitable_t;
  inline constexpr as_awaitable_t as_awaitable{};
#endif
} // namespace stdexec
