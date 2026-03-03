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
#include "__completion_signatures_of.hpp"
#include "__concepts.hpp"
#include "__connect.hpp"
#include "__meta.hpp"
#include "__queries.hpp"
#include "__tag_invoke.hpp"
#include "__type_traits.hpp"

#include <exception>
#include <functional>  // for std::identity
#include <system_error>
#include <variant>

namespace STDEXEC
{
#if !STDEXEC_NO_STDCPP_COROUTINES()
  namespace __detail
  {
    template <std::size_t _Count>
    extern __q<__decayed_std_tuple> const __as_single;

    template <>
    inline constexpr __q<__midentity> __as_single<1>;

    template <>
    inline constexpr __mconst<void> __as_single<0>;

    template <class... _Values>
    using __single_value = __minvoke<decltype(__as_single<sizeof...(_Values)>), _Values...>;

    template <class _Sender, class _Promise>
    using __value_t = __decay_t<
      __value_types_of_t<_Sender, env_of_t<_Promise&>, __q<__single_value>, __msingle_or<void>>>;

    inline constexpr auto __get_await_completion_adaptor =
      __with_default{get_await_completion_adaptor, std::identity{}};

    template <class _Sender>
    using __adapt_completion_t = __result_of<__get_await_completion_adaptor, env_of_t<_Sender>>;

    template <class _Sender>
    constexpr auto __adapt_sender_for_await(_Sender&& __sndr)
      noexcept(__nothrow_callable<__adapt_completion_t<_Sender>, _Sender>) -> decltype(auto)
    {
      return __get_await_completion_adaptor(get_env(__sndr))(static_cast<_Sender&&>(__sndr));
    }

    template <class _Sender>
    using __adapted_sender_t =
      __remove_rvalue_reference_t<__call_result_t<__adapt_completion_t<_Sender>, _Sender>>;
  }  // namespace __detail

  /////////////////////////////////////////////////////////////////////////////
  // STDEXEC::as_awaitable [exec.as.awaitable]
  namespace __as_awaitable
  {
    struct __void
    {};

    template <class _Value>
    using __value_or_void_t = __if_c<__same_as<_Value, void>, __void, _Value>;

    template <class _Value>
    using __expected_t =
      std::variant<std::monostate, __value_or_void_t<_Value>, std::exception_ptr>;

    template <class _Tag, class _Sender, class... _Env>
    concept __completes_inline_for = __never_sends<_Tag, _Sender, _Env...>
                                  || STDEXEC::__completes_inline<_Tag, env_of_t<_Sender>, _Env...>;

    template <class _Sender, class... _Env>
    concept __completes_inline = __completes_inline_for<set_value_t, _Sender, _Env...>
                              && __completes_inline_for<set_error_t, _Sender, _Env...>
                              && __completes_inline_for<set_stopped_t, _Sender, _Env...>;

    template <class _Value>
    struct __receiver_base
    {
      using receiver_concept = receiver_t;

      template <class... _Us>
      void set_value(_Us&&... __us) noexcept
      {
        STDEXEC_TRY
        {
          __result_.template emplace<1>(static_cast<_Us&&>(__us)...);
        }
        STDEXEC_CATCH_ALL
        {
          __result_.template emplace<2>(std::current_exception());
        }
      }

      template <class _Error>
      void set_error(_Error&& __err) noexcept
      {
        if constexpr (__decays_to<_Error, std::exception_ptr>)
          __result_.template emplace<2>(static_cast<_Error&&>(__err));
        else if constexpr (__decays_to<_Error, std::error_code>)
          __result_.template emplace<2>(std::make_exception_ptr(std::system_error(__err)));
        else
          __result_.template emplace<2>(std::make_exception_ptr(static_cast<_Error&&>(__err)));
      }

      __expected_t<_Value>& __result_;
    };

    template <class _Promise, class _Value>
    struct __sync_receiver : __receiver_base<_Value>
    {
      constexpr explicit __sync_receiver(__expected_t<_Value>&             __result,
                                         __std::coroutine_handle<_Promise> __continuation) noexcept
        : __receiver_base<_Value>{__result}
        , __continuation_{__continuation}
      {}

      void set_stopped() noexcept
      {
        // no-op: the __result_ variant will remain engaged with the monostate
        // alternative, which signals that the operation was stopped.
      }

      // Forward get_env query to the coroutine promise
      constexpr auto get_env() const noexcept -> env_of_t<_Promise&>
      {
        return STDEXEC::get_env(__continuation_.promise());
      }

      __std::coroutine_handle<_Promise> __continuation_;
    };

    // The receiver type used to connect to senders that could complete asynchronously.
    template <class _Promise, class _Value>
    struct __async_receiver : __sync_receiver<_Promise, _Value>
    {
      constexpr explicit __async_receiver(__expected_t<_Value>&             __result,
                                          __std::coroutine_handle<_Promise> __continuation) noexcept
        : __sync_receiver<_Promise, _Value>{__result, __continuation}
      {}

      template <class... _Us>
      void set_value(_Us&&... __us) noexcept
      {
        this->__sync_receiver<_Promise, _Value>::set_value(static_cast<_Us&&>(__us)...);
        this->__continuation_.resume();
      }

      template <class _Error>
      void set_error(_Error&& __err) noexcept
      {
        this->__sync_receiver<_Promise, _Value>::set_error(static_cast<_Error&&>(__err));
        this->__continuation_.resume();
      }

      constexpr void set_stopped() noexcept
      {
        STDEXEC_TRY
        {
          // Resuming the stopped continuation unwinds the coroutine stack until we reach
          // a promise that can handle the stopped signal. The coroutine referred to by
          // __continuation_ will never be resumed.
          __std::coroutine_handle<> __on_stopped =
            this->__continuation_.promise().unhandled_stopped();
          __on_stopped.resume();
        }
        STDEXEC_CATCH_ALL
        {
          this->__result_.template emplace<2>(std::current_exception());
          this->__continuation_.resume();
        }
      }
    };

    template <class _Sender, class _Promise>
    using __sync_receiver_t = __sync_receiver<_Promise, __detail::__value_t<_Sender, _Promise>>;

    template <class _Sender, class _Promise>
    using __async_receiver_t = __async_receiver<_Promise, __detail::__value_t<_Sender, _Promise>>;

    template <class _Value>
    struct __sender_awaitable_base
    {
      static constexpr auto await_ready() noexcept -> bool
      {
        return false;
      }

      constexpr auto await_resume() -> _Value
      {
        // If the operation completed with set_stopped (as denoted by the monostate
        // alternative being active), we should not be resuming this coroutine at all.
        STDEXEC_ASSERT(__result_.index() != 0);
        if (__result_.index() == 2)
        {
          // The operation completed with set_error, so we need to rethrow the exception.
          std::rethrow_exception(std::move(std::get<2>(__result_)));
        }
        // The operation completed with set_value, so we can just return the value, which
        // may be void.
        return static_cast<std::add_rvalue_reference_t<_Value>>(std::get<1>(__result_));
      }

     protected:
      __expected_t<_Value> __result_{};
    };

    //////////////////////////////////////////////////////////////////////////////////////
    // __sender_awaitable: awaitable type returned by as_awaitable when given a sender
    // that does not have an as_awaitable member function
    template <class _Promise, class _Sender>
    struct __sender_awaitable : __sender_awaitable_base<__detail::__value_t<_Sender, _Promise>>
    {
      constexpr explicit __sender_awaitable(_Sender&&                         __sndr,
                                            __std::coroutine_handle<_Promise> __hcoro)
        noexcept(__nothrow_connectable<_Sender, __receiver_t>)
        : __opstate_(STDEXEC::connect(static_cast<_Sender&&>(__sndr),
                                      __receiver_t(this->__result_, __hcoro)))
      {}

      constexpr void await_suspend(__std::coroutine_handle<_Promise>) noexcept
      {
        STDEXEC::start(__opstate_);
      }

     private:
      using __receiver_t = __async_receiver_t<_Sender, _Promise>;
      connect_result_t<_Sender, __receiver_t> __opstate_;
    };

    // When the sender is known to complete inline, we can connect and start the operation
    // in await_suspend.
    template <class _Promise, class _Sender>
      requires __completes_inline<_Sender, env_of_t<_Promise&>>
    struct __sender_awaitable<_Promise, _Sender>
      : __sender_awaitable_base<__detail::__value_t<_Sender, _Promise>>
    {
      constexpr explicit __sender_awaitable(_Sender&& sndr, __ignore)
        noexcept(__nothrow_move_constructible<_Sender>)
        : __sndr_(static_cast<_Sender&&>(sndr))
      {}

      bool await_suspend(__std::coroutine_handle<_Promise> __hcoro)
      {
        {
          auto __opstate = STDEXEC::connect(static_cast<_Sender&&>(__sndr_),
                                            __receiver_t(this->__result_, __hcoro));
          // The following call to start will complete synchronously, writing its result
          // into the __result_ variant.
          STDEXEC::start(__opstate);
        }

        if (this->__result_.index() == 0)
        {
          // The operation completed with set_stopped, so we need to call
          // unhandled_stopped() on the promise to propagate the stop signal. That will
          // result in the coroutine being torn down, so beware. We then resume the
          // returned coroutine handle (which may be a noop_coroutine).
          __std::coroutine_handle<> __on_stopped = __hcoro.promise().unhandled_stopped();
          __on_stopped.resume();

          // By returning true, we indicate that the coroutine should not be resumed
          // (because it no longer exists).
          return true;
        }

        // The operation completed with set_value or set_error, so we can just resume the
        // current coroutine. await_resume with either return the value or throw as
        // appropriate.
        return false;
      }

     private:
      using __receiver_t = __sync_receiver_t<_Sender, _Promise>;
      _Sender __sndr_;
    };

    template <class _Sender, class _Promise>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
    __sender_awaitable(_Sender&&, __std::coroutine_handle<_Promise>)
      -> __sender_awaitable<_Promise, _Sender>;

    template <class _Sender, class _Promise>
    concept __awaitable_adapted_sender = sender_in<_Sender, env_of_t<_Promise&>>
                                      && __minvocable_q<__detail::__value_t, _Sender, _Promise>
                                      && requires(_Promise& __promise) {
                                           {
                                             __promise.unhandled_stopped()
                                           } -> __std::convertible_to<__std::coroutine_handle<>>;
                                         };

    template <class _Sender, class _Promise>
    concept __awaitable_sender =
      __awaitable_adapted_sender<__detail::__adapted_sender_t<_Sender>, _Promise>;

    struct __unspecified
    {
      constexpr auto get_return_object() noexcept -> __unspecified;
      constexpr auto initial_suspend() noexcept -> __unspecified;
      constexpr auto final_suspend() noexcept -> __unspecified;
      constexpr void unhandled_exception() noexcept;
      constexpr void return_void() noexcept;
      constexpr auto unhandled_stopped() noexcept -> __std::coroutine_handle<>;
    };

    template <class _Sender, class _Promise>
    concept __incompatible_sender = sender<_Sender>
                                 && __merror<__detail::__value_t<_Sender, _Promise>>;
  }  // namespace __as_awaitable

  struct as_awaitable_t
  {
    template <class _Tp, class _Promise>
    static consteval auto __get_declfn() noexcept
    {
      using namespace __as_awaitable;
      if constexpr (__connect_await::__has_as_awaitable_member<_Tp, _Promise>)
      {
        using __result_t = decltype(__declval<_Tp>().as_awaitable(__declval<_Promise&>()));
        constexpr bool __is_nothrow = noexcept(
          __declval<_Tp>().as_awaitable(__declval<_Promise&>()));
        return __declfn<__result_t, __is_nothrow>();
      }
      else if constexpr (__awaitable<_Tp, __unspecified>)  // NOT __awaitable<_Tp, _Promise> !!
      {                                                    // NOLINT(bugprone-branch-clone)
        return __declfn<_Tp&&>();
      }
      else if constexpr (__awaitable_sender<_Tp, _Promise>)
      {
        using __result_t            = decltype(  //
          __sender_awaitable{__detail::__adapt_sender_for_await(__declval<_Tp>()),
                             __std::coroutine_handle<_Promise>()});
        constexpr bool __is_nothrow = noexcept(
          __sender_awaitable{__detail::__adapt_sender_for_await(__declval<_Tp>()),
                             __std::coroutine_handle<_Promise>()});
        return __declfn<__result_t, __is_nothrow>();
      }
      else if constexpr (__incompatible_sender<_Tp, _Promise>)
      {
        // NOT TO SPEC: It's a sender, but it isn't a sender in the current promise's
        // environment, so we can return the error type that results from trying to
        // compute the sender's value type:
        return __declfn<__detail::__value_t<_Tp, _Promise>>();
      }
      else
      {
        return __declfn<_Tp&&>();
      }
    }

    template <class _Tp, class _Promise, auto _DeclFn = __get_declfn<_Tp, _Promise>()>
      requires __callable<__mtypeof<_DeclFn>>
    auto operator()(_Tp&& __t, _Promise& __promise) const noexcept(noexcept(_DeclFn()))
      -> decltype(_DeclFn())
    {
      using namespace __as_awaitable;
      if constexpr (__connect_await::__has_as_awaitable_member<_Tp, _Promise>)
      {
        return static_cast<_Tp&&>(__t).as_awaitable(__promise);
      }
      else if constexpr (__awaitable<_Tp, __unspecified>)  // NOT __awaitable<_Tp, _Promise> !!
      {                                                    // NOLINT(bugprone-branch-clone)
        return static_cast<_Tp&&>(__t);
      }
      else if constexpr (__awaitable_sender<_Tp, _Promise>)
      {
        auto __hcoro = __std::coroutine_handle<_Promise>::from_promise(__promise);
        return __sender_awaitable{__detail::__adapt_sender_for_await(static_cast<_Tp&&>(__t)),
                                  __hcoro};
      }
      else if constexpr (__incompatible_sender<_Tp, _Promise>)
      {
        return __detail::__value_t<_Tp, _Promise>();
      }
      else
      {
        return static_cast<_Tp&&>(__t);
      }
    }

    template <class _Tp, class _Promise, auto _DeclFn = __get_declfn<_Tp, _Promise>()>
      requires __callable<__mtypeof<_DeclFn>> || __tag_invocable<as_awaitable_t, _Tp, _Promise&>
    [[deprecated("the use of tag_invoke for as_awaitable is deprecated")]]
    auto operator()(_Tp&& __t, _Promise& __promise) const
      noexcept(__nothrow_tag_invocable<as_awaitable_t, _Tp, _Promise&>)
        -> __tag_invoke_result_t<as_awaitable_t, _Tp, _Promise&>
    {
      using __result_t = __tag_invoke_result_t<as_awaitable_t, _Tp, _Promise&>;
      static_assert(__awaitable<__result_t, _Promise>);
      return __tag_invoke(*this, static_cast<_Tp&&>(__t), __promise);
    }
  };

  inline constexpr as_awaitable_t as_awaitable{};
#endif
}  // namespace STDEXEC
