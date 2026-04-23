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

#include "../coroutine.hpp"
#include "../functional.hpp"
#include "__atomic.hpp"
#include "__awaitable.hpp"
#include "__completion_signatures_of.hpp"
#include "__concepts.hpp"
#include "__connect.hpp"
#include "__meta.hpp"
#include "__queries.hpp"
#include "__type_traits.hpp"
#include "__variant.hpp"

#include <exception>
#include <functional>  // for std::identity
#include <system_error>
#include <thread>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")
STDEXEC_PRAGMA_IGNORE_MSVC(4714)  // marked as __forceinline not inlined

namespace STDEXEC
{
#if !STDEXEC_NO_STDCPP_COROUTINES()
  /////////////////////////////////////////////////////////////////////////////
  // STDEXEC::as_awaitable [exec.as.awaitable]

  namespace __as_awaitable
  {
    template <std::size_t _Count>
    extern __q<__decayed_std_tuple> const __as_single;

    template <>
    inline constexpr __q<__midentity> __as_single<1>;

    template <>
    inline constexpr __mconst<void> __as_single<0>;

    template <class... _Values>
    using __single_value_t = __minvoke<decltype(__as_single<sizeof...(_Values)>), _Values...>;

    template <class _Sender, class _Promise>
    using __value_t = __decay_t<
      __value_types_of_t<_Sender, env_of_t<_Promise&>, __q<__single_value_t>, __msingle_or<void>>>;

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

    struct __void
    {};

    template <class _Value>
    using __value_or_void_t = __if_c<__same_as<_Value, void>, __void, _Value>;

    template <class _Value>
    using __expected_t = __variant<__value_or_void_t<_Value>, std::exception_ptr>;

    using __connect_await::__has_as_awaitable_member;

    template <class _Tag, class _Sender, class... _Env>
    concept __completes_inline_for = __never_sends<_Tag, _Sender, _Env...>
                                  || STDEXEC::__completes_inline<_Tag, env_of_t<_Sender>, _Env...>;

    template <class _Sender, class... _Env>
    concept __completes_inline = __completes_inline_for<set_value_t, _Sender, _Env...>
                              && __completes_inline_for<set_error_t, _Sender, _Env...>
                              && __completes_inline_for<set_stopped_t, _Sender, _Env...>;

    template <class _Value, bool _Inline>
    struct __sender_awaiter_base;

    template <class _Value>
    struct __sender_awaiter_base<_Value, true>
    {
      static constexpr auto await_ready() noexcept -> bool
      {
        return false;
      }

      constexpr auto await_resume() -> _Value
      {
        // If the operation completed with set_stopped (as denoted by the result variant
        // being valueless), we should not be resuming this coroutine at all.
        STDEXEC_ASSERT(!__result_.__is_valueless());
        if (__result_.index() == 1)
        {
          // The operation completed with set_error, so we need to rethrow the exception.
          std::rethrow_exception(std::move(__var::__get<1>(__result_)));
        }
        // The operation completed with set_value, so we can just return the value, which
        // may be void.
        using __reference_t = std::add_rvalue_reference_t<_Value>;
        return static_cast<__reference_t>(__var::__get<0>(__result_));
      }

      [[nodiscard]]
      constexpr auto __get_continuation() const noexcept -> __std::coroutine_handle<>
      {
        // If the operation was stopped, so there is no result. We should not be resuming
        // the __continuation_; rather, we should use the unhandled_stopped()
        // continuation. Otherwise, so we should resume the __continuation_ as normal.
        return __result_.__is_valueless() ? __continuation_.unhandled_stopped()
                                          : __continuation_.handle();
      }

      __coroutine_handle<> __continuation_;
      __expected_t<_Value> __result_{__no_init};
    };

    // When the sender is not statically known to complete inline, we need to use atomic
    // state to guard against too many inline completions causing a stack overflow.
    template <class _Value>
    struct __sender_awaiter_base<_Value, false> : __sender_awaiter_base<_Value, true>
    {
      std::atomic<int>      __refcount_{2};
      std::thread::id const __starting_thread_{std::this_thread::get_id()};
    };

    template <class _Value>
    struct __receiver_base
    {
      using receiver_concept = receiver_tag;

      template <class... _Us>
      void set_value(_Us&&... __us) noexcept
      {
        STDEXEC_TRY
        {
          __awaiter_.__result_.template emplace<0>(static_cast<_Us&&>(__us)...);
        }
        STDEXEC_CATCH_ALL
        {
          __awaiter_.__result_.template emplace<1>(std::current_exception());
        }
      }

      template <class _Error>
      void set_error(_Error&& __err) noexcept
      {
        if constexpr (__decays_to<_Error, std::exception_ptr>)
          __awaiter_.__result_.template emplace<1>(static_cast<_Error&&>(__err));
        else if constexpr (__decays_to<_Error, std::error_code>)
          __awaiter_.__result_.template emplace<1>(
            std::make_exception_ptr(std::system_error(__err)));
        else
          __awaiter_.__result_.template emplace<1>(
            std::make_exception_ptr(static_cast<_Error&&>(__err)));
      }

      __sender_awaiter_base<_Value, true>& __awaiter_;
    };

    template <class _Promise, class _Value>
    struct __sync_receiver : __receiver_base<_Value>
    {
      using __awaiter_t = __sender_awaiter_base<_Value, true>;

      constexpr explicit __sync_receiver(__awaiter_t& __awaiter) noexcept
        : __receiver_base<_Value>{__awaiter}
      {}

      void set_stopped() noexcept
      {
        // no-op: the __result_ variant will remain valueless, which signals that the
        // operation was stopped.
      }

      // Forward get_env query to the coroutine promise
      [[nodiscard]]
      constexpr auto get_env() const noexcept -> env_of_t<_Promise&>
      {
        auto __hcoro = STDEXEC::__coroutine_handle_cast<_Promise>(
          this->__awaiter_.__continuation_.handle());
        return STDEXEC::get_env(__hcoro.promise());
      }
    };

    // The receiver type used to connect to senders that could complete asynchronously.
    template <class _Promise, class _Value>
    struct __async_receiver : __sync_receiver<_Promise, _Value>
    {
      using __awaiter_t = __sender_awaiter_base<_Value, false>;

      constexpr explicit __async_receiver(__awaiter_t& __awaiter) noexcept
        : __sync_receiver<_Promise, _Value>{__awaiter}
      {}

      template <class... _Us>
      void set_value(_Us&&... __us) noexcept
      {
        this->__sync_receiver<_Promise, _Value>::set_value(static_cast<_Us&&>(__us)...);
        __done();
      }

      template <class _Error>
      void set_error(_Error&& __err) noexcept
      {
        this->__sync_receiver<_Promise, _Value>::set_error(static_cast<_Error&&>(__err));
        __done();
      }

      constexpr void set_stopped() noexcept
      {
        __done();
      }

     private:
      void __done() noexcept
      {
        auto&      __awaiter         = static_cast<__awaiter_t&>(this->__awaiter_);
        bool const __on_other_thread = std::this_thread::get_id() != __awaiter.__starting_thread_;
        // It is possible that `await_suspend` hasn't even returned yet. `await_suspend`
        // wants to read state from the awaitable after calling `start` on the opstate, so
        // we need to wait until `start` has returned before completing the operation.
        // Otherwise, `await_suspend` might be reading from the awaiter after it has been
        // destroyed.
        int const __old_refs = __awaiter.__refcount_.fetch_sub(1, __std::memory_order_acq_rel);

        if (__on_other_thread)
        {
          // If we are completing on a different thread than the one that started the
          // operation, we must wait until `await_suspend` has decremented the refcount
          // and is no longer accessing the awaiter before we can safely resume the
          // continuation.
          __awaiter.__refcount_.wait(1, __std::memory_order_acquire);
        }

        if (__old_refs == 1)
        {
          // We get here if `await_suspend` has already decremented the refcount, which
          // means that this operation is completing asynchronously. We need to resume the
          // continuation from here because `await_suspend` will not.
          __awaiter.__get_continuation().resume();
        }
      }
    };

    template <class _Sender, class _Promise>
    using __sync_receiver_t = __sync_receiver<_Promise, __value_t<_Sender, _Promise>>;

    template <class _Sender, class _Promise>
    using __async_receiver_t = __async_receiver<_Promise, __value_t<_Sender, _Promise>>;

    //////////////////////////////////////////////////////////////////////////////////////
    // __sender_awaiter: awaitable type returned by as_awaitable when given a sender
    // that does not have an as_awaitable member function
    template <class _Promise, sender_in<env_of_t<_Promise&>> _Sender>
    struct __sender_awaiter : __sender_awaiter_base<__value_t<_Sender, _Promise>, false>
    {
      using __value_t = __as_awaitable::__value_t<_Sender, _Promise>;

      constexpr explicit __sender_awaiter(_Sender&&                         __sndr,
                                          __std::coroutine_handle<_Promise> __hcoro)
        noexcept(__nothrow_connectable<_Sender, __receiver_t>)
        : __sender_awaiter_base<__value_t, false>{__hcoro}
        , __opstate_(STDEXEC::connect(static_cast<_Sender&&>(__sndr), __receiver_t(*this)))
      {}

      constexpr auto
      await_suspend([[maybe_unused]] __std::coroutine_handle<> __continuation) noexcept
        -> STDEXEC_PP_IF(STDEXEC_GCC(), bool, __std::coroutine_handle<>)
      {
        STDEXEC_ASSERT(this->__continuation_.handle() == __continuation);

        // Start the operation.
        STDEXEC::start(__opstate_);

        int const __old_refcount = this->__refcount_.fetch_sub(1, __std::memory_order_acq_rel);
        this->__refcount_.notify_one();

        if (__old_refcount == 1)
        {
          // If the refcount was 1 before the decrement, then the operation has already
          // completed (either synchronously or asynchronously) and we are responsible for
          // resuming the continuation. Otherwise, we can let the receiver resume the
          // continuation when the operation completes.
          __continuation = this->__get_continuation();
          return STDEXEC_PP_IF(STDEXEC_GCC(), (__continuation.resume(), false), __continuation);
        }
        else
        {
          // Otherwise, the operation has not completed yet, so we need to suspend the
          // current coroutine. The continuation will be resumed when the operation
          // completes.
          return STDEXEC_PP_IF(STDEXEC_GCC(), true, std::noop_coroutine());
        }
      }

     private:
      using __receiver_t = __async_receiver_t<_Sender, _Promise>;
      connect_result_t<_Sender, __receiver_t> __opstate_;
    };

    // When the sender is known to complete inline, we can connect and start the operation
    // in await_suspend.
    template <class _Promise, sender_in<env_of_t<_Promise&>> _Sender>
      requires __completes_inline<_Sender, env_of_t<_Promise&>>
    struct __sender_awaiter<_Promise, _Sender>
      : __sender_awaiter_base<__value_t<_Sender, _Promise>, true>
    {
      using __value_t = __as_awaitable::__value_t<_Sender, _Promise>;

      constexpr explicit __sender_awaiter(_Sender&&                         __sndr,
                                          __std::coroutine_handle<_Promise> __hcoro)
        noexcept(__nothrow_move_constructible<_Sender>)
        : __sender_awaiter_base<__value_t, true>{__hcoro}
        , __sndr_(static_cast<_Sender&&>(__sndr))
      {}

      auto await_suspend([[maybe_unused]] __std::coroutine_handle<> __continuation)
        -> STDEXEC_PP_IF(STDEXEC_GCC(), bool, __std::coroutine_handle<>)
      {
        STDEXEC_ASSERT(this->__continuation_.handle() == __continuation);
        {
          auto __opstate = STDEXEC::connect(static_cast<_Sender&&>(__sndr_), __receiver_t(*this));
          // The following call to start will complete synchronously, writing its result
          // into the __result_ variant.
          STDEXEC::start(__opstate);
        }

        __continuation = this->__get_continuation();
        return STDEXEC_PP_IF(STDEXEC_GCC(), (__continuation.resume(), false), __continuation);
      }

     private:
      using __receiver_t = __sync_receiver_t<_Sender, _Promise>;
      _Sender __sndr_;
    };

    template <class _Sender, class _Promise>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
    __sender_awaiter(_Sender&&, __std::coroutine_handle<_Promise>)
      -> __sender_awaiter<_Promise, _Sender>;

    template <class _Sender, class _Promise>
    concept __awaitable_adapted_sender = sender_in<_Sender, env_of_t<_Promise&>>
                                      && __minvocable_q<__value_t, _Sender, _Promise>
                                      && requires(_Promise& __promise) {
                                           {
                                             __promise.unhandled_stopped()
                                           } -> __std::convertible_to<__std::coroutine_handle<>>;
                                         };

    template <class _Sender, class _Promise>
    concept __awaitable_sender = __awaitable_adapted_sender<__adapted_sender_t<_Sender>, _Promise>;

    template <class _Sender, class _Promise>
    concept __incompatible_sender = sender<_Sender> && __merror<__value_t<_Sender, _Promise>>;

    // clang-format off
    template <class _Ty, class _Promise>
    concept __simple_awaitable = requires(_Ty&& __value, _Promise& __promise) {
      { STDEXEC::__get_awaiter(static_cast<_Ty&&>(__value)) } -> __awaiter<_Promise>;
    };
    // clang-format on

    template <class _Sender, class _Promise>
    concept __has_transform_as_awaitable_member =
      sender_in<_Sender, env_of_t<_Promise>>
      && __has_as_awaitable_member<transform_sender_result_t<_Sender, env_of_t<_Promise>>,
                                   _Promise>;

    template <class _Sender, class _Promise>
    concept __awaitable_transform_sender =  //
      sender_in<_Sender, env_of_t<_Promise>>
      && __awaitable_sender<transform_sender_result_t<_Sender, env_of_t<_Promise>>, _Promise>;

    inline constexpr auto __with_member =  //
      []<class _Promise, __has_as_awaitable_member<_Promise> _Tp>(_Tp&& __t, auto& __promise)
        STDEXEC_AUTO_RETURN(static_cast<_Tp&&>(__t).as_awaitable(__promise));

    inline constexpr auto __with_transform_member =  //
      []<class _Promise, __has_transform_as_awaitable_member<_Promise> _Tp>(_Tp&&     __t,
                                                                            _Promise& __promise)
        STDEXEC_AUTO_RETURN(
          STDEXEC::transform_sender(static_cast<_Tp&&>(__t), STDEXEC::get_env(__promise))
            .as_awaitable(__promise));

    inline constexpr auto __with_await =  //
      []<class _Promise, __simple_awaitable<_Promise> _Tp>(_Tp&& __t, _Promise&)
        STDEXEC_AUTO_RETURN(static_cast<_Tp&&>(__t));

    inline constexpr auto __with_sender =  //
      []<class _Promise, __awaitable_transform_sender<_Promise> _Tp>(_Tp&& __t, _Promise& __promise)
        STDEXEC_AUTO_RETURN(__sender_awaiter{
          __as_awaitable::__adapt_sender_for_await(
            STDEXEC::transform_sender(static_cast<_Tp&&>(__t), STDEXEC::get_env(__promise))),
          __std::coroutine_handle<_Promise>::from_promise(__promise)});

    // NOT TO SPEC: It's a sender, but it isn't a sender in the current promise's
    // environment, so we can return the error type that results from trying to
    // compute the sender's value type:
    inline constexpr auto __with_incompatible_sender =  //
      []<class _Promise, __incompatible_sender<_Promise> _Tp>(_Tp&&, _Promise&)
    {
      return __value_t<_Tp, _Promise>{};
    };

    inline constexpr auto __identity =  //
      []<class _Tp>(_Tp&& __t, __ignore) noexcept -> decltype(auto)
    {
      return static_cast<_Tp&&>(__t);
    };

    inline constexpr auto __as_awaitable_impl =  //
      __first_callable{__with_member,
                       __with_transform_member,
                       __with_await,
                       __with_sender,
                       __with_incompatible_sender,
                       __identity};

  }  // namespace __as_awaitable

  struct as_awaitable_t : decltype(__as_awaitable::__as_awaitable_impl)
  {};

  inline constexpr as_awaitable_t as_awaitable{};
#endif
}  // namespace STDEXEC

STDEXEC_PRAGMA_POP()
