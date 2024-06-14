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

// include these after __execution_fwd.hpp
#include "__basic_sender.hpp"
#include "__cpo.hpp"
#include "__env.hpp"
#include "__intrusive_ptr.hpp"
#include "__intrusive_slist.hpp"
#include "__meta.hpp"
#include "__transform_completion_signatures.hpp"
#include "__tuple.hpp"
#include "__variant.hpp"

#include "../stop_token.hpp"
#include "../functional.hpp"

#include <exception>
#include <mutex>
#include <optional>

namespace stdexec {
  ////////////////////////////////////////////////////////////////////////////
  // shared components of split and ensure_started
  //
  // The split and ensure_started algorithms are very similar in implementation.
  // The salient differences are:
  //
  // split: the input async operation is always connected. It is only
  //   started when one of the split senders is connected and started.
  //   split senders are copyable, so there are multiple operation states
  //   to be notified on completion. These are stored in an instrusive
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
  namespace __shared {
    template <class _BaseEnv>
    using __env_t = //
      __env::__join_t<
        __env::__with<inplace_stop_token, get_stop_token_t>,
        _BaseEnv>; // BUGBUG NOT TO SPEC

    template <class _Receiver>
    auto __make_notify_visitor(_Receiver& __rcvr) noexcept {
      return [&]<class _Tuple>(_Tuple&& __tupl) noexcept -> void {
        __tupl.apply(
          [&](auto __tag, auto&&... __args) noexcept -> void {
            __tag(static_cast<_Receiver&&>(__rcvr), __forward_like<_Tuple>(__args)...);
          },
          __tupl);
      };
    }

    struct __local_state_base : __immovable {
      using __notify_fn = void(__local_state_base*) noexcept;

      __notify_fn* __notify_{};
      __local_state_base* __next_{};
    };

    template <class _CvrefSender, class _Env>
    struct __shared_state;

    // The operation state of ensure_started, and each operation state of split, has one of these,
    // created when the sender is connected. There are 0 or more of them for each underlying async
    // operation. It is what ensure_started- and split-sender's `get_state` fn returns. It holds a
    // ref count to the shared state.
    template <class _CvrefSender, class _Receiver>
    struct __local_state
      : __local_state_base
      , __enable_receiver_from_this<_CvrefSender, _Receiver> {
      using __tag_t = tag_of_t<_CvrefSender>;
      using __stok_t = stop_token_of_t<env_of_t<_Receiver>>;
      static_assert(__one_of<__tag_t, __split::__split_t, __ensure_started::__ensure_started_t>);

      explicit __local_state(_CvrefSender&& __sndr) noexcept
        : __local_state::__local_state_base{{}, &__notify<tag_of_t<_CvrefSender>>}
        , __sh_state_(__get_sh_state(__sndr)) {
      }

      ~__local_state() {
        __sh_state_t::__detach(__sh_state_);
      }

      // Stop request callback:
      void operator()() noexcept {
        // We reach here when a split/ensure_started sender has received a stop request from the
        // receiver to which it is connected.
        if (std::unique_lock __lock{__sh_state_->__mutex_}) {
          // Remove this operation from the waiters list. Removal can fail if:
          //   1. It was already removed by another thread, or
          //   2. It hasn't been added yet (see `start` below), or
          //   3. The underlying operation has already completed.
          //
          // In each case, the right thing to do is nothing. If (1) then we raced with another
          // thread and lost. In that case, the other thread will take care of it. If (2) then
          // `start` will take care of it. If (3) then this stop request is safe to ignore.
          if (!__sh_state_->__waiters_.remove(this))
            return;
        }

        // The following code and the __notify function cannot both execute. This is because the
        // __notify function is called from the shared state's __notify_waiters function, which
        // first sets __waiters_ to the completed state. As a result, the attempt to remove `this`
        // from the waiters list above will fail and this stop request is ignored.
        __sh_state_t::__detach(__sh_state_);
        stdexec::set_stopped(static_cast<_Receiver&&>(this->__receiver()));
      }

      // This is called from __shared_state::__notify_waiters when the input async operation
      // completes; or, if it has already completed when start is called, it is called from start:
      // __notify cannot race with __on_stop_request. See comment in __on_stop_request.
      template <class _Tag>
      static void __notify(__local_state_base* __base) noexcept {
        auto* const __self = static_cast<__local_state*>(__base);

        // The split algorithm sends by T const&. ensure_started sends by T&&.
        constexpr bool __is_split = same_as<__split::__split_t, _Tag>;
        using __variant_t = decltype(__self->__sh_state_->__results_);
        using __cv_variant_t = __if_c<__is_split, const __variant_t&, __variant_t>;

        __self->__on_stop_.reset();

        auto __visitor = __make_notify_visitor(__self->__receiver());
        __variant_t::visit(
          __visitor, static_cast<__cv_variant_t&&>(__self->__sh_state_->__results_));
      }

      static auto __get_sh_state(_CvrefSender& __sndr) noexcept {
        return __sndr.apply(static_cast<_CvrefSender&&>(__sndr), __detail::__get_data()).__sh_state_;
      }

      using __sh_state_ptr_t = __result_of<__get_sh_state, _CvrefSender&>;
      using __sh_state_t = typename __sh_state_ptr_t::element_type;

      std::optional<stop_callback_for_t<__stok_t, __local_state&>> __on_stop_{};
      __sh_state_ptr_t __sh_state_;
    };

    template <class _CvrefSenderId, class _EnvId>
    struct __receiver {
      using _CvrefSender = stdexec::__cvref_t<_CvrefSenderId>;
      using _Env = stdexec::__t<_EnvId>;

      struct __t {
        using receiver_concept = receiver_t;
        using __id = __receiver;

        template <class... _As>
        STDEXEC_ATTRIBUTE((always_inline))
        void
          set_value(_As&&... __as) noexcept {
          __sh_state_->__complete(set_value_t(), static_cast<_As&&>(__as)...);
        }

        template <class _Error>
        STDEXEC_ATTRIBUTE((always_inline))
        void
          set_error(_Error&& __err) noexcept {
          __sh_state_->__complete(set_error_t(), static_cast<_Error&&>(__err));
        }

        STDEXEC_ATTRIBUTE((always_inline))
        void
          set_stopped() noexcept {
          __sh_state_->__complete(set_stopped_t());
        }

        auto get_env() const noexcept -> const __env_t<_Env>& {
          return __sh_state_->__env_;
        }

        // The receiver does not hold a reference to the shared state.
        __shared_state<_CvrefSender, _Env>* __sh_state_;
      };
    };

    inline __local_state_base* __get_tombstone() noexcept {
      static __local_state_base __tombstone_{{}, nullptr, nullptr};
      return &__tombstone_;
    }

    template <class _CvrefSender, class _Env>
    struct __shared_state
      : private __enable_intrusive_from_this<__shared_state<_CvrefSender, _Env>, 2> {
      using __receiver_t = __t<__receiver<__cvref_id<_CvrefSender>, __id<_Env>>>;
      using __waiters_list_t = __intrusive_slist<&__local_state_base::__next_>;

      using __variant_t = //
        __transform_completion_signatures<
          __completion_signatures_of_t<_CvrefSender, _Env>,
          __mbind_front_q<__decayed_tuple, set_value_t>::__f,
          __mbind_front_q<__decayed_tuple, set_error_t>::__f,
          __tuple_for<set_error_t, std::exception_ptr>,
          __munique<__mbind_front_q<__variant_for, __tuple_for<set_stopped_t>>>::__f,
          __tuple_for<set_error_t, std::exception_ptr>>;

      static constexpr std::size_t __started_bit = 0;
      static constexpr std::size_t __completed_bit = 1;

      inplace_stop_source __stop_source_{};
      __env_t<_Env> __env_;
      __variant_t __results_{}; // Defaults to the "set_stopped" state
      std::mutex __mutex_;      // This mutex guards access to __waiters_.
      __waiters_list_t __waiters_{};
      connect_result_t<_CvrefSender, __receiver_t> __shared_op_;

      explicit __shared_state(_CvrefSender&& __sndr, _Env __env)
        : __env_(__env::__join(
          __env::__with(__stop_source_.get_token(), get_stop_token),
          static_cast<_Env&&>(__env)))
        , __shared_op_(connect(static_cast<_CvrefSender&&>(__sndr), __receiver_t{this})) {
        // add one ref count to account for the case where there are no watchers left but the
        // shared op is still running.
        this->__inc_ref();
      }

      // The caller of this wants to release their reference to the shared state. The ref
      // count must be at least 2 at this point: one owned by the caller, and one added in the
      // __shared_state ctor.
      static void __detach(__intrusive_ptr<__shared_state, 2>& __ptr) noexcept {
        // Ask the intrusive ptr to stop managing the reference count so we can manage it manually.
        if (auto* __self = __ptr.__release_()) {
          auto __old = __self->__dec_ref();
          STDEXEC_ASSERT(__count(__old) >= 2);

          if (__count(__old) == 2) {
            // The last watcher has released its reference. Asked the shared op to stop.
            static_cast<__shared_state*>(__self)->__stop_source_.request_stop();

            // Additionally, if the shared op was never started, or if it has already completed,
            // then the shared state is no longer needed. Decrement the ref count to 0 here, which
            // will delete __self.
            if (!__bit<__started_bit>(__old) || __bit<__completed_bit>(__old)) {
              __self->__dec_ref();
            }
          }
        }
      }

      /// @post The started bit is set in the shared state's ref count, OR the __waiters_ list
      /// is set to the known "tombstone" value indicating completion.
      void __try_start() noexcept {
        // With the split algorithm, multiple split senders can be started simultaneously, but
        // only one should start the shared async operation. If the "started" bit is set, then
        // someone else has already started the shared operation. Do nothing.
        if (this->template __is_set<__started_bit>()) {
          return;
        } else if (__bit<__started_bit>(this->template __set_bit<__started_bit>())) {
          return;
        } else if (__stop_source_.stop_requested()) {
          // Stop has already been requested. Rather than starting the operation, complete with
          // set_stopped immediately.
          // 1. Sets __waiters_ to a known "tombstone" value
          // 2. Notifies all the waiters that the operation has stopped
          // 3. Sets the "completed" bit in the ref count.
          __notify_waiters();
          return;
        } else {
          stdexec::start(__shared_op_);
        }
      }

      template <class _StopToken>
      bool __try_add_waiter(__local_state_base* __waiter, _StopToken __stok) noexcept {
        std::unique_lock __lock{__mutex_};
        if (__waiters_.front() == __get_tombstone()) {
          // The work has already completed. Notify the waiter immediately.
          __lock.unlock();
          __waiter->__notify_(__waiter);
          return true;
        } else if (__stok.stop_requested()) {
          // Stop has been requested. Do not add the waiter.
          return false;
        } else {
          // Add the waiter to the list.
          __waiters_.push_front(__waiter);
          return true;
        }
      }

      /// @brief This is called when the shared async operation completes.
      /// @post __waiters_ is set to a known "tombstone" value.
      template <class _Tag, class... _As>
      void __complete(_Tag, _As&&... __as) noexcept {
        try {
          using __tuple_t = __decayed_tuple<_Tag, _As...>;
          __results_.template emplace<__tuple_t>(_Tag(), static_cast<_As&&>(__as)...);
        } catch (...) {
          using __tuple_t = __decayed_tuple<set_error_t, std::exception_ptr>;
          __results_.template emplace<__tuple_t>(set_error, std::current_exception());
        }

        __notify_waiters();
      }

      /// @brief This is called when the shared async operation completes.
      /// @post __waiters_ is set to a known "tombstone" value.
      void __notify_waiters() noexcept {
        __waiters_list_t __waiters_copy{__get_tombstone()};

        // Set the waiters list to a known "tombstone" value that we can check later.
        {
          std::lock_guard __lock{__mutex_};
          __waiters_.swap(__waiters_copy);
        }

        STDEXEC_ASSERT(__waiters_copy.front() != __get_tombstone());
        for (__local_state_base* __item: __waiters_copy) {
          __item->__notify_(__item);
        }

        // Set the "completed" bit in the ref count. If the ref count is 1, then there are no more
        // waiters. Release the final reference.
        if (__count(this->template __set_bit<__completed_bit>()) == 1) {
          this->__dec_ref(); // release the extra ref count, deletes this
        }
      }
    };

    template <class _Cvref, class _CvrefSender, class _Env>
    using __make_completions = //
      __try_make_completion_signatures<
        // NOT TO SPEC:
        // See https://github.com/cplusplus/sender-receiver/issues/23
        _CvrefSender,
        __env_t<_Env>,
        completion_signatures<
          set_error_t(__minvoke<_Cvref, std::exception_ptr>),
          set_stopped_t()>, // NOT TO SPEC
        __transform<_Cvref, __mcompose<__q<completion_signatures>, __qf<set_value_t>>>,
        __transform<_Cvref, __mcompose<__q<completion_signatures>, __qf<set_error_t>>>>;

    // split completes with const T&. ensure_started completes with T&&.
    template <class _Tag>
    using __cvref_results_t = //
      __mcompose<__if_c<same_as<_Tag, __split::__split_t>, __cpclr, __cp>, __q<__decay_t>>;

    // NOTE: the use of __mapply in the return type below takes advantage of the fact that _ShState
    // denotes an instance of the __shared_state template, which is parameterized on the
    // cvref-qualified sender and the environment.
    template <class _Tag, class _ShState>
    using __completions = //
      __mapply<__mbind_front_q<__make_completions, __cvref_results_t<_Tag>>, _ShState>;

    template <class _CvrefSender, class _Env, bool _Copyable = true>
    struct __box {
      using __tag_t = __if_c<_Copyable, __split::__split_t, __ensure_started::__ensure_started_t>;
      using __sh_state_t = __shared_state<_CvrefSender, _Env>;

      __box(__tag_t, __intrusive_ptr<__sh_state_t, 2> __sh_state) noexcept
        : __sh_state_(std::move(__sh_state)) {
      }

      __box(__box&&) noexcept = default;
      __box(const __box&) noexcept
        requires _Copyable
      = default;

      ~__box() {
        __sh_state_t::__detach(__sh_state_);
      }

      __intrusive_ptr<__sh_state_t, 2> __sh_state_;
    };

    template <class _CvrefSender, class _Env>
    __box(__split::__split_t, __intrusive_ptr<__shared_state<_CvrefSender, _Env>, 2>) //
      ->__box<_CvrefSender, _Env, true>;

    template <class _CvrefSender, class _Env>
    __box(
      __ensure_started::__ensure_started_t,
      __intrusive_ptr<__shared_state<_CvrefSender, _Env>, 2>) -> __box<_CvrefSender, _Env, false>;

    template <class _Tag>
    struct __shared_impl : __sexpr_defaults {
      static constexpr auto get_state = //
        []<class _CvrefSender, class _Receiver>(_CvrefSender&& __sndr, _Receiver&) noexcept
        -> __local_state<_CvrefSender, _Receiver> {
        static_assert(sender_expr_for<_CvrefSender, _Tag>);
        return __local_state<_CvrefSender, _Receiver>{static_cast<_CvrefSender&&>(__sndr)};
      };

      static constexpr auto get_completion_signatures = //
        []<class _Self, class _OtherEnv>(const _Self&, _OtherEnv&&) noexcept
        -> __completions<_Tag, typename __data_of<_Self>::__sh_state_t> {
        static_assert(sender_expr_for<_Self, _Tag>);
        return {};
      };

      static constexpr auto start = //
        []<class _Sender, class _Receiver>(
          __local_state<_Sender, _Receiver>& __self,
          _Receiver& __rcvr) noexcept -> void {
        using __sh_state_t = typename __local_state<_Sender, _Receiver>::__sh_state_t;
        // Scenario: there are no more split senders, this is the only operation state, the
        // underlying operation has not yet been started, and the receiver's stop token is already
        // in the "stop requested" state. Then registering the stop callback will call
        // __on_stop_request on __self synchronously. It may also be called asynchronously at
        // any point after the callback is registered. Beware. We are guaranteed, however, that
        // __on_stop_request will not complete the operation or decrement the shared state's ref
        // count until after __self has been added to the waiters list.
        const auto __stok = stdexec::get_stop_token(stdexec::get_env(__rcvr));
        __self.__on_stop_.emplace(__stok, __self);

        // We haven't put __self in the waiters list yet and we are holding a ref count to
        // __sh_state_, so nothing can happen to the __sh_state_ here.

        // Start the shared op. As an optimization, skip it if the receiver's stop token has already
        // been signaled.
        if (!__stok.stop_requested()) {
          __self.__sh_state_->__try_start();
          if (__self.__sh_state_->__try_add_waiter(&__self, __stok)) {
            // successfully added the waiter
            return;
          }
        }

        // Otherwise, failed to add the waiter because of a stop-request.
        // Complete synchronously with set_stopped().
        __self.__on_stop_.reset();
        __sh_state_t::__detach(__self.__sh_state_);
        stdexec::set_stopped(static_cast<_Receiver&&>(__rcvr));
      };
    };
  } // namespace __shared
} // namespace stdexec
