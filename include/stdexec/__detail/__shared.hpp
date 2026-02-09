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
#include "../stop_token.hpp"
#include "__atomic.hpp"
#include "__basic_sender.hpp"
#include "__env.hpp"
#include "__intrusive_slist.hpp"
#include "__memory.hpp"
#include "__meta.hpp"
#include "__optional.hpp"
#include "__queries.hpp"
#include "__receivers.hpp"
#include "__transform_completion_signatures.hpp"
#include "__tuple.hpp"
#include "__variant.hpp" // IWYU pragma: keep

#include <exception>
#include <mutex>
#include <utility>

////////////////////////////////////////////////////////////////////////////
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
namespace STDEXEC::__shared {
  template <class _Env, class _Variant>
  struct __shared_state_base;

  template <class _CvSender, class _Env>
  struct __shared_state;

  template <class _Env>
  using __env_t = __join_env_t<prop<get_stop_token_t, inplace_stop_token>, _Env>;

  struct __notify_fn {
    template <class _Receiver, class _Tag, class... _Args>
    constexpr void operator()(_Receiver& __rcvr, _Tag, _Args&&... __args) const noexcept {
      _Tag()(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
    }
  };

  struct __notify_visitor {
    template <class _Receiver, class _Tuple>
    constexpr void operator()(_Receiver& __rcvr, _Tuple&& __tupl) const noexcept {
      STDEXEC::__apply(__notify_fn(), static_cast<_Tuple&&>(__tupl), __rcvr);
    };
  };

  ////////////////////////////////////////////////////////////////////////////////////////
  template <class _Env, class _Variant>
  struct __receiver {
    using receiver_concept = receiver_t;
    template <class... _As>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr void set_value(_As&&... __as) noexcept {
      __sh_state_->__complete(set_value_t(), static_cast<_As&&>(__as)...);
    }

    template <class _Error>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr void set_error(_Error&& __err) noexcept {
      __sh_state_->__complete(set_error_t(), static_cast<_Error&&>(__err));
    }

    STDEXEC_ATTRIBUTE(always_inline)
    constexpr void set_stopped() noexcept {
      __sh_state_->__complete(set_stopped_t());
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> const __env_t<_Env>& {
      return __sh_state_->__env_;
    }

    // The receiver does not hold a reference to the shared state.
    __shared_state_base<_Env, _Variant>* __sh_state_;
  };

  ////////////////////////////////////////////////////////////////////////////////////////
  template <class _CvSender, class _Env>
  using __result_variant_t = __transform_completion_signatures_t<
    __completion_signatures_of_t<_CvSender, _Env>,
    __mbind_front_q<__decayed_tuple, set_value_t>::__f,
    __mbind_front_q<__decayed_tuple, set_error_t>::__f,
    __tuple<set_stopped_t>,
    __munique<__qq<__variant>>::__f,
    __tuple<set_error_t, std::exception_ptr>,
    __tuple<set_stopped_t>
  >;

  ////////////////////////////////////////////////////////////////////////////////////////
  template <class _CvChild, class _Env>
  [[nodiscard]]
  constexpr auto __mk_sh_state(_CvChild&& __child, _Env __env)
    -> std::shared_ptr<__shared_state<_CvChild, _Env>> {
    using __sh_state_t = __shared_state<_CvChild, _Env>;
    auto __tmp = __with_default(get_allocator, std::allocator<void>{})(__env);
    auto __alloc = STDEXEC::__rebind_allocator<__sh_state_t>(__tmp);
    return std::allocate_shared<__sh_state_t>(
      __alloc, static_cast<_CvChild&&>(__child), static_cast<_Env&&>(__env));
  }

  ////////////////////////////////////////////////////////////////////////////////////////
  struct __local_state_base : __immovable {
    __local_state_base() = default;
    constexpr virtual void __notify() noexcept {
      // should never be called
    }
    __local_state_base* __next_ = nullptr;
  };

  ////////////////////////////////////////////////////////////////////////////////////////
  // The operation state of ensure_started, and each operation state of split, has one of these,
  // created when the sender is connected. There are 0 or more of them for each underlying async
  // operation. It is what ensure_started- and split-sender's `connect` fn returns. It holds a
  // ref count to the shared state.
  template <class _Tag, class _CvChild, class _Env, class _Receiver>
  struct __local_state final : __local_state_base {
    using __stok_t = stop_token_of_t<env_of_t<_Receiver>>;
    using __sh_state_t = __shared_state<_CvChild, _Env>;
    using __sh_state_ptr_t = std::shared_ptr<__sh_state_t>;

    static_assert(__one_of<_Tag, split_t, ensure_started_t>);

    constexpr explicit __local_state(__sh_state_ptr_t __sh_state, _Receiver __rcvr) noexcept
      : __local_state::__local_state_base{}
      , __rcvr_(static_cast<_Receiver&&>(__rcvr))
      , __sh_state_(std::move(__sh_state)) {
    }

    constexpr ~__local_state() {
      if (__sh_state_) {
        __sh_state_->__detach();
      }
    }

    constexpr void start() noexcept {
      // Scenario: there are no more split senders, this is the only operation state, the
      // underlying operation has not yet been started, and the receiver's stop token is already
      // in the "stop requested" state. Then registering the stop callback will call
      // __local_state::operator() on *this synchronously. It may also be called asynchronously
      // at any point after the callback is registered. Beware. We are guaranteed, however, that
      // __local_state::operator() will not complete the operation or decrement the shared state's
      // ref count until after *this has been added to the waiters list.
      const auto __stok = STDEXEC::get_stop_token(STDEXEC::get_env(__rcvr_));
      __on_stop_.emplace(__stok, *this);

      // We haven't put __state in the waiters list yet and we are holding a ref count to
      // __sh_state_, so nothing can happen to the __sh_state_ here.

      // Start the shared op. As an optimization, skip it if the receiver's stop token has already
      // been signaled.
      if (!__stok.stop_requested()) {
        __sh_state_->__try_start();
        if (__sh_state_->__try_add_waiter(this, __stok)) {
          // successfully added the waiter
          return;
        }
      }

      // Otherwise, failed to add the waiter because of a stop-request.
      // Complete synchronously with set_stopped().
      __on_stop_.reset();
      std::exchange(__sh_state_, {})->__detach();
      STDEXEC::set_stopped(static_cast<_Receiver&&>(__rcvr_));
    }

    // Stop request callback:
    constexpr void operator()() noexcept {
      // We reach here when a split/ensure_started sender has received a stop request from the
      // receiver to which it is connected.
      if (std::unique_lock __lock{__sh_state_->__mutex_}) {
        // Remove this operation from the waiters list. Removal can fail if:
        //   1. It was already removed by another thread, or
        //   2. It hasn't been added yet (see `start` below), or
        //   3. The underlying operation has already completed.

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
      std::exchange(__sh_state_, {})->__detach();
      STDEXEC::set_stopped(static_cast<_Receiver&&>(__rcvr_));
    }

    // This is called from __shared_state::__notify_waiters when the input async operation
    // completes; or, if it has already completed when start is called, it is called from start:
    // __notify cannot race with __local_state::operator(). See comment in
    // __local_state::operator().
    constexpr void __notify() noexcept final {
      // The split algorithm sends by T const&. ensure_started sends by T&&.
      constexpr bool __is_split = __std::same_as<split_t, _Tag>;
      using __variant_t = decltype(__sh_state_->__results_);
      using __cv_variant_t = __if_c<__is_split, const __variant_t&, __variant_t>;

      __on_stop_.reset();

      STDEXEC::__visit(
        __notify_visitor(), static_cast<__cv_variant_t&&>(__sh_state_->__results_), __rcvr_);
    }

    _Receiver __rcvr_;
    __optional<stop_callback_for_t<__stok_t, __local_state&>> __on_stop_{};
    __sh_state_ptr_t __sh_state_;
  };

  ////////////////////////////////////////////////////////////////////////////////////////
  //! Base class for heap-allocatable shared state for `split` and `ensure_started`.
  template <class _Env, class _Variant>
  struct __shared_state_base : __local_state_base {
    using __waiters_list_t = __intrusive_slist<&__local_state_base::__next_>;

    constexpr explicit __shared_state_base(_Env __env)
      : __env_(
          __env::__join(
            prop{get_stop_token, __stop_source_.get_token()},
            static_cast<_Env&&>(__env))) {
      __results_.template emplace<__tuple<set_stopped_t>>();
    }

    virtual ~__shared_state_base() = 0;

    /// @brief This is called when the shared async operation completes.
    /// @post __waiters_ is set to a known "tombstone" value.
    template <class _Tag, class... _As>
    void __complete(_Tag, _As&&... __as) noexcept {
      STDEXEC_TRY {
        using __tuple_t = __decayed_tuple<_Tag, _As...>;
        __results_.template emplace<__tuple_t>(_Tag(), static_cast<_As&&>(__as)...);
      }
      STDEXEC_CATCH_ALL {
        if constexpr (!__nothrow_decay_copyable<_As...>) {
          using __tuple_t = __decayed_tuple<set_error_t, std::exception_ptr>;
          __results_.template emplace<__tuple_t>(set_error, std::current_exception());
        }
      }

      __notify_waiters();
    }

    /// @brief This is called when the shared async operation completes.
    /// @post __waiters_ is set to a known "tombstone" value.
    void __notify_waiters() noexcept {
      __waiters_list_t __waiters_copy{this};

      // Set the waiters list to a known "tombstone" value that we can check later.
      {
        std::lock_guard __lock{this->__mutex_};
        this->__waiters_.swap(__waiters_copy);
      }

      STDEXEC_ASSERT(__waiters_copy.front() != this);
      for (auto __itr = __waiters_copy.begin(); __itr != __waiters_copy.end();) {
        __local_state_base* __item = *__itr;

        // We must increment the iterator before calling notify, since notify may end up
        // triggering *__item to be destructed on another thread, and the intrusive slist's
        // iterator increment relies on __item.
        ++__itr;
        __item->__notify();
      }

      // Set the "is running" bit in the ref count to zero. Delete the shared state if the
      // ref-count is now zero.
      __set_completed();
    }

    constexpr virtual void __set_completed() noexcept = 0;

    std::mutex __mutex_{}; // This mutex guards access to __waiters_.
    __waiters_list_t __waiters_{};
    inplace_stop_source __stop_source_{};
    __env_t<_Env> __env_;
    _Variant __results_{__no_init}; // Initialized to the "set_stopped" state in the ctor.
  };

  template <class _Env, class _Variant>
  __shared_state_base<_Env, _Variant>::~__shared_state_base() = default;

  ////////////////////////////////////////////////////////////////////////////////////////
  //! Heap-allocatable shared state for `stdexec::split` and `stdexec::ensure_started`.
  template <class _CvSender, class _Env>
  struct STDEXEC_ATTRIBUTE(empty_bases) __shared_state final
    : std::enable_shared_from_this<__shared_state<_CvSender, _Env>>
    , __shared_state_base<_Env, __result_variant_t<_CvSender, _Env>> {
    using __receiver_t = __receiver<_Env, __result_variant_t<_CvSender, _Env>>;
    using __waiters_list_t = __shared_state::__shared_state_base::__waiters_list_t;

    constexpr explicit __shared_state(_CvSender&& __sndr, _Env __env)
      : __shared_state::__shared_state_base(static_cast<_Env&&>(__env))
      , __shared_op_(STDEXEC::connect(static_cast<_CvSender&&>(__sndr), __receiver_t{this})) {
    }

    bool __set_started() noexcept {
      if (!__started_.test_and_set(__std::memory_order_acq_rel)) {
        // we were the first to start the operation
        std::shared_ptr<__shared_state> __expected{};
        return __started_ref_.compare_exchange_strong(
          __expected,
          this->shared_from_this(),
          __std::memory_order_acq_rel,
          __std::memory_order_acquire);
      }
      return false; // already started
    }

    /// @post The `__started_` atomic flag is set in the shared state's ref count, OR the
    /// __waiters_ list is set to the known "tombstone" value indicating completion.
    void __try_start() noexcept {
      // With the split algorithm, multiple split senders can be started simultaneously,
      // but only one should start the shared async operation. If __set_started() reports
      // that the operation has already been started, do nothing.
      if (this->__set_started()) {
        // we are the first to start the underlying operation
        if (this->__stop_source_.stop_requested()) {
          // Stop has already been requested. Rather than starting the operation, complete with
          // set_stopped immediately.
          // 1. Sets __waiters_ to a known "tombstone" value.
          // 2. Notifies all the waiters that the operation has stopped.
          // 3. Sets the "is running" bit in the ref count to 0.
          this->__notify_waiters();
        } else {
          STDEXEC::start(__shared_op_);
        }
      }
    }

    template <class _StopToken>
    auto __try_add_waiter(__local_state_base* __waiter, _StopToken __stok) noexcept -> bool {
      std::unique_lock __lock{this->__mutex_};
      if (this->__waiters_.front() == this) {
        // The work has already completed. Notify the waiter immediately.
        __lock.unlock();
        __waiter->__notify();
        return true;
      } else if (__stok.stop_requested()) {
        // Stop has been requested. Do not add the waiter.
        return false;
      } else {
        // Add the waiter to the list.
        this->__waiters_.push_front(__waiter);
        return true;
      }
    }

    constexpr void __detach() noexcept {
      // increments the use count:
      auto __started_ptr = __started_ref_.load();
      // If the use count is 3 (one for *this, one for __started_ref_, and one for
      // __started_ptr), then we are the final "consumer". Ask the operation to stop
      // early.
      if (3 == __started_ptr.use_count()) {
        this->__stop_source_.request_stop();
      }
    }

    constexpr virtual void __set_completed() noexcept final {
      __started_ref_.store({}, __std::memory_order_release);
    }

    __std::atomic_flag __started_{};
    // this shared_ptr is non-null when the operation is running:
    __std::__atomic_shared_ptr<__shared_state> __started_ref_;
    connect_result_t<_CvSender, __receiver_t> __shared_op_;
  };

  template <class _Cv, class _CvSender, class _Env>
  using __make_completions_t = __try_make_completion_signatures<
    _CvSender,
    __env_t<_Env>,
    completion_signatures<set_error_t(__mcall1<_Cv, std::exception_ptr>), set_stopped_t()>,
    __mtransform<_Cv, __mcompose<__qq<completion_signatures>, __qf<set_value_t>>>,
    __mtransform<_Cv, __mcompose<__qq<completion_signatures>, __qf<set_error_t>>>
  >;

  // split completes with const T&. ensure_started completes with T&&.
  template <class _Tag>
  using __cvref_results_t =
    __mcompose<__if_c<__same_as<_Tag, split_t>, __cpclr, __cp>, __q1<__decay_t>>;

  template <class _Tag>
  struct __impls : __sexpr_defaults {
    template <class _CvChild, class _Env>
    static consteval auto __get_completion_signatures() {
      // Use the senders decay-copyability as a proxy for whether it is lvalue-connectable.
      // TODO: update this for constant evaluation
      if constexpr (__decay_copyable<_CvChild>) {
        return __make_completions_t<__cvref_results_t<_Tag>, _CvChild, _Env>();
      } else {
        return STDEXEC::__throw_compile_time_error<
          _WHAT_(_SENDER_TYPE_IS_NOT_DECAY_COPYABLE_),
          _WHERE_(_IN_ALGORITHM_, _Tag),
          _WITH_PRETTY_SENDER_<_CvChild>
        >();
      }
    }

    template <class _CvSender>
    static consteval auto get_completion_signatures() {
      static_assert(sender_expr_for<_CvSender, _Tag>);
      return __get_completion_signatures<__child_of<_CvSender>, __decay_t<__data_of<_CvSender>>>();
    };
  };

  /// This class is a split sender when _Tag is split_t, and an ensure_started sender when
  /// _Tag is ensure_started_t.
  template <class _Tag, class _CvChild, class _Env>
  struct __sndr : __if_c<__same_as<_Tag, split_t>, __empty, __move_only> {
    using sender_concept = sender_t;
    using __tag_t = _Tag;

    constexpr explicit __sndr(_Tag, _CvChild&& __child, _Env __env)
      : __sh_state_(
          __shared::__mk_sh_state(static_cast<_CvChild&&>(__child), static_cast<_Env&&>(__env))) {
    }

    __sndr(__sndr&&) = default;
    __sndr& operator=(__sndr&&) = default;

    __sndr(const __sndr&) = default;
    __sndr& operator=(const __sndr&) = default;

    constexpr ~__sndr() {
      if (__sh_state_) {
        __sh_state_->__detach();
      }
    }

    template <class>
    static consteval auto get_completion_signatures() {
      return __impls<_Tag>::template __get_completion_signatures<_CvChild, _Env>();
    }

    template <class _Receiver>
    constexpr auto connect(_Receiver __rcvr) && noexcept //
      -> __local_state<_Tag, _CvChild, _Env, _Receiver> {
      return __local_state<_Tag, _CvChild, _Env, _Receiver>{
        std::move(__sh_state_), static_cast<_Receiver&&>(__rcvr)};
    }

    template <class _Receiver>
      requires __same_as<_Tag, split_t>
    constexpr auto connect(_Receiver __rcvr) const & noexcept //
      -> __local_state<_Tag, _CvChild, _Env, _Receiver> {
      return __local_state<_Tag, _CvChild, _Env, _Receiver>{
        __sh_state_, static_cast<_Receiver&&>(__rcvr)};
    }

   private:
    friend struct __ensure_started::ensure_started_t;
    std::shared_ptr<__shared_state<_CvChild, _Env>> __sh_state_;
  };

  template <class _Tag, class _CvChild, class _Env>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
    __sndr(_Tag, _CvChild&&, _Env) -> __sndr<_Tag, _CvChild, _Env>;

} // namespace STDEXEC::__shared
