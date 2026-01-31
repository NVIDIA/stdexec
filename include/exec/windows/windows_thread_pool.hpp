/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Copyright (c) 2025 NVIDIA Corporation
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

#if !__has_include(<windows.h>)
#  error "windows.h not found."
#else
// windows.h must be included before threadpoolapiset.h
// clang-format off
#include <windows.h>
#include <threadpoolapiset.h>
// clang-format on

#  include "../../stdexec/__detail/__atomic.hpp"
#  include "../../stdexec/__detail/__manual_lifetime.hpp"
#  include "../../stdexec/__detail/__operation_states.hpp"
#  include "../../stdexec/__detail/__receivers.hpp"
#  include "../../stdexec/__detail/__schedulers.hpp"
#  include "../../stdexec/__detail/__stop_token.hpp"
#  include "../timed_scheduler.hpp" // IWYU pragma: keep
#  include "./filetime_clock.hpp"

#  include <system_error>
#  include <utility>

namespace exec::__win32 {
  class windows_thread_pool {
    struct attrs;
    class scheduler;
    class schedule_sender;
    class schedule_op_base;

    template <class Rcvr>
    struct _schedule_op {
      class type;
    };
    template <class Rcvr>
    using schedule_op = _schedule_op<Rcvr>::type;

    template <class StopToken>
    struct _cancellable_schedule_op_base {
      class type;
      using __t = type;
    };
    template <class StopToken>
    using cancellable_schedule_op_base = _cancellable_schedule_op_base<StopToken>::type;

    template <class Rcvr>
    struct _cancellable_schedule_op {
      class type;
      using __t = type;
    };
    template <class Rcvr>
    using cancellable_schedule_op = _cancellable_schedule_op<Rcvr>::type;

    template <class StopToken>
    struct _time_schedule_op {
      class type;
      using __t = type;
    };
    template <class StopToken>
    using time_schedule_op = _time_schedule_op<StopToken>::type;

    template <class Rcvr>
    struct _schedule_at_op {
      class type;
      using __t = type;
    };
    template <class Rcvr>
    using schedule_at_op = _schedule_at_op<Rcvr>::type;

    template <class Duration, class Rcvr>
    struct _schedule_after_op {
      class type;
      using __t = type;
    };
    template <class Duration, class Rcvr>
    using schedule_after_op = _schedule_after_op<Duration, Rcvr>::type;

    class schedule_at_sender;

    template <class Duration>
    struct _schedule_after {
      class sender;
      using __t = sender;
    };
    template <class Duration>
    using schedule_after_sender = _schedule_after<Duration>::sender;

    using clock_type = filetime_clock;

   public:
    // Initialise to use the process' default thread-pool.
    windows_thread_pool() noexcept;

    // Construct to an independend thread-pool with a dynamic number of
    // threads that varies between a min and a max number of threads.
    explicit windows_thread_pool(std::uint32_t minThreadCount, std::uint32_t maxThreadCount);

    ~windows_thread_pool();

    auto get_scheduler() noexcept -> scheduler;

   private:
    PTP_POOL threadPool_;
  };

  /////////////////////////
  // Non-cancellable schedule() operation

  class windows_thread_pool::schedule_op_base {
   public:
    schedule_op_base(schedule_op_base &&) = delete;
    auto operator=(schedule_op_base &&) -> schedule_op_base & = delete;

    ~schedule_op_base();

    void start() noexcept;

   protected:
    schedule_op_base(windows_thread_pool &pool, PTP_WORK_CALLBACK workCallback);

   private:
    TP_CALLBACK_ENVIRON environ_;
    PTP_WORK work_;
  };

  template <class Rcvr>
  class windows_thread_pool::_schedule_op<Rcvr>::type final
    : public windows_thread_pool::schedule_op_base {
   public:
    explicit type(windows_thread_pool &pool, Rcvr rcvr)
      : schedule_op_base(pool, &work_callback)
      , rcvr_(std::move(rcvr)) {
    }

   private:
    static void CALLBACK
      work_callback(PTP_CALLBACK_INSTANCE, void *workContext, PTP_WORK) noexcept {
      auto &op = *static_cast<type *>(workContext);
      STDEXEC::set_value(std::move(op.rcvr_));
    }

    Rcvr rcvr_;
  };

  ///////////////////////////
  // Cancellable schedule() operation

  template <class StopToken>
  class windows_thread_pool::_cancellable_schedule_op_base<StopToken>::type {
   public:
    using operation_state_concept = STDEXEC::operation_state_t;
    type(type &&) = delete;
    auto operator=(type &&) -> type & = delete;

    ~type() {
      ::CloseThreadpoolWork(work_);
      ::DestroyThreadpoolEnvironment(&environ_);
      delete state_;
    }

   protected:
    explicit type(windows_thread_pool &pool, bool isStopPossible) {
      ::InitializeThreadpoolEnvironment(&environ_);
      ::SetThreadpoolCallbackPool(&environ_, pool.threadPool_);

      work_ = ::CreateThreadpoolWork(
        isStopPossible ? &stoppable_work_callback : &unstoppable_work_callback,
        static_cast<void *>(this),
        &environ_);
      if (work_ == nullptr) {
        DWORD errorCode = ::GetLastError();
        ::DestroyThreadpoolEnvironment(&environ_);
        throw std::system_error{
          static_cast<int>(errorCode), std::system_category(), "CreateThreadpoolWork()"};
      }

      if (isStopPossible) {
        state_ = new (std::nothrow) STDEXEC::__std::atomic<std::uint32_t>(not_started);
        if (state_ == nullptr) {
          ::CloseThreadpoolWork(work_);
          ::DestroyThreadpoolEnvironment(&environ_);
          throw std::bad_alloc{};
        }
      } else {
        state_ = nullptr;
      }
    }

    void start_impl(const StopToken &stopToken) & noexcept {
      if (state_ != nullptr) {
        // Short-circuit all of this if stopToken.stop_requested() is already
        // true.
        //
        // TODO: this means done can be delivered on "the wrong thread"
        if (stopToken.stop_requested()) {
          set_stopped_impl();
          return;
        }

        stopCallback_.__construct(stopToken, stop_requested_callback{*this});

        // Take a copy of the 'state' pointer prior to submitting the
        // work as the operation-state may have already been destroyed
        // on another thread by the time SubmitThreadpoolWork() returns.
        auto *state = state_;

        ::SubmitThreadpoolWork(work_);

        // Signal that SubmitThreadpoolWork() has returned and that it is
        // now safe for the stop-request to request cancellation of the
        // work items.
        const auto prevState =
          state->fetch_add(submit_complete_flag, STDEXEC::__std::memory_order_acq_rel);
        if ((prevState & stop_requested_flag) != 0) {
          // stop was requested before the call to SubmitThreadpoolWork()
          // returned and before the work started executing. It was not
          // safe for the request_stop() method to cancel the work before
          // it had finished being submitted so it has delegated responsibility
          // for cancelling the just-submitted work to us to do once we
          // finished submitting the work.
          complete_with_done();
        } else if ((prevState & running_flag) != 0) {
          // Otherwise, it's possible that the work item may have started
          // running on another thread already, prior to us returning.
          // If this is the case then, to avoid leaving us with a
          // dangling reference to the 'state' when the operation-state
          // is destroyed, it will detach the 'state' from the operation-state
          // and delegate the delete of the 'state' to us.
          delete state;
        }
      } else {
        // A stop-request is not possible so skip the extra
        // synchronisation needed to support it.
        ::SubmitThreadpoolWork(work_);
      }
    }

   private:
    static void CALLBACK
      unstoppable_work_callback(PTP_CALLBACK_INSTANCE, void *workContext, PTP_WORK) noexcept {
      auto &op = *static_cast<type *>(workContext);
      op.set_value_impl();
    }

    static void CALLBACK
      stoppable_work_callback(PTP_CALLBACK_INSTANCE, void *workContext, PTP_WORK) noexcept {
      auto &op = *static_cast<type *>(workContext);

      // Signal that the work callback has started executing.
      auto prevState = op.state_->fetch_add(starting_flag, STDEXEC::__std::memory_order_acq_rel);
      if ((prevState & stop_requested_flag) != 0) {
        // request_stop() is already running and is waiting for this callback
        // to finish executing. So we return immediately here without doing
        // anything further so that we don't introduce a deadlock.
        // In particular, we don't want to try to deregister the stop-callback
        // which will block waiting for the request_stop() method to return.
        return;
      }

      // Note that it's possible that stop might be requested after setting
      // the 'starting' flag but before we deregister the stop callback.
      // We're going to ignore these stop-requests as we already won the race
      // in the fetch_add() above and ignoring them simplifies some of the
      // cancellation logic.

      op.stopCallback_.__destroy();

      prevState = op.state_->fetch_add(running_flag, STDEXEC::__std::memory_order_acq_rel);
      if (prevState == starting_flag) {
        // start() method has not yet finished submitting the work
        // on another thread and so is still accessing the 'state'.
        // This means we don't want to let the operation-state destructor
        // free the state memory. Instead, we have just delegated
        // responsibility for freeing this memory to the start() method
        // and we clear the start_ member here to prevent the destructor
        // from freeing it.
        op.state_ = nullptr;
      }

      op.set_value_impl();
    }

    void request_stop() noexcept {
      auto prevState = state_->load(STDEXEC::__std::memory_order_relaxed);
      do {
        STDEXEC_ASSERT((prevState & running_flag) == 0);
        if ((prevState & starting_flag) != 0) {
          // Work callback won the race and will be waiting for
          // us to return so it can deregister the stop-callback.
          // Return immediately so we don't deadlock.
          return;
        }
      } while (!state_->compare_exchange_weak(
        prevState,
        prevState | stop_requested_flag,
        STDEXEC::__std::memory_order_acq_rel,
        STDEXEC::__std::memory_order_relaxed));

      STDEXEC_ASSERT((prevState & starting_flag) == 0);

      if ((prevState & submit_complete_flag) != 0) {
        // start() has finished calling SubmitThreadpoolWork() and the work has
        // not yet started executing the work so it's safe for this method to now
        // try and cancel the work. While it's possible that the work callback
        // will start executing concurrently on a thread-pool thread, we are
        // guaranteed that it will see our write of the stop_requested_flag and
        // will promptly return without blocking.
        complete_with_done();
      } else {
        // Otherwise, as the start() method has not yet finished calling
        // SubmitThreadpoolWork() we can't safely call
        // WaitForThreadpoolWorkCallbacks(). In this case we are delegating
        // responsibility for calling complete_with_done() to start() method when
        // it eventually returns from SubmitThreadpoolWork().
      }
    }

    void complete_with_done() noexcept {
      const BOOL cancelPending = TRUE;
      ::WaitForThreadpoolWorkCallbacks(work_, cancelPending);

      // Destruct the stop-callback before calling set_stopped() as the call
      // to set_stopped() will invalidate the stop-token and we need to
      // make sure that
      stopCallback_.__destroy();

      // Now that the work has been successfully cancelled we can
      // call the receiver's set_stopped().
      set_stopped_impl();
    }

    virtual void set_stopped_impl() noexcept = 0;
    virtual void set_value_impl() noexcept = 0;

    struct stop_requested_callback {
      type &op_;

      void operator()() noexcept {
        op_.request_stop();
      }
    };

    /////////////////
    // Flags to use for state_ member

    // Initial state. start() not yet called.
    static constexpr std::uint32_t not_started = 0;

    // Flag set once start() has finished calling ThreadpoolSubmitWork()
    static constexpr std::uint32_t submit_complete_flag = 1;

    // Flag set by request_stop()
    static constexpr std::uint32_t stop_requested_flag = 2;

    // Flag set by cancellable_work_callback() when it starts executing.
    // This is before deregistering the stop-callback.
    static constexpr std::uint32_t starting_flag = 4;

    // Flag set by cancellable_work_callback() after having deregistered
    // the stop-callback, just before it calls the receiver.
    static constexpr std::uint32_t running_flag = 8;

    PTP_WORK work_;
    TP_CALLBACK_ENVIRON environ_;
    STDEXEC::__std::atomic<std::uint32_t> *state_;
    STDEXEC::__manual_lifetime<STDEXEC::stop_callback_for_t<StopToken, stop_requested_callback>>
      stopCallback_;
  };

  template <class Rcvr>
  class windows_thread_pool::_cancellable_schedule_op<Rcvr>::type final
    : public windows_thread_pool::cancellable_schedule_op_base<
        STDEXEC::stop_token_of_t<STDEXEC::env_of_t<Rcvr>>
      > {
    using base = windows_thread_pool::cancellable_schedule_op_base<
      STDEXEC::stop_token_of_t<STDEXEC::env_of_t<Rcvr>>
    >;

   public:
    explicit type(windows_thread_pool &pool, Rcvr rcvr)
      : base(pool, STDEXEC::get_stop_token(rcvr).stop_possible())
      , rcvr_(std::move(rcvr)) {
    }

    void start() noexcept {
      this->start_impl(STDEXEC::get_stop_token(STDEXEC::get_env(rcvr_)));
    }

   private:
    void set_value_impl() noexcept override {
      STDEXEC::set_value(std::move(rcvr_));
    }

    void set_stopped_impl() noexcept override {
      if constexpr (!STDEXEC::unstoppable_token<STDEXEC::stop_token_of_t<STDEXEC::env_of_t<Rcvr>>>) {
        STDEXEC::set_stopped(std::move(rcvr_));
      } else {
        STDEXEC_ASSERT(false);
      }
    }

    Rcvr rcvr_;
  };

  ////////////////////////////////////////////////////
  // schedule senders' attributes
  struct windows_thread_pool::attrs {
    [[nodiscard]]
    auto
      query(STDEXEC::get_completion_scheduler_t<STDEXEC::set_value_t>) const noexcept -> scheduler;

    windows_thread_pool *pool_;
  };

  ////////////////////////////////////////////////////
  // schedule() sender

  class windows_thread_pool::schedule_sender {
   public:
    using sender_concept = STDEXEC::sender_t;
    using completion_signatures = STDEXEC::completion_signatures<
      STDEXEC::set_value_t(),
      STDEXEC::set_error_t(std::exception_ptr),
      STDEXEC::set_stopped_t()
    >;

    template <class Rcvr> //
      requires STDEXEC::receiver_of<Rcvr, completion_signatures>
            && STDEXEC::unstoppable_token<STDEXEC::stop_token_of_t<STDEXEC::env_of_t<Rcvr>>>
    auto connect(Rcvr rcvr) const -> schedule_op<Rcvr> {
      return schedule_op<Rcvr>{*pool_, static_cast<Rcvr &&>(rcvr)};
    }

    template <class Rcvr> //
      requires STDEXEC::receiver_of<Rcvr, completion_signatures>
            && (!STDEXEC::unstoppable_token<STDEXEC::stop_token_of_t<STDEXEC::env_of_t<Rcvr>>>)
    auto connect(Rcvr rcvr) const -> cancellable_schedule_op<Rcvr> {
      return cancellable_schedule_op<Rcvr>{*pool_, static_cast<Rcvr &&>(rcvr)};
    }

    [[nodiscard]]
    auto get_env() const noexcept -> attrs {
      return attrs{pool_};
    }

   private:
    friend scheduler;

    explicit schedule_sender(windows_thread_pool &pool) noexcept
      : pool_(&pool) {
    }

    windows_thread_pool *pool_;
  };

  /////////////////////////////////
  // time_schedule_op

  template <class StopToken>
  class windows_thread_pool::_time_schedule_op<StopToken>::type {
   protected:
    explicit type(windows_thread_pool &pool, bool isStopPossible) {
      ::InitializeThreadpoolEnvironment(&environ_);
      ::SetThreadpoolCallbackPool(&environ_, pool.threadPool_);

      // Give the optimiser a hand for cases where the parameter
      // can never be true.
      if constexpr (STDEXEC::unstoppable_token<StopToken>) {
        isStopPossible = false;
      }

      timer_ = ::CreateThreadpoolTimer(
        isStopPossible ? &stoppable_timer_callback : &timer_callback,
        static_cast<void *>(this),
        &environ_);
      if (timer_ == nullptr) {
        DWORD errorCode = ::GetLastError();
        ::DestroyThreadpoolEnvironment(&environ_);
        throw std::system_error{
          static_cast<int>(errorCode), std::system_category(), "CreateThreadpoolTimer()"};
      }

      if (isStopPossible) {
        state_ = new (std::nothrow) STDEXEC::__std::atomic<std::uint32_t>{not_started};
        if (state_ == nullptr) {
          ::CloseThreadpoolTimer(timer_);
          ::DestroyThreadpoolEnvironment(&environ_);
          throw std::bad_alloc{};
        }
      }
    }

   public:
    using operation_state_concept = STDEXEC::operation_state_t;

    ~type() {
      ::CloseThreadpoolTimer(timer_);
      ::DestroyThreadpoolEnvironment(&environ_);
      delete state_;
    }

   protected:
    void start_impl(const StopToken &stopToken, FILETIME dueTime) noexcept {
      auto startTimer = [&]() noexcept {
        const DWORD periodInMs = 0;   // Single-shot
        const DWORD maxDelayInMs = 0; // Max delay to allow timer coalescing
        ::SetThreadpoolTimer(timer_, &dueTime, periodInMs, maxDelayInMs);
      };

      if constexpr (!STDEXEC::unstoppable_token<StopToken>) {
        auto *const state = state_;
        if (state != nullptr) {
          // Short-circuit extra work submitting the
          // timer if stop has already been requested.
          //
          // TODO: this means done can be delivered on "the wrong thread"
          if (stopToken.stop_requested()) {
            set_stopped_impl();
            return;
          }

          stopCallback_.__construct(stopToken, stop_requested_callback{*this});

          startTimer();

          const auto prevState =
            state->fetch_add(submit_complete_flag, STDEXEC::__std::memory_order_acq_rel);
          if ((prevState & stop_requested_flag) != 0) {
            complete_with_done();
          } else if ((prevState & running_flag) != 0) {
            delete state;
          }

          return;
        }
      }

      startTimer();
    }

   private:
    virtual void set_value_impl() noexcept = 0;
    virtual void set_stopped_impl() noexcept = 0;

    static void CALLBACK timer_callback(
      [[maybe_unused]] PTP_CALLBACK_INSTANCE instance,
      void *timerContext,
      [[maybe_unused]] PTP_TIMER timer) noexcept {
      type &op = *static_cast<type *>(timerContext);
      op.set_value_impl();
    }

    static void CALLBACK stoppable_timer_callback(
      [[maybe_unused]] PTP_CALLBACK_INSTANCE instance,
      void *timerContext,
      [[maybe_unused]] PTP_TIMER timer) noexcept {
      type &op = *static_cast<type *>(timerContext);

      auto prevState = op.state_->fetch_add(starting_flag, STDEXEC::__std::memory_order_acq_rel);
      if ((prevState & stop_requested_flag) != 0) {
        return;
      }

      op.stopCallback_.__destroy();

      prevState = op.state_->fetch_add(running_flag, STDEXEC::__std::memory_order_acq_rel);
      if (prevState == starting_flag) {
        op.state_ = nullptr;
      }

      op.set_value_impl();
    }

    void request_stop() noexcept {
      auto prevState = state_->load(STDEXEC::__std::memory_order_relaxed);
      do {
        STDEXEC_ASSERT((prevState & running_flag) == 0);
        if ((prevState & starting_flag) != 0) {
          return;
        }
      } while (!state_->compare_exchange_weak(
        prevState,
        prevState | stop_requested_flag,
        STDEXEC::__std::memory_order_acq_rel,
        STDEXEC::__std::memory_order_relaxed));

      STDEXEC_ASSERT((prevState & starting_flag) == 0);

      if ((prevState & submit_complete_flag) != 0) {
        complete_with_done();
      }
    }

    void complete_with_done() noexcept {
      const BOOL cancelPending = TRUE;
      ::WaitForThreadpoolTimerCallbacks(timer_, cancelPending);

      stopCallback_.__destroy();

      set_stopped_impl();
    }

    struct stop_requested_callback {
      type &op_;

      void operator()() noexcept {
        op_.request_stop();
      }
    };

    /////////////////
    // Flags to use for state_ member

    // Initial state. start() not yet called.
    static constexpr std::uint32_t not_started = 0;

    // Flag set once start() has finished calling ThreadpoolSubmitWork()
    static constexpr std::uint32_t submit_complete_flag = 1;

    // Flag set by request_stop()
    static constexpr std::uint32_t stop_requested_flag = 2;

    // Flag set by cancellable_work_callback() when it starts executing.
    // This is before deregistering the stop-callback.
    static constexpr std::uint32_t starting_flag = 4;

    // Flag set by cancellable_work_callback() after having deregistered
    // the stop-callback, just before it calls the receiver.
    static constexpr std::uint32_t running_flag = 8;

    PTP_TIMER timer_;
    TP_CALLBACK_ENVIRON environ_;
    STDEXEC::__std::atomic<std::uint32_t> *state_{nullptr};
    STDEXEC::__manual_lifetime<typename StopToken::template callback_type<stop_requested_callback>>
      stopCallback_;
  };

  /////////////////////////////////
  // schedule_at() operation

  template <class Rcvr>
  class windows_thread_pool::_schedule_at_op<Rcvr>::type final
    : public windows_thread_pool::time_schedule_op<
        STDEXEC::stop_token_of_t<STDEXEC::env_of_t<Rcvr>>
      > {
    using base =
      windows_thread_pool::time_schedule_op<STDEXEC::stop_token_of_t<STDEXEC::env_of_t<Rcvr>>>;

   public:
    explicit type(
      windows_thread_pool &pool,
      windows_thread_pool::clock_type::time_point dueTime,
      Rcvr rcvr)
      : base(pool, STDEXEC::get_stop_token(rcvr).stop_possible())
      , dueTime_(dueTime)
      , rcvr_(std::move(rcvr)) {
    }

    void start() noexcept {
      ULARGE_INTEGER ticks;
      ticks.QuadPart = dueTime_.get_ticks();

      FILETIME ft;
      ft.dwLowDateTime = ticks.LowPart;
      ft.dwHighDateTime = ticks.HighPart;

      this->start_impl(STDEXEC::get_stop_token(STDEXEC::get_env(rcvr_)), ft);
    }

   private:
    void set_value_impl() noexcept override {
      STDEXEC::set_value(std::move(rcvr_));
    }

    void set_stopped_impl() noexcept override {
      STDEXEC::set_stopped(std::move(rcvr_));
    }

    windows_thread_pool::clock_type::time_point dueTime_;
    Rcvr rcvr_;
  };

  class windows_thread_pool::schedule_at_sender {
   public:
    using sender_concept = STDEXEC::sender_t;
    using completion_signatures = STDEXEC::completion_signatures<
      STDEXEC::set_value_t(),
      STDEXEC::set_error_t(std::exception_ptr),
      STDEXEC::set_stopped_t()
    >;
    explicit schedule_at_sender(windows_thread_pool &pool, filetime_clock::time_point dueTime)
      : pool_(&pool)
      , dueTime_(dueTime) {
    }

    template <class Rcvr>
      requires STDEXEC::receiver_of<Rcvr, completion_signatures>
    auto connect(Rcvr rcvr) const -> schedule_at_op<Rcvr> {
      return schedule_at_op<Rcvr>{*pool_, dueTime_, static_cast<Rcvr &&>(rcvr)};
    }

    [[nodiscard]]
    auto get_env() const noexcept -> attrs {
      return attrs{pool_};
    }

   private:
    windows_thread_pool *pool_;
    filetime_clock::time_point dueTime_;
  };

  //////////////////////////////////
  // schedule_after()

  template <class Duration, class Rcvr>
  class windows_thread_pool::_schedule_after_op<Duration, Rcvr>::type final
    : public windows_thread_pool::time_schedule_op<
        STDEXEC::stop_token_of_t<STDEXEC::env_of_t<Rcvr>>
      > {
    using base =
      windows_thread_pool::time_schedule_op<STDEXEC::stop_token_of_t<STDEXEC::env_of_t<Rcvr>>>;

   public:
    explicit type(windows_thread_pool &pool, Duration duration, Rcvr rcvr)
      : base(pool, STDEXEC::get_stop_token(rcvr).stop_possible())
      , duration_(duration)
      , rcvr_(std::move(rcvr)) {
    }

    void start() noexcept {
      auto dueTime = filetime_clock::now() + duration_;

      ULARGE_INTEGER ticks;
      ticks.QuadPart = dueTime.get_ticks();

      FILETIME ft;
      ft.dwLowDateTime = ticks.LowPart;
      ft.dwHighDateTime = ticks.HighPart;

      this->start_impl(STDEXEC::get_stop_token(STDEXEC::get_env(rcvr_)), ft);
    }

   private:
    void set_value_impl() noexcept override {
      STDEXEC::set_value(std::move(rcvr_));
    }

    void set_stopped_impl() noexcept override {
      STDEXEC::set_stopped(std::move(rcvr_));
    }

    Duration duration_;
    Rcvr rcvr_;
  };

  template <class Duration>
  class windows_thread_pool::_schedule_after<Duration>::sender {
   public:
    using sender_concept = STDEXEC::sender_t;
    using completion_signatures = STDEXEC::completion_signatures<
      STDEXEC::set_value_t(),
      STDEXEC::set_error_t(std::exception_ptr),
      STDEXEC::set_stopped_t()
    >;

    explicit sender(windows_thread_pool &pool, Duration duration)
      : pool_(&pool)
      , duration_(duration) {
    }

    template <class Rcvr>
      requires STDEXEC::receiver_of<Rcvr, completion_signatures>
    auto connect(Rcvr rcvr) const -> schedule_after_op<Duration, Rcvr> {
      return schedule_after_op<Duration, Rcvr>{*pool_, duration_, static_cast<Rcvr &&>(rcvr)};
    }

    [[nodiscard]]
    auto get_env() const noexcept -> attrs {
      return attrs{pool_};
    }

   private:
    windows_thread_pool *pool_;
    Duration duration_;
  };

  /////////////////////////////////
  // scheduler

  class windows_thread_pool::scheduler {
   public:
    using time_point = filetime_clock::time_point;

    [[nodiscard]]
    auto schedule() const noexcept -> schedule_sender {
      return schedule_sender{*pool_};
    }

    [[nodiscard]]
    static auto now() noexcept -> time_point {
      return filetime_clock::now();
    }

    [[nodiscard]]
    auto schedule_at(time_point tp) const noexcept -> schedule_at_sender {
      return schedule_at_sender{*pool_, tp};
    }

    template <class Duration>
    [[nodiscard]]
    auto schedule_after(Duration d) noexcept -> schedule_after_sender<Duration> {
      return schedule_after_sender<Duration>{*pool_, std::move(d)};
    }

    friend auto operator==(scheduler a, scheduler b) noexcept -> bool {
      return a.pool_ == b.pool_;
    }

    friend auto operator!=(scheduler a, scheduler b) noexcept -> bool {
      return a.pool_ != b.pool_;
    }

   private:
    friend windows_thread_pool;

    explicit scheduler(windows_thread_pool &pool) noexcept
      : pool_(&pool) {
    }

    windows_thread_pool *pool_;
  };

  inline auto windows_thread_pool::attrs::query(
    STDEXEC::get_completion_scheduler_t<STDEXEC::set_value_t>) const noexcept -> scheduler {
    return scheduler{*pool_};
  }

  /////////////////////////
  // scheduler methods

  inline auto windows_thread_pool::get_scheduler() noexcept -> windows_thread_pool::scheduler {
    return scheduler{*this};
  }

  /////////////////////////

  inline windows_thread_pool::windows_thread_pool() noexcept
    : threadPool_(nullptr) {
  }

  inline windows_thread_pool::windows_thread_pool(
    std::uint32_t minThreadCount,
    std::uint32_t maxThreadCount)
    : threadPool_(::CreateThreadpool(nullptr)) {
    if (threadPool_ == nullptr) {
      DWORD errorCode = ::GetLastError();
      throw std::system_error{
        static_cast<int>(errorCode), std::system_category(), "CreateThreadPool()"};
    }

    ::SetThreadpoolThreadMaximum(threadPool_, maxThreadCount);
    if (!::SetThreadpoolThreadMinimum(threadPool_, minThreadCount)) {
      DWORD errorCode = ::GetLastError();
      ::CloseThreadpool(threadPool_);
      throw std::system_error{
        static_cast<int>(errorCode), std::system_category(), "SetThreadpoolThreadMinimum()"};
    }
  }

  inline windows_thread_pool::~windows_thread_pool() {
    if (threadPool_ != nullptr) {
      ::CloseThreadpool(threadPool_);
    }
  }

  inline windows_thread_pool::schedule_op_base::~schedule_op_base() {
    ::CloseThreadpoolWork(work_);
    ::DestroyThreadpoolEnvironment(&environ_);
  }

  inline void windows_thread_pool::schedule_op_base::start() noexcept {
    ::SubmitThreadpoolWork(work_);
  }

  inline windows_thread_pool::schedule_op_base::schedule_op_base(
    windows_thread_pool &pool,
    PTP_WORK_CALLBACK workCallback) {
    ::InitializeThreadpoolEnvironment(&environ_);
    ::SetThreadpoolCallbackPool(&environ_, pool.threadPool_);
    work_ = ::CreateThreadpoolWork(workCallback, this, &environ_);
    if (work_ == nullptr) {
      // TODO: Should we just cache the error and deliver via set_error(rcvr_,
      // std::error_code{}) upon start()?
      DWORD errorCode = ::GetLastError();
      ::DestroyThreadpoolEnvironment(&environ_);
      throw std::system_error{
        static_cast<int>(errorCode), std::system_category(), "CreateThreadpoolWork()"};
    }
  }
} // namespace exec::__win32

namespace exec {
  using __win32::windows_thread_pool;

  static_assert(
    STDEXEC::scheduler<decltype(windows_thread_pool{}.get_scheduler())>,
    "windows_thread_pool::scheduler must model STDEXEC::scheduler");
} // namespace exec

#endif
