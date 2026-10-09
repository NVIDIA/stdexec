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
#  include "../../stdexec/__detail/__bulk.hpp"
#  include "../../stdexec/__detail/__connect.hpp"
#  include "../../stdexec/__detail/__domain.hpp"
#  include "../../stdexec/__detail/__env.hpp"
#  include "../../stdexec/__detail/__get_completion_signatures.hpp"
#  include "../../stdexec/__detail/__manual_lifetime.hpp"
#  include "../../stdexec/__detail/__operation_states.hpp"
#  include "../../stdexec/__detail/__receivers.hpp"
#  include "../../stdexec/__detail/__schedulers.hpp"
#  include "../../stdexec/__detail/__stop_token.hpp"
#  include "../../stdexec/__detail/__transform_sender.hpp"
#  include "../../stdexec/__detail/__tuple.hpp"
#  include "../../stdexec/__detail/__variant.hpp"
#  include "../completion_signatures.hpp"
#  include "../sender_for.hpp"
#  include "../timed_scheduler.hpp"  // IWYU pragma: keep
#  include "./filetime_clock.hpp"

#  include <algorithm>
#  include <cstdint>
#  include <exception>
#  include <functional>
#  include <system_error>
#  include <thread>
#  include <type_traits>
#  include <utility>

namespace experimental::execution::__win32
{
  class windows_thread_pool
  {
    struct attrs;
    class schedule_sender;
    class schedule_op_base;

    template <class Rcvr>
    struct _schedule_op
    {
      class type;
    };
    template <class Rcvr>
    using schedule_op = _schedule_op<Rcvr>::type;

    template <class StopToken>
    struct _cancellable_schedule_op_base
    {
      class type;
      using __t = type;
    };
    template <class StopToken>
    using cancellable_schedule_op_base = _cancellable_schedule_op_base<StopToken>::type;

    template <class Rcvr>
    struct _cancellable_schedule_op
    {
      class type;
      using __t = type;
    };
    template <class Rcvr>
    using cancellable_schedule_op = _cancellable_schedule_op<Rcvr>::type;

    template <class StopToken>
    struct _time_schedule_op
    {
      class type;
      using __t = type;
    };
    template <class StopToken>
    using time_schedule_op = _time_schedule_op<StopToken>::type;

    template <class Rcvr>
    struct _schedule_at_op
    {
      class type;
      using __t = type;
    };
    template <class Rcvr>
    using schedule_at_op = _schedule_at_op<Rcvr>::type;

    template <class Duration, class Rcvr>
    struct _schedule_after_op
    {
      class type;
      using __t = type;
    };
    template <class Duration, class Rcvr>
    using schedule_after_op = _schedule_after_op<Duration, Rcvr>::type;

    class schedule_at_sender;

    template <class Duration>
    struct _schedule_after
    {
      class sender;
      using __t = sender;
    };
    template <class Duration>
    using schedule_after_sender = _schedule_after<Duration>::sender;

    using clock_type = filetime_clock;

    struct transform_bulk;

    template <bool Parallelize, std::integral Shape, class Fun, class Sender>
    class bulk_sender;

    template <bool Parallelize, class Shape, class Fun, class CvSender, class Rcvr>
    struct bulk_shared_state;

    template <bool Parallelize, class Shape, class Fun, class CvSender, class Rcvr>
    struct bulk_receiver;

    template <bool Parallelize, std::integral Shape, class Fun, class CvSender, class Rcvr>
    struct bulk_op;

   public:
    class scheduler;
    struct domain;

    // Initialise to use the process' default thread-pool.
    windows_thread_pool() noexcept;

    // Construct to an independend thread-pool with a dynamic number of
    // threads that varies between a min and a max number of threads.
    explicit windows_thread_pool(std::uint32_t minThreadCount, std::uint32_t maxThreadCount);

    ~windows_thread_pool();

    auto get_scheduler() noexcept -> scheduler;

   private:
    // The maximum number of execution agents a bulk operation is split into.
    [[nodiscard]]
    auto available_parallelism() const noexcept -> std::uint32_t
    {
      return availableParallelism_;
    }

    PTP_POOL      threadPool_;
    std::uint32_t availableParallelism_;
  };

  //////////////////////////////////////////////////////////////////////////////
  // Non-cancellable schedule() operation

  class windows_thread_pool::schedule_op_base
  {
   public:
    schedule_op_base(schedule_op_base &&)                     = delete;
    auto operator=(schedule_op_base &&) -> schedule_op_base & = delete;

    ~schedule_op_base();

    void start() noexcept;

   protected:
    schedule_op_base(windows_thread_pool &pool, PTP_WORK_CALLBACK workCallback);

   private:
    TP_CALLBACK_ENVIRON environ_;
    PTP_WORK            work_;
  };

  template <class Rcvr>
  class windows_thread_pool::_schedule_op<Rcvr>::type final
    : public windows_thread_pool::schedule_op_base
  {
   public:
    explicit type(windows_thread_pool &pool, Rcvr rcvr)
      : schedule_op_base(pool, &work_callback)
      , rcvr_(std::move(rcvr))
    {}

   private:
    static void CALLBACK work_callback(PTP_CALLBACK_INSTANCE, void *workContext, PTP_WORK) noexcept
    {
      auto &op = *static_cast<type *>(workContext);
      STDEXEC::set_value(std::move(op.rcvr_));
    }

    Rcvr rcvr_;
  };

  //////////////////////////////////////////////////////////////////////////////
  // Cancellable schedule() operation

  template <class StopToken>
  class windows_thread_pool::_cancellable_schedule_op_base<StopToken>::type
  {
   public:
    using operation_state_concept     = STDEXEC::operation_state_tag;
    type(type &&)                     = delete;
    auto operator=(type &&) -> type & = delete;

    ~type()
    {
      ::CloseThreadpoolWork(work_);
      ::DestroyThreadpoolEnvironment(&environ_);
      delete state_;
    }

   protected:
    explicit type(windows_thread_pool &pool, bool isStopPossible)
    {
      ::InitializeThreadpoolEnvironment(&environ_);
      ::SetThreadpoolCallbackPool(&environ_, pool.threadPool_);

      work_ = ::CreateThreadpoolWork(isStopPossible ? &stoppable_work_callback
                                                    : &unstoppable_work_callback,
                                     static_cast<void *>(this),
                                     &environ_);
      if (work_ == nullptr)
      {
        DWORD errorCode = ::GetLastError();
        ::DestroyThreadpoolEnvironment(&environ_);
        throw std::system_error{static_cast<int>(errorCode),
                                std::system_category(),
                                "CreateThreadpoolWork()"};
      }

      if (isStopPossible)
      {
        state_ = new (std::nothrow) STDEXEC::__std::atomic<std::uint32_t>(not_started);
        if (state_ == nullptr)
        {
          ::CloseThreadpoolWork(work_);
          ::DestroyThreadpoolEnvironment(&environ_);
          throw std::bad_alloc{};
        }
      }
      else
      {
        state_ = nullptr;
      }
    }

    void start_impl(StopToken const &stopToken) & noexcept
    {
      if (state_ != nullptr)
      {
        // Short-circuit all of this if stopToken.stop_requested() is already
        // true.
        //
        // TODO: this means done can be delivered on "the wrong thread"
        if (stopToken.stop_requested())
        {
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
        auto const prevState = state->fetch_add(submit_complete_flag,
                                                STDEXEC::__std::memory_order_acq_rel);
        if ((prevState & stop_requested_flag) != 0)
        {
          // stop was requested before the call to SubmitThreadpoolWork()
          // returned and before the work started executing. It was not
          // safe for the request_stop() method to cancel the work before
          // it had finished being submitted so it has delegated responsibility
          // for cancelling the just-submitted work to us to do once we
          // finished submitting the work.
          complete_with_done();
        }
        else if ((prevState & running_flag) != 0)
        {
          // Otherwise, it's possible that the work item may have started
          // running on another thread already, prior to us returning.
          // If this is the case then, to avoid leaving us with a
          // dangling reference to the 'state' when the operation-state
          // is destroyed, it will detach the 'state' from the operation-state
          // and delegate the delete of the 'state' to us.
          delete state;
        }
      }
      else
      {
        // A stop-request is not possible so skip the extra
        // synchronisation needed to support it.
        ::SubmitThreadpoolWork(work_);
      }
    }

   private:
    static void CALLBACK unstoppable_work_callback(PTP_CALLBACK_INSTANCE,
                                                   void *workContext,
                                                   PTP_WORK) noexcept
    {
      auto &op = *static_cast<type *>(workContext);
      op.set_value_impl();
    }

    static void CALLBACK stoppable_work_callback(PTP_CALLBACK_INSTANCE,
                                                 void *workContext,
                                                 PTP_WORK) noexcept
    {
      auto &op = *static_cast<type *>(workContext);

      // Signal that the work callback has started executing.
      auto prevState = op.state_->fetch_add(starting_flag, STDEXEC::__std::memory_order_acq_rel);
      if ((prevState & stop_requested_flag) != 0)
      {
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
      if (prevState == starting_flag)
      {
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

    void request_stop() noexcept
    {
      auto prevState = state_->load(STDEXEC::__std::memory_order_relaxed);
      do
      {
        STDEXEC_ASSERT((prevState & running_flag) == 0);
        if ((prevState & starting_flag) != 0)
        {
          // Work callback won the race and will be waiting for
          // us to return so it can deregister the stop-callback.
          // Return immediately so we don't deadlock.
          return;
        }
      }
      while (!state_->compare_exchange_weak(prevState,
                                            prevState | stop_requested_flag,
                                            STDEXEC::__std::memory_order_acq_rel,
                                            STDEXEC::__std::memory_order_relaxed));

      STDEXEC_ASSERT((prevState & starting_flag) == 0);

      if ((prevState & submit_complete_flag) != 0)
      {
        // start() has finished calling SubmitThreadpoolWork() and the work has
        // not yet started executing the work so it's safe for this method to
        // now try and cancel the work. While it's possible that the work
        // callback will start executing concurrently on a thread-pool thread,
        // we are guaranteed that it will see our write of the
        // stop_requested_flag and will promptly return without blocking.
        complete_with_done();
      }
      else
      {
        // Otherwise, as the start() method has not yet finished calling
        // SubmitThreadpoolWork() we can't safely call
        // WaitForThreadpoolWorkCallbacks(). In this case we are delegating
        // responsibility for calling complete_with_done() to start() method
        // when it eventually returns from SubmitThreadpoolWork().
      }
    }

    void complete_with_done() noexcept
    {
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
    virtual void set_value_impl() noexcept   = 0;

    struct stop_requested_callback
    {
      type &op_;

      void operator()() noexcept
      {
        op_.request_stop();
      }
    };

    ////////////////////////////////////////////////////////////////////////////
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

    PTP_WORK                               work_;
    TP_CALLBACK_ENVIRON                    environ_;
    STDEXEC::__std::atomic<std::uint32_t> *state_;
    STDEXEC::__manual_lifetime<STDEXEC::stop_callback_for_t<StopToken, stop_requested_callback>>
      stopCallback_;
  };

  template <class Rcvr>
  class windows_thread_pool::_cancellable_schedule_op<Rcvr>::type final
    : public windows_thread_pool::cancellable_schedule_op_base<
        STDEXEC::stop_token_of_t<STDEXEC::env_of_t<Rcvr>>>
  {
    using base = windows_thread_pool::cancellable_schedule_op_base<
      STDEXEC::stop_token_of_t<STDEXEC::env_of_t<Rcvr>>>;

   public:
    explicit type(windows_thread_pool &pool, Rcvr rcvr)
      : base(pool, STDEXEC::get_stop_token(rcvr).stop_possible())
      , rcvr_(std::move(rcvr))
    {}

    void start() noexcept
    {
      this->start_impl(STDEXEC::get_stop_token(STDEXEC::get_env(rcvr_)));
    }

   private:
    void set_value_impl() noexcept override
    {
      STDEXEC::set_value(std::move(rcvr_));
    }

    void set_stopped_impl() noexcept override
    {
      if constexpr (!STDEXEC::unstoppable_token<STDEXEC::stop_token_of_t<STDEXEC::env_of_t<Rcvr>>>)
      {
        STDEXEC::set_stopped(std::move(rcvr_));
      }
      else
      {
        STDEXEC_ASSERT(false);
      }
    }

    Rcvr rcvr_;
  };

  //////////////////////////////////////////////////////////////////////////////
  // schedule senders' attributes
  struct windows_thread_pool::attrs
  {
    [[nodiscard]]
    auto
      query(STDEXEC::get_completion_scheduler_t<STDEXEC::set_value_t>) const noexcept -> scheduler;

    windows_thread_pool *pool_;
  };

  //////////////////////////////////////////////////////////////////////////////
  // schedule() sender

  class windows_thread_pool::schedule_sender
  {
   public:
    using sender_concept = STDEXEC::sender_tag;
    using completion_signatures =
      STDEXEC::completion_signatures<STDEXEC::set_value_t(),
                                     STDEXEC::set_error_t(std::exception_ptr),
                                     STDEXEC::set_stopped_t()>;

    template <class Rcvr>  //
      requires STDEXEC::receiver_of<Rcvr, completion_signatures>
            && STDEXEC::unstoppable_token<STDEXEC::stop_token_of_t<STDEXEC::env_of_t<Rcvr>>>
    auto connect(Rcvr rcvr) const -> schedule_op<Rcvr>
    {
      return schedule_op<Rcvr>{*pool_, static_cast<Rcvr &&>(rcvr)};
    }

    template <class Rcvr>  //
      requires STDEXEC::receiver_of<Rcvr, completion_signatures>
            && (!STDEXEC::unstoppable_token<STDEXEC::stop_token_of_t<STDEXEC::env_of_t<Rcvr>>>)
    auto connect(Rcvr rcvr) const -> cancellable_schedule_op<Rcvr>
    {
      return cancellable_schedule_op<Rcvr>{*pool_, static_cast<Rcvr &&>(rcvr)};
    }

    [[nodiscard]]
    auto get_env() const noexcept -> attrs
    {
      return attrs{pool_};
    }

   private:
    friend scheduler;

    explicit schedule_sender(windows_thread_pool &pool) noexcept
      : pool_(&pool)
    {}

    windows_thread_pool *pool_;
  };

  //////////////////////////////////////////////////////////////////////////////
  // time_schedule_op

  template <class StopToken>
  class windows_thread_pool::_time_schedule_op<StopToken>::type
  {
   protected:
    explicit type(windows_thread_pool &pool, bool isStopPossible)
    {
      ::InitializeThreadpoolEnvironment(&environ_);
      ::SetThreadpoolCallbackPool(&environ_, pool.threadPool_);

      // Give the optimiser a hand for cases where the parameter
      // can never be true.
      if constexpr (STDEXEC::unstoppable_token<StopToken>)
      {
        isStopPossible = false;
      }

      timer_ = ::CreateThreadpoolTimer(isStopPossible ? &stoppable_timer_callback : &timer_callback,
                                       static_cast<void *>(this),
                                       &environ_);
      if (timer_ == nullptr)
      {
        DWORD errorCode = ::GetLastError();
        ::DestroyThreadpoolEnvironment(&environ_);
        throw std::system_error{static_cast<int>(errorCode),
                                std::system_category(),
                                "CreateThreadpoolTimer()"};
      }

      if (isStopPossible)
      {
        state_ = new (std::nothrow) STDEXEC::__std::atomic<std::uint32_t>{not_started};
        if (state_ == nullptr)
        {
          ::CloseThreadpoolTimer(timer_);
          ::DestroyThreadpoolEnvironment(&environ_);
          throw std::bad_alloc{};
        }
      }
    }

   public:
    using operation_state_concept = STDEXEC::operation_state_tag;

    ~type()
    {
      ::CloseThreadpoolTimer(timer_);
      ::DestroyThreadpoolEnvironment(&environ_);
      delete state_;
    }

   protected:
    void start_impl(StopToken const &stopToken, FILETIME dueTime) noexcept
    {
      auto startTimer = [&]() noexcept
      {
        const DWORD periodInMs   = 0;  // Single-shot
        const DWORD maxDelayInMs = 0;  // Max delay to allow timer coalescing
        ::SetThreadpoolTimer(timer_, &dueTime, periodInMs, maxDelayInMs);
      };

      if constexpr (!STDEXEC::unstoppable_token<StopToken>)
      {
        auto *const state = state_;
        if (state != nullptr)
        {
          // Short-circuit extra work submitting the
          // timer if stop has already been requested.
          //
          // TODO: this means done can be delivered on "the wrong thread"
          if (stopToken.stop_requested())
          {
            set_stopped_impl();
            return;
          }

          stopCallback_.__construct(stopToken, stop_requested_callback{*this});

          startTimer();

          auto const prevState = state->fetch_add(submit_complete_flag,
                                                  STDEXEC::__std::memory_order_acq_rel);
          if ((prevState & stop_requested_flag) != 0)
          {
            complete_with_done();
          }
          else if ((prevState & running_flag) != 0)
          {
            delete state;
          }

          return;
        }
      }

      startTimer();
    }

   private:
    virtual void set_value_impl() noexcept   = 0;
    virtual void set_stopped_impl() noexcept = 0;

    static void CALLBACK timer_callback([[maybe_unused]] PTP_CALLBACK_INSTANCE instance,
                                        void                                  *timerContext,
                                        [[maybe_unused]] PTP_TIMER             timer) noexcept
    {
      type &op = *static_cast<type *>(timerContext);
      op.set_value_impl();
    }

    static void CALLBACK stoppable_timer_callback([[maybe_unused]] PTP_CALLBACK_INSTANCE instance,
                                                  void                      *timerContext,
                                                  [[maybe_unused]] PTP_TIMER timer) noexcept
    {
      type &op = *static_cast<type *>(timerContext);

      auto prevState = op.state_->fetch_add(starting_flag, STDEXEC::__std::memory_order_acq_rel);
      if ((prevState & stop_requested_flag) != 0)
      {
        return;
      }

      op.stopCallback_.__destroy();

      prevState = op.state_->fetch_add(running_flag, STDEXEC::__std::memory_order_acq_rel);
      if (prevState == starting_flag)
      {
        op.state_ = nullptr;
      }

      op.set_value_impl();
    }

    void request_stop() noexcept
    {
      auto prevState = state_->load(STDEXEC::__std::memory_order_relaxed);
      do
      {
        STDEXEC_ASSERT((prevState & running_flag) == 0);
        if ((prevState & starting_flag) != 0)
        {
          return;
        }
      }
      while (!state_->compare_exchange_weak(prevState,
                                            prevState | stop_requested_flag,
                                            STDEXEC::__std::memory_order_acq_rel,
                                            STDEXEC::__std::memory_order_relaxed));

      STDEXEC_ASSERT((prevState & starting_flag) == 0);

      if ((prevState & submit_complete_flag) != 0)
      {
        complete_with_done();
      }
    }

    void complete_with_done() noexcept
    {
      const BOOL cancelPending = TRUE;
      ::WaitForThreadpoolTimerCallbacks(timer_, cancelPending);

      stopCallback_.__destroy();

      set_stopped_impl();
    }

    struct stop_requested_callback
    {
      type &op_;

      void operator()() noexcept
      {
        op_.request_stop();
      }
    };

    ////////////////////////////////////////////////////////////////////////////
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

    PTP_TIMER                              timer_;
    TP_CALLBACK_ENVIRON                    environ_;
    STDEXEC::__std::atomic<std::uint32_t> *state_{nullptr};
    STDEXEC::__manual_lifetime<typename StopToken::template callback_type<stop_requested_callback>>
      stopCallback_;
  };

  //////////////////////////////////////////////////////////////////////////////
  // schedule_at() operation

  template <class Rcvr>
  class windows_thread_pool::_schedule_at_op<Rcvr>::type final
    : public windows_thread_pool::time_schedule_op<
        STDEXEC::stop_token_of_t<STDEXEC::env_of_t<Rcvr>>>
  {
    using base =
      windows_thread_pool::time_schedule_op<STDEXEC::stop_token_of_t<STDEXEC::env_of_t<Rcvr>>>;

   public:
    explicit type(windows_thread_pool                        &pool,
                  windows_thread_pool::clock_type::time_point dueTime,
                  Rcvr                                        rcvr)
      : base(pool, STDEXEC::get_stop_token(rcvr).stop_possible())
      , dueTime_(dueTime)
      , rcvr_(std::move(rcvr))
    {}

    void start() noexcept
    {
      ULARGE_INTEGER ticks;
      ticks.QuadPart = dueTime_.get_ticks();

      FILETIME ft;
      ft.dwLowDateTime  = ticks.LowPart;
      ft.dwHighDateTime = ticks.HighPart;

      this->start_impl(STDEXEC::get_stop_token(STDEXEC::get_env(rcvr_)), ft);
    }

   private:
    void set_value_impl() noexcept override
    {
      STDEXEC::set_value(std::move(rcvr_));
    }

    void set_stopped_impl() noexcept override
    {
      STDEXEC::set_stopped(std::move(rcvr_));
    }

    windows_thread_pool::clock_type::time_point dueTime_;
    Rcvr                                        rcvr_;
  };

  class windows_thread_pool::schedule_at_sender
  {
   public:
    using sender_concept = STDEXEC::sender_tag;
    using completion_signatures =
      STDEXEC::completion_signatures<STDEXEC::set_value_t(),
                                     STDEXEC::set_error_t(std::exception_ptr),
                                     STDEXEC::set_stopped_t()>;
    explicit schedule_at_sender(windows_thread_pool &pool, filetime_clock::time_point dueTime)
      : pool_(&pool)
      , dueTime_(dueTime)
    {}

    template <class Rcvr>
      requires STDEXEC::receiver_of<Rcvr, completion_signatures>
    auto connect(Rcvr rcvr) const -> schedule_at_op<Rcvr>
    {
      return schedule_at_op<Rcvr>{*pool_, dueTime_, static_cast<Rcvr &&>(rcvr)};
    }

    [[nodiscard]]
    auto get_env() const noexcept -> attrs
    {
      return attrs{pool_};
    }

   private:
    windows_thread_pool       *pool_;
    filetime_clock::time_point dueTime_;
  };

  //////////////////////////////////////////////////////////////////////////////
  // schedule_after()

  template <class Duration, class Rcvr>
  class windows_thread_pool::_schedule_after_op<Duration, Rcvr>::type final
    : public windows_thread_pool::time_schedule_op<
        STDEXEC::stop_token_of_t<STDEXEC::env_of_t<Rcvr>>>
  {
    using base =
      windows_thread_pool::time_schedule_op<STDEXEC::stop_token_of_t<STDEXEC::env_of_t<Rcvr>>>;

   public:
    explicit type(windows_thread_pool &pool, Duration duration, Rcvr rcvr)
      : base(pool, STDEXEC::get_stop_token(rcvr).stop_possible())
      , duration_(duration)
      , rcvr_(std::move(rcvr))
    {}

    void start() noexcept
    {
      auto dueTime = filetime_clock::now() + duration_;

      ULARGE_INTEGER ticks;
      ticks.QuadPart = dueTime.get_ticks();

      FILETIME ft;
      ft.dwLowDateTime  = ticks.LowPart;
      ft.dwHighDateTime = ticks.HighPart;

      this->start_impl(STDEXEC::get_stop_token(STDEXEC::get_env(rcvr_)), ft);
    }

   private:
    void set_value_impl() noexcept override
    {
      STDEXEC::set_value(std::move(rcvr_));
    }

    void set_stopped_impl() noexcept override
    {
      STDEXEC::set_stopped(std::move(rcvr_));
    }

    Duration duration_;
    Rcvr     rcvr_;
  };

  template <class Duration>
  class windows_thread_pool::_schedule_after<Duration>::sender
  {
   public:
    using sender_concept = STDEXEC::sender_tag;
    using completion_signatures =
      STDEXEC::completion_signatures<STDEXEC::set_value_t(),
                                     STDEXEC::set_error_t(std::exception_ptr),
                                     STDEXEC::set_stopped_t()>;

    explicit sender(windows_thread_pool &pool, Duration duration)
      : pool_(&pool)
      , duration_(duration)
    {}

    template <class Rcvr>
      requires STDEXEC::receiver_of<Rcvr, completion_signatures>
    auto connect(Rcvr rcvr) const -> schedule_after_op<Duration, Rcvr>
    {
      return schedule_after_op<Duration, Rcvr>{*pool_, duration_, static_cast<Rcvr &&>(rcvr)};
    }

    [[nodiscard]]
    auto get_env() const noexcept -> attrs
    {
      return attrs{pool_};
    }

   private:
    windows_thread_pool *pool_;
    Duration             duration_;
  };

  //////////////////////////////////////////////////////////////////////////////
  // bulk_chunked() / bulk_unchunked()

  struct CANNOT_DISPATCH_THE_BULK_ALGORITHM_TO_THE_WINDOWS_THREAD_POOL_SCHEDULER;
  struct BECAUSE_THERE_IS_NO_WINDOWS_THREAD_POOL_SCHEDULER_IN_THE_ENVIRONMENT;
  struct ADD_A_CONTINUES_ON_TRANSITION_TO_THE_WINDOWS_THREAD_POOL_SCHEDULER_BEFORE_THE_BULK_ALGORITHM;

  struct windows_thread_pool::transform_bulk
  {
    template <STDEXEC::__one_of<STDEXEC::bulk_chunked_t, STDEXEC::bulk_unchunked_t> Tag,
              class Data,
              class CvSender>
    auto operator()(Tag, Data &&data, CvSender &&sndr) const
    {
      auto [pol, shape, fun] = static_cast<Data &&>(data);
      using policy_t         = STDEXEC::__decay_t<decltype(pol.__get())>;
      constexpr bool parallelize =
        STDEXEC::__same_as<policy_t, STDEXEC::parallel_policy>
        || STDEXEC::__same_as<policy_t, STDEXEC::parallel_unsequenced_policy>;

      if constexpr (STDEXEC::__same_as<Tag, STDEXEC::bulk_unchunked_t>)
      {
        // Turn a bulk_unchunked into a bulk_chunked operation
        using fun_t = STDEXEC::__bulk::__as_bulk_chunked_fn<decltype(fun)>;
        using sender_t =
          bulk_sender<parallelize, decltype(shape), fun_t, STDEXEC::__decay_t<CvSender>>;
        return sender_t{*pool_, static_cast<CvSender &&>(sndr), shape, fun_t{std::move(fun)}};
      }
      else
      {
        using fun_t = decltype(fun);
        using sender_t =
          bulk_sender<parallelize, decltype(shape), fun_t, STDEXEC::__decay_t<CvSender>>;
        return sender_t{*pool_, static_cast<CvSender &&>(sndr), shape, std::move(fun)};
      }
    }

    windows_thread_pool *pool_;
  };

  struct windows_thread_pool::domain : STDEXEC::default_domain
  {
    // transform the generic bulk_chunked/bulk_unchunked senders into a parallel
    // windows_thread_pool bulk sender
    template <experimental::execution::sender_for Sender, class Env>
      requires STDEXEC::__one_of<STDEXEC::tag_of_t<Sender>,
                                 STDEXEC::bulk_chunked_t,
                                 STDEXEC::bulk_unchunked_t>
    auto transform_sender(STDEXEC::set_value_t, Sender &&sndr, Env const &env) const noexcept
    {
      if constexpr (STDEXEC::__completes_on<Sender, windows_thread_pool::scheduler, Env>)
      {
        auto sched = STDEXEC::get_completion_scheduler<STDEXEC::set_value_t>(STDEXEC::get_env(sndr),
                                                                             env);
        static_assert(std::is_same_v<decltype(sched), windows_thread_pool::scheduler>);
        return STDEXEC::__apply(transform_bulk{sched.pool_}, static_cast<Sender &&>(sndr));
      }
      else
      {
        return STDEXEC::__not_a_sender<
          STDEXEC::_WHAT_(CANNOT_DISPATCH_THE_BULK_ALGORITHM_TO_THE_WINDOWS_THREAD_POOL_SCHEDULER),
          STDEXEC::_WHY_(BECAUSE_THERE_IS_NO_WINDOWS_THREAD_POOL_SCHEDULER_IN_THE_ENVIRONMENT),
          STDEXEC::_WHERE_(STDEXEC::_IN_ALGORITHM_, STDEXEC::tag_of_t<Sender>),
          STDEXEC::_TO_FIX_THIS_ERROR_(
            ADD_A_CONTINUES_ON_TRANSITION_TO_THE_WINDOWS_THREAD_POOL_SCHEDULER_BEFORE_THE_BULK_ALGORITHM),
          STDEXEC::_WITH_PRETTY_SENDER_<Sender>,
          STDEXEC::_WITH_ENVIRONMENT_(Env)>();
      }
    }
  };

  template <bool Parallelize, std::integral Shape, class Fun, class Sender>
  class windows_thread_pool::bulk_sender
  {
    template <class Self, class Rcvr>
    using bulk_op_t = bulk_op<Parallelize, Shape, Fun, STDEXEC::__copy_cvref_t<Self, Sender>, Rcvr>;

   public:
    using sender_concept = STDEXEC::sender_tag;

    explicit bulk_sender(windows_thread_pool &pool, Sender sndr, Shape shape, Fun fun)
      noexcept(STDEXEC::__nothrow_move_constructible<Sender, Fun>)
      : pool_(&pool)
      , sndr_(static_cast<Sender &&>(sndr))
      , shape_(shape)
      , fun_(static_cast<Fun &&>(fun))
    {}

    template <STDEXEC::__decays_to<bulk_sender> Self, STDEXEC::receiver Rcvr>
    STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self &&self, Rcvr rcvr) -> bulk_op_t<Self, Rcvr>
    {
      return bulk_op_t<Self, Rcvr>{*self.pool_,
                                   self.shape_,
                                   self.fun_,
                                   static_cast<Self &&>(self).sndr_,
                                   static_cast<Rcvr &&>(rcvr)};
    }
    STDEXEC_EXPLICIT_THIS_END(connect)

    template <STDEXEC::__decays_to<bulk_sender> Self, class... Env>
    static consteval auto get_completion_signatures()
    {
      using namespace STDEXEC;
      return experimental::execution::transform_completion_signatures(
        STDEXEC::get_completion_signatures<__copy_cvref_t<Self, Sender>, Env...>(),
        []<class... Args>()
        {
          if constexpr (!__decay_copyable<Args...>)
          {
            return experimental::execution::throw_compile_time_error<
              _WHAT_(_PREDECESSOR_RESULTS_ARE_NOT_DECAY_COPYABLE_),
              _WHERE_(_IN_ALGORITHM_, bulk_chunked_t),
              _WITH_ARGUMENTS_(Args...),
              _WITH_PRETTY_SENDER_<__copy_cvref_t<Self, Sender>>,
              _WITH_ENVIRONMENT_(Env...)>();
          }
          else if constexpr (!__callable<Fun &, Shape, Shape, __decay_t<Args> &...>)
          {
            return experimental::execution::throw_compile_time_error<
              _WHAT_(_FUNCTION_IS_NOT_CALLABLE_WITH_THE_GIVEN_ARGUMENTS_),
              _WHERE_(_IN_ALGORITHM_, bulk_chunked_t),
              _WITH_FUNCTION_(Fun &),
              _WITH_ARGUMENTS_(Shape, Shape, __decay_t<Args> & ...)>();
          }
          else if constexpr (__nothrow_callable<Fun &, Shape, Shape, __decay_t<Args> &...>
                             && __nothrow_decay_copyable<Args...>)
          {
            return completion_signatures<set_value_t(__decay_t<Args>...)>();
          }
          else
          {
            return completion_signatures<set_value_t(__decay_t<Args>...),
                                         set_error_t(std::exception_ptr)>();
          }
        });
    }

    [[nodiscard]]
    auto get_env() const noexcept -> STDEXEC::env_of_t<Sender const &>
    {
      return STDEXEC::get_env(sndr_);
    }

   private:
    windows_thread_pool *pool_;
    Sender               sndr_;
    Shape                shape_;
    Fun                  fun_;
  };

  // The state shared by all the execution agents of a bulk operation.
  //
  // A single thread-pool work object is created for the whole operation and is
  // submitted once per execution agent. Each invocation of the work callback
  // claims the next agent index and runs the bulk function over that agent's
  // share of [0, shape). The agent that finishes last completes the operation.
  template <bool Parallelize, class Shape, class Fun, class CvSender, class Rcvr>
  struct windows_thread_pool::bulk_shared_state
  {
    using variant_t = STDEXEC::__value_types_of_t<CvSender,
                                                  STDEXEC::env_of_t<Rcvr>,
                                                  STDEXEC::__qq<STDEXEC::__decayed_tuple>,
                                                  STDEXEC::__qq<STDEXEC::__variant>>;

    explicit bulk_shared_state(windows_thread_pool &pool, Rcvr rcvr, Shape shape, Fun fun)
      : rcvr_(static_cast<Rcvr &&>(rcvr))
      , shape_(shape)
      , fun_(static_cast<Fun &&>(fun))
      , num_agents_(num_agents_required(pool, shape))
      , agent_with_exception_(num_agents_)
    {
      ::InitializeThreadpoolEnvironment(&environ_);
      ::SetThreadpoolCallbackPool(&environ_, pool.threadPool_);
      work_ = ::CreateThreadpoolWork(&work_callback, static_cast<void *>(this), &environ_);
      if (work_ == nullptr)
      {
        DWORD errorCode = ::GetLastError();

        ::DestroyThreadpoolEnvironment(&environ_);
        throw std::system_error{static_cast<int>(errorCode),
                                std::system_category(),
                                "CreateThreadpoolWork()"};
      }
    }

    bulk_shared_state(bulk_shared_state &&) = delete;

    ~bulk_shared_state()
    {
      // This may run inside the work callback of the last agent (when the
      // receiver destroys the operation from set_value()). This is fine: the
      // work object is then released once the outstanding callbacks return.
      ::CloseThreadpoolWork(work_);
      ::DestroyThreadpoolEnvironment(&environ_);
    }

    //! The number of agents required is the minimum of `shape` and the
    //! available parallelism of the pool, or 1 if the policy is sequenced.
    static auto
    num_agents_required(windows_thread_pool &pool, Shape shape) noexcept -> std::uint32_t
    {
      if (!(Shape{} < shape))
      {
        return 0;
      }

      if constexpr (Parallelize)
      {
        using ushape_t = std::make_unsigned_t<Shape>;
        return static_cast<std::uint32_t>(
          (std::min) (static_cast<std::uint64_t>(static_cast<ushape_t>(shape)),
                      static_cast<std::uint64_t>(pool.available_parallelism())));
      }
      else
      {
        return 1;
      }
    }

    //! Splits `[0, n)` into `size` chunks, distributing `n % size` evenly
    //! between the first ranks, and returns the chunk of `rank`.
    static auto
    even_share(Shape n, std::uint32_t rank, std::uint32_t size) noexcept -> std::pair<Shape, Shape>
    {
      using ushape_t           = std::make_unsigned_t<Shape>;
      auto const avg_per_agent = static_cast<ushape_t>(n) / size;
      auto const n_big_share   = avg_per_agent + 1;
      auto const big_shares    = static_cast<ushape_t>(n) % size;
      auto const is_big_share  = rank < big_shares;
      auto const begin         = is_big_share
                                 ? n_big_share * rank
                                 : n_big_share * big_shares + (rank - big_shares) * avg_per_agent;
      auto const end           = begin + (is_big_share ? n_big_share : avg_per_agent);
      return {static_cast<Shape>(begin), static_cast<Shape>(end)};
    }

    void submit() noexcept
    {
      for (std::uint32_t i = 0; i != num_agents_; ++i)
      {
        ::SubmitThreadpoolWork(work_);
      }
    }

    void complete() noexcept
    {
      STDEXEC::__visit(
        [this](auto &tupl) noexcept
        {
          STDEXEC::__apply([this](auto &...args) noexcept
                           { STDEXEC::set_value(static_cast<Rcvr &&>(rcvr_), std::move(args)...); },
                           tupl);
        },
        data_);
    }

    static void CALLBACK work_callback(PTP_CALLBACK_INSTANCE, void *workContext, PTP_WORK) noexcept
    {
      auto &self = *static_cast<bulk_shared_state *>(workContext);

      // Every submission of the work object invokes this callback exactly once,
      // so each invocation gets a distinct agent index.
      std::uint32_t const agent = self.next_agent_.fetch_add(1,
                                                             STDEXEC::__std::memory_order_relaxed);
      STDEXEC_ASSERT(agent < self.num_agents_);

      auto const [begin, end] = even_share(self.shape_, agent, self.num_agents_);
      auto const applicator   = std::bind_front(std::ref(self.fun_), begin, end);
      auto const computation  = std::bind_front(STDEXEC::__apply, applicator);

      if constexpr (noexcept(STDEXEC::__visit(computation, self.data_)))
      {
        STDEXEC::__visit(computation, self.data_);
        if (self.finished_agents_.fetch_add(1, STDEXEC::__std::memory_order_acq_rel) + 1
            == self.num_agents_)  // last agent?
        {
          self.complete();
        }
      }
      else
      {
        STDEXEC_TRY
        {
          STDEXEC::__visit(computation, self.data_);
        }
        STDEXEC_CATCH_ALL
        {
          std::uint32_t expected = self.num_agents_;
          if (self.agent_with_exception_.compare_exchange_strong(
                expected,
                agent,
                STDEXEC::__std::memory_order_relaxed,
                STDEXEC::__std::memory_order_relaxed))
          {
            self.exception_ = std::current_exception();
          }
        }

        if (self.finished_agents_.fetch_add(1, STDEXEC::__std::memory_order_acq_rel) + 1
            == self.num_agents_)  // last agent?
        {
          if (self.exception_)
          {
            STDEXEC::set_error(static_cast<Rcvr &&>(self.rcvr_), std::move(self.exception_));
          }
          else
          {
            self.complete();
          }
        }
      }
    }

    variant_t                             data_{STDEXEC::__no_init};
    Rcvr                                  rcvr_;
    Shape                                 shape_;
    Fun                                   fun_;
    std::uint32_t                         num_agents_;
    STDEXEC::__std::atomic<std::uint32_t> next_agent_{0};
    STDEXEC::__std::atomic<std::uint32_t> finished_agents_{0};
    STDEXEC::__std::atomic<std::uint32_t> agent_with_exception_;
    std::exception_ptr                    exception_;
    PTP_WORK                              work_;
    TP_CALLBACK_ENVIRON                   environ_;
  };

  template <bool Parallelize, class Shape, class Fun, class CvSender, class Rcvr>
  struct windows_thread_pool::bulk_receiver
  {
    using receiver_concept = STDEXEC::receiver_tag;

    template <class... As>
    void set_value(As &&...as) noexcept
    {
      STDEXEC_TRY
      {
        shared_state_.data_.template emplace<STDEXEC::__decayed_tuple<As...>>(
          static_cast<As &&>(as)...);
      }
      STDEXEC_CATCH_ALL
      {
        if constexpr (!STDEXEC::__nothrow_decay_copyable<As...>)
        {
          STDEXEC::set_error(static_cast<Rcvr &&>(shared_state_.rcvr_), std::current_exception());
          return;
        }
      }

      if (shared_state_.num_agents_ != 0)
      {
        shared_state_.submit();
      }
      else
      {
        shared_state_.complete();
      }
    }

    template <class Error>
    void set_error(Error &&error) noexcept
    {
      STDEXEC::set_error(static_cast<Rcvr &&>(shared_state_.rcvr_), static_cast<Error &&>(error));
    }

    void set_stopped() noexcept
    {
      STDEXEC::set_stopped(static_cast<Rcvr &&>(shared_state_.rcvr_));
    }

    [[nodiscard]]
    auto get_env() const noexcept -> STDEXEC::env_of_t<Rcvr>
    {
      return STDEXEC::get_env(shared_state_.rcvr_);
    }

    bulk_shared_state<Parallelize, Shape, Fun, CvSender, Rcvr> &shared_state_;
  };

  template <bool Parallelize, std::integral Shape, class Fun, class CvSender, class Rcvr>
  struct windows_thread_pool::bulk_op
  {
    using operation_state_concept = STDEXEC::operation_state_tag;
    using receiver_t              = bulk_receiver<Parallelize, Shape, Fun, CvSender, Rcvr>;
    using shared_state_t          = bulk_shared_state<Parallelize, Shape, Fun, CvSender, Rcvr>;
    using inner_op_t              = STDEXEC::connect_result_t<CvSender, receiver_t>;

    explicit bulk_op(windows_thread_pool &pool, Shape shape, Fun fun, CvSender &&sndr, Rcvr rcvr)
      : shared_state_(pool, static_cast<Rcvr &&>(rcvr), shape, static_cast<Fun &&>(fun))
      , inner_op_(STDEXEC::connect(static_cast<CvSender &&>(sndr), receiver_t{shared_state_}))
    {}

    void start() & noexcept
    {
      STDEXEC::start(inner_op_);
    }

    shared_state_t shared_state_;
    inner_op_t     inner_op_;
  };

  //////////////////////////////////////////////////////////////////////////////
  // scheduler

  class windows_thread_pool::scheduler
  {
   public:
    using scheduler_concept = STDEXEC::scheduler_tag;
    using time_point        = filetime_clock::time_point;

    [[nodiscard]]
    auto schedule() const noexcept -> schedule_sender
    {
      return schedule_sender{*pool_};
    }

    [[nodiscard]]
    static auto now() noexcept -> time_point
    {
      return filetime_clock::now();
    }

    [[nodiscard]]
    auto schedule_at(time_point tp) const noexcept -> schedule_at_sender
    {
      return schedule_at_sender{*pool_, tp};
    }

    template <class Duration>
    [[nodiscard]]
    auto schedule_after(Duration d) noexcept -> schedule_after_sender<Duration>
    {
      return schedule_after_sender<Duration>{*pool_, std::move(d)};
    }

    [[nodiscard]]
    static constexpr auto
    query(STDEXEC::get_forward_progress_guarantee_t) noexcept -> STDEXEC::forward_progress_guarantee
    {
      return STDEXEC::forward_progress_guarantee::parallel;
    }

    [[nodiscard]]
    auto
    query(STDEXEC::get_completion_scheduler_t<STDEXEC::set_value_t>) const noexcept -> scheduler
    {
      return *this;
    }

    [[nodiscard]]
    static constexpr auto
    query(STDEXEC::get_completion_domain_t<STDEXEC::set_value_t>) noexcept -> domain
    {
      return {};
    }

    friend auto operator==(scheduler a, scheduler b) noexcept -> bool
    {
      return a.pool_ == b.pool_;
    }

    friend auto operator!=(scheduler a, scheduler b) noexcept -> bool
    {
      return a.pool_ != b.pool_;
    }

   private:
    friend windows_thread_pool;
    friend domain;

    explicit scheduler(windows_thread_pool &pool) noexcept
      : pool_(&pool)
    {}

    windows_thread_pool *pool_;
  };

  inline auto windows_thread_pool::attrs::query(
    STDEXEC::get_completion_scheduler_t<STDEXEC::set_value_t>) const noexcept -> scheduler
  {
    return scheduler{*pool_};
  }

  //////////////////////////////////////////////////////////////////////////////
  // scheduler methods

  inline auto windows_thread_pool::get_scheduler() noexcept -> windows_thread_pool::scheduler
  {
    return scheduler{*this};
  }

  //////////////////////////////////////////////////////////////////////////////

  inline windows_thread_pool::windows_thread_pool() noexcept
    : threadPool_(nullptr)
    , availableParallelism_((std::max) (1u, std::thread::hardware_concurrency()))
  {}

  inline windows_thread_pool::windows_thread_pool(std::uint32_t minThreadCount,
                                                  std::uint32_t maxThreadCount)
    : threadPool_(::CreateThreadpool(nullptr))
    , availableParallelism_(
        (std::max) ((std::min) (maxThreadCount, std::thread::hardware_concurrency()),
                    std::uint32_t{1}))
  {
    if (threadPool_ == nullptr)
    {
      DWORD errorCode = ::GetLastError();
      throw std::system_error{static_cast<int>(errorCode),
                              std::system_category(),
                              "CreateThreadPool()"};
    }

    ::SetThreadpoolThreadMaximum(threadPool_, maxThreadCount);
    if (!::SetThreadpoolThreadMinimum(threadPool_, minThreadCount))
    {
      DWORD errorCode = ::GetLastError();
      ::CloseThreadpool(threadPool_);
      throw std::system_error{static_cast<int>(errorCode),
                              std::system_category(),
                              "SetThreadpoolThreadMinimum()"};
    }
  }

  inline windows_thread_pool::~windows_thread_pool()
  {
    if (threadPool_ != nullptr)
    {
      ::CloseThreadpool(threadPool_);
    }
  }

  inline windows_thread_pool::schedule_op_base::~schedule_op_base()
  {
    ::CloseThreadpoolWork(work_);
    ::DestroyThreadpoolEnvironment(&environ_);
  }

  inline void windows_thread_pool::schedule_op_base::start() noexcept
  {
    ::SubmitThreadpoolWork(work_);
  }

  inline windows_thread_pool::schedule_op_base::schedule_op_base(windows_thread_pool &pool,
                                                                 PTP_WORK_CALLBACK    workCallback)
  {
    ::InitializeThreadpoolEnvironment(&environ_);
    ::SetThreadpoolCallbackPool(&environ_, pool.threadPool_);
    work_ = ::CreateThreadpoolWork(workCallback, this, &environ_);
    if (work_ == nullptr)
    {
      // TODO: Should we just cache the error and deliver via set_error(rcvr_,
      // std::error_code{}) upon start()?
      DWORD errorCode = ::GetLastError();
      ::DestroyThreadpoolEnvironment(&environ_);
      throw std::system_error{static_cast<int>(errorCode),
                              std::system_category(),
                              "CreateThreadpoolWork()"};
    }
  }
}  // namespace experimental::execution::__win32

namespace experimental::execution
{
  using __win32::windows_thread_pool;

  static_assert(STDEXEC::scheduler<decltype(windows_thread_pool{}.get_scheduler())>,
                "windows_thread_pool::scheduler must model " STDEXEC_PP_STRINGIZE(STDEXEC) "::"
                                                                                           "schedul"
                                                                                           "er");
}  // namespace experimental::execution

namespace exec = experimental::execution;

#endif
