/*
 * Copyright (c) 2023 Maikel Nadolski
 * Copyright (c) 2023 NVIDIA Corporation
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

#include "../io_uring_context.hpp"
#include "../../__detail/__bit_cast.hpp"

#include <cstring>
#include <system_error>

#include <sys/eventfd.h>
#include <sys/syscall.h>

#ifndef STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
#include <sys/timerfd.h>
#include <sys/uio.h>
#endif

namespace exec { namespace __io_uring {
  using __task_queue = stdexec::__intrusive_queue<&__task::__next_>;

  inline void __throw_on_error(int __ec) {
    if (__ec) {
      throw std::system_error(__ec, std::system_category());
    }
  }

  inline void __throw_error_code_if(bool __cond, int __ec) {
    if (__cond) {
      throw std::system_error(__ec, std::system_category());
    }
  }

  inline safe_file_descriptor __io_uring_setup(unsigned __entries, ::io_uring_params& __params) {
    int rc = (int) ::syscall(__NR_io_uring_setup, __entries, &__params);
    __throw_error_code_if(rc < 0, -rc);
    return safe_file_descriptor{rc};
  }

  inline int __io_uring_enter(
    int __ring_fd,
    unsigned int __to_submit,
    unsigned int __min_complete,
    unsigned int __flags) {
    return (int) ::syscall(
      __NR_io_uring_enter, __ring_fd, __to_submit, __min_complete, __flags, nullptr, 0);
  }

  inline memory_mapped_region __map_region(int __fd, ::off_t __offset, std::size_t __size) {
    void* __ptr = ::mmap(
      nullptr, __size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, __fd, __offset);
    __throw_error_code_if(__ptr == MAP_FAILED, errno);
    return memory_mapped_region{__ptr, __size};
  }

  inline __context_base::__context_base(unsigned __entries, unsigned __flags)
    : __params_{.flags = __flags}
    , __ring_fd_{__io_uring_setup(__entries, __params_)}
    , __eventfd_{::eventfd(0, EFD_CLOEXEC)} {
    if (!__eventfd_) {
      throw std::system_error(errno, std::system_category());
    }
    auto __sring_sz = __params_.sq_off.array + __params_.sq_entries * sizeof(unsigned);
    auto __cring_sz = __params_.cq_off.cqes + __params_.cq_entries * sizeof(::io_uring_cqe);
    auto __sqes_sz = __params_.sq_entries * sizeof(::io_uring_sqe);
    if (__params_.features & IORING_FEAT_SINGLE_MMAP) {
      __sring_sz = std::max(__sring_sz, __cring_sz);
      __cring_sz = __sring_sz;
    }
    __submission_queue_region_ = __map_region(__ring_fd_, IORING_OFF_SQ_RING, __sring_sz);
    __submission_queue_entries_ = __map_region(__ring_fd_, IORING_OFF_SQES, __sqes_sz);
    if (!(__params_.features & IORING_FEAT_SINGLE_MMAP)) {
      __completion_queue_region_ = __map_region(__ring_fd_, IORING_OFF_CQ_RING, __cring_sz);
    }
  }

  template <class _Ty>
  inline _Ty at_offset_as(void* __pointer, __u32 __offset) {
    return reinterpret_cast<_Ty>(static_cast<std::byte*>(__pointer) + __offset);
  }

  __completion_queue::__completion_queue(
    const memory_mapped_region& __region,
    const ::io_uring_params& __params)
    : __head_{*at_offset_as<__u32*>(__region.data(), __params.cq_off.head)}
    , __tail_{*at_offset_as<__u32*>(__region.data(), __params.cq_off.tail)}
    , __entries_{at_offset_as<::io_uring_cqe*>(__region.data(), __params.cq_off.cqes)}
    , __mask_{*at_offset_as<__u32*>(__region.data(), __params.cq_off.ring_mask)} {
  }

  inline int __completion_queue::complete(__task_queue __ready) noexcept {
    __u32 __head = __head_.load(std::memory_order_relaxed);
    __u32 __tail = __tail_.load(std::memory_order_acquire);
    int __count = 0;
    while (__head != __tail) {
      const __u32 __index = __head & __mask_;
      const ::io_uring_cqe& __cqe = __entries_[__index];
      __task* __op = bit_cast<__task*>(__cqe.user_data);
      __op->__vtable_->__complete_(__op, __cqe);
      ++__head;
      ++__count;
      __tail = __tail_.load(std::memory_order_acquire);
    }
    __head_.store(__head, std::memory_order_release);
    while (!__ready.empty()) {
      __task* __op = __ready.pop_front();
      ::io_uring_cqe __dummy_cqe{.user_data = bit_cast<__u64>(__op)};
      __op->__vtable_->__complete_(__op, __dummy_cqe);
    }
    return __count;
  }

  __submission_queue::__submission_queue(
    const memory_mapped_region& __region,
    const memory_mapped_region& __sqes_region,
    const ::io_uring_params& __params)
    : __head_{*at_offset_as<__u32*>(__region.data(), __params.sq_off.head)}
    , __tail_{*at_offset_as<__u32*>(__region.data(), __params.sq_off.tail)}
    , __array_{at_offset_as<__u32*>(__region.data(), __params.sq_off.array)}
    , __entries_{static_cast<::io_uring_sqe*>(__sqes_region.data())}
    , __mask_{*at_offset_as<__u32*>(__region.data(), __params.sq_off.ring_mask)}
    , __n_total_slots_{__params.sq_entries} {
  }

  inline void __stop(__task* __op) noexcept {
    ::io_uring_cqe __cqe{};
    __cqe.res = -ECANCELED;
    __cqe.user_data = bit_cast<__u64>(__op);
    __op->__vtable_->__complete_(__op, __cqe);
  }

  inline __submission_result __submission_queue::submit(
    __task_queue __tasks,
    __u32 __max_submissions,
    bool __is_stopped) noexcept {
    __u32 __tail = __tail_.load(std::memory_order_relaxed);
    __u32 __head = __head_.load(std::memory_order_acquire);
    __u32 __current_count = __tail - __head;
    STDEXEC_ASSERT(__current_count <= __n_total_slots_);
    __max_submissions = std::min(__max_submissions, __n_total_slots_ - __current_count);
    __u32 __count = 0;
    __task_queue __ready{};
    __task_queue __pending{};
    __task* __op = nullptr;
    while (!__tasks.empty() && __count < __max_submissions) {
      const __u32 __index = __tail & __mask_;
      ::io_uring_sqe& __sqe = __entries_[__index];
      __op = __tasks.pop_front();
      STDEXEC_ASSERT(__op->__vtable_);
      if (__op->__vtable_->__ready_(__op)) {
        __ready.push_back(__op);
      } else {
        __op->__vtable_->__submit_(__op, __sqe);
#ifdef STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
        if (__is_stopped && __sqe.opcode != IORING_OP_ASYNC_CANCEL) {
#else
        if (__is_stopped) {
#endif
          __stop(__op);
        } else {
          __sqe.user_data = bit_cast<__u64>(__op);
          __array_[__index] = __index;
          ++__count;
          ++__tail;
        }
      }
    }
    __tail_.store(__tail, std::memory_order_release);
    while (!__tasks.empty()) {
      __op = __tasks.pop_front();
      if (__op->__vtable_->__ready_(__op)) {
        __ready.push_back(__op);
      } else {
        __pending.push_back(__op);
      }
    }
    return {
      .__n_submitted = __count,
      .__pending = (__task_queue&&) __pending,
      .__ready = (__task_queue&&) __ready};
  }

  bool __wakeup_operation::__ready_(__task*) noexcept {
    return false;
  }

  void __wakeup_operation::__submit_(__task* __pointer, ::io_uring_sqe& __entry) noexcept {
    __wakeup_operation& __self = *static_cast<__wakeup_operation*>(__pointer);
    std::memset(&__entry, 0, sizeof(__entry));
    __entry.fd = __self.__eventfd_;
    __entry.addr = bit_cast<__u64>(&__self.__buffer_);
#ifdef STDEXEC_HAS_IORING_OP_READ
    __entry.opcode = IORING_OP_READ;
    __entry.len = sizeof(__self.__buffer_);
#else
    __entry.opcode = IORING_OP_READV;
    __entry.len = 1;
#endif
  }

  void __wakeup_operation::__complete_(__task* __pointer, const ::io_uring_cqe& __entry) noexcept {
    __wakeup_operation& __self = *static_cast<__wakeup_operation*>(__pointer);
    __self.start();
  }

  __wakeup_operation::__wakeup_operation(__context* __context, int __eventfd)
    : __task{__vtable}
    , __context_{__context}
    , __eventfd_{__eventfd} {
  }

  void __wakeup_operation::start() noexcept {
    if (!__context_->__stop_source_->stop_requested()) {
      __context_->__pending_.push_front(this);
    }
  }

  inline __context::__context(unsigned __entries, unsigned __flags)
    : __context_base(std::max(__entries, 2u), __flags)
    , __completion_queue_{__completion_queue_region_ ? __completion_queue_region_ : __submission_queue_region_, __params_}
    , __submission_queue_{__submission_queue_region_, __submission_queue_entries_, __params_}
    , __wakeup_operation_{this, __eventfd_} {
  }

  inline void __context::wakeup() {
    std::uint64_t __wakeup = 1;
    __throw_error_code_if(::write(__eventfd_, &__wakeup, sizeof(__wakeup)) == -1, errno);
  }

  inline void __context::request_stop() {
    __stop_source_->request_stop();
    wakeup();
  }

  inline bool __context::stop_requested() const noexcept {
    return __stop_source_->stop_requested();
  }

  inline stdexec::in_place_stop_token __context::get_stop_token() const noexcept {
    return __stop_source_->get_token();
  }

  inline bool __context::is_running() const noexcept {
    return __is_running_.load(std::memory_order_relaxed);
  }

  inline bool __context::submit(__task* __op) noexcept {
    // As long as the number of in-flight submissions is not __no_new_submissions, we can
    // increment the counter and push the operation onto the queue.
    // If the number of in-flight submissions is __no_new_submissions, we have already
    // finished the stop operation of the io context and we can immediately stop the operation inline.
    // Remark: As long as the stopping is in progress we can still submit new operations.
    // But no operation will be submitted to io uring unless it is a cancellation operation.
    int __n = 0;
    while (__n != __no_new_submissions
           && !__n_submissions_in_flight_.compare_exchange_weak(
             __n, __n + 1, std::memory_order_acquire, std::memory_order_relaxed))
      ;
    if (__n == __no_new_submissions) {
      __stop(__op);
      return false;
    } else {
      __requests_.push_front(__op);
      [[maybe_unused]] int __prev = __n_submissions_in_flight_.fetch_sub(
        1, std::memory_order_relaxed);
      STDEXEC_ASSERT(__prev > 0);
      return true;
    }
  }

  inline __scheduler __context::get_scheduler() noexcept {
    return __scheduler{this};
  }

  inline void __context::run() {
    bool expected_running = false;
    // Only one thread of execution is allowed to drive the io context.
    if (!__is_running_.compare_exchange_strong(expected_running, true, std::memory_order_relaxed)) {
      throw std::runtime_error("exec::io_uring_context::run() called on a running context");
    } else {
      // Check whether we restart the context after a context-wide stop.
      // We have to reset the stop source in this case.
      int __in_flight = __n_submissions_in_flight_.load(std::memory_order_relaxed);
      if (__in_flight == __no_new_submissions) {
        __stop_source_.emplace();
        // Make emplacement of stop source visible to other threads and open the door for new submissions.
        __n_submissions_in_flight_.store(0, std::memory_order_release);
      }
    }
    std::ptrdiff_t __n_submitted = 0;
    const __u32 __max_completions = __params_.cq_entries;
    __wakeup_operation_.start();
    __pending_.append(__requests_.pop_all());
    while (__n_submitted > 0 || !__pending_.empty()) {
      STDEXEC_ASSERT(__n_submitted <= static_cast<std::ptrdiff_t>(__max_completions));
      __submission_result __result = __submission_queue_.submit(
        (__task_queue&&) __pending_,
        __max_completions - static_cast<__u32>(__n_submitted),
        __stop_source_->stop_requested());
      __n_submitted += __result.__n_submitted;
      STDEXEC_ASSERT(__n_submitted <= static_cast<std::ptrdiff_t>(__max_completions));
      __pending_ = (__task_queue&&) __result.__pending;
      while (!__result.__ready.empty()) {
        __n_submitted -= __completion_queue_.complete((__task_queue&&) __result.__ready);
        STDEXEC_ASSERT(0 <= __n_submitted);
        __pending_.append(__requests_.pop_all());
        __result = __submission_queue_.submit(
          (__task_queue&&) __pending_,
          __max_completions - static_cast<__u32>(__n_submitted),
          __stop_source_->stop_requested());
        __n_submitted += __result.__n_submitted;
        __pending_ = (__task_queue&&) __result.__pending;
      }
      if (__n_submitted <= 0) {
        break;
      }
      constexpr int __min_complete = 1;
      int rc = __io_uring_enter(__ring_fd_, __n_submitted, __min_complete, IORING_ENTER_GETEVENTS);
      __throw_error_code_if(rc < 0, -rc);
      __n_submitted -= __completion_queue_.complete((__task_queue&&) __result.__ready);
      __pending_.append(__requests_.pop_all());
    }
    STDEXEC_ASSERT(__n_submitted == 0);
    STDEXEC_ASSERT(__pending_.empty());
    STDEXEC_ASSERT_FN(__stop_source_->stop_requested());
    // try to shutdown the request queue
    int __n_in_flight_expected = 0;
    while (!__n_submissions_in_flight_.compare_exchange_weak(
      __n_in_flight_expected, __no_new_submissions, std::memory_order_relaxed)) {
      if (__n_in_flight_expected == __no_new_submissions) {
        break;
      }
      __n_in_flight_expected = 0;
    }
    STDEXEC_ASSERT(
      __n_submissions_in_flight_.load(std::memory_order_relaxed) == __no_new_submissions);
    // There could have been requests in flight. Complete all of them
    // and then stop it, finally.
    __pending_.append(__requests_.pop_all());
    __submission_result __result = __submission_queue_.submit(
      (__task_queue&&) __pending_, __max_completions, true);
    STDEXEC_ASSERT(__result.__n_submitted == 0);
    STDEXEC_ASSERT(__result.__pending.empty());
    __completion_queue_.complete((__task_queue&&) __result.__ready);
    __is_running_.store(false, std::memory_order_relaxed);
  }

  template <class _Op>
  concept __io_task = requires(_Op& __op, ::io_uring_sqe& __sqe, const ::io_uring_cqe& __cqe) {
    { __op.ready() } noexcept -> std::convertible_to<bool>;
    { __op.submit(__sqe) } noexcept;
    { __op.complete(__cqe) } noexcept;
  };

  template <class _Derived>
  struct __io_task_base : __task {
    static bool __ready_(__task* __pointer) noexcept {
      _Derived* __self = static_cast<_Derived*>(__pointer);
      return __self->ready();
    }

    static void __submit_(__task* __pointer, ::io_uring_sqe& __sqe) noexcept {
      _Derived* __self = static_cast<_Derived*>(__pointer);
      __self->submit(__sqe);
    }

    static void __complete_(__task* __pointer, const ::io_uring_cqe& __cqe) noexcept {
      _Derived* __self = static_cast<_Derived*>(__pointer);
      __self->complete(__cqe);
    }

    static constexpr __task_vtable __vtable{&__ready_, &__submit_, &__complete_};

    __io_task_base()
      : __task{__vtable} {
    }
  };

  template <class _ReceiverId>
  class __schedule_operation : public __io_task_base<__schedule_operation<_ReceiverId>> {
    using _Receiver = stdexec::__t<_ReceiverId>;
    __context* __context_;
    _Receiver __receiver_;

    friend void tag_invoke(stdexec::start_t, __schedule_operation& __self) noexcept {
      if (__self.__context_->submit(&__self)) {
        __self.__context_->wakeup();
      }
    }

   public:
    __schedule_operation(__context* __context, _Receiver&& __receiver)
      : __context_{__context}
      , __receiver_{(_Receiver&&) __receiver} {
    }

    static constexpr std::true_type ready() noexcept {
      return {};
    }

    static constexpr void submit(::io_uring_sqe& __entry) noexcept {
    }

    void complete(const ::io_uring_cqe& __cqe) noexcept {
      auto token = stdexec::get_stop_token(stdexec::get_env(__receiver_));
      if (__cqe.res == -ECANCELED || __context_->stop_requested() || token.stop_requested()) {
        stdexec::set_stopped((_Receiver&&) __receiver_);
      } else {
        stdexec::set_value((_Receiver&&) __receiver_);
      }
    }
  };

  class __schedule_env {
   public:
    __scheduler __sched_;
   private:
    friend __scheduler tag_invoke(
      stdexec::get_completion_scheduler_t<stdexec::set_value_t>,
      const __schedule_env& __env) noexcept {
      return __env.__sched_;
    }
  };

  class __schedule_sender {
   public:
    __schedule_env __env_;

   private:
    friend __schedule_env
      tag_invoke(stdexec::get_env_t, const __schedule_sender& __sender) noexcept {
      return __sender.__env_;
    }

    template <class _Env>
    friend stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_stopped_t()>
      tag_invoke(stdexec::get_completion_signatures_t, const __schedule_sender&, _Env) noexcept {
      return {};
    }

    template <class _Receiver>
    friend __schedule_operation<stdexec::__id<_Receiver>>
      tag_invoke(stdexec::connect_t, const __schedule_sender& __sender, _Receiver&& __receiver) {
      return {__sender.__env_.__sched_.__context_, (_Receiver&&) __receiver};
    }
  };

  inline __schedule_sender tag_invoke(stdexec::schedule_t, const __scheduler& __sched) {
    return __schedule_sender{.__env_ = {.__sched_ = __sched}};
  }

  template <class _Derived, class _Receiver>
  struct __stoppable_task_base;

  template <class _Derived, class _Receiver>
  class __stop_operation : public __io_task_base<__stop_operation<_Derived, _Receiver>> {
    _Derived* __op_;
   public:
    static constexpr std::false_type ready() noexcept {
      return {};
    }

    void submit(::io_uring_sqe& __sqe) noexcept {
#ifdef STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
      if constexpr (requires(_Derived* __op, ::io_uring_sqe& __sqe) { __op->submit_stop(__sqe); }) {
        __op_->submit_stop(__sqe);
      } else {
        __sqe = ::io_uring_sqe{
          .opcode = IORING_OP_ASYNC_CANCEL, //
          .addr = bit_cast<__u64>(__op_)    //
        };
      }
#else
      __op_->submit_stop(__sqe);
#endif
    }

    void complete(const ::io_uring_cqe&) noexcept {
      if (__op_->__n_ops_.fetch_sub(1, std::memory_order_relaxed) == 1) {
        _Receiver& __receiver = __op_->__receiver_;
        __op_->__on_context_stop_.reset();
        __op_->__on_receiver_stop_.reset();
        stdexec::set_stopped((_Receiver&&) __receiver);
      }
    }

   public:
    explicit __stop_operation(_Derived* __op) noexcept
      : __op_{__op} {
    }

    void start() noexcept {
      int expected = 1;
      if (__op_->__n_ops_.compare_exchange_strong(expected, 2, std::memory_order_relaxed)) {
        if (__op_->__context_->submit(this)) {
          __op_->__context_->wakeup();
        }
      }
    }
  };

  template <class _Derived, class _Receiver>
  struct __stoppable_task_base : __io_task_base<_Derived> {
    struct __stop_callback {
      __stoppable_task_base* __self_;

      void operator()() noexcept {
        __self_->__stop_operation_.start();
      }
    };

    using __on_context_stop_t = std::optional<stdexec::in_place_stop_callback<__stop_callback>>;
    using __on_receiver_stop_t = std::optional<typename stdexec::stop_token_of_t<
      stdexec::env_of_t<_Receiver>&>::template callback_type<__stop_callback>>;

    __context* __context_;
    _Receiver __receiver_;
    __stop_operation<_Derived, _Receiver> __stop_operation_;
    std::atomic<int> __n_ops_{0};
    __on_context_stop_t __on_context_stop_{};
    __on_receiver_stop_t __on_receiver_stop_{};

    explicit __stoppable_task_base(__context* __context, _Receiver&& __receiver)
      : __context_{__context}
      , __receiver_{(_Receiver&&) __receiver}
      , __stop_operation_{static_cast<_Derived*>(this)} {
    }

    void prepare_submission() {
      if (__n_ops_.fetch_add(1, std::memory_order_relaxed) == 0) {
        __on_context_stop_.emplace(__context_->get_stop_token(), __stop_callback{this});
        __on_receiver_stop_.emplace(
          stdexec::get_stop_token(stdexec::get_env(__receiver_)), __stop_callback{this});
      }
    }
  };

  template <class _ReceiverId>
  class __schedule_after_operation
    : public __stoppable_task_base<
        __schedule_after_operation<_ReceiverId>,
        stdexec::__t<_ReceiverId>> {
    using _Receiver = stdexec::__t<_ReceiverId>;

#ifdef STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
    struct __kernel_timespec {
      __s64 __tv_sec;
      __s64 __tv_nsec;
    };

    __kernel_timespec __duration_;

    static constexpr __kernel_timespec
      __duration_to_timespec(std::chrono::nanoseconds dur) noexcept {
      auto secs = std::chrono::duration_cast<std::chrono::seconds>(dur);
      dur -= secs;
      secs = std::max(secs, std::chrono::seconds{0});
      dur = std::clamp(dur, std::chrono::nanoseconds{0}, std::chrono::nanoseconds{999'999'999});
      return __kernel_timespec{secs.count(), dur.count()};
    }
#else
    safe_file_descriptor __timerfd_;
    ::itimerspec __duration_;
    std::uint64_t __n_expirations_{0};
    ::iovec __iov_{&__n_expirations_, sizeof(__n_expirations_)};

    static constexpr ::itimerspec __duration_to_timespec(std::chrono::nanoseconds __nsec) noexcept {
      ::itimerspec __timerspec{};
      ::clock_gettime(CLOCK_REALTIME, &__timerspec.it_value);
      __nsec = std::chrono::nanoseconds{__timerspec.it_value.tv_nsec} + __nsec;
      auto __sec = std::chrono::duration_cast<std::chrono::seconds>(__nsec);
      __nsec -= __sec;
      __timerspec.it_value.tv_sec += __sec.count();
      __timerspec.it_value.tv_nsec = __nsec.count();
      STDEXEC_ASSERT(
        0 <= __timerspec.it_value.tv_nsec && __timerspec.it_value.tv_nsec < 1'000'000'000);
      return __timerspec;
    }
#endif

   public:
    static constexpr std::false_type ready() noexcept {
      return {};
    }

#ifndef STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
    void submit_stop(::io_uring_sqe& __sqe) noexcept {
      __duration_.it_value.tv_sec = 1;
      __duration_.it_value.tv_nsec = 0;
      ::timerfd_settime(
        __timerfd_, TFD_TIMER_ABSTIME | TFD_TIMER_CANCEL_ON_SET, &__duration_, nullptr);
      __sqe = ::io_uring_sqe{.opcode = IORING_OP_NOP};
    }
#endif

    void submit(::io_uring_sqe& __sqe) noexcept {
      this->prepare_submission();
#ifdef STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
      ::io_uring_sqe __sqe_{};
      __sqe_.opcode = IORING_OP_TIMEOUT;
      __sqe_.addr = bit_cast<__u64>(&__duration_);
      __sqe_.len = 1;
      __sqe = __sqe_;
#else
      ::io_uring_sqe __sqe_{};
      __sqe_.opcode = IORING_OP_READV;
      __sqe_.fd = __timerfd_;
      __sqe_.addr = bit_cast<__u64>(&__iov_);
      __sqe_.len = 1;
      __sqe = __sqe_;
#endif
    }

    void complete(const ::io_uring_cqe& __cqe) noexcept {
      if (this->__n_ops_.fetch_sub(1, std::memory_order_relaxed) != 1) {
        return;
      }
      this->__on_context_stop_.reset();
      this->__on_receiver_stop_.reset();
      auto token = stdexec::get_stop_token(stdexec::get_env(this->__receiver_));
      if (__cqe.res == -ECANCELED || this->__context_->stop_requested() || token.stop_requested()) {
        stdexec::set_stopped((_Receiver&&) this->__receiver_);
#ifdef STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
      } else if (__cqe.res == -ETIME || __cqe.res == 0) {
#else
      } else if (__cqe.res == 8) {
#endif
        stdexec::set_value((_Receiver&&) this->__receiver_);
      } else {
        stdexec::set_error(
          (_Receiver&&) this->__receiver_,
          std::make_exception_ptr(std::system_error(-__cqe.res, std::system_category())));
      }
    }

   private:
    friend void tag_invoke(stdexec::start_t, __schedule_after_operation& __self) noexcept {
      if (__self.__context_->submit(&__self)) {
        __self.__context_->wakeup();
      }
    }

    using __base_t =
      __stoppable_task_base<__schedule_after_operation<_ReceiverId>, stdexec::__t<_ReceiverId>>;

   public:
    __schedule_after_operation(
      __context* __context,
      std::chrono::nanoseconds __duration,
      _Receiver&& __receiver)
      : __base_t{__context, (_Receiver&&) __receiver}
#ifdef STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
      , __duration_{__duration_to_timespec(__duration)}
#else
      , __timerfd_{::timerfd_create(CLOCK_REALTIME, 0)}
      , __duration_{__duration_to_timespec(__duration)}
#endif
    {
#ifndef STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
      int __rc = ::timerfd_settime(
        __timerfd_, TFD_TIMER_ABSTIME | TFD_TIMER_CANCEL_ON_SET, &__duration_, nullptr);
      __throw_error_code_if(__rc < 0, errno);
#endif
    }
  };

  class __schedule_after_sender {
   public:
    __schedule_env __env_;
    std::chrono::nanoseconds __duration_;

   private:
    friend __schedule_env
      tag_invoke(stdexec::get_env_t, const __schedule_after_sender& __sender) noexcept {
      return __sender.__env_;
    }

    template <class _Env>
    friend stdexec::completion_signatures<
      stdexec::set_value_t(),
      stdexec::set_error_t(std::exception_ptr),
      stdexec::set_stopped_t()>
      tag_invoke(
        stdexec::get_completion_signatures_t,
        const __schedule_after_sender&,
        _Env) noexcept {
      return {};
    }

    template <class _Receiver>
    friend __schedule_after_operation<stdexec::__id<_Receiver>> tag_invoke(
      stdexec::connect_t,
      const __schedule_after_sender& __sender,
      _Receiver&& __receiver) {
      return {__sender.__env_.__sched_.__context_, __sender.__duration_, (_Receiver&&) __receiver};
    }
  };

  inline std::chrono::time_point<std::chrono::steady_clock>
    tag_invoke(exec::now_t, const __scheduler& __sched) noexcept {
    return std::chrono::steady_clock::now();
  }

  inline __schedule_after_sender tag_invoke(
    exec::schedule_after_t,
    const __scheduler& __sched,
    std::chrono::nanoseconds __duration) {
    return __schedule_after_sender{.__env_ = {.__sched_ = __sched}, .__duration_ = __duration};
  }

  template <class _Clock, class _Duration>
  __schedule_after_sender tag_invoke(
    exec::schedule_at_t,
    const __scheduler& __sched,
    const std::chrono::time_point<_Clock, _Duration>& __time_point) {
    auto __duration = __time_point - _Clock::now();
    if (__duration < std::chrono::nanoseconds(1)) {
      __duration = std::chrono::nanoseconds(1);
    }
    return __schedule_after_sender{.__env_ = {.__sched_ = __sched}, .__duration_ = __duration};
  }
}}
