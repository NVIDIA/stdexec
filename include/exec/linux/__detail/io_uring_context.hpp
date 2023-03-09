/*
 * Copyright (C) 2023 Maikel Nadolski
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
      __op->__vtable_->__complete_(__op, &__cqe);
      ++__head;
      ++__count;
      __tail = __tail_.load(std::memory_order_acquire);
    }
    __head_.store(__head, std::memory_order_release);
    while (!__ready.empty()) {
      __task* __op = __ready.pop_front();
      __op->__vtable_->__complete_(__op, nullptr);
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
    __op->__vtable_->__complete_(__op, &__cqe);
  }

  inline __submission_result
    __submission_queue::submit(__task_queue __tasks, bool __is_stopped) noexcept {
    __u32 __tail = __tail_.load(std::memory_order_relaxed);
    __u32 __head = __head_.load(std::memory_order_acquire);
    __u32 __total_count = __tail - __head;
    __u32 __count = 0;
    __task_queue __ready{};
    __task_queue __pending{};
    __task* __op = nullptr;
    while (!__tasks.empty() && __total_count < __n_total_slots_) {
      const __u32 __index = __tail & __mask_;
      ::io_uring_sqe& __sqe = __entries_[__index];
      __op = __tasks.pop_front();
      STDEXEC_ASSERT(__op->__vtable_);
      if (__op->__vtable_->__ready_(__op)) {
        __ready.push_back(__op);
      } else {
        __op->__vtable_->__submit_(__op, &__sqe);
#ifdef STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
        if (__is_stopped && __sqe.opcode != IORING_OP_ASYNC_CANCEL) {
#else
        if (__is_stopped && __sqe.opcode != IORING_OP_NOP) {
#endif
          __stop(__op);
        } else {
          __sqe.user_data = bit_cast<__u64>(__op);
          __array_[__index] = __index;
          ++__total_count;
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
      .__n_submitted_ = __count,
      .__pending_ = (__task_queue&&) __pending,
      .__ready_ = (__task_queue&&) __ready};
  }

  bool __wakeup_operation::__ready_(void*) noexcept {
    return false;
  }

  void __wakeup_operation::__submit_(void* __pointer, ::io_uring_sqe* __entry) noexcept {
    __wakeup_operation& __self = *static_cast<__wakeup_operation*>(__pointer);
    std::memset(__entry, 0, sizeof(*__entry));
    __entry->fd = __self.__eventfd_;
    __entry->addr = bit_cast<__u64>(&__self.__buffer_);
#ifdef STDEXEC_HAS_IORING_OP_READ
    __entry->opcode = IORING_OP_READ;
    __entry->len = sizeof(__self.__buffer_);
#else
    __entry->opcode = IORING_OP_READV;
    __entry->len = 1;
#endif
  }

  void __wakeup_operation::__complete_(void* __pointer, const ::io_uring_cqe* __entry) noexcept {
    __wakeup_operation& __self = *static_cast<__wakeup_operation*>(__pointer);
    __self.start();
  }

  __wakeup_operation::__wakeup_operation(__context* __context, int __eventfd)
    : __task{__vtable}
    , __context_{__context}
    , __eventfd_{__eventfd} {
  }

  void __wakeup_operation::start() noexcept {
    if (!__context_->__stop_source_.stop_requested()) {
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
    __stop_source_.request_stop();
    wakeup();
  }

  inline bool __context::stop_requested() {
    return __stop_source_.stop_requested();
  }

  inline stdexec::in_place_stop_token __context::get_stop_token() const noexcept {
    return __stop_source_.get_token();
  }

  inline void __context::submit(__task* __op) noexcept {
    __requests_.push_front(__op);
  }

  inline __scheduler __context::get_scheduler() noexcept {
    return __scheduler{this};
  }

  inline void __context::run() {
    __wakeup_operation_.start();
    __pending_.append(__requests_.pop_all());
    while (__n_submitted_ > 0 || !__pending_.empty()) {
      __submission_result __result = __submission_queue_.submit(
        (__task_queue&&) __pending_, __stop_source_.stop_requested());
      __n_submitted_ += __result.__n_submitted_;
      __pending_ = (__task_queue&&) __result.__pending_;
      while (!__result.__ready_.empty()) {
        __n_submitted_ -= __completion_queue_.complete((__task_queue&&) __result.__ready_);
        __pending_.append(__requests_.pop_all());
        __result = __submission_queue_.submit(
          (__task_queue&&) __pending_, __stop_source_.stop_requested());
        __n_submitted_ += __result.__n_submitted_;
        __pending_ = (__task_queue&&) __result.__pending_;
      }
      if (__n_submitted_ <= 0) {
        break;
      }
      constexpr int __min_complete = 1;
      int rc = __io_uring_enter(__ring_fd_, __n_submitted_, __min_complete, IORING_ENTER_GETEVENTS);
      __throw_error_code_if(rc < 0, -rc);
      __n_submitted_ -= __completion_queue_.complete((__task_queue&&) __result.__ready_);
      __pending_.append(__requests_.pop_all());
    }
  }

  template <class _Op>
  concept __io_task = requires(_Op& __op, ::io_uring_sqe* __sqe, ::io_uring_cqe* __cqe) {
    { __op.ready() } noexcept -> std::convertible_to<bool>;
    { __op.submit(__sqe) } noexcept;
    { __op.complete(__cqe) } noexcept;
  };

  template <class _BaseOperation>
  struct __io_task_base : __task {
    static bool __ready_(void* __pointer) noexcept {
      _BaseOperation* __self = static_cast<_BaseOperation*>(__pointer);
      return __self->ready();
    }

    static void __submit_(void* __pointer, ::io_uring_sqe* __sqe) noexcept {
      _BaseOperation* __self = static_cast<_BaseOperation*>(__pointer);
      __self->submit(__sqe);
    }

    static void __complete_(void* __pointer, const ::io_uring_cqe* __cqe) noexcept {
      _BaseOperation* __self = static_cast<_BaseOperation*>(__pointer);
      __self->complete(__cqe);
    }

    static constexpr __task_vtable __vtable{&__ready_, &__submit_, &__complete_};

    __io_task_base()
      : __task{__vtable} {
    }
  };

  template <class _ReceiverId>
  class __schedule_operation : __io_task_base<__schedule_operation<_ReceiverId>> {
    using _Receiver = stdexec::__t<_ReceiverId>;
    __context* __context_;
    _Receiver __receiver_;

    static constexpr std::true_type ready() noexcept {
      return {};
    }

    static constexpr void submit(::io_uring_sqe* __entry) noexcept {
    }

    void complete(const ::io_uring_cqe*) noexcept {
      if (__context_->stop_requested()) {
        stdexec::set_stopped((_Receiver&&) __receiver_);
      } else {
        stdexec::set_value((_Receiver&&) __receiver_);
      }
    }

    friend void tag_invoke(stdexec::start_t, __schedule_operation& __self) noexcept {
      if (__self.__context_->stop_requested()) {
        stdexec::set_stopped((_Receiver&&) __self.__receiver_);
      } else {
        __self.__context_->submit(&__self);
        __self.__context_->wakeup();
      }
    }

   public:
    __schedule_operation(__context* __context, _Receiver&& __receiver)
      : __context_{__context}
      , __receiver_{(_Receiver&&) __receiver} {
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

    void submit(::io_uring_sqe* __sqe) noexcept {
#ifdef STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
      if constexpr (requires(_Derived* __op, ::io_uring_sqe* __sqe) { __op->submit_stop(__sqe); }) {
        __op_->submit_stop(__sqe);
      } else {
        *__sqe = ::io_uring_sqe{
          .opcode = IORING_OP_ASYNC_CANCEL, //
          .addr = bit_cast<__u64>(__op_)    //
        };
      }
#else
      __op_->submit_stop(__sqe);
#endif
    }

    void complete(const ::io_uring_cqe* __entry) noexcept {
      if (__op_->__n_ops_.fetch_sub(1, std::memory_order_relaxed) == 1) {
        auto& __receiver = __op_->__receiver_;
        __op_->__on_context_stop_.reset();
        __op_->__on_receiver_stop_.reset();
        stdexec::set_stopped((std::decay_t<decltype(__receiver)>&&) __receiver);
      }
    }

   public:
    explicit __stop_operation(_Derived* __op) noexcept
      : __op_{__op} {
    }

    void start() noexcept {
      int expected = 1;
      if (__op_->__n_ops_.compare_exchange_strong(expected, 2, std::memory_order_relaxed)) {
        __op_->__context_->submit(this);
        __op_->__context_->wakeup();
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

    void start() {
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
      auto __sec = std::chrono::duration_cast<std::chrono::seconds>(__nsec);
      __nsec -= __sec;
      __timerspec.it_value.tv_sec += __sec.count();
      __timerspec.it_value.tv_nsec += __nsec.count();
      return __timerspec;
    }
#endif

   public:
    static constexpr std::false_type ready() noexcept {
      return {};
    }

#ifndef STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
    void submit_stop(::io_uring_sqe* __sqe) noexcept {
      __duration_.it_value.tv_sec = 1;
      __duration_.it_value.tv_nsec = 0;
      ::timerfd_settime(
        __timerfd_, TFD_TIMER_ABSTIME | TFD_TIMER_CANCEL_ON_SET, &__duration_, nullptr);
      __sqe->opcode = IORING_OP_NOP;
    }
#endif

    void submit(::io_uring_sqe* __sqe) noexcept {
#ifdef STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
      ::io_uring_sqe __sqe_{};
      __sqe_.opcode = IORING_OP_TIMEOUT;
      __sqe_.addr = bit_cast<__u64>(&__duration_);
      __sqe_.len = 1;
      *__sqe = __sqe_;
#else
      ::io_uring_sqe __sqe_{};
      __sqe_.opcode = IORING_OP_READV;
      __sqe_.fd = __timerfd_;
      __sqe_.addr = bit_cast<__u64>(&__iov_);
      __sqe_.len = 1;
      *__sqe = __sqe_;
#endif
    }

    void complete(const ::io_uring_cqe* __cqe) noexcept {
      if (this->__n_ops_.fetch_sub(1, std::memory_order_relaxed) != 1) {
        return;
      }
      this->__on_context_stop_.reset();
      this->__on_receiver_stop_.reset();
      auto token = stdexec::get_stop_token(stdexec::get_env(this->__receiver_));
      if (
        this->__context_->stop_requested() || //
        token.stop_requested() || __cqe->res == -ECANCELED) {
        stdexec::set_stopped((_Receiver&&) this->__receiver_);
#ifdef STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
      } else if (__cqe->res == -ETIME || __cqe->res == 0) {
#else
      } else if (__cqe->res == 8) {
#endif
        stdexec::set_value((_Receiver&&) this->__receiver_);
      } else {
        stdexec::set_error(
          (_Receiver&&) this->__receiver_,
          std::make_exception_ptr(std::system_error(-__cqe->res, std::system_category())));
      }
    }

   private:
    friend void tag_invoke(stdexec::start_t, __schedule_after_operation& __self) noexcept {
      if (__self.__context_->stop_requested()) {
        stdexec::set_stopped((_Receiver&&) __self.__receiver_);
      } else {
        __self.start();
        __self.__context_->submit(&__self);
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
      int __rc = ::timerfd_settime(
        __timerfd_, TFD_TIMER_ABSTIME | TFD_TIMER_CANCEL_ON_SET, &__duration_, nullptr);
      __throw_error_code_if(__rc < 0, errno);
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

  inline __schedule_after_sender tag_invoke(
    exec::schedule_after_t,
    const __scheduler& __sched,
    std::chrono::nanoseconds __duration) {
    return __schedule_after_sender{.__env_ = {.__sched_ = __sched}, .__duration_ = __duration};
  }
}}