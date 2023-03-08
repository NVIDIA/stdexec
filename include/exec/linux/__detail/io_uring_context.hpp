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

#include <cstring>

#include <system_error>

#if !__has_include(<linux/io_uring.h>)
#error "io_uring.h not found. Your kernel is probably too old."
#else
#include <linux/io_uring.h>
#endif

#include <sys/eventfd.h>
#include <sys/syscall.h>

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
      const ::io_uring_cqe __cqe = __entries_[__index];
      __task* __op = std::bit_cast<__task*>(__cqe.user_data);
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

  inline __submission_result __submission_queue::submit(__task_queue __tasks) noexcept {
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
        __sqe.user_data = std::bit_cast<__u64>(__op);
        __array_[__index] = __index;
        ++__total_count;
        ++__count;
        ++__tail;
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
    __entry->addr = std::bit_cast<__u64>(&__self.__buffer_);
#ifdef STDEXEC_IORING_OP_READ
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

  inline void __context::submit(__task* __op) noexcept {
    __requests_.push_front(__op);
  }

  inline __scheduler __context::get_scheduler() noexcept {
    return __scheduler{this};
  }

  inline void __context::run() {
    __wakeup_operation_.start();
    while (__n_submitted_ > 0 || !__pending_.empty()) {
      __pending_.append(__requests_.pop_all());
      __submission_result __result = __submission_queue_.submit((__task_queue&&) __pending_);
      __n_submitted_ += __result.__n_submitted_;
      __pending_ = (__task_queue&&) __result.__pending_;
      while (!__result.__ready_.empty()) {
        __n_submitted_ -= __completion_queue_.complete((__task_queue&&) __result.__ready_);
        __pending_.append(__requests_.pop_all());
        __result = __submission_queue_.submit((__task_queue&&) __pending_);
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
    }
  }
}}