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

#if !__has_include(<linux/io_uring.h>)
#  error "io_uring.h not found. Your kernel is probably too old."
#else
#  include <linux/io_uring.h>

#  include "../../stdexec/execution.hpp"
#  include "../timed_scheduler.hpp"

#  include "../__detail/__atomic_intrusive_queue.hpp"
#  include "../__detail/__atomic_ref.hpp"
#  include "../__detail/__bit_cast.hpp"

#  include "./safe_file_descriptor.hpp"
#  include "./memory_mapped_region.hpp"

#  include "../scope.hpp"

#  if !__has_include(<linux/version.h>)
#    error "linux/version.h not found. Do you use Linux?"
#  else
#    include <linux/version.h>

#    if LINUX_VERSION_CODE < KERNEL_VERSION(5, 5, 0)
#      warning "Your kernel is too old to support io_uring with cancellation support."
#      include <sys/timerfd.h>
#    else
#      define STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
#    endif

#    if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
#      define STDEXEC_HAS_IORING_OP_READ
#    endif

#    include <sys/uio.h>
#    include <sys/eventfd.h>
#    include <sys/syscall.h>

#    include <algorithm>
#    include <cstring>

namespace exec {
  namespace __io_uring {
    using namespace stdexec::tags;

    inline void __throw_error_code_if(bool __cond, int __ec) {
      if (__cond) {
        STDEXEC_THROW(std::system_error(__ec, std::system_category()));
      }
    }

    inline auto
      __io_uring_setup(unsigned __entries, ::io_uring_params& __params) -> safe_file_descriptor {
      int rc = static_cast<int>(::syscall(__NR_io_uring_setup, __entries, &__params));
      __throw_error_code_if(rc < 0, -rc);
      return safe_file_descriptor{rc};
    }

    inline auto __io_uring_enter(
      int __ring_fd,
      unsigned int __to_submit,
      unsigned int __min_complete,
      unsigned int __flags) -> int {
      int rc = static_cast<int>(::syscall(
        __NR_io_uring_enter, __ring_fd, __to_submit, __min_complete, __flags, nullptr, 0));
      if (rc == -1) {
        return -errno;
      } else {
        return rc;
      }
    }

    inline auto
      __map_region(int __fd, ::off_t __offset, std::size_t __size) -> memory_mapped_region {
      void* __ptr =
        ::mmap(nullptr, __size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, __fd, __offset);
      __throw_error_code_if(__ptr == MAP_FAILED, errno);
      return memory_mapped_region{__ptr, __size};
    }

    // This base class maps the kernel's io_uring data structures into the process.
    struct __context_base : stdexec::__immovable {
      explicit __context_base(unsigned __entries, unsigned __flags = 0)
        : __params_{__context_base::__init_params(__flags)}
        , __ring_fd_{__io_uring_setup(__entries, __params_)}
        , __eventfd_{::eventfd(0, EFD_CLOEXEC)} {
        __throw_error_code_if(!__eventfd_, errno);
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

      static auto __init_params(unsigned __flags) noexcept -> ::io_uring_params {
        ::io_uring_params __params{};
        __params.flags = __flags;
        return __params;
      }

      // memory mapped regions for submission and completion queue
      memory_mapped_region __submission_queue_region_{};
      memory_mapped_region __completion_queue_region_{};
      memory_mapped_region __submission_queue_entries_{};
      ::io_uring_params __params_{};
      // file descriptor for io_uring
      safe_file_descriptor __ring_fd_{};
      // file descriptor for the wakeup event
      safe_file_descriptor __eventfd_{};
    };

    struct __task;

    // Each io operation provides the following interface:
    struct __task_vtable {
      // If this function returns true, the __submit_ function will not be called.
      // The task is marked ready to be completed and will be returned by the
      // by the context's completion queue.
      // If this function returns false, the __submit_ function will be called.
      bool (*__ready_)(__task*) noexcept;
      // This function is called to submit the task to the io_uring.
      // Its purpose is to fill the io_uring_sqe structure to describe the io
      // operation and its completion condition.
      void (*__submit_)(__task*, ::io_uring_sqe&) noexcept;
      // This function is called when the io operation is completed.
      // The status of the operation is passed as a parameter.
      void (*__complete_)(__task*, const ::io_uring_cqe&) noexcept;
    };

    // This is the base class for all io operations.
    // It provides the vtable and the next pointer for the intrusive queues.
    struct __task : stdexec::__immovable {
      const __task_vtable* __vtable_;
      __task* __next_{nullptr};

      explicit __task(const __task_vtable& __vtable)
        : __vtable_{&__vtable} {
      }
    };

    using __task_queue = stdexec::__intrusive_queue<&__task::__next_>;
    using __atomic_task_queue = __atomic_intrusive_queue<&__task::__next_>;

    template <class _Ty>
    inline auto __at_offset_as(void* __pointer, __u32 __offset) -> _Ty {
      return reinterpret_cast<_Ty>(static_cast<std::byte*>(__pointer) + __offset);
    }

    struct __submission_result {
      __u32 __n_submitted;
      __task_queue __pending;
      __task_queue __ready;
    };

    inline void __stop(__task* __op) noexcept {
      ::io_uring_cqe __cqe{};
      __cqe.res = -ECANCELED;
      __cqe.user_data = bit_cast<__u64>(__op);
      __op->__vtable_->__complete_(__op, __cqe);
    }

    // This class implements the io_uring submission queue.
    class __submission_queue {
      __atomic_ref<__u32> __head_;
      __atomic_ref<__u32> __tail_;
      __u32* __array_;
      ::io_uring_sqe* __entries_;
      __u32 __mask_;
      __u32 __n_total_slots_;
     public:
      explicit __submission_queue(
        const memory_mapped_region& __region,
        const memory_mapped_region& __sqes_region,
        const ::io_uring_params& __params)
        : __head_{*__at_offset_as<__u32*>(__region.data(), __params.sq_off.head)}
        , __tail_{*__at_offset_as<__u32*>(__region.data(), __params.sq_off.tail)}
        , __array_{__at_offset_as<__u32*>(__region.data(), __params.sq_off.array)}
        , __entries_{static_cast<::io_uring_sqe*>(__sqes_region.data())}
        , __mask_{*__at_offset_as<__u32*>(__region.data(), __params.sq_off.ring_mask)}
        , __n_total_slots_{__params.sq_entries} {
      }

      // This function submits the given queue of tasks to the io_uring.

      // Each task that is ready to be completed is moved to the __ready queue.
      // If the submission queue gets full before all tasks are submitted, the
      // remaining tasks are moved to the __pending queue.
      // If is_stopped is true, no new tasks are submitted to the io_uring unless it is a cancellation.
      // If is_stopped is true and a task is not ready to be completed, the task is completed with
      // an io_uring_cqe object with the result field set to -ECANCELED.
      auto submit(__task_queue __tasks, __u32 __max_submissions, bool __is_stopped) noexcept
        -> __submission_result {
        __u32 __tail = __tail_.load(std::memory_order_relaxed);
        __u32 __head = __head_.load(std::memory_order_acquire);
        __u32 __current_count = __tail - __head;
        STDEXEC_ASSERT(__current_count <= __n_total_slots_);
        __max_submissions = std::min(__max_submissions, __n_total_slots_ - __current_count);
        __submission_result __result{};
        __task* __op = nullptr;
        while (!__tasks.empty() && __result.__n_submitted < __max_submissions) {
          const __u32 __index = __tail & __mask_;
          ::io_uring_sqe& __sqe = __entries_[__index];
          __op = __tasks.pop_front();
          STDEXEC_ASSERT(__op->__vtable_);
          if (__op->__vtable_->__ready_(__op)) {
            __result.__ready.push_back(__op);
          } else {
            __op->__vtable_->__submit_(__op, __sqe);
#    ifdef STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
            __is_stopped = __is_stopped && __sqe.opcode != IORING_OP_ASYNC_CANCEL;
#    endif
            if (__is_stopped) {
              __stop(__op);
            } else {
              __sqe.user_data = bit_cast<__u64>(__op);
              __array_[__index] = __index;
              ++__result.__n_submitted;
              ++__tail;
            }
          }
        }
        __tail_.store(__tail, std::memory_order_release);
        while (!__tasks.empty()) {
          __op = __tasks.pop_front();
          if (__op->__vtable_->__ready_(__op)) {
            __result.__ready.push_back(__op);
          } else {
            __result.__pending.push_back(__op);
          }
        }
        return __result;
      }
    };

    class __completion_queue {
      __atomic_ref<__u32> __head_;
      __atomic_ref<__u32> __tail_;
      ::io_uring_cqe* __entries_;
      __u32 __mask_;
     public:
      explicit __completion_queue(
        const memory_mapped_region& __region,
        const ::io_uring_params& __params) noexcept
        : __head_{*__at_offset_as<__u32*>(__region.data(), __params.cq_off.head)}
        , __tail_{*__at_offset_as<__u32*>(__region.data(), __params.cq_off.tail)}
        , __entries_{__at_offset_as<::io_uring_cqe*>(__region.data(), __params.cq_off.cqes)}
        , __mask_{*__at_offset_as<__u32*>(__region.data(), __params.cq_off.ring_mask)} {
      }

      // This function first completes all tasks that are ready in the completion queue of the io_uring.
      // Then it completes all tasks that are ready in the given queue of ready tasks.
      // The function returns the number of previously submitted completed tasks.
      auto complete(stdexec::__intrusive_queue<&__task::__next_> __ready = __task_queue{}) noexcept
        -> int {
        __u32 __head = __head_.load(std::memory_order_relaxed);
        __u32 __tail = __tail_.load(std::memory_order_acquire);
        int __count = 0;
        while (__head != __tail) {
          const __u32 __index = __head & __mask_;
          const ::io_uring_cqe& __cqe = __entries_[__index];
          auto* __op = bit_cast<__task*>(__cqe.user_data);
          __op->__vtable_->__complete_(__op, __cqe);
          ++__head;
          ++__count;
          __tail = __tail_.load(std::memory_order_acquire);
        }
        __head_.store(__head, std::memory_order_release);
        while (!__ready.empty()) {
          __task* __op = __ready.pop_front();
          ::io_uring_cqe __dummy_cqe{};
          __dummy_cqe.user_data = bit_cast<__u64>(__op);
          __op->__vtable_->__complete_(__op, __dummy_cqe);
        }
        return __count;
      }
    };

    class __context;

    struct __wakeup_operation : __task {
      __context* __context_ = nullptr;
      int __eventfd_ = -1;
#    ifdef STDEXEC_HAS_IORING_OP_READ
      std::uint64_t __buffer_ = 0;
#    else
      std::uint64_t __value_ = 0;
      ::iovec __buffer_ = {.iov_base = &__value_, .iov_len = sizeof(__value_)};
#    endif

      static auto __ready_(__task*) noexcept -> bool {
        return false;
      }

      static void __submit_(__task* __pointer, ::io_uring_sqe& __entry) noexcept {
        __wakeup_operation& __self = *static_cast<__wakeup_operation*>(__pointer);
        __entry = ::io_uring_sqe{};
        __entry.fd = __self.__eventfd_;
        __entry.addr = bit_cast<__u64>(&__self.__buffer_);
#    ifdef STDEXEC_HAS_IORING_OP_READ
        __entry.opcode = IORING_OP_READ;
        __entry.len = sizeof(__self.__buffer_);
#    else
        __entry.opcode = IORING_OP_READV;
        __entry.len = 1;
#    endif
      }

      static void __complete_(__task* __pointer, const ::io_uring_cqe&) noexcept {
        __wakeup_operation& __self = *static_cast<__wakeup_operation*>(__pointer);
        __self.start();
      }

      static constexpr __task_vtable __vtable{&__ready_, &__submit_, &__complete_};

      __wakeup_operation(__context* __ctx, int __eventfd)
        : __task{__vtable}
        , __context_{__ctx}
        , __eventfd_{__eventfd} {
      }

      void start() & noexcept;
    };

    class __scheduler;

    enum class until {
      stopped,
      empty
    };

    class __context : __context_base {
     public:
      explicit __context(unsigned __entries = 1024, unsigned __flags = 0)
        : __context_base(std::max(__entries, 2u), __flags)
        , __completion_queue_{__completion_queue_region_ ? __completion_queue_region_ : __submission_queue_region_, __params_}
        , __submission_queue_{__submission_queue_region_, __submission_queue_entries_, __params_}
        , __wakeup_operation_{this, __eventfd_} {
      }

      auto try_wakeup() noexcept -> std::error_code {
        std::uint64_t __wakeup = 1;
        if (::write(__eventfd_, &__wakeup, sizeof(__wakeup)) == -1) {
          return {errno, std::system_category()};
        }
        return {};
      }

      void wakeup() {
        if (auto __ec = try_wakeup()) {
          STDEXEC_THROW(std::system_error{__ec});
        }
      }

      /// @brief Resets the io context to its initial state.
      void reset() {
        if (__is_running_.load(std::memory_order_relaxed) || __n_total_submitted_ > 0) {
          STDEXEC_THROW(
            std::runtime_error("exec::io_uring_context::reset() called on a running context"));
        }
        __n_submissions_in_flight_.store(0, std::memory_order_relaxed);
        __stop_source_.reset();
        __stop_source_.emplace();
      }

      auto request_stop() noexcept -> std::error_code {
        __stop_source_->request_stop();
        return try_wakeup();
      }

      auto stop_requested() const noexcept -> bool {
        return __stop_source_->stop_requested();
      }

      auto get_stop_token() const noexcept -> stdexec::inplace_stop_token {
        return __stop_source_->get_token();
      }

      auto is_running() const noexcept -> bool {
        return __is_running_.load(std::memory_order_relaxed);
      }

      /// @brief  Breaks out of the run loop of the io context without stopping the context.
      void finish() {
        __break_loop_.store(true, std::memory_order_release);
        wakeup();
      }

      /// \brief Submits the given task to the io_uring.
      /// \returns true if the task was submitted, false if this io context and this task is have been stopped.
      auto submit(__task* __op) noexcept -> bool {
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
          [[maybe_unused]]
          int __prev = __n_submissions_in_flight_.fetch_sub(1, std::memory_order_relaxed);
          STDEXEC_ASSERT(__prev > 0);
          return true;
        }
      }

      /// @brief Submit any pending tasks and complete any ready tasks.
      ///
      /// This function is not thread-safe and must only be called from the thread that drives the io context.
      void run_some() noexcept {
        __n_total_submitted_ -= __completion_queue_.complete();
        STDEXEC_ASSERT(
          0 <= __n_total_submitted_
          && std::cmp_less_equal(__n_total_submitted_, __params_.cq_entries));
        __u32 __max_submissions = __params_.cq_entries - static_cast<__u32>(__n_total_submitted_);
        __pending_.append(__requests_.pop_all_reversed());
        __submission_result __result = __submission_queue_.submit(
          static_cast<__task_queue&&>(__pending_),
          __max_submissions,
          __stop_source_->stop_requested());
        __n_total_submitted_ += __result.__n_submitted;
        __n_newly_submitted_ += __result.__n_submitted;
        STDEXEC_ASSERT(std::cmp_less_equal(__n_total_submitted_, __params_.cq_entries));
        __pending_ = static_cast<__task_queue&&>(__result.__pending);
        while (!__result.__ready.empty()) {
          __n_total_submitted_ -= __completion_queue_
                                    .complete(static_cast<__task_queue&&>(__result.__ready));
          STDEXEC_ASSERT(0 <= __n_total_submitted_);
          __pending_.append(__requests_.pop_all_reversed());
          __max_submissions = __params_.cq_entries - static_cast<__u32>(__n_total_submitted_);
          __result = __submission_queue_.submit(
            static_cast<__task_queue&&>(__pending_),
            __max_submissions,
            __stop_source_->stop_requested());
          __n_total_submitted_ += __result.__n_submitted;
          __n_newly_submitted_ += __result.__n_submitted;
          STDEXEC_ASSERT(std::cmp_less_equal(__n_total_submitted_, __params_.cq_entries));
          __pending_ = static_cast<__task_queue&&>(__result.__pending);
        }
      }

      void run_until_stopped() {
        bool expected_running = false;
        // Only one thread of execution is allowed to drive the io context.
        if (!__is_running_
               .compare_exchange_strong(expected_running, true, std::memory_order_relaxed)) {
          STDEXEC_THROW(
            std::runtime_error("exec::io_uring_context::run() called on a running context"));
        } else {
          // Check whether we restart the context after a context-wide stop.
          // We have to reset the stop source in this case.
          int __in_flight = __n_submissions_in_flight_.load(std::memory_order_relaxed);
          if (__in_flight == __no_new_submissions) {
            __stop_source_.emplace();
            // Make emplacement of stop source visible to other threads and open the door for new submissions.
            __n_submissions_in_flight_.store(0, std::memory_order_release);
          } else {
            // This can only happen for the very first pass of run_until_stopped()
            __wakeup_operation_.start();
          }
        }
        scope_guard __not_running{
          [&]() noexcept { __is_running_.store(false, std::memory_order_relaxed); }};
        __pending_.append(__requests_.pop_all_reversed());
        while (__n_total_submitted_ > 0 || !__pending_.empty()) {
          run_some();
          if (
            __n_total_submitted_ == 0
            || (__n_total_submitted_ == 1 && __break_loop_.load(std::memory_order_acquire))) {
            __break_loop_.store(false, std::memory_order_relaxed);
            break;
          }
          constexpr int __min_complete = 1;
          STDEXEC_ASSERT(
            0 <= __n_total_submitted_
            && std::cmp_less_equal(__n_total_submitted_, __params_.cq_entries));
          int rc = __io_uring_enter(
            __ring_fd_,
            static_cast<unsigned>(__n_newly_submitted_),
            __min_complete,
            IORING_ENTER_GETEVENTS);
          __throw_error_code_if(rc < 0 && rc != -EINTR, -rc);
          if (rc != -EINTR) {
            STDEXEC_ASSERT(rc <= __n_newly_submitted_);
            __n_newly_submitted_ -= rc;
          }
          __n_total_submitted_ -= __completion_queue_.complete();
          STDEXEC_ASSERT(0 <= __n_total_submitted_);
          __pending_.append(__requests_.pop_all_reversed());
        }
        STDEXEC_ASSERT(__n_total_submitted_ <= 1);
        if (__stop_source_->stop_requested() && __pending_.empty()) {
          STDEXEC_ASSERT(__n_total_submitted_ == 0);
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
          __pending_.append(__requests_.pop_all_reversed());
          __submission_result __result = __submission_queue_.submit(
            static_cast<__task_queue&&>(__pending_), __params_.cq_entries, true);
          STDEXEC_ASSERT(__result.__n_submitted == 0);
          STDEXEC_ASSERT(__result.__pending.empty());
          __completion_queue_.complete(static_cast<__task_queue&&>(__result.__ready));
        }
      }

      struct __on_stop {
        __context& __context_;

        void operator()() const noexcept {
          if (auto __ec = __context_.request_stop()) {
            std::terminate();
          }
        }
      };

      template <class _Rcvr>
      struct __run_op {
        using __id = __run_op;
        using __t = __run_op;
        _Rcvr __rcvr_;
        __context& __context_;
        until __mode_;

        using __on_stopped_callback = stdexec::stop_callback_for_t<
          stdexec::stop_token_of_t<stdexec::env_of_t<_Rcvr&>>,
          __on_stop
        >;

        void start() & noexcept {
          std::optional<__on_stopped_callback> __callback(
            std::in_place,
            stdexec::get_stop_token(stdexec::get_env(__rcvr_)),
            __on_stop{__context_});
          STDEXEC_TRY {
            if (__mode_ == until::stopped) {
              __context_.run_until_stopped();
            } else {
              __context_.run_until_empty();
            }
          }
          STDEXEC_CATCH_ALL {
            __callback.reset();
            stdexec::set_error(static_cast<_Rcvr&&>(__rcvr_), std::current_exception());
          }
          __callback.reset();
          if (__context_.stop_requested()) {
            stdexec::set_stopped(static_cast<_Rcvr&&>(__rcvr_));
          } else {
            stdexec::set_value(static_cast<_Rcvr&&>(__rcvr_));
          }
        }
      };

      class __run_sender {
       public:
        using sender_concept = stdexec::sender_t;
        using completion_signatures = stdexec::completion_signatures<
          stdexec::set_value_t(),
          stdexec::set_error_t(std::exception_ptr),
          stdexec::set_stopped_t()
        >;

        template <stdexec::receiver_of<completion_signatures> _Rcvr>
        auto connect(_Rcvr __rcvr) const noexcept -> __run_op<_Rcvr> {
          return {static_cast<_Rcvr&&>(__rcvr), *__context_, __mode_};
        }

       private:
        friend class __context;
        __context* __context_;
        until __mode_;

        explicit __run_sender(__context* __context, until __mode) noexcept
          : __context_{__context}
          , __mode_{__mode} {
        }
      };

      auto run(until __mode = until::stopped) -> __run_sender {
        return __run_sender{this, __mode};
      }

      void run_until_empty() {
        __break_loop_.store(true, std::memory_order_relaxed);
        run_until_stopped();
      }

      auto get_scheduler() noexcept -> __scheduler;

     private:
      friend struct __wakeup_operation;

      // This constant is used for __n_submissions_in_flight to indicate that no new submissions
      // to this context will be completed by this context.
      static constexpr int __no_new_submissions = -1;

      std::atomic<bool> __is_running_{false};
      std::atomic<int> __n_submissions_in_flight_{0};
      std::atomic<bool> __break_loop_{false};
      std::ptrdiff_t __n_total_submitted_{0};
      std::ptrdiff_t __n_newly_submitted_{0};
      std::optional<stdexec::inplace_stop_source> __stop_source_{std::in_place};
      __completion_queue __completion_queue_;
      __submission_queue __submission_queue_;
      __task_queue __pending_{};
      __atomic_task_queue __requests_{};
      __wakeup_operation __wakeup_operation_;
    };

    inline void __wakeup_operation::start() & noexcept {
      if (!__context_->__stop_source_->stop_requested()) {
        __context_->__pending_.push_front(this);
      }
    }

    template <class _Op>
    concept __io_task = requires(_Op& __op, ::io_uring_sqe& __sqe, const ::io_uring_cqe& __cqe) {
      { __op.context() } noexcept -> std::convertible_to<__context&>;
      { __op.ready() } noexcept -> std::convertible_to<bool>;
      { __op.submit(__sqe) } noexcept;
      { __op.complete(__cqe) } noexcept;
    };

    template <class _Op>
    concept __stoppable_task = __io_task<_Op> && requires(_Op& __op) {
      {
        static_cast<_Op &&>(__op).receiver()
      } noexcept -> stdexec::receiver_of<stdexec::completion_signatures<stdexec::set_stopped_t()>>;
    };

    template <__stoppable_task _Op>
    using __receiver_of_t = stdexec::__decay_t<decltype(std::declval<_Op&>().receiver())>;

    template <__io_task _Base>
    struct __io_task_facade : __task {
      static auto __ready_(__task* __pointer) noexcept -> bool {
        auto* __self = static_cast<__io_task_facade*>(__pointer);
        return __self->__base_.ready();
      }

      static void __submit_(__task* __pointer, ::io_uring_sqe& __sqe) noexcept {
        auto* __self = static_cast<__io_task_facade*>(__pointer);
        __self->__base_.submit(__sqe);
      }

      static void __complete_(__task* __pointer, const ::io_uring_cqe& __cqe) noexcept {
        auto* __self = static_cast<__io_task_facade*>(__pointer);
        __self->__base_.complete(__cqe);
      }

      static constexpr __task_vtable __vtable{&__ready_, &__submit_, &__complete_};

      template <class... _Args>
        requires stdexec::constructible_from<_Base, std::in_place_t, __task*, _Args...>
      __io_task_facade(std::in_place_t, _Args&&... __args)
        noexcept(stdexec::__nothrow_constructible_from<_Base, __task*, _Args...>)
        : __task{__vtable}
        , __base_(std::in_place, static_cast<__task*>(this), static_cast<_Args&&>(__args)...) {
      }

      template <class... _Args>
        requires stdexec::constructible_from<_Base, _Args...>
      __io_task_facade(std::in_place_t, _Args&&... __args)
        noexcept(stdexec::__nothrow_constructible_from<_Base, _Args...>)
        : __task{__vtable}
        , __base_(static_cast<_Args&&>(__args)...) {
      }

      auto base() noexcept -> _Base& {
        return __base_;
      }

      void start() & noexcept {
        __context& __context = __base_.context();
        if (__context.submit(this)) {
          if (auto __ec = __context.try_wakeup()) {
            std::terminate(); // TODO: handle error
          }
        }
      }

     private:
      _Base __base_;
    };

    template <class _ReceiverId>
    struct __schedule_operation {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __impl {
        __context& __context_;
        STDEXEC_ATTRIBUTE(no_unique_address) _Receiver __receiver_;

        __impl(__context& __context, _Receiver&& __receiver)
          : __context_{__context}
          , __receiver_{static_cast<_Receiver&&>(__receiver)} {
        }

        [[nodiscard]]
        auto context() const noexcept -> __context& {
          return __context_;
        }

        static constexpr auto ready() noexcept -> std::true_type {
          return {};
        }

        static constexpr void submit(::io_uring_sqe&) noexcept {
        }

        void complete(const ::io_uring_cqe& __cqe) noexcept {
          auto token = stdexec::get_stop_token(stdexec::get_env(__receiver_));
          if (__cqe.res == -ECANCELED || __context_.stop_requested() || token.stop_requested()) {
            stdexec::set_stopped(static_cast<_Receiver&&>(__receiver_));
          } else {

            stdexec::set_value(static_cast<_Receiver&&>(__receiver_));
          }
        }
      };

      using __t = __io_task_facade<__impl>;
    };

    template <class _Base>
    struct __stop_operation {
      class __t : public __task {
        _Base* __op_;
       public:
        static auto __ready_(__task*) noexcept -> bool {
          return false;
        }

        static void __submit_(__task* __pointer, ::io_uring_sqe& __sqe) noexcept {
          __t* __self = static_cast<__t*>(__pointer);
          __self->submit(__sqe);
        }

        static void __complete_(__task* __pointer, const ::io_uring_cqe& __cqe) noexcept {
          __t* __self = static_cast<__t*>(__pointer);
          __self->complete(__cqe);
        }

        void submit(::io_uring_sqe& __sqe) noexcept {
#    ifdef STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
          if constexpr (requires(_Base* __op, ::io_uring_sqe& __sqe) {
                          __op->submit_stop(__sqe);
                        }) {
            __op_->submit_stop(__sqe);
          } else {
            std::memset(&__sqe, 0, sizeof(__sqe));
            __sqe.opcode = IORING_OP_ASYNC_CANCEL;
            __sqe.addr = bit_cast<__u64>(__op_->__parent_);
          }
#    else
          __op_->submit_stop(__sqe);
#    endif
        }

        void complete(const ::io_uring_cqe&) noexcept {
          if (__op_->__n_ops_.fetch_sub(1, std::memory_order_relaxed) == 1) {
            __op_->__on_context_stop_.reset();
            __op_->__on_receiver_stop_.reset();
            stdexec::set_stopped((static_cast<_Base&&>(*__op_)).receiver());
          }
        }

        static constexpr __task_vtable __vtable{&__ready_, &__submit_, &__complete_};

        explicit __t(_Base* __op) noexcept
          : __task(__vtable)
          , __op_{__op} {
        }

        void start() & noexcept {
          int expected = 1;
          if (__op_->__n_ops_.compare_exchange_strong(expected, 2, std::memory_order_relaxed)) {
            if (__op_->context().submit(this)) {
              __op_->context().wakeup();
            }
          }
        }
      };
    };

    template <class _Base, bool _False>
    struct __impl_base {
      __task* __parent_;
      _Base __base_;

      template <class... _Args>
      __impl_base(__task* __parent, std::in_place_t, _Args&&... __args)
        noexcept(stdexec::__nothrow_constructible_from<_Base, _Args...>)
        : __parent_{__parent}
        , __base_(static_cast<_Args&&>(__args)...) {
      }
    };

    template <class _Base>
    struct __impl_base<_Base, true> {
      __task* __parent_;
      _Base __base_;

      template <class... _Args>
      __impl_base(__task* __parent, std::in_place_t, _Args&&... __args)
        noexcept(stdexec::__nothrow_constructible_from<_Base, _Args...>)
        : __parent_{__parent}
        , __base_(static_cast<_Args&&>(__args)...) {
      }

      void submit_stop(::io_uring_sqe& __sqe) noexcept {
        __base_.submit_stop(__sqe);
      }
    };

    template <__stoppable_task _Base>
    struct __stoppable_task_facade {
      using _Receiver = __receiver_of_t<_Base>;

      template <class _Ty>
      static constexpr bool __has_submit_stop_v = requires(_Ty& __base, ::io_uring_sqe& __sqe) {
        __base.submit_stop(__sqe);
      };

      using __base_t = __impl_base<_Base, __has_submit_stop_v<_Base>>;

      struct __impl : __base_t {
        struct __stop_callback {
          __impl* __self_;

          void operator()() noexcept {
            __self_->__stop_operation_.start();
          }
        };

        using __on_context_stop_t = std::optional<stdexec::inplace_stop_callback<__stop_callback>>;
        using __on_receiver_stop_t = std::optional<typename stdexec::stop_token_of_t<
          stdexec::env_of_t<_Receiver>&
        >::template callback_type<__stop_callback>>;

        stdexec::__t<__stop_operation<__impl>> __stop_operation_;
        std::atomic<int> __n_ops_{0};
        __on_context_stop_t __on_context_stop_{};
        __on_receiver_stop_t __on_receiver_stop_{};

        template <class... _Args>
          requires stdexec::constructible_from<_Base, _Args...>
        __impl(std::in_place_t, __task* __parent, _Args&&... __args)
          noexcept(stdexec::__nothrow_constructible_from<_Base, _Args...>)
          : __base_t(__parent, std::in_place, static_cast<_Args&&>(__args)...)
          , __stop_operation_{this} {
        }

        auto context() noexcept -> __context& {
          return this->__base_.context();
        }

        auto receiver() & noexcept -> _Receiver& {
          return this->__base_.receiver();
        }

        auto receiver() && noexcept -> _Receiver&& {
          return static_cast<_Receiver&&>(this->__base_.receiver());
        }

        [[nodiscard]]
        auto ready() const noexcept -> bool {
          return this->__base_.ready();
        }

        void submit(::io_uring_sqe& __sqe) noexcept {
          [[maybe_unused]]
          int prev = __n_ops_.fetch_add(1, std::memory_order_relaxed);
          STDEXEC_ASSERT(prev == 0);
          __context& __context_ = this->__base_.context();
          _Receiver& __receiver = this->__base_.receiver();
          __on_context_stop_.emplace(__context_.get_stop_token(), __stop_callback{this});
          __on_receiver_stop_
            .emplace(stdexec::get_stop_token(stdexec::get_env(__receiver)), __stop_callback{this});
          this->__base_.submit(__sqe);
        }

        void complete(const ::io_uring_cqe& __cqe) noexcept {
          if (__n_ops_.fetch_sub(1, std::memory_order_relaxed) == 1) {
            __on_context_stop_.reset();
            __on_receiver_stop_.reset();
            _Receiver& __receiver = this->__base_.receiver();
            __context& __context_ = this->__base_.context();
            auto token = stdexec::get_stop_token(stdexec::get_env(__receiver));
            if (__cqe.res == -ECANCELED || __context_.stop_requested() || token.stop_requested()) {
              stdexec::set_stopped(static_cast<_Receiver&&>(__receiver));
            } else {
              this->__base_.complete(__cqe);
            }
          }
        }
      };

      using __t = __io_task_facade<__impl>;
    };

    template <class _Base>
    using __stoppable_task_facade_t = stdexec::__t<__stoppable_task_facade<_Base>>;

    template <class _Receiver>
    struct __stoppable_op_base {
      __context& __context_;
      _Receiver __receiver_;

      auto receiver() & noexcept -> _Receiver& {
        return __receiver_;
      }

      auto receiver() && noexcept -> _Receiver&& {
        return static_cast<_Receiver&&>(__receiver_);
      }

      auto context() noexcept -> __context& {
        return __context_;
      }
    };

    template <class _ReceiverId>
    struct __schedule_after_operation {
      using _Receiver = stdexec::__t<_ReceiverId>;

      class __impl : public __stoppable_op_base<_Receiver> {
#    ifdef STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
        struct __kernel_timespec {
          __s64 __tv_sec;
          __s64 __tv_nsec;
        };

        __kernel_timespec __duration_;

        static constexpr auto
          __duration_to_timespec(std::chrono::nanoseconds dur) noexcept -> __kernel_timespec {
          auto secs = std::chrono::duration_cast<std::chrono::seconds>(dur);
          dur -= secs;
          secs = std::max(secs, std::chrono::seconds{0});
          dur = std::clamp(dur, std::chrono::nanoseconds{0}, std::chrono::nanoseconds{999'999'999});
          return __kernel_timespec{secs.count(), dur.count()};
        }
#    else
        safe_file_descriptor __timerfd_;
        ::itimerspec __duration_;
        std::uint64_t __n_expirations_{0};
        ::iovec __iov_{&__n_expirations_, sizeof(__n_expirations_)};

        static constexpr ::itimerspec
          __duration_to_timespec(std::chrono::nanoseconds __nsec) noexcept {
          ::itimerspec __timerspec{};
          ::clock_gettime(CLOCK_REALTIME, &__timerspec.it_value);
          __nsec = std::chrono::nanoseconds{__timerspec.it_value.tv_nsec} + __nsec;
          auto __sec = std::chrono::duration_cast<std::chrono::seconds>(__nsec);
          __nsec -= __sec;
          __nsec =
            std::clamp(__nsec, std::chrono::nanoseconds{0}, std::chrono::nanoseconds{999'999'999});
          __timerspec.it_value.tv_sec += __sec.count();
          __timerspec.it_value.tv_nsec = __nsec.count();
          STDEXEC_ASSERT(
            0 <= __timerspec.it_value.tv_nsec && __timerspec.it_value.tv_nsec < 1'000'000'000);
          return __timerspec;
        }
#    endif

       public:
        static constexpr auto ready() noexcept -> std::false_type {
          return {};
        }

#    ifndef STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
        void submit_stop(::io_uring_sqe& __sqe) noexcept {
          __duration_.it_value.tv_sec = 1;
          __duration_.it_value.tv_nsec = 0;
          ::timerfd_settime(
            __timerfd_, TFD_TIMER_ABSTIME | TFD_TIMER_CANCEL_ON_SET, &__duration_, nullptr);
          __sqe = ::io_uring_sqe{.opcode = IORING_OP_NOP};
        }
#    endif

        void submit(::io_uring_sqe& __sqe) noexcept {
#    ifdef STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
          ::io_uring_sqe __sqe_{};
          __sqe_.opcode = IORING_OP_TIMEOUT;
          __sqe_.addr = bit_cast<__u64>(&__duration_);
          __sqe_.len = 1;
          __sqe = __sqe_;
#    else
          ::io_uring_sqe __sqe_{};
          __sqe_.opcode = IORING_OP_READV;
          __sqe_.fd = __timerfd_;
          __sqe_.addr = bit_cast<__u64>(&__iov_);
          __sqe_.len = 1;
          __sqe = __sqe_;
#    endif
        }

        void complete(const ::io_uring_cqe& __cqe) noexcept {
          bool const __success =
#    ifdef STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
            __cqe.res == -ETIME || __cqe.res == 0
#    else
            __cqe.res == sizeof(std::uint64_t)
#    endif
            ;
          if (__success) {
            stdexec::set_value(static_cast<_Receiver&&>(this->__receiver_));
          } else {
            STDEXEC_ASSERT(__cqe.res < 0);
            stdexec::set_error(
              static_cast<_Receiver&&>(this->__receiver_),
              std::make_exception_ptr(std::system_error(-__cqe.res, std::system_category())));
          }
        }

        __impl(__context& __context, std::chrono::nanoseconds __duration, _Receiver&& __receiver)
          : __stoppable_op_base<_Receiver>{__context, static_cast<_Receiver&&>(__receiver)}
#    ifdef STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
          , __duration_{__duration_to_timespec(__duration)}
#    else
          , __timerfd_{::timerfd_create(CLOCK_REALTIME, 0)}
          , __duration_{__duration_to_timespec(__duration)}
#    endif
        {
#    ifndef STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
          int __rc = ::timerfd_settime(
            __timerfd_, TFD_TIMER_ABSTIME | TFD_TIMER_CANCEL_ON_SET, &__duration_, nullptr);
          __throw_error_code_if(__rc < 0, errno);
#    endif
        }
      };

      using __t = __stoppable_task_facade_t<__impl>;
    };

    class __scheduler {
     public:
      __context* __context_;

      auto operator==(const __scheduler&) const -> bool = default;

      struct __schedule_env {
        [[nodiscard]]
        auto query(stdexec::get_completion_scheduler_t<stdexec::set_value_t>) const noexcept
          -> __scheduler {
          return __scheduler{__context_};
        }

        __context* __context_;
      };

      class __schedule_sender {
        using __completion_sigs =
          stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_stopped_t()>;

        __schedule_env __env_;

       public:
        using sender_concept = stdexec::sender_t;
        using __id = __schedule_sender;
        using __t = __schedule_sender;

        explicit __schedule_sender(__schedule_env __env) noexcept
          : __env_{__env} {
        }

        [[nodiscard]]
        auto get_env() const noexcept -> __schedule_env {
          return __env_;
        }

        [[nodiscard]]
        auto get_completion_signatures(stdexec::__ignore = {}) const noexcept -> __completion_sigs {
          return {};
        }

        template <stdexec::receiver_of<__completion_sigs> _Receiver>
        auto connect(_Receiver __receiver)
          const & -> stdexec::__t<__schedule_operation<stdexec::__id<_Receiver>>> {
          return stdexec::__t<__schedule_operation<stdexec::__id<_Receiver>>>(
            std::in_place, *__env_.__context_, static_cast<_Receiver&&>(__receiver));
        }
      };

      class __schedule_after_sender {
        using __completion_sigs = stdexec::completion_signatures<
          stdexec::set_value_t(),
          stdexec::set_error_t(std::exception_ptr),
          stdexec::set_stopped_t()
        >;

       public:
        using sender_concept = stdexec::sender_t;
        using __id = __schedule_after_sender;
        using __t = __schedule_after_sender;

        __schedule_env __env_;
        std::chrono::nanoseconds __duration_;

        [[nodiscard]]
        auto get_env() const noexcept -> __schedule_env {
          return __env_;
        }

        template <class... _Env>
        static auto get_completion_signatures(const __schedule_after_sender&, _Env&&...) noexcept
          -> __completion_sigs {
          return {};
        }

        template <stdexec::receiver_of<__completion_sigs> _Receiver>
        auto connect(_Receiver __receiver)
          const & -> stdexec::__t<__schedule_after_operation<stdexec::__id<_Receiver>>> {
          return stdexec::__t<__schedule_after_operation<stdexec::__id<_Receiver>>>(
            std::in_place, *__env_.__context_, __duration_, static_cast<_Receiver&&>(__receiver));
        }
      };

      [[nodiscard]]
      auto schedule() const -> __schedule_sender {
        return __schedule_sender{__schedule_env{__context_}};
      }

      [[nodiscard]]
      auto now() const noexcept -> std::chrono::time_point<std::chrono::steady_clock> {
        return std::chrono::steady_clock::now();
      }

      [[nodiscard]]
      auto schedule_after(std::chrono::nanoseconds __duration) const -> __schedule_after_sender {
        return __schedule_after_sender{.__env_ = {__context_}, .__duration_ = __duration};
      }

      template <class _Clock, class _Duration>
      [[nodiscard]]
      auto schedule_at(const std::chrono::time_point<_Clock, _Duration>& __time_point) const
        -> __schedule_after_sender {
        auto __duration = __time_point - _Clock::now();
        return __schedule_after_sender{.__env_ = {__context_}, .__duration_ = __duration};
      }
    };

    inline auto __context::get_scheduler() noexcept -> __scheduler {
      return __scheduler{this};
    }
  } // namespace __io_uring

  using __io_uring::until;
  using io_uring_context = __io_uring::__context;
  using io_uring_scheduler = __io_uring::__scheduler;

  static_assert(__timed_scheduler<io_uring_scheduler>);

} // namespace exec

#  endif // if __has_include(<linux/verison.h>)
#endif   // if __has_include(<linux/io_uring.h>)
