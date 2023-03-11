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

#include "../../stdexec/execution.hpp"

#include "../__detail/__atomic_intrusive_queue.hpp"
#include "../__detail/__atomic_ref.hpp"

#include "./safe_file_descriptor.hpp"
#include "./memory_mapped_region.hpp"

#if !__has_include(<linux/io_uring.h>)
#error "io_uring.h not found. Your kernel is probably too old."
#else
#include <linux/io_uring.h>
#endif

#if !__has_include(<linux/version.h>)
#error "linux/version.h not found. Do you use Linux?"
#else
#include <linux/version.h>
#endif

#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 5, 0)
#warning "Your kernel is too old to support io_uring with cancellation support."
#else
#define STDEXEC_HAS_IO_URING_ASYNC_CANCELLATION
#endif

#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 6, 0)
#include <sys/uio.h>
#else
#define STDEXEC_HAS_IORING_OP_READ
#endif

namespace exec {
  struct schedule_after_t {
    template <class _Scheduler, class _Duration>
      requires stdexec::tag_invocable<schedule_after_t, const _Scheduler&, _Duration>
    auto operator()(const _Scheduler& __scheduler, _Duration __duration) const
      -> stdexec::tag_invoke_result_t<schedule_after_t, const _Scheduler&, _Duration> {
      return tag_invoke(*this, __scheduler, __duration);
    }
  };

  inline constexpr schedule_after_t schedule_after{};

  namespace __io_uring {
    // This base class maps the kernel's io_uring data structures into the process.
    struct __context_base : stdexec::__immovable {
      explicit __context_base(unsigned __entries, unsigned __flags = 0);

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
      void (*__submit_)(__task*, ::io_uring_sqe*) noexcept;
      // This function is called when the io operation is completed.
      // The status of the operation is passed as a parameter.
      void (*__complete_)(__task*, const ::io_uring_cqe*) noexcept;
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

    struct __submission_result {
      __u32 __n_submitted;
      stdexec::__intrusive_queue<&__task::__next_> __pending;
      stdexec::__intrusive_queue<&__task::__next_> __ready;
    };

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
        const ::io_uring_params& __params);

      // This function submits the given queue of tasks to the io_uring.
      //
      // Each task that is ready to be completed is moved to the __ready queue.
      // If the submission queue gets full before all tasks are submitted, the
      // remaining tasks are moved to the __pending queue.
      // If is_stopped is true, no new tasks are submitted to the io_uring unless it is a cancellation.
      // If is_stopped is true and a task is not ready to be completed, the task is completed with
      // an io_uring_cqe object with the result field set to -ECANCELED.
      __submission_result
        submit(stdexec::__intrusive_queue<&__task::__next_> __task, bool is_stopped) noexcept;
    };

    class __completion_queue {
      __atomic_ref<__u32> __head_;
      __atomic_ref<__u32> __tail_;
      ::io_uring_cqe* __entries_;
      __u32 __mask_;
     public:
      explicit __completion_queue(
        const memory_mapped_region& __region,
        const ::io_uring_params& __params);

      // This function first completes all tasks that are ready in the completion queue of the io_uring.
      // Then it completes all tasks that are ready in the given queue of ready tasks.
      // The function returns the number of previously submitted completed tasks.
      int complete(stdexec::__intrusive_queue<&__task::__next_> __ready) noexcept;
    };

    class __context;

    struct __wakeup_operation : __task {
      __context* __context_ = nullptr;
      int __eventfd_ = -1;
#ifdef STDEXEC_HAS_IORING_OP_READ
      std::uint64_t __buffer_ = 0;
#else
      std::uint64_t __value_ = 0;
      ::iovec __buffer_ = {.iov_base = &__value_, .iov_len = sizeof(__value_)};
#endif

      static bool __ready_(__task*) noexcept;

      static void __submit_(__task* __pointer, ::io_uring_sqe* __entry) noexcept;

      static void __complete_(__task* __pointer, const ::io_uring_cqe* __entry) noexcept;

      static constexpr __task_vtable __vtable{&__ready_, &__submit_, &__complete_};

      __wakeup_operation(__context* __context_, int __eventfd);

      void start() noexcept;
    };

    class __scheduler;

    class __context : __context_base {
     public:
      explicit __context(unsigned __entries = 1024, unsigned __flags = 0);

      void run();

      void wakeup();

      void request_stop();

      bool stop_requested() const noexcept;

      bool is_running() const noexcept;

      stdexec::in_place_stop_token get_stop_token() const noexcept;

      /// \brief Submits the given task to the io_uring.
      /// \returns true if the task was submitted, false if this io context and this task is have been stopped.
      bool submit(__task* __op) noexcept;

      __scheduler get_scheduler() noexcept;

     private:
      friend struct __wakeup_operation;

      // This constant is used for __n_submissions_in_flight to indicate that no new submissions
      // to this context will be completed by this context.
      static constexpr int __no_new_submissions = -1;

      std::atomic<bool> __is_running_{false};
      std::atomic<int> __n_submissions_in_flight_{0};
      std::optional<stdexec::in_place_stop_source> __stop_source_{std::in_place};
      __completion_queue __completion_queue_;
      __submission_queue __submission_queue_;
      stdexec::__intrusive_queue<&__task::__next_> __pending_{};
      __atomic_intrusive_queue<&__task::__next_> __requests_{};
      __wakeup_operation __wakeup_operation_;
    };

    class __schedule_after_sender;
    class __schedule_sender;

    class __scheduler {
     public:
      __context* __context_;
     private:
      friend __schedule_sender tag_invoke(stdexec::schedule_t, const __scheduler& __sched);

      friend __schedule_after_sender tag_invoke(
        exec::schedule_after_t,
        const __scheduler& __sched,
        std::chrono::nanoseconds __duration);
    };
  }

  using io_uring_context = __io_uring::__context;
}

#include "./__detail/io_uring_context.hpp"