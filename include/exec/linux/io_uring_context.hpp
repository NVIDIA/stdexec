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

#include "../../stdexec/execution.hpp"

#include "__detail/atomic_intrusive_queue.hpp"
#include "safe_file_descriptor.hpp"
#include "memory_mapped_region.hpp"

#if !__has_include(<linux/io_uring.h>)
#error "io_uring.h not found. Your kernel is probably too old."
#else
#include <linux/io_uring.h>
#endif

namespace exec {
  namespace __io_uring {
    struct __context_base : stdexec::__immovable {
      explicit __context_base(unsigned __entries, unsigned __flags = 0);

      memory_mapped_region __submission_queue_region_{};
      memory_mapped_region __completion_queue_region_{};
      memory_mapped_region __submission_queue_entries_{};
      ::io_uring_params __params_{};
      safe_file_descriptor __ring_fd_{};
      safe_file_descriptor __eventfd_{};
    };

    struct __task_vtable {
      bool (*__ready_)(void*) noexcept;
      // __suspend_?
      void (*__submit_)(void*, ::io_uring_sqe*) noexcept;
      // __resume_?
      void (*__complete_)(void*, const ::io_uring_cqe*) noexcept;
    };

    struct __task {
      const __task_vtable* __vtable_;
      __task* __next_;
    };

    struct __submission_result {
      int __n_submitted_;
      stdexec::__intrusive_queue<&__task::__next_> __pending_;
      stdexec::__intrusive_queue<&__task::__next_> __ready_;
    };

    class __submission_queue {
      std::atomic_ref<__u32> __head_;
      std::atomic_ref<__u32> __tail_;
      std::atomic_ref<__u32> __mask_;
      ::io_uring_sqe* __entries_;
     public:
      explicit __submission_queue(
        const memory_mapped_region& __region,
        const memory_mapped_region& __sqes_region,
        const ::io_uring_params& __params);

      __submission_result submit(stdexec::__intrusive_queue<&__task::__next_> __task) noexcept;
    };

    class __completion_queue {
      std::atomic_ref<__u32> __head_;
      std::atomic_ref<__u32> __tail_;
      std::atomic_ref<__u32> __mask_;
      ::io_uring_cqe* __entries_;
     public:
      explicit __completion_queue(
        const memory_mapped_region& __region,
        const ::io_uring_params& __params);

      int complete(stdexec::__intrusive_queue<&__task::__next_> __ready) noexcept;
    };

    class __context;

    struct __wakeup_operation : __task {
      __context* __context_ = nullptr;
      int __eventfd_ = -1;
      uint64_t __buffer_ = 0;

      static bool __ready_(void*) noexcept;

      static void __submit_(void* __pointer, ::io_uring_sqe* __entry) noexcept;

      static void __complete_(void* __pointer, const ::io_uring_cqe* __entry) noexcept;

      static constexpr __task_vtable __vtable{&__ready_, &__submit_, &__complete_};

      __wakeup_operation(__context* __context_, int __eventfd);

      void start() noexcept;
    };

    class __context : __context_base {
     public:
      explicit __context(unsigned __entries, unsigned __flags = 0);

      void run();

      void submit(__task* __op) noexcept;

     private:
      friend struct __wakeup_operation;

      stdexec::in_place_stop_source __stop_source_{};
      __completion_queue __completion_queue_;
      __submission_queue __submission_queue_;
      stdexec::__intrusive_queue<&__task::__next_> __pending_{};
      __atomic_intrusive_queue<&__task::__next_> __requests_{};
      std::ptrdiff_t __n_submitted_{};
      __wakeup_operation __wakeup_operation_;
    };
  }

  using io_uring_context = __io_uring::__context;
}

#include "__detail/io_uring_context.hpp"