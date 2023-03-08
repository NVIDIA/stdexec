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

#include "./__detail/atomic_intrusive_queue.hpp"
#include "./__detail/__atomic_ref.hpp"
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

#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 6, 0)
#include <sys/uio.h>
#else
#define STDEXEC_IORING_OP_READ
#endif

#include <cstring>

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

    struct __task : stdexec::__immovable {
      const __task_vtable* __vtable_;
      __task* __next_{nullptr};

      explicit __task(const __task_vtable& __vtable)
        : __vtable_{&__vtable} {
      }
    };

    struct __submission_result {
      __u32 __n_submitted_;
      stdexec::__intrusive_queue<&__task::__next_> __pending_;
      stdexec::__intrusive_queue<&__task::__next_> __ready_;
    };

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

      __submission_result submit(stdexec::__intrusive_queue<&__task::__next_> __task) noexcept;
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

      int complete(stdexec::__intrusive_queue<&__task::__next_> __ready) noexcept;
    };

    class __context;

    struct __wakeup_operation : __task {
      __context* __context_ = nullptr;
      int __eventfd_ = -1;
#ifdef STDEXEC_IORING_OP_READ
      std::uint64_t __buffer_ = 0;
#else
      std::uint64_t __value_ = 0;
      ::iovec __buffer_ = {.iov_base = &__value_, .iov_len = sizeof(__value_)};
#endif


      static bool __ready_(void*) noexcept;

      static void __submit_(void* __pointer, ::io_uring_sqe* __entry) noexcept;

      static void __complete_(void* __pointer, const ::io_uring_cqe* __entry) noexcept;

      static constexpr __task_vtable __vtable{&__ready_, &__submit_, &__complete_};

      __wakeup_operation(__context* __context_, int __eventfd);

      void start() noexcept;
    };

    class __scheduler;

    class __context : __context_base {
     public:
      explicit __context(unsigned __entries, unsigned __flags = 0);

      void run();

      void wakeup();

      void request_stop();

      bool stop_requested();

      void submit(__task* __op) noexcept;

      __scheduler get_scheduler() noexcept;

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

    template <class _ReceiverId>
    class __schedule_operation : __task {
      using _Receiver = stdexec::__t<_ReceiverId>;
      __context* __context_;
      _Receiver __receiver_;

      static bool __ready_(void*) noexcept {
        return true;
      }

      static void __submit_(void*, ::io_uring_sqe*) noexcept {
      }

      static void __complete_(void* __pointer, const ::io_uring_cqe*) noexcept {
        auto __self = static_cast<__schedule_operation*>(__pointer);
        if (__self->__context_->stop_requested()) {
          stdexec::set_stopped((_Receiver&&) __self->__receiver_);
        } else {
          stdexec::set_value((_Receiver&&) __self->__receiver_);
        }
      }

      static constexpr __task_vtable __vtable{&__ready_, &__submit_, &__complete_};

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
        : __task(__vtable)
        , __context_{__context}
        , __receiver_{(_Receiver&&) __receiver} {
      }
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

    template <class _ReceiverId>
    class __schedule_after_operation : __task {
      using _Receiver = stdexec::__t<_ReceiverId>;
      __context* __context_;
      _Receiver __receiver_;
      ::timespec __duration_;

      static constexpr ::timespec __duration_to_timespec(std::chrono::nanoseconds dur) noexcept {
        auto secs = std::chrono::duration_cast<std::chrono::seconds>(dur);
        dur -= secs;
        return ::timespec{secs.count(), dur.count()};
      }

      static bool __ready_(void*) noexcept {
        return false;
      }

      static void __submit_(void* __pointer, ::io_uring_sqe* __sqe) noexcept {
        auto __self = static_cast<__schedule_after_operation*>(__pointer);
        std::memset(__sqe, 0, sizeof(*__sqe));
        __sqe->opcode = IORING_OP_TIMEOUT;
        __sqe->addr = std::bit_cast<__u64>(&__self->__duration_);
        __sqe->len = 1;
      }

      static void __complete_(void* __pointer, const ::io_uring_cqe* __cqe) noexcept {
        auto __self = static_cast<__schedule_after_operation*>(__pointer);
        if (__self->__context_->stop_requested() || __cqe->res == -ECANCELED) {
          stdexec::set_stopped((_Receiver&&) __self->__receiver_);
        } else if (__cqe->res == -ETIME || __cqe->res == 0) {
          stdexec::set_value((_Receiver&&) __self->__receiver_);
        } else {
          stdexec::set_error(
            (_Receiver&&) __self->__receiver_,
            std::make_exception_ptr(std::system_error(-__cqe->res, std::system_category())));
        }
      }

      static constexpr __task_vtable __vtable{&__ready_, &__submit_, &__complete_};

      friend void tag_invoke(stdexec::start_t, __schedule_after_operation& __self) noexcept {
        if (__self.__context_->stop_requested()) {
          stdexec::set_stopped((_Receiver&&) __self.__receiver_);
        } else {
          __self.__context_->submit(&__self);
          __self.__context_->wakeup();
        }
      }

     public:
      __schedule_after_operation(
        __context* __context,
        std::chrono::nanoseconds __duration,
        _Receiver&& __receiver)
        : __task(__vtable)
        , __context_{__context}
        , __receiver_{(_Receiver&&) __receiver}
        , __duration_{__duration_to_timespec(__duration)} {
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
        return {
          __sender.__env_.__sched_.__context_, __sender.__duration_, (_Receiver&&) __receiver};
      }
    };

    inline __schedule_after_sender tag_invoke(
      exec::schedule_after_t,
      const __scheduler& __sched,
      std::chrono::nanoseconds __duration) {
      return __schedule_after_sender{.__env_ = {.__sched_ = __sched}, .__duration_ = __duration};
    }
  }

  using io_uring_context = __io_uring::__context;
}

#include "./__detail/io_uring_context.hpp"