/*
 * Copyright (c) 2023 Lee Howes, Lucian Radu Teodorescu
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

#include "stdexec/execution.hpp"
#include "exec/static_thread_pool.hpp"

STDEXEC_PRAGMA_PUSH()
// "warning: empty struct has size 0 in C, size 1 in C++"
// The intended use of these two structs is as a base class for other structs.
// When used like that, the size occupied by these structs is 0 because of the
// empty base class optimization.
STDEXEC_PRAGMA_IGNORE_GNU("-Wextern-c-compat")

#include "__system_context_if.h"

STDEXEC_PRAGMA_POP()

namespace exec::__detail {
  using stdexec::__declval;
  using __pool_scheduler_t =
    stdexec::__decay_t<decltype(__declval<exec::static_thread_pool&>().get_scheduler())>;

  /// Receiver that calls the callback when the operation completes.
  template <class _Sender, bool _IsBulk>
  struct __operation;

  template <class _Sender, bool _IsBulk>
  struct __recv {
    using receiver_concept = stdexec::receiver_t;

    /// The callback to be called.
    system_context_completion_callback __cb_;

    /// The data to be passed to the callback.
    void* __data_;

    /// The owning operation state, to be destructed when the operation completes.
    __operation<_Sender, _IsBulk>* __op_;

    void set_value() noexcept {
      __cb_(__data_, 0, nullptr);
    }

    void set_error(std::exception_ptr __ptr) noexcept {
      __cb_(__data_, 2, *reinterpret_cast<void**>(&__ptr));
    }

    void set_stopped() noexcept {
      __cb_(__data_, 1, nullptr);
    }
  };

  template <class _Sender, bool _IsBulk>
  struct __operation
    : stdexec::__if_c<_IsBulk, system_bulk_operation_state, system_operation_state> {
    /// The inner operation state, that results out of connecting the underlying sender with the receiver.
    stdexec::connect_result_t<_Sender, __recv<_Sender, _IsBulk>> __inner_op_;
    /// True if the operation is on the heap, false if it is in the preallocated space.
    bool __on_heap_;

    /// Try to construct the operation in the preallocated memory if it fits, otherwise allocate a new operation.
    static __operation* __construct_maybe_alloc(
      void* __preallocated,
      size_t __psize,
      _Sender __sndr,
      system_context_completion_callback __cb,
      void* __data) {
      if (__preallocated == nullptr || __psize < sizeof(__operation)) {
        return ::new __operation(std::move(__sndr), __cb, __data, true);
      } else {
        return ::new (__preallocated) __operation(std::move(__sndr), __cb, __data, false);
      }
    }

    /// Destructs the operation; frees any allocated memory.
    static void operator delete(__operation* __self, std::destroying_delete_t) noexcept {
      auto on_heap = __self->__on_heap_;
      std::destroy_at(__self);
      if (on_heap) {
        ::operator delete(__self);
      }
    }

   private:
    __operation(
      _Sender __sndr,
      system_context_completion_callback __cb,
      void* __data,
      bool __on_heap) noexcept(stdexec::__nothrow_connectable<_Sender, __recv<_Sender, _IsBulk>>) //
      : __inner_op_(
          stdexec::connect(std::move(__sndr), __recv<_Sender, _IsBulk>{__cb, __data, this}))
      , __on_heap_(__on_heap) {
    }
  };

  struct __system_scheduler_impl : exec::system_scheduler_base {
   private:
    /// Scheduler from the underlying thread pool.
    __pool_scheduler_t __pool_scheduler_;

    struct __bulk_functor {
      system_context_bulk_item_callback __cb_item_;
      void* __data_;

      void operator()(uint64_t __idx) const noexcept {
        __cb_item_(__data_, __idx);
      }
    };

   public:
    static constexpr stdexec::forward_progress_guarantee forward_progress_guarantee =
      stdexec::forward_progress_guarantee::parallel;

    explicit __system_scheduler_impl(exec::static_thread_pool& __pool) noexcept
      : exec::system_scheduler_base(this)
      , __pool_scheduler_{__pool.get_scheduler()} {
    }

    using schedule_operation = //
      __operation<stdexec::__result_of<stdexec::schedule, __pool_scheduler_t>, false>;

    using bulk_schedule_operation = //
      __operation<
        stdexec::__result_of<
          stdexec::bulk,
          stdexec::__result_of<stdexec::schedule, __pool_scheduler_t&>,
          uint64_t,
          __bulk_functor>,
        true>;

    system_operation_state* schedule(
      void* __preallocated,
      uint32_t __psize,
      system_context_completion_callback __cb,
      void* __data) noexcept {
      auto __sndr = stdexec::schedule(__pool_scheduler_);
      auto __os = schedule_operation::__construct_maybe_alloc(
        __preallocated, __psize, std::move(__sndr), __cb, __data);
      stdexec::start(__os->__inner_op_);
      return __os;
    }

    system_bulk_operation_state* bulk_schedule(
      void* __preallocated,
      uint32_t __psize,
      system_context_completion_callback __cb,
      system_context_bulk_item_callback __cb_item,
      void* __data,
      uint64_t __size) noexcept {
      auto __sndr = stdexec::bulk(
        stdexec::schedule(__pool_scheduler_), __size, __bulk_functor{__cb_item, __data});
      auto __os = bulk_schedule_operation::__construct_maybe_alloc(
        __preallocated, __psize, std::move(__sndr), __cb, __data);
      stdexec::start(__os->__inner_op_);
      return __os;
    }
  };

  /// Default implementation of a system context, based on `static_thread_pool`
  struct __system_context_impl : exec::system_context_base {
    __system_context_impl() noexcept
      : exec::system_context_base(this) {
    }

    system_scheduler_interface* get_scheduler() noexcept {
      return &__scheduler_;
    }

   private:
    /// The underlying thread pool.
    exec::static_thread_pool __pool_{};

    /// The system scheduler implementation.
    __system_scheduler_impl __scheduler_{__pool_};
  };
} // namespace exec::__detail
