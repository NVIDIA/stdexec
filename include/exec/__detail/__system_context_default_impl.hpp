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

#include "__system_context_if.h"
#include "stdexec/execution.hpp"
#include "exec/static_thread_pool.hpp"

namespace exec::__system_context_default_impl {

  // TODO: move default implementation to weak pointers

  using __pool_scheduler_t = decltype(std::declval<exec::static_thread_pool>().get_scheduler());

  /// Receiver that calls the callback when the operation completes.
  template <class __Sender>
  struct __operation;

  template <class __Sender>
  struct __recv {
    using receiver_concept = stdexec::receiver_t;

    /// The callback to be called.
    __exec_system_context_completion_callback_t __cb_;

    /// The data to be passed to the callback.
    void* __data_;

    /// The owning operation state, to be destructed when the operation completes.
    __operation<__Sender>* __op_;
    /// True if the operation is on the heap, false if it is in the preallocated space.
    bool __on_heap_;

    friend void tag_invoke(stdexec::set_value_t, __recv&& __self) noexcept {
      __self.__cb_(__self.__data_, 0, nullptr);
      __self.__destruct();
    }

    friend void tag_invoke(stdexec::set_stopped_t, __recv&& __self) noexcept {
      __self.__cb_(__self.__data_, 1, nullptr);
      __self.__destruct();
    }

    friend void
      tag_invoke(stdexec::set_error_t, __recv&& __self, std::exception_ptr __ptr) noexcept {
      __self.__cb_(__self.__data_, 2, *reinterpret_cast<void**>(&__ptr));
      __self.__destruct();
    }

    void __destruct() {
      if (__on_heap_) {
        delete __op_;
      } else {
        __op_->~__operation();
      }
    }
  };

  template <typename __Sender>
  struct __operation {
    stdexec::connect_result_t<__Sender, __recv<__Sender>> __inner_op_;

    /// Try to construct the operation in the preallocated memory if it fits, otherwise allocate a new operation.
    static __operation* __construct_maybe_alloc(
      void* __preallocated,
      size_t __psize,
      __Sender __sndr,
      __exec_system_context_completion_callback_t __cb,
      void* __data) {
      if (__preallocated == nullptr || __psize < sizeof(__operation)) {
        return new __operation(std::move(__sndr), __cb, __data, true);
      } else {
        return new (__preallocated) __operation(std::move(__sndr), __cb, __data, false);
      }
    }

   private:
    __operation(
      __Sender __sndr,
      __exec_system_context_completion_callback_t __cb,
      void* __data,
      bool __on_heap)
      : __inner_op_(
        stdexec::connect(std::move(__sndr), __recv<__Sender>{__cb, __data, this, __on_heap})) {
    }
  };

  struct __system_scheduler_impl : __exec_system_scheduler_interface {
    __system_scheduler_impl(exec::static_thread_pool& __pool)
      : __pool_scheduler_{__pool.get_scheduler()} {
      __forward_progress_guarantee = 1; // parallel
      __schedule_operation_size = sizeof(__schedule_operation_t),
      __schedule_operation_alignment = alignof(__schedule_operation_t),
      __schedule = __schedule_impl;
      __bulk_schedule_operation_size = sizeof(__bulk_schedule_operation_t),
      __bulk_schedule_operation_alignment = alignof(__bulk_schedule_operation_t),
      __bulk_schedule = __bulk_schedule_impl;
    }

   private:
    /// Scheduler from the underlying thread pool.
    __pool_scheduler_t __pool_scheduler_;

    struct __bulk_functor {
      __exec_system_context_bulk_item_callback_t __cb_item_;
      void* __data_;

      void operator()(long __idx) const noexcept {
        __cb_item_(__data_, __idx);
      }
    };

    using __schedule_operation_t =
      __operation<decltype(stdexec::schedule(std::declval<__pool_scheduler_t>()))>;

    using __bulk_schedule_operation_t = __operation<decltype(stdexec::bulk(
      stdexec::schedule(std::declval<__pool_scheduler_t>()),
      std::declval<long>(),
      std::declval<__bulk_functor>()))>;

    static void __schedule_impl(
      __exec_system_scheduler_interface* __self,
      void* __preallocated,
      uint32_t __psize,
      __exec_system_context_completion_callback_t __cb,
      void* __data) noexcept {

      auto __this = static_cast<__system_scheduler_impl*>(__self);
      auto __sndr = stdexec::schedule(__this->__pool_scheduler_);
      auto __os = __schedule_operation_t::__construct_maybe_alloc(
        __preallocated, __psize, std::move(__sndr), __cb, __data);
      stdexec::start(__os->__inner_op_);
    }

    static void __bulk_schedule_impl(
      __exec_system_scheduler_interface* __self,
      void* __preallocated,
      uint32_t __psize,
      __exec_system_context_completion_callback_t __cb,
      __exec_system_context_bulk_item_callback_t __cb_item,
      void* __data,
      long __size) noexcept {

      auto __this = static_cast<__system_scheduler_impl*>(__self);
      auto __sndr = stdexec::bulk(
        stdexec::schedule(__this->__pool_scheduler_), __size, __bulk_functor{__cb_item, __data});
      auto __os = __bulk_schedule_operation_t::__construct_maybe_alloc(
        __preallocated, __psize, std::move(__sndr), __cb, __data);
      stdexec::start(__os->__inner_op_);
    }
  };

  /// Default implementation of a system context, based on `static_thread_pool`
  struct __system_context_impl : __exec_system_context_interface {
    __system_context_impl() {
      __version = 202402;
      __get_scheduler = __get_scheduler_impl;
    }

   private:
    /// The underlying thread pool.
    exec::static_thread_pool __pool_{};

    /// The system scheduler implementation.
    __system_scheduler_impl __scheduler_{__pool_};

    static long __get_version_impl(__exec_system_context_interface*) noexcept {
      return 202402L;
    }

    static __exec_system_scheduler_interface* __get_scheduler_impl(
      __exec_system_context_interface* __self) noexcept {
      return &static_cast<__system_context_impl*>(__self)->__scheduler_;
    }
  };

  /// Gets the default system context implementation.
  static __system_context_impl* __get_exec_system_context_impl() {
    static __system_context_impl __impl_;
    return &__impl_;
  }

} // namespace exec::__system_context_default_impl
