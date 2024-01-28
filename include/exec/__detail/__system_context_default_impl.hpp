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

  struct __system_scheduler_impl : __exec_system_scheduler_interface {
    __system_scheduler_impl(exec::static_thread_pool& __pool)
      : __pool_scheduler_{__pool.get_scheduler()} {
      __get_forward_progress_guarantee = __get_forward_progress_guarantee_impl;
      __schedule = __schedule_impl;
      __bulk_schedule = __bulk_schedule_impl;
    }

   private:
    /// Scheduler from the underlying thread pool.
    decltype(std::declval<exec::static_thread_pool>().get_scheduler()) __pool_scheduler_;

    static int __get_forward_progress_guarantee_impl(__exec_system_scheduler_interface*) {
      return 1; // parallel
    }

    static void __schedule_impl(
      __exec_system_scheduler_interface* __self,
      __exec_system_context_schedule_callback_t __cb,
      void* __data);

    static void __bulk_schedule_impl(
      __exec_system_scheduler_interface* __self,
      __exec_system_context_schedule_callback_t __cb,
      __exec_system_context_bulk_item_callback_t __cb_item,
      void* __data,
      long __size);
  };

  /// Default implementation of a system context, based on `static_thread_pool`
  struct __system_context_impl : __exec_system_context_interface {
    __system_context_impl() {
      __get_scheduler = __get_scheduler_impl;
    }

   private:
    /// The underlying thread pool.
    exec::static_thread_pool __pool_{};

    /// The system scheduler implementation.
    __system_scheduler_impl __scheduler_{__pool_};

    static __exec_system_scheduler_interface*
      __get_scheduler_impl(__exec_system_context_interface* __self) noexcept {
      return &static_cast<__system_context_impl*>(__self)->__scheduler_;
    }
  };

  /// Receiver that calls the callback when the operation completes.
  struct __recv {
    using receiver_concept = stdexec::receiver_t;

    /// The callback to be called.
    __exec_system_context_schedule_callback_t __cb_;

    /// The data to be passed to the callback.
    void* __data_;

    friend void tag_invoke(stdexec::set_value_t, __recv&& __self) noexcept {
      __self.__cb_(__self.__data_, 0, nullptr);
    }

    friend void tag_invoke(stdexec::set_stopped_t, __recv&& __self) noexcept {
      __self.__cb_(__self.__data_, 1, nullptr);
    }

    friend void
      tag_invoke(stdexec::set_error_t, __recv&& __self, std::exception_ptr __ptr) noexcept {
      __self.__cb_(__self.__data_, 2, *reinterpret_cast<void**>(&__ptr));
    }
  };

  inline void __system_scheduler_impl::__schedule_impl(
    __exec_system_scheduler_interface* __self,
    __exec_system_context_schedule_callback_t __cb,
    void* __data) {
    auto __this = static_cast<__system_scheduler_impl*>(__self);
    auto __sender = stdexec::schedule(__this->__pool_scheduler_);
    using operation_state_t = stdexec::connect_result_t<decltype(__sender), __recv>;
    auto __os = new operation_state_t(stdexec::connect(std::move(__sender), __recv{__cb, __data}));
    // TODO: stop leaking
    // TODO: we have: size=64, alignment=8
    stdexec::start(*__os);
  }

  inline void __system_scheduler_impl::__bulk_schedule_impl(
    __exec_system_scheduler_interface* __self,
    __exec_system_context_schedule_callback_t __cb,
    __exec_system_context_bulk_item_callback_t __cb_item,
    void* __data,
    long __size) {
    auto __this = static_cast<__system_scheduler_impl*>(__self);
    auto __sender = stdexec::bulk(
      stdexec::schedule(__this->__pool_scheduler_), __size, [__cb_item, __data](long __idx) {
        __cb_item(__data, __idx);
      });
    using operation_state_t = stdexec::connect_result_t<decltype(__sender), __recv>;
    auto __os = new operation_state_t(stdexec::connect(std::move(__sender), __recv{__cb, __data}));
    // TODO: stop leaking
    // TODO: we have: size=???, alignment=???
    stdexec::start(*__os);
  }

  /// Gets the default system context implementation.
  static __system_context_impl* __get_exec_system_context_impl() {
    static __system_context_impl __impl_;
    return &__impl_;
  }

}
