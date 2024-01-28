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

#include "__system_context_if.hpp"
#include "stdexec/execution.hpp"
#include "exec/static_thread_pool.hpp"

namespace exec { namespace __system_context_default_impl {

  namespace __if = __system_context_interface;

  // Low-level APIs
  // Phase 2 will move to pointers and ref counting ala COM
  // Phase 3 will move these to weak symbols and allow replacement in tests
  // Default implementation based on static_thread_pool
  struct __exec_system_context_impl : public __if::__exec_system_context_interface {
    exec::static_thread_pool __pool_;

    __if::__exec_system_scheduler_interface* get_scheduler() noexcept override;
  };

  struct __exec_system_scheduler_impl : public __if::__exec_system_scheduler_interface {
    __exec_system_scheduler_impl(
      __exec_system_context_impl* __ctx,
      decltype(__ctx->__pool_.get_scheduler()) __pool_scheduler)
      : __ctx_{__ctx}
      , __pool_scheduler_{__pool_scheduler} {
    }

    __exec_system_context_impl* __ctx_;
    decltype(__ctx_->__pool_.get_scheduler()) __pool_scheduler_;

    void schedule(__if::__exec_system_context_schedule_callback_t __cb, void* __data) override;

    void bulk_schedule(
      __if::__exec_system_context_schedule_callback_t __cb,
      __if::__exec_system_context_bulk_item_callback_t __cb_item,
      void* __data,
      long __size) override;

    stdexec::forward_progress_guarantee get_forward_progress_guarantee() const override {
      return stdexec::forward_progress_guarantee::parallel;
    }

    bool equals(const __if::__exec_system_scheduler_interface* __rhs) const override {
      auto __rhs_impl = dynamic_cast<const __exec_system_scheduler_impl*>(__rhs);
      return __rhs_impl && __rhs_impl->__ctx_ == __ctx_;
    }
  };

  inline __if::__exec_system_scheduler_interface*
    __exec_system_context_impl::get_scheduler() noexcept {
    // TODO: ref counting etc
    return new __exec_system_scheduler_impl(this, __pool_.get_scheduler());
  }

  // TODO: rename
  struct __recv {
    using receiver_concept = stdexec::receiver_t;
    __if::__exec_system_context_schedule_callback_t __cb_;
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

  inline void __exec_system_scheduler_impl::schedule(
    __if::__exec_system_context_schedule_callback_t __cb,
    void* __data) {
    auto __sender = stdexec::schedule(__pool_scheduler_);
    using operation_state_t = stdexec::connect_result_t<decltype(__sender), __recv>;
    auto __os = new operation_state_t(stdexec::connect(std::move(__sender), __recv{__cb, __data}));
    // TODO: stop leaking
    // TODO: we have: size=64, alignment=8
    stdexec::start(*__os);
  }

  inline void __exec_system_scheduler_impl::bulk_schedule(
    __if::__exec_system_context_schedule_callback_t __cb,
    __if::__exec_system_context_bulk_item_callback_t __cb_item,
    void* __data,
    long __size) {
    // TODO
    auto __sender = stdexec::bulk(
      stdexec::schedule(__pool_scheduler_), __size, [__cb_item, __data](long __idx) {
        __cb_item(__data, __idx);
        // TODO: error, stopped
      });

    using operation_state_t = stdexec::connect_result_t<decltype(__sender), __recv>;
    auto __os = new operation_state_t(stdexec::connect(std::move(__sender), __recv{__cb, __data}));
    // TODO: stop leaking
    // TODO: we have: size=???, alignment=???
    stdexec::start(*__os);
  }

  // Phase 1 implementation, single implementation
  // TODO: Make a weak symbol and replace in a test
  static __exec_system_context_impl* __get_exec_system_context_impl() {
    static __exec_system_context_impl __impl_;

    return &__impl_;
  }

  // TODO: Move everything above here to a detail header and wrap in a
  // namespace to represent extern "C"


}}