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

    __if::__exec_system_sender_interface* bulk(
      __if::__exec_system_bulk_shape __shp,
      __if::__exec_system_bulk_function_object __fn) override;

    stdexec::forward_progress_guarantee get_forward_progress_guarantee() const override {
      return stdexec::forward_progress_guarantee::parallel;
    }

    bool equals(const __if::__exec_system_scheduler_interface* __rhs) const override {
      auto __rhs_impl = dynamic_cast<const __exec_system_scheduler_impl*>(__rhs);
      return __rhs_impl && __rhs_impl->__ctx_ == __ctx_;
    }
  };

  struct __exec_system_operation_state_impl;
  using __exec_pool_sender_t =
    decltype(stdexec::schedule(std::declval<__exec_system_scheduler_impl>().__pool_scheduler_));

  struct __exec_system_pool_receiver {
    using receiver_concept = stdexec::receiver_t;

    friend void tag_invoke(stdexec::set_value_t, __exec_system_pool_receiver&&) noexcept;

    friend void tag_invoke(stdexec::set_stopped_t, __exec_system_pool_receiver&&) noexcept;

    friend void
      tag_invoke(stdexec::set_error_t, __exec_system_pool_receiver&&, std::exception_ptr) noexcept;

    friend stdexec::empty_env
      tag_invoke(stdexec::get_env_t, const __exec_system_pool_receiver&) noexcept {
      return {};
    }

    __exec_system_operation_state_impl* __os_ = nullptr;
  };

  struct __exec_system_operation_state_impl : public __if::__exec_system_operation_state_interface {
    __exec_system_operation_state_impl(
      __exec_pool_sender_t&& __pool_sender,
      __if::__exec_system_receiver&& __recv)
      : __recv_{std::move(__recv)}
      , __pool_operation_state_{[&]() {
        return stdexec::connect(std::move(__pool_sender), __exec_system_pool_receiver{this});
      }()} {
    }

    __exec_system_operation_state_impl(const __exec_system_operation_state_impl&) = delete;
    __exec_system_operation_state_impl(__exec_system_operation_state_impl&&) = delete;
    __exec_system_operation_state_impl&
      operator=(const __exec_system_operation_state_impl&) = delete;
    __exec_system_operation_state_impl& operator=(__exec_system_operation_state_impl&&) = delete;

    void start() noexcept override {
      stdexec::start(__pool_operation_state_);
    }

    __if::__exec_system_receiver __recv_;
    stdexec::connect_result_t<__exec_pool_sender_t, __exec_system_pool_receiver>
      __pool_operation_state_;
  };

  inline void tag_invoke(stdexec::set_value_t, __exec_system_pool_receiver&& __recv) noexcept {
    __if::__exec_system_receiver& __system_recv = __recv.__os_->__recv_;
    __system_recv.set_value(__system_recv.__cpp_recv_);
  }

  inline void tag_invoke(stdexec::set_stopped_t, __exec_system_pool_receiver&& __recv) noexcept {
    __if::__exec_system_receiver& __system_recv = __recv.__os_->__recv_;
    __recv.__os_->__recv_.set_stopped(&(__system_recv.__cpp_recv_));
  }

  inline void tag_invoke(
    stdexec::set_error_t,
    __exec_system_pool_receiver&& __recv,
    std::exception_ptr __ptr) noexcept {
    __if::__exec_system_receiver& __system_recv = __recv.__os_->__recv_;
    __recv.__os_->__recv_.set_error(&(__system_recv.__cpp_recv_), &__ptr);
  }

  struct __exec_system_sender_impl : public __if::__exec_system_sender_interface {
    __exec_system_sender_impl(
      __exec_system_scheduler_impl* __scheduler,
      __exec_pool_sender_t&& __pool_sender)
      : __scheduler_{__scheduler}
      , __pool_sender_(std::move(__pool_sender)) {
    }

    __if::__exec_system_operation_state_interface*
      connect(__if::__exec_system_receiver __recv) noexcept override {
      return new __exec_system_operation_state_impl(std::move(__pool_sender_), std::move(__recv));
    }

    __if::__exec_system_scheduler_interface* get_completion_scheduler() noexcept override {
      return __scheduler_;
    };

    __exec_system_scheduler_impl* __scheduler_;
    __exec_pool_sender_t __pool_sender_;
  };


  struct __exec_system_bulk_operation_state_impl;

  struct __exec_system_bulk_pool_receiver {
    using receiver_concept = stdexec::receiver_t;

    friend void tag_invoke(stdexec::set_value_t, __exec_system_bulk_pool_receiver&&) noexcept;

    friend void tag_invoke(stdexec::set_stopped_t, __exec_system_bulk_pool_receiver&&) noexcept;

    friend void tag_invoke(
      stdexec::set_error_t,
      __exec_system_bulk_pool_receiver&&,
      std::exception_ptr) noexcept;

    friend stdexec::empty_env
      tag_invoke(stdexec::get_env_t, const __exec_system_bulk_pool_receiver&) noexcept {
      return {};
    }

    __exec_system_bulk_operation_state_impl* __os_ = nullptr;
  };

  auto __exec_pool_operation_state(
    __exec_system_bulk_operation_state_impl* __self,
    __exec_pool_sender_t&& __ps,
    __if::__exec_system_bulk_shape __shp,
    __if::__exec_system_bulk_function_object __fn) {
    return stdexec::connect(
      stdexec::bulk(
        std::move(__ps), __shp, [__fn](long __idx) { __fn.__fn(__fn.__fn_state, __idx); }),
      __exec_system_bulk_pool_receiver{__self});
  }

  struct __exec_system_bulk_operation_state_impl
    : public __if::__exec_system_operation_state_interface {
    __exec_system_bulk_operation_state_impl(
      __exec_pool_sender_t&& __pool_sender,
      __if::__exec_system_bulk_shape __bulk_shape,
      __if::__exec_system_bulk_function_object __bulk_function,
      __if::__exec_system_receiver&& __recv)
      : __recv_{std::move(__recv)}
      , __bulk_function_{__bulk_function}
      , __pool_operation_state_{__exec_pool_operation_state(
          this,
          std::move(__pool_sender),
          __bulk_shape,
          __bulk_function_)} {
    }

    __exec_system_bulk_operation_state_impl(
      const __exec_system_bulk_operation_state_impl&) = delete;
    __exec_system_bulk_operation_state_impl(__exec_system_bulk_operation_state_impl&&) = delete;
    __exec_system_bulk_operation_state_impl&
      operator=(const __exec_system_bulk_operation_state_impl&) = delete;
    __exec_system_bulk_operation_state_impl&
      operator=(__exec_system_bulk_operation_state_impl&&) = delete;

    void start() noexcept override {
      stdexec::start(__pool_operation_state_);
    }

    __if::__exec_system_receiver __recv_;
    __if::__exec_system_bulk_function_object __bulk_function_;
    stdexec::__result_of<
      __exec_pool_operation_state,
      __exec_system_bulk_operation_state_impl*,
      __exec_pool_sender_t,
      __if::__exec_system_bulk_shape,
      __if::__exec_system_bulk_function_object>
      __pool_operation_state_;
  };

  inline void tag_invoke(stdexec::set_value_t, __exec_system_bulk_pool_receiver&& __recv) noexcept {
    __if::__exec_system_receiver& __system_recv = __recv.__os_->__recv_;
    __system_recv.set_value((__system_recv.__cpp_recv_));
  }

  inline void
    tag_invoke(stdexec::set_stopped_t, __exec_system_bulk_pool_receiver&& __recv) noexcept {
    __if::__exec_system_receiver& __system_recv = __recv.__os_->__recv_;
    __recv.__os_->__recv_.set_stopped(&(__system_recv.__cpp_recv_));
  }

  inline void tag_invoke(
    stdexec::set_error_t,
    __exec_system_bulk_pool_receiver&& __recv,
    std::exception_ptr __ptr) noexcept {
    __if::__exec_system_receiver& __system_recv = __recv.__os_->__recv_;
    __recv.__os_->__recv_.set_error(&(__system_recv.__cpp_recv_), &__ptr);
  }

  // A bulk sender is just a system sender viewed externally.
  // TODO: a bulk operation state is just a system operation state viewed externally
  struct __exec_system_bulk_sender_impl : public __if::__exec_system_sender_interface {
    __exec_system_bulk_sender_impl(
      __exec_system_scheduler_impl* __scheduler,
      __if::__exec_system_bulk_shape __bulk_shape,
      __if::__exec_system_bulk_function_object __bulk_function,
      __exec_pool_sender_t&& __pool_sender)
      : __scheduler_{__scheduler}
      , __bulk_shape_{__bulk_shape}
      , __bulk_function_{__bulk_function}
      , __pool_sender_(std::move(__pool_sender)) {
    }

    __if::__exec_system_operation_state_interface*
      connect(__if::__exec_system_receiver __recv) noexcept override {
      return new __exec_system_bulk_operation_state_impl(
        std::move(__pool_sender_), __bulk_shape_, __bulk_function_, std::move(__recv));
    }

    __if::__exec_system_scheduler_interface* get_completion_scheduler() noexcept override {
      return __scheduler_;
    };

    __exec_system_scheduler_impl* __scheduler_;
    __if::__exec_system_bulk_shape __bulk_shape_;
    __if::__exec_system_bulk_function_object __bulk_function_;
    __exec_pool_sender_t __pool_sender_;
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

  void __exec_system_scheduler_impl::schedule(
    __if::__exec_system_context_schedule_callback_t __cb,
    void* __data) {
    auto __sender = stdexec::schedule(__pool_scheduler_);
    using operation_state_t = stdexec::connect_result_t<decltype(__sender), __recv>;
    auto __os = new operation_state_t(stdexec::connect(std::move(__sender), __recv{__cb, __data}));
    // TODO: stop leaking
    // TODO: we have: size=64, alignment=8
    stdexec::start(*__os);
  }

  inline __if::__exec_system_sender_interface* __exec_system_scheduler_impl::bulk(
    __if::__exec_system_bulk_shape __shp,
    __if::__exec_system_bulk_function_object __fn) {
    // This is bulk off a system_scheduler, so we need to start with schedule.
    // TODO: a version later will key off a bulk *sender* and would behave slightly
    // differently.
    // In both cases pass in the result of schedule, or the predecessor though.
    return new __exec_system_bulk_sender_impl(
      this, __shp, __fn, stdexec::schedule(__pool_scheduler_));
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