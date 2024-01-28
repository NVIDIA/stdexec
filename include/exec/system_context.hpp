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
#include "__detail/__system_context_default_impl.hpp"

namespace exec {
  namespace __if = __system_context_interface;

  namespace __detail {
    /// Transforms from a C API signal to the `set_xxx` completion signal.
    template <typename __R>
    inline void __pass_to_receiver(int __completion_type, void* __exception, __R&& __recv) {
      if (__completion_type == 0) {
        stdexec::set_value(std::forward<__R>(__recv));
      } else if (__completion_type == 1) {
        stdexec::set_stopped(std::forward<__R>(__recv));
      } else if (__completion_type == 2) {
        stdexec::set_error(
          std::forward<__R>(__recv),
          std::move(*reinterpret_cast<std::exception_ptr*>(&__exception)));
      }
    }
  }

  class system_scheduler;
  class system_sender;
  template <stdexec::sender __S, std::integral __Size, class __Fn>
  struct system_bulk_sender;

  /// Provides a view on some global underlying execution context supporting parallel forward progress.
  class system_context {
   public:
    /// Initializes the system context with the default implementation.
    system_context();
    ~system_context() = default;

    system_context(const system_context&) = delete;
    system_context(system_context&&) = delete;
    system_context& operator=(const system_context&) = delete;
    system_context& operator=(system_context&&) = delete;

    // Returns a scheduler that can add work to the underlying execution context.
    system_scheduler get_scheduler();

    /// Returns the maximum number of threads the context may support; this is just a hint.
    size_t max_concurrency() const noexcept;

   private:
    /// The actual implementation of the system context.
    __if::__exec_system_context_interface* __impl_{nullptr};
  };

  /// A scheduler that can add work to the system context.
  class system_scheduler {
   public:
    system_scheduler() = delete;

    /// Returns `true` iff `*this` refers to the same scheduler as `__rhs`.
    bool operator==(const system_scheduler& __rhs) const noexcept {
      return __scheduler_interface_->equals(__rhs.__scheduler_interface_);
    }

    /// Implementation detail. Constructs the scheduler to wrap `__impl`.
    system_scheduler(__if::__exec_system_scheduler_interface* __impl)
      : __scheduler_interface_(__impl) {
    }

   private:
    /// Schedules new work, returning the sender that signals the start of the work.
    friend system_sender tag_invoke(stdexec::schedule_t, const system_scheduler&) noexcept;

    /// Returns the forward progress guarantee of `this`.
    friend stdexec::forward_progress_guarantee
      tag_invoke(stdexec::get_forward_progress_guarantee_t, const system_scheduler&) noexcept;

    /// Schedules new bulk work, calling `__fun` with the index of each chunk in range `[0, __size]`,
    /// and the value(s) resulting from completing `__previous`; returns a sender that completes
    /// when all chunks complete.
    template <stdexec::sender __S, std::integral __Size, class __Fn>
    friend system_bulk_sender<__S, __Size, __Fn> tag_invoke(
      stdexec::bulk_t,
      const system_scheduler& __self,
      __S&& __previous,
      __Size __size,
      __Fn __fun) noexcept;

    /// The underlying implementation of the scheduler.
    __if::__exec_system_scheduler_interface* __scheduler_interface_;
  };

  /// Describes the environment of this sender.
  struct __system_scheduler_env {
    /// Returns the system scheduler as the completion scheduler for `set_value_t`.
    friend system_scheduler tag_invoke(
      stdexec::get_completion_scheduler_t<stdexec::set_value_t>,
      const __system_scheduler_env& __self) noexcept {
      return {__self.__scheduler_impl_};
    }

    /// Returns the system scheduler as the completion scheduler for `set_stopped_t`.
    friend system_scheduler tag_invoke(
      stdexec::get_completion_scheduler_t<stdexec::set_stopped_t>,
      const __system_scheduler_env& __self) noexcept {
      return {__self.__scheduler_impl_};
    }

    /// The underlying implementation of the scheduler we are using.
    __if::__exec_system_scheduler_interface* __scheduler_impl_;
  };

  /// The sender used to schedule new work in the system context.
  class system_sender {
   public:
    /// Marks this type as being a sender; not to spec.
    using sender_concept = stdexec::sender_t;
    /// Declares the completion signals sent by `this`.
    using completion_signatures = stdexec::completion_signatures<
      stdexec::set_value_t(),
      stdexec::set_stopped_t(),
      stdexec::set_error_t(std::exception_ptr) >;

    /// Implementation detail. Constructs the sender to wrap `__impl`.
    system_sender(__if::__exec_system_scheduler_interface* __impl)
      : __scheduler_impl_{__impl} {
    }

   private:
    /// The operation state used to execute the work described by this sender.
    template <class __S, class __R>
    struct __op {
      /// Constructs `this` from `__recv` and `__scheduler_impl`.
      __op(__R&& __recv, __if::__exec_system_scheduler_interface* __scheduler_impl)
        : __recv_{std::move(__recv)}
        , __scheduler_impl_{__scheduler_impl} {
      }

      __op(const __op&) = delete;
      __op(__op&&) = delete;
      __op& operator=(const __op&) = delete;
      __op& operator=(__op&&) = delete;

      /// Starts the work stored in `this`.
      friend void tag_invoke(stdexec::start_t, __op& __self) noexcept {
        __self.__scheduler_impl_->schedule(__cb, &__self);
      }

      static void __cb(void* __data, int __completion_type, void* __exception) {
        auto __self = static_cast<__op*>(__data);
        __detail::__pass_to_receiver(__completion_type, __exception, std::move(__self->__recv_));
      }

      /// Object that receives completion from the work described by the sender.
      __R __recv_;
      /// The underlying implementation of the scheduler.
      __if::__exec_system_scheduler_interface* __scheduler_impl_{nullptr};
    };

    /// Connects `__self` to `__r`, returning the operation state containing the work to be done.
    template <class __R>
    friend auto tag_invoke(stdexec::connect_t, system_sender&& __self, __R&& __r) noexcept(
      std::is_nothrow_constructible_v<std::remove_cvref_t<__R>, __R>)
      -> __op<system_sender, std::remove_cvref_t<__R>> {

      return {std::move(__r), __self.__scheduler_impl_};
    }

    /// Gets the environment of this sender.
    friend __system_scheduler_env
      tag_invoke(stdexec::get_env_t, const system_sender& __self) noexcept {
      return {__self.__scheduler_impl_};
    }

    /// The underlying implementation of the system scheduler.
    __if::__exec_system_scheduler_interface* __scheduler_impl_{nullptr};
  };

  /// The state needed to execute the bulk sender created from system context.
  template <stdexec::sender __Previous, std::integral __Size, class __Fn, class __R>
  struct __bulk_state {
    /// The sender object that describes the work to be done.
    system_bulk_sender<__Previous, __Size, __Fn> __snd_;
    /// The receiver object that receives completion from the work described by the sender.
    __R __recv_;
    /// The arguments passed from the previous receiver to the function object of the bulk sender,
    void* __arguments_data_{nullptr};
  };

  /// Receiver that is used in "bulk" to connect toe the input sender of the bulk operation.
  template <stdexec::sender __Previous, std::integral __Size, class __Fn, class __R>
  struct __bulk_intermediate_receiver {
    /// Declare that this is a `receiver`.
    using receiver_concept = stdexec::receiver_t;

    /// The type of the object that holds relevant data for the entire bulk operation.
    using __bulk_state_t = __bulk_state<__Previous, __Size, __Fn, __R>;

    /// Object that holds the relevant data for the entire bulk operation.
    __bulk_state_t& __state_;

    /// Invoked when the previous sender completes with a value to trigger multiple operations on the system scheduler.
    template <class... __As>
    friend void tag_invoke(
      stdexec::set_value_t,
      __bulk_intermediate_receiver&& __self,
      __As&&... __as) noexcept {

      // Heap allocate input data in shared state as needed
      std::tuple<__As...>* __inputs = new std::tuple<__As...>{__as...};
      __self.__state_.__arguments_data_ = __inputs;
      // TODO: fix memory leak

      auto __type_erased_fn = [](void* __state_, long __idx) {
        auto* __state = static_cast<__bulk_state_t*>(__state_);
        std::apply(
          [&](auto&&... __args) { __state->__snd_.__fun_(__idx, __args...); },
          *static_cast<std::tuple<__As...>*>(__state->__arguments_data_));
      };

      // Schedule the bulk work on the system scheduler.
      __self.__state_.__snd_.__scheduler_impl_->bulk_schedule(
        __cb, __type_erased_fn, &__self.__state_, __self.__state_.__snd_.__size_);
    }

    /// Invoked when the previous sender completes with "stopped" to stop the entire work.
    friend void tag_invoke(stdexec::set_stopped_t, __bulk_intermediate_receiver&& __self) noexcept {
      stdexec::set_stopped(std::move(__self.__state_.__recv_));
    }

    /// Invoked when the previous sender completes with error to forward the error to the connected receiver.
    friend void tag_invoke(
      stdexec::set_error_t,
      __bulk_intermediate_receiver&& __self,
      std::exception_ptr __ptr) noexcept {
      stdexec::set_error(std::move(__self.__state_.__recv_), std::move(__ptr));
    }

    /// Gets the environment of this receiver; returns the environment of the connected receiver.
    friend auto
      tag_invoke(stdexec::get_env_t, const __bulk_intermediate_receiver& __self) noexcept {
      return stdexec::get_env(__self.__state_.__recv_);
    }

    /// Called by the system scheduler when the bulk operation completes.
    static void __cb(void* __data, int __completion_type, void* __exception) {
      auto __state = static_cast<__bulk_state_t*>(__data);
      __detail::__pass_to_receiver(__completion_type, __exception, std::move(__state->__recv_));
    }
  };

  /// The operation state object for the system bulk sender.
  template <stdexec::sender __Previous, std::integral __Size, class __Fn, class __R>
  struct __bulk_op {
    /// The inner operation state, which is the result of connecting the previous sender to the bulk intermediate receiver.
    using __inner_op_state = stdexec::
      connect_result_t<__Previous, __bulk_intermediate_receiver<__Previous, __Size, __Fn, __R>>;

    /// Constructs `this` from `__snd` and `__recv`, using the object returned by `__initFunc` to start the operation.
    ///
    /// Using a functor to initialize the operation state allows the use of `this` to get the
    /// underlying implementation object.
    template <class __InitF>
    __bulk_op(
      system_bulk_sender<__Previous, __Size, __Fn>&& __snd,
      __R&& __recv,
      __InitF&& __initFunc)
      : __state_{std::move(__snd), std::move(__recv)}
      , __previous_operation_state_{__initFunc(*this)} {
    }

    __bulk_op(const __bulk_op&) = delete;
    __bulk_op(__bulk_op&&) = delete;
    __bulk_op& operator=(const __bulk_op&) = delete;
    __bulk_op& operator=(__bulk_op&&) = delete;

    /// Starts the work stored in `__self`.
    friend void tag_invoke(stdexec::start_t, __bulk_op& __self) noexcept {
      // Start previous operation state.
      // Bulk operation will be started when the previous sender completes.
      stdexec::start(__self.__previous_operation_state_);
    }

    /// The state of this bulk operation.
    __bulk_state<__Previous, __Size, __Fn, __R> __state_;
    /// The operation state object of the previous computation.
    __inner_op_state __previous_operation_state_;
  };

  /// The sender used to schedule bulk work in the system context.
  template <stdexec::sender __Previous, std::integral __Size, class __Fn>
  struct system_bulk_sender {
    /// Marks this type as being a sender; not to spec.
    using sender_concept = stdexec::sender_t;
    /// Declares the completion signals sent by `this`.
    using completion_signatures = stdexec::completion_signatures<
      stdexec::set_value_t(),
      stdexec::set_stopped_t(),
      stdexec::set_error_t(std::exception_ptr) >;

    // TODO: This can complete with different values... should propagate from __Previous

    /// Constructs `this`.
    system_bulk_sender(
      __if::__exec_system_scheduler_interface* __scheduler_impl,
      __Previous __previous,
      __Size __size,
      __Fn&& __fun)
      : __scheduler_impl_{__scheduler_impl}
      , __previous_{std::move(__previous)}
      , __size_{std::move(__size)}
      , __fun_{std::move(__fun)} {
    }

    /// Connects `__self` to `__r`, returning the operation state containing the work to be done.
    template <class __R>
    friend auto tag_invoke(stdexec::connect_t, system_bulk_sender&& __self, __R&& __r) //
      noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<__R>, __R>)
        -> __bulk_op<__Previous, __Size, __Fn, __R> {

      return {std::move(__self), std::move(__r), [](auto& __op) {
                // Connect bulk input receiver with the previous operation and store in the operating state.
                return stdexec::connect(
                  std::move(__op.__state_.__snd_.__previous_),
                  __bulk_intermediate_receiver<__Previous, __Size, __Fn, __R>{__op.__state_});
              }};
    }

    /// Gets the environment of this sender.
    friend __system_scheduler_env
      tag_invoke(stdexec::get_env_t, const system_bulk_sender& __snd) noexcept {
      return {__snd.__scheduler_impl_};
    }

    /// The underlying implementation of the scheduler we are using.
    __if::__exec_system_scheduler_interface* __scheduler_impl_{nullptr};
    /// The previous sender, the one that produces the input value for the bulk function.
    __Previous __previous_;
    /// The size of the bulk operation.
    __Size __size_;
    /// The function to be executed to perform the bulk work.
    __Fn __fun_;
  };

  inline system_context::system_context() {
    __impl_ = __system_context_default_impl::__get_exec_system_context_impl();
    // TODO error handling
  }

  inline system_scheduler system_context::get_scheduler() {
    return system_scheduler{__impl_->get_scheduler()};
  }

  inline size_t system_context::max_concurrency() const noexcept {
    return std::thread::hardware_concurrency();
  }

  system_sender tag_invoke(stdexec::schedule_t, const system_scheduler& sched) noexcept {
    return {sched.__scheduler_interface_};
  }

  stdexec::forward_progress_guarantee
    tag_invoke(stdexec::get_forward_progress_guarantee_t, const system_scheduler& __self) noexcept {
    return __self.__scheduler_interface_->get_forward_progress_guarantee();
  }

  template <stdexec::sender __S, std::integral __Size, class __Fn>
  system_bulk_sender<__S, __Size, __Fn> tag_invoke(
    stdexec::bulk_t,
    const system_scheduler& __self,
    __S&& __previous,
    __Size __size,
    __Fn __fun) noexcept {
    return system_bulk_sender<__S, __Size, __Fn>{
      __self.__scheduler_interface_, (__S&&) __previous, __size, (__Fn&&) __fun};
  }


} // namespace exec
