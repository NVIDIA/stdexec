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
#include "__detail/__system_context_if.h"
#include "__detail/__weak_attribute.hpp"

#ifndef __EXEC__SYSTEM_CONTEXT__SCHEDULE_OP_SIZE
#  define __EXEC__SYSTEM_CONTEXT__SCHEDULE_OP_SIZE 80
#endif
#ifndef __EXEC__SYSTEM_CONTEXT__SCHEDULE_OP_ALIGN
#  define __EXEC__SYSTEM_CONTEXT__SCHEDULE_OP_ALIGN 8
#endif
#ifndef __EXEC__SYSTEM_CONTEXT__BULK_SCHEDULE_OP_SIZE
#  define __EXEC__SYSTEM_CONTEXT__BULK_SCHEDULE_OP_SIZE 168
#endif
#ifndef __EXEC__SYSTEM_CONTEXT__BULK_SCHEDULE_OP_ALIGN
#  define __EXEC__SYSTEM_CONTEXT__BULK_SCHEDULE_OP_ALIGN 8
#endif

// TODO: make these configurable by providing policy to the system context

/// Gets the default system context implementation.
extern "C" __EXEC_WEAK_ATTRIBUTE __exec_system_context_interface* __get_exec_system_context_impl();

/// Sets the default system context implementation.
extern "C" __EXEC_WEAK_ATTRIBUTE void
  __set_exec_system_context_impl(__exec_system_context_interface* __instance);

namespace exec {
  namespace __detail {
    using namespace stdexec::tags;

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

    /// Same as a above, but allows passing arguments to set_value.
    template <typename __R, typename... __SetValueArgs>
    inline void __pass_to_receiver_with_args(
      int __completion_type,
      void* __exception,
      __R&& __recv,
      __SetValueArgs&&... __setValueArgs) {
      if (__completion_type == 0) {
        stdexec::set_value(std::forward<__R>(__recv), std::move(__setValueArgs)...);
      } else if (__completion_type == 1) {
        stdexec::set_stopped(std::forward<__R>(__recv));
      } else if (__completion_type == 2) {
        stdexec::set_error(
          std::forward<__R>(__recv),
          std::move(*reinterpret_cast<std::exception_ptr*>(&__exception)));
      }
    }

    /// The type large enough to store the data produced by a sender.
    template <typename __Sender>
    using __sender_data_t = decltype(stdexec::sync_wait(std::declval<__Sender>()).value());

  } // namespace __detail

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
    __exec_system_context_interface* __impl_{nullptr};
  };

  /// The execution domain of the system_scheduler, used for the purposes of customizing
  /// sender algorithms such as `bulk`.
  struct system_scheduler_domain : stdexec::default_domain {
    /// Schedules new bulk work, calling `__fun` with the index of each chunk in range `[0, __size]`,
    /// and the value(s) resulting from completing `__previous`; returns a sender that completes
    /// when all chunks complete.
    template <stdexec::sender_expr_for<stdexec::bulk_t> __Sender, class __Env>
    auto transform_sender(__Sender&& __sndr, const __Env& __env) const noexcept;
  };

  namespace __detail {
    template <class T>
    auto __make_system_scheduler_from(T, __exec_system_scheduler_interface*) noexcept;

    /// Describes the environment of this sender.
    struct __system_scheduler_env {
      /// Returns the system scheduler as the completion scheduler for `set_value_t`.
      template <stdexec::__none_of<stdexec::set_error_t> _Tag>
      STDEXEC_MEMFN_DECL(auto query)(
        this const __system_scheduler_env& __self,
        stdexec::get_completion_scheduler_t<_Tag>) noexcept {
        return __detail::__make_system_scheduler_from(_Tag(), __self.__scheduler_);
      }

      /// The underlying implementation of the scheduler we are using.
      __exec_system_scheduler_interface* __scheduler_;
    };

    /// The operation state used to execute the work described by this sender.
    template <class __S, class __R>
    struct __system_op {
      /// Constructs `this` from `__recv` and `__scheduler_impl`.
      __system_op(__R&& __recv, __exec_system_scheduler_interface* __scheduler_impl)
        : __recv_{std::move(__recv)}
        , __scheduler_{__scheduler_impl} {
      }

      ~__system_op() {
        __scheduler_->__destruct_schedule_operation(__scheduler_, __impl_os_);
      }

      __system_op(const __system_op&) = delete;
      __system_op(__system_op&&) = delete;
      __system_op& operator=(const __system_op&) = delete;
      __system_op& operator=(__system_op&&) = delete;

      /// Starts the work stored in `this`.
      STDEXEC_MEMFN_DECL(
        void start)(this __system_op& __self) noexcept {
        __self.__impl_os_ = __self.__scheduler_->__schedule(
          __self.__scheduler_,
          &__self.__preallocated_,
          sizeof(__self.__preallocated_),
          __cb,
          &__self);
      }

      static void __cb(void* __data, int __completion_type, void* __exception) {
        auto __self = static_cast<__system_op*>(__data);
        __detail::__pass_to_receiver(__completion_type, __exception, std::move(__self->__recv_));
      }

      /// Object that receives completion from the work described by the sender.
      __R __recv_;
      /// The underlying implementation of the scheduler.
      __exec_system_scheduler_interface* __scheduler_{nullptr};
      /// The operating state on the implementation side.
      void* __impl_os_;

      /// Preallocated space for storing the operation state on the implementation size.
      struct alignas(__EXEC__SYSTEM_CONTEXT__SCHEDULE_OP_ALIGN) __preallocated {
        char __data[__EXEC__SYSTEM_CONTEXT__SCHEDULE_OP_SIZE];
      } __preallocated_;
    };
  } // namespace __detail

  /// The sender used to schedule new work in the system context.
  class system_sender {
   public:
    /// Marks this type as being a sender; not to spec.
    using sender_concept = stdexec::sender_t;
    /// Declares the completion signals sent by `this`.
    using completion_signatures = stdexec::completion_signatures<
      stdexec::set_value_t(),
      stdexec::set_stopped_t(),
      stdexec::set_error_t(std::exception_ptr)>;

    /// Implementation detail. Constructs the sender to wrap `__impl`.
    system_sender(__exec_system_scheduler_interface* __impl)
      : __scheduler_{__impl} {
    }

   private:
    STDEXEC_MEMFN_FRIEND(get_env);
    STDEXEC_MEMFN_FRIEND(connect);

    /// Connects `__self` to `__r`, returning the operation state containing the work to be done.
    template <stdexec::receiver __R>
    STDEXEC_MEMFN_DECL(auto connect)(this system_sender&& __self, __R&& __r) //
      noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<__R>, __R>)
        -> __detail::__system_op<system_sender, std::remove_cvref_t<__R>> {

      return {std::move(__r), __self.__scheduler_};
    }

    /// Gets the environment of this sender.
    STDEXEC_MEMFN_DECL(
      auto get_env)(this const system_sender& __self) noexcept -> __detail::__system_scheduler_env {
      return {__self.__scheduler_};
    }

    /// The underlying implementation of the system scheduler.
    __exec_system_scheduler_interface* __scheduler_{nullptr};
  };

  /// A scheduler that can add work to the system context.
  class system_scheduler {
   public:
    system_scheduler() = delete;

    /// Returns `true` iff `*this` refers to the same scheduler as the argument.
    bool operator==(const system_scheduler&) const noexcept = default;

    /// Implementation detail. Constructs the scheduler to wrap `__impl`.
    system_scheduler(__exec_system_scheduler_interface* __impl)
      : __scheduler_(__impl) {
    }

   private:
    template <stdexec::sender, std::integral, class>
    friend struct system_bulk_sender;
    STDEXEC_MEMFN_FRIEND(schedule);

    /// Schedules new work, returning the sender that signals the start of the work.
    STDEXEC_MEMFN_DECL(
      system_sender schedule)(this const system_scheduler& sched) noexcept {
      return {sched.__scheduler_};
    }

    /// Returns the forward progress guarantee of `this`.
    STDEXEC_MEMFN_DECL(
      stdexec::forward_progress_guarantee
      query)(this const system_scheduler&, stdexec::get_forward_progress_guarantee_t) noexcept;

    /// Returns the execution domain of `this`.
    STDEXEC_MEMFN_DECL(
      system_scheduler_domain query)(this const system_scheduler&, stdexec::get_domain_t) noexcept {
      return {};
    }

    /// The underlying implementation of the scheduler.
    __exec_system_scheduler_interface* __scheduler_;
  };

  namespace __detail {
    template <class T>
    auto __make_system_scheduler_from(T, __exec_system_scheduler_interface* p) noexcept {
      return system_scheduler{p};
    }

    /// The state needed to execute the bulk sender created from system context.
    template <stdexec::sender __Previous, std::integral __Size, class __Fn, class __R>
    struct __bulk_state {
      /// The sender object that describes the work to be done.
      system_bulk_sender<__Previous, __Size, __Fn> __snd_;
      /// The receiver object that receives completion from the work described by the sender.
      __R __recv_;
      /// The operating state on the implementation side.
      void* __impl_os_ = nullptr;
      /// Storage for the arguments passed from the previous receiver to the function object of the bulk sender.
      alignas(__detail::__sender_data_t<__Previous>) unsigned char __arguments_data_[sizeof(
        __detail::__sender_data_t<__Previous>)]{};

      /// Preallocated space for storing the operation state on the implementation size.
      struct alignas(__EXEC__SYSTEM_CONTEXT__BULK_SCHEDULE_OP_ALIGN) __preallocated {
        char __data[__EXEC__SYSTEM_CONTEXT__BULK_SCHEDULE_OP_SIZE];
      } __preallocated_{};

      ~__bulk_state() {
        __snd_.__scheduler_->__destruct_bulk_schedule_operation(__snd_.__scheduler_, __impl_os_);
      }
    }; // namespace __detail

    /// Receiver that is used in "bulk" to connect toe the input sender of the bulk operation.
    template <stdexec::sender __Previous, std::integral __Size, class __Fn, class __R>
    struct __bulk_intermediate_receiver {
      /// Declare that this is a `receiver`.
      using receiver_concept = stdexec::receiver_t;

      /// The type of the object that holds relevant data for the entire bulk operation.
      using __bulk_state_t = __bulk_state<__Previous, __Size, __Fn, __R>;

      /// Object that holds the relevant data for the entire bulk operation.
      __bulk_state_t& __state_;

      template <class... __As>
      void __set_value(__As&&... __as) noexcept {
        // Store the input data in the shared state, in the preallocated buffer.
        static_assert(sizeof(std::tuple<__As...>) <= sizeof(__state_.__arguments_data_));
        new (&__state_.__arguments_data_)
          std::tuple<stdexec::__decay_t<__As>...>{std::move(__as)...};

        // The function that needs to be applied to each item in the bulk operation.
        auto __type_erased_item_fn = +[](void* __state_arg, unsigned long __idx) {
          auto* __state = static_cast<__bulk_state_t*>(__state_arg);
          std::apply(
            [&](auto&&... __args) { __state->__snd_.__fun_(__idx, __args...); },
            *reinterpret_cast<std::tuple<__As...>*>(&__state->__arguments_data_));
        };

        // The function that needs to be applied when all items are complete.
        auto __type_erased_cb_fn =
          +[](void* __state_arg, int __completion_type, void* __exception) {
            auto __state = static_cast<__bulk_state_t*>(__state_arg);
            std::apply(
              [&](auto&&... __args) {
                __detail::__pass_to_receiver_with_args(
                  __completion_type, __exception, std::move(__state->__recv_), __args...);
              },
              *reinterpret_cast<std::tuple<__As...>*>(&__state->__arguments_data_));
          };

        // Schedule the bulk work on the system scheduler.
        __state_.__impl_os_ = __state_.__snd_.__scheduler_->__bulk_schedule(
          __state_.__snd_.__scheduler_,                       // self
          &__state_.__preallocated_,                          // preallocated
          sizeof(__state_.__preallocated_),                   // psize
          __type_erased_cb_fn,                                // cb
          __type_erased_item_fn,                              // cb_item
          &__state_,                                          // data
          static_cast<unsigned long>(__state_.__snd_.__size_) // size
        );
      }

      /// Invoked when the previous sender completes with a value to trigger multiple operations on the system scheduler.
      template <class... __As>
      STDEXEC_MEMFN_DECL(void set_value)(this __bulk_intermediate_receiver&& __self, __As&&... __as) noexcept {
        __self.__set_value(static_cast<__As&&>(__as)...);
      }

      /// Invoked when the previous sender completes with "stopped" to stop the entire work.
      STDEXEC_MEMFN_DECL(
        void set_stopped)(this __bulk_intermediate_receiver&& __self) noexcept {
        stdexec::set_stopped(std::move(__self.__state_.__recv_));
      }

      /// Invoked when the previous sender completes with error to forward the error to the connected receiver.
      STDEXEC_MEMFN_DECL(
        void
        set_error)(this __bulk_intermediate_receiver&& __self, std::exception_ptr __ptr) noexcept {
        stdexec::set_error(std::move(__self.__state_.__recv_), std::move(__ptr));
      }

      /// Gets the environment of this receiver; returns the environment of the connected receiver.
      STDEXEC_MEMFN_DECL(
        auto get_env)(this const __bulk_intermediate_receiver& __self) noexcept -> decltype(auto) {
        return stdexec::get_env(__self.__state_.__recv_);
      }
    };

    /// The operation state object for the system bulk sender.
    template <stdexec::sender __Previous, std::integral __Size, class __Fn, class __R>
    struct __system_bulk_op {
      /// The inner operation state, which is the result of connecting the previous sender to the bulk intermediate receiver.
      using __inner_op_state = stdexec::
        connect_result_t<__Previous, __bulk_intermediate_receiver<__Previous, __Size, __Fn, __R>>;

      /// Constructs `this` from `__snd` and `__recv`, using the object returned by `__initFunc` to start the operation.
      ///
      /// Using a functor to initialize the operation state allows the use of `this` to get the
      /// underlying implementation object.
      template <class __InitF>
      __system_bulk_op(
        system_bulk_sender<__Previous, __Size, __Fn>&& __snd,
        __R&& __recv,
        __InitF&& __initFunc)
        : __state_{std::move(__snd), std::move(__recv)}
        , __previous_operation_state_{__initFunc(*this)} {
      }

      __system_bulk_op(const __system_bulk_op&) = delete;
      __system_bulk_op(__system_bulk_op&&) = delete;
      __system_bulk_op& operator=(const __system_bulk_op&) = delete;
      __system_bulk_op& operator=(__system_bulk_op&&) = delete;

      /// Starts the work stored in `__self`.
      STDEXEC_MEMFN_DECL(
        void start)(this __system_bulk_op& __self) noexcept {
        // Start previous operation state.
        // Bulk operation will be started when the previous sender completes.
        stdexec::start(__self.__previous_operation_state_);
      }

      /// The state of this bulk operation.
      __bulk_state<__Previous, __Size, __Fn, __R> __state_;
      /// The operation state object of the previous computation.
      __inner_op_state __previous_operation_state_;
    };
  } // namespace __detail

  /// The sender used to schedule bulk work in the system context.
  template <stdexec::sender __Previous, std::integral __Size, class __Fn>
  struct system_bulk_sender {
    /// Marks this type as being a sender; not to spec.
    using sender_concept = stdexec::sender_t;

    /// Constructs `this`.
    system_bulk_sender(system_scheduler __sched, __Previous __previous, __Size __size, __Fn&& __fun)
      : __scheduler_{__sched.__scheduler_}
      , __previous_{std::move(__previous)}
      , __size_{std::move(__size)}
      , __fun_{std::move(__fun)} {
    }

   private:
    STDEXEC_MEMFN_FRIEND(get_env);
    STDEXEC_MEMFN_FRIEND(connect);
    STDEXEC_MEMFN_FRIEND(get_completion_signatures);
    /// Meta-function that returns the completion signatures of `this`.
    template <typename __Self, typename __Env>
    using __completions_t = stdexec::make_completion_signatures<               //
      stdexec::__copy_cvref_t<__Self, __Previous>,                             //
      __Env,                                                                   //
      stdexec::completion_signatures<stdexec::set_error_t(std::exception_ptr)> //
      >;
    template <stdexec::sender, std::integral, class, class>
    friend struct __detail::__bulk_state;
    template <stdexec::sender, std::integral, class, class>
    friend struct __detail::__bulk_intermediate_receiver;

    /// Connects `__self` to `__r`, returning the operation state containing the work to be done.
    template <stdexec::receiver __R>
    STDEXEC_MEMFN_DECL(auto connect)(this system_bulk_sender&& __self, __R&& __r) //
      noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<__R>, __R>)
        -> __detail::__system_bulk_op<__Previous, __Size, __Fn, __R> {
      using __receiver_t = __detail::__bulk_intermediate_receiver<__Previous, __Size, __Fn, __R>;
      return {std::move(__self), std::move(__r), [](auto& __op) {
                // Connect bulk input receiver with the previous operation and store in the operating state.
                return stdexec::connect(
                  std::move(__op.__state_.__snd_.__previous_), __receiver_t{__op.__state_});
              }};
    }

    /// Gets the completion signatures for this sender.
    template <stdexec::__decays_to<system_bulk_sender> __Self, class __Env>
    STDEXEC_MEMFN_DECL(auto get_completion_signatures)(this __Self&&, __Env&&) -> __completions_t<__Self, __Env> {
      return {};
    }

    /// Gets the environment of this sender.
    STDEXEC_MEMFN_DECL(
      auto
      get_env)(this const system_bulk_sender& __snd) noexcept -> __detail::__system_scheduler_env {
      return {__snd.__scheduler_};
    }

    /// The underlying implementation of the scheduler we are using.
    __exec_system_scheduler_interface* __scheduler_{nullptr};
    /// The previous sender, the one that produces the input value for the bulk function.
    __Previous __previous_;
    /// The size of the bulk operation.
    __Size __size_;
    /// The function to be executed to perform the bulk work.
    __Fn __fun_;
  };

  inline system_context::system_context() {
    __impl_ = __get_exec_system_context_impl();
    // TODO error handling
  }

  inline system_scheduler system_context::get_scheduler() {
    return system_scheduler{__impl_->__get_scheduler(__impl_)};
  }

  inline size_t system_context::max_concurrency() const noexcept {
    return std::thread::hardware_concurrency();
  }

  stdexec::forward_progress_guarantee
    tag_invoke(stdexec::get_forward_progress_guarantee_t, const system_scheduler& __self) noexcept {
    switch (__self.__scheduler_->__forward_progress_guarantee) {
    case 0:
      return stdexec::forward_progress_guarantee::concurrent;
    case 1:
      return stdexec::forward_progress_guarantee::parallel;
    case 2:
      return stdexec::forward_progress_guarantee::weakly_parallel;
    default:
      return stdexec::forward_progress_guarantee::parallel;
    }
  }

  struct __transform_system_bulk_sender {
    template <class __Data, class __Previous>
    auto operator()(stdexec::bulk_t, __Data&& __data, __Previous&& __previous) const noexcept {
      auto [__shape, __fn] = static_cast<__Data&&>(__data);
      return system_bulk_sender<__Previous, decltype(__shape), decltype(__fn)>{
        __sched_, static_cast<__Previous&&>(__previous), __shape, std::move(__fn)};
    }

    system_scheduler __sched_;
  };

  template <class __Sender>
  struct __not_a_sender {
    using sender_concept = stdexec::sender_t;
  };

  template <stdexec::sender_expr_for<stdexec::bulk_t> __Sender, class __Env>
  auto
    system_scheduler_domain::transform_sender(__Sender&& __sndr, const __Env& __env) const noexcept {
    if constexpr (stdexec::__completes_on<__Sender, system_scheduler>) {
      auto __sched =
        stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(__sndr));
      return stdexec::__sexpr_apply(
        static_cast<__Sender&&>(__sndr), __transform_system_bulk_sender{__sched});
    } else if constexpr (stdexec::__starts_on<__Sender, system_scheduler, __Env>) {
      auto __sched = stdexec::get_scheduler(__env);
      return stdexec::__sexpr_apply(
        static_cast<__Sender&&>(__sndr), __transform_system_bulk_sender{__sched});
    } else {
      static_assert( //
        stdexec::__starts_on<__Sender, system_scheduler, __Env>
          || stdexec::__completes_on<__Sender, system_scheduler>,
        "No system_scheduler instance can be found in the sender's or receiver's "
        "environment on which to schedule bulk work.");
      return __not_a_sender<stdexec::__name_of<__Sender>>();
    }
  }
} // namespace exec

#if defined(__EXEC__SYSTEM_CONTEXT__HEADER_ONLY)
#  define __EXEC__SYSTEM_CONTEXT__INLINE inline
#  include "__detail/__system_context_default_impl_entry.hpp"
#endif
