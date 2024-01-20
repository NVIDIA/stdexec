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

  class system_scheduler;
  class system_sender;
  template <stdexec::sender __S, std::integral __Shape, class __Fn>
  struct system_bulk_sender;

  class system_context {
   public:
    system_context() {
      __impl_ = __system_context_default_impl::__get_exec_system_context_impl();
      // TODO error handling
    }

    system_context(const system_context&) = delete;
    system_context(system_context&&) = delete;
    system_context& operator=(const system_context&) = delete;
    system_context& operator=(system_context&&) = delete;

    system_scheduler get_scheduler();

    size_t max_concurrency() const noexcept;

   private:
    __if::__exec_system_context_interface* __impl_ = nullptr;
  };

  class system_scheduler {
   public:
    // Pointer that we ref count?
    system_scheduler(__if::__exec_system_scheduler_interface* __scheduler_interface)
      : __scheduler_interface_(__scheduler_interface) {
    }

    bool operator==(const system_scheduler& __rhs) const noexcept {
      return __scheduler_interface_->equals(__rhs.__scheduler_interface_);
    }

   private:
    friend system_sender tag_invoke(stdexec::schedule_t, const system_scheduler&) noexcept;

    friend stdexec::forward_progress_guarantee
      tag_invoke(stdexec::get_forward_progress_guarantee_t, const system_scheduler&) noexcept;

    template <stdexec::sender __S, std::integral __Shape, class __Fn>
    friend system_bulk_sender<__S, __Shape, __Fn> tag_invoke( //
      stdexec::bulk_t,                                        //
      const system_scheduler& __sch,                          //
      __S&& __sndr,                                           //
      __Shape __shape,                                        //
      __Fn __fun)                                             //
      noexcept;

    __if::__exec_system_scheduler_interface* __scheduler_interface_;
    friend class system_context;
  };

  class system_sender {
   public:
    using is_sender = void;
    using completion_signatures = stdexec::completion_signatures<
      stdexec::set_value_t(),
      stdexec::set_stopped_t(),
      stdexec::set_error_t(std::exception_ptr) >;

    system_sender(__if::__exec_system_sender_interface* __sender_impl)
      : __sender_impl_{__sender_impl} {
    }

   private:
    template <class __S, class __R>
    struct __op {
      template <class __F>
      __op(system_sender&& __snd, __R&& __recv, __F&& __initFunc)
        : __snd_{std::move(__snd)}
        , __recv_{std::move(__recv)}
        , __os_{__initFunc(*this)} {
      }

      __op(const __op&) = delete;
      __op(__op&&) = delete;
      __op& operator=(const __op&) = delete;
      __op& operator=(__op&&) = delete;

      friend void tag_invoke(stdexec::start_t, __op& __op_v) noexcept {
        if (auto __os = __op_v.__os_) {
          __os->start();
        }
      }

      __S __snd_;
      __R __recv_;
      __if::__exec_system_operation_state_interface* __os_ = nullptr;
    };

    template <class __R>
    friend auto tag_invoke(stdexec::connect_t, system_sender&& __snd, __R&& __rec) //
      noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<__R>, __R>)
        -> __op<system_sender, std::remove_cvref_t<__R>> {

      return __op<system_sender, std::remove_cvref_t<__R>>{
        std::move(__snd), std::move(__rec), [](auto& __op) {
          __if::__exec_system_receiver __receiver_impl{
            &__op.__recv_,
            [](void* __cpp_recv) noexcept {
              stdexec::set_value(std::move(*static_cast<__R*>(__cpp_recv)));
            },
            [](void* __cpp_recv) noexcept {
              stdexec::set_stopped(std::move(*static_cast<__R*>(__cpp_recv)));
            },
            [](void* __cpp_recv, void* __exception) noexcept {
              stdexec::set_error(
                std::move(*static_cast<__R*>(__cpp_recv)),
                std::move(*reinterpret_cast<std::exception_ptr*>(&__exception)));
            }};

          return __op.__snd_.__sender_impl_->connect(std::move(__receiver_impl));
        }};
    }

    struct __env {
      friend system_scheduler tag_invoke(
        stdexec::get_completion_scheduler_t<stdexec::set_value_t>,
        const __env& __self) //
        noexcept {
        return {__self.__scheduler_impl_};
      }

      friend system_scheduler tag_invoke(
        stdexec::get_completion_scheduler_t<stdexec::set_stopped_t>,
        const __env& __self) //
        noexcept {
        return {__self.__scheduler_impl_};
      }

      __if::__exec_system_scheduler_interface* __scheduler_impl_;
    };

    friend __env tag_invoke(stdexec::get_env_t, const system_sender& __snd) noexcept {
      return {__snd.__sender_impl_->get_completion_scheduler()};
    }

    __if::__exec_system_sender_interface* __sender_impl_ = nullptr;
  };

  template <stdexec::sender __Pred, std::integral __Shape, class __Fn, class __R>
  struct __bulk_state {
    system_bulk_sender<__Pred, __Shape, __Fn> __snd_;
    __R __recv_;
    void* __arg_data_ = nullptr;
    __if::__exec_system_operation_state_interface* __os_ = nullptr;
  };

  template <stdexec::sender __Pred, std::integral __Shape, class __Fn, class __R>
  struct __bulk_recv {
    using receiver_concept = stdexec::receiver_t;

    __bulk_state<__Pred, __Shape, __Fn, __R>& __state_;

    template <class... __As>
    friend void tag_invoke(stdexec::set_value_t, __bulk_recv&& __self, __As&&... __as) noexcept {

      // Heap allocate input data in shared state as needed
      std::tuple<__As...>* __inputs = new std::tuple<__As...>{__as...};
      __self.__state_.__arg_data_ = __inputs;

      // Construct bulk operation with type conversions to use C ABI state
      auto __sched = __self.__state_.__snd_.__scheduler_impl_;
      if (__sched) {
        __if::__exec_system_bulk_function_object __fn{
          &__self.__state_, [](void* __state_, long __idx) {
            __bulk_state<__Pred, __Shape, __Fn, __R>* __state =
              static_cast<__bulk_state<__Pred, __Shape, __Fn, __R>*>(__state_);

            std::apply(
              [&](auto&&... __args) { __state->__snd_.__fun_(__idx, __args...); },
              *static_cast<std::tuple<__As...>*>(__state->__arg_data_));
          }};

        auto* __sender = __sched->bulk(__self.__state_.__snd_.__shape_, __fn);
        // Connect to a type-erasing receiver to call our receiver on completion
        __self.__state_.__os_ = __sender->connect(__if::__exec_system_receiver{
          &__self.__state_.__recv_,
          [](void* __cpp_recv) noexcept {
            stdexec::set_value(std::move(*static_cast<__R*>(__cpp_recv)));
          },
          [](void* __cpp_recv) noexcept {
            stdexec::set_stopped(std::move(*static_cast<__R*>(__cpp_recv)));
          },
          [](void* __cpp_recv, void* __exception) noexcept {
            stdexec::set_error(
              std::move(*static_cast<__R*>(__cpp_recv)),
              std::move(*static_cast<std::exception_ptr*>(__exception)));
          }});
        __self.__state_.__os_->start();
      }
    }

    friend void tag_invoke(stdexec::set_stopped_t, __bulk_recv&& __self) noexcept {
      stdexec::set_stopped(std::move(__self.__state_.__recv_));
    }

    friend void
      tag_invoke(stdexec::set_error_t, __bulk_recv&& __self, std::exception_ptr __ptr) noexcept {
      stdexec::set_error(std::move(__self.__state_.__recv_), std::move(__ptr));
    }

    friend auto tag_invoke(stdexec::get_env_t, const __bulk_recv& __self) noexcept {
      return stdexec::get_env(__self.__state_.__recv_);
    }
  };

  template <stdexec::sender __Pred, std::integral __Shape, class __Fn, class __R>
  struct __bulk_op {
    using __inner_op_state =
      stdexec::connect_result_t<__Pred, __bulk_recv<__Pred, __Shape, __Fn, __R>>;

    template <class __InitF>
    __bulk_op(system_bulk_sender<__Pred, __Shape, __Fn>&& __snd, __R&& __recv, __InitF&& __initFunc)
      : __state_{std::move(__snd), std::move(__recv)}
      , __pred_operation_state_{__initFunc(*this)} {
    }

    __bulk_op(const __bulk_op&) = delete;
    __bulk_op(__bulk_op&&) = delete;
    __bulk_op& operator=(const __bulk_op&) = delete;
    __bulk_op& operator=(__bulk_op&&) = delete;

    friend void tag_invoke(stdexec::start_t, __bulk_op& __op) noexcept {
      if (auto __os = __op.__state_.__os_) {
        __os->start();
      }
      // Start inner operation state
      // Bulk operation will be started when that completes
      stdexec::start(__op.__pred_operation_state_);
    }

    __bulk_state<__Pred, __Shape, __Fn, __R> __state_;
    __inner_op_state __pred_operation_state_;
  };

  template <stdexec::sender __Pred, std::integral __Shape, class __Fn>
  struct system_bulk_sender {
    using __Sender = __Pred;
    using __Fun = __Fn;
    using is_sender = void;
    using completion_signatures = stdexec::completion_signatures<
      stdexec::set_value_t(),
      stdexec::set_stopped_t(),
      stdexec::set_error_t(std::exception_ptr) >;

    // TODO: This can complete with different values... should propagate from __Pred

    system_bulk_sender(
      __if::__exec_system_scheduler_interface* __scheduler_impl,
      __Sender __pred,
      __Shape __shape,
      __Fun&& __fun)
      : __scheduler_impl_{__scheduler_impl}
      , __pred_{std::move(__pred)}
      , __shape_{std::move(__shape)}
      , __fun_{std::move(__fun)} {
    }

    template <class __R>
    friend auto tag_invoke(stdexec::connect_t, system_bulk_sender&& __snd, __R&& __rec) //
      noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<__R>, __R>)
        -> __bulk_op<__Pred, __Shape, __Fn, __R> {

      return __bulk_op<__Pred, __Shape, __Fn, __R>{
        std::move(__snd), std::move(__rec), [](auto& __op) {
          // Connect bulk input receiver with the previous operation and store in the OS
          return stdexec::connect(
            std::move(__op.__state_.__snd_.__pred_),
            __bulk_recv<__Pred, __Shape, __Fn, __R>{__op.__state_});
        }};
    }

    struct __env {
      friend system_scheduler tag_invoke(
        stdexec::get_completion_scheduler_t<stdexec::set_value_t>,
        const __env& __self) //
        noexcept {
        return {__self.__scheduler_impl_};
      }

      friend system_scheduler tag_invoke(
        stdexec::get_completion_scheduler_t<stdexec::set_stopped_t>,
        const __env& __self) //
        noexcept {
        return {__self.__scheduler_impl_};
      }

      __if::__exec_system_scheduler_interface* __scheduler_impl_;
    };

    friend __env tag_invoke(stdexec::get_env_t, const system_bulk_sender& __snd) noexcept {
      // If we trigger this customization we know what the completion scheduler will be
      return {__snd.__scheduler_impl_};
    }

    __if::__exec_system_scheduler_interface* __scheduler_impl_ = nullptr;
    __Sender __pred_;
    __Shape __shape_;
    __Fun __fun_;
  };

  inline system_scheduler system_context::get_scheduler() {
    return system_scheduler{__impl_->get_scheduler()};
  }

  inline size_t system_context::max_concurrency() const noexcept {
    return std::thread::hardware_concurrency();
  }

  system_sender tag_invoke(stdexec::schedule_t, const system_scheduler& sched) noexcept {
    return system_sender(sched.__scheduler_interface_->schedule());
  }

  stdexec::forward_progress_guarantee tag_invoke(
    stdexec::get_forward_progress_guarantee_t,
    const system_scheduler& __sched) noexcept {
    return __sched.__scheduler_interface_->get_forward_progress_guarantee();
  }

  template <stdexec::sender __S, std::integral __Shape, class __Fn>
  system_bulk_sender<__S, __Shape, __Fn> tag_invoke( //
    stdexec::bulk_t,                                 //
    const system_scheduler& __sch,                   //
    __S&& __pred,                                    //
    __Shape __shape,                                 //
    __Fn __fun)                                      //
    noexcept {
    return system_bulk_sender<__S, __Shape, __Fn>{
      __sch.__scheduler_interface_, (__S&&) __pred, __shape, (__Fn&&) __fun};
  }


} // namespace exec
