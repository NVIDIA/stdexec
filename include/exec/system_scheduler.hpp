/*
 * Copyright (c) 2023 Lee Howes
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
// For the default implementation, test will override
#include "exec/static_thread_pool.hpp"


struct __exec_system_scheduler_interface;
struct __exec_system_sender_interface;
struct __exec_system_scheduler_impl;
struct __exec_system_sender_impl;


// Virtual interfaces to underlying implementations for initial simplicit
// TODO: Potentially move to custom vtable implementations
struct __exec_system_context_interface {
  virtual __exec_system_scheduler_interface* get_scheduler() noexcept = 0;
};

// bulk function for scheduler to transmit from, will wrap actual function stub stored in real type
using __exec_system_bulk_shape = long;
using __exec_system_bulk_fn = void(void*, __exec_system_bulk_shape);
struct __exec_system_bulk_function_object {
  void* __fn_state = nullptr;
  __exec_system_bulk_fn* __fn = nullptr;
};


struct __exec_system_scheduler_interface {
  virtual stdexec::forward_progress_guarantee get_forward_progress_guarantee() const = 0;
  virtual __exec_system_sender_interface* schedule() = 0;
  // TODO: Move chaining in here to support chaining after a system_sender or other system_bulk_sender
  // or don't do anything that specific?
  virtual __exec_system_sender_interface* bulk(__exec_system_bulk_shape __shp, __exec_system_bulk_function_object __fn) = 0;
  virtual bool equals(const __exec_system_scheduler_interface* __rhs) const = 0;
};

struct __exec_system_operation_state_interface {
  virtual void start() noexcept = 0;
};

struct __exec_system_receiver {
  void* __cpp_recv_ = nullptr;
  void (*set_value)(void* __cpp_recv) noexcept;
  void (*set_stopped)(void* __cpp_recv) noexcept;
  // Type-erase the exception pointer for extern-c-ness
  void (*set_error)(void* __cpp_recv, void* __exception) noexcept;

};

struct __exec_system_sender_interface {
  virtual __exec_system_operation_state_interface* connect(__exec_system_receiver __recv) noexcept = 0;
  virtual __exec_system_scheduler_interface* get_completion_scheduler() noexcept = 0;
};






// Low-level APIs
// Phase 2 will move to pointers and ref counting ala COM
// Phase 3 will move these to weak symbols and allow replacement in tests
// Default implementation based on static_thread_pool
struct __exec_system_context_impl : public __exec_system_context_interface {
  exec::static_thread_pool __pool_;

  __exec_system_scheduler_interface* get_scheduler() noexcept override;
};


struct __exec_system_scheduler_impl : public __exec_system_scheduler_interface {
  __exec_system_scheduler_impl(
      __exec_system_context_impl* __ctx, decltype(__ctx->__pool_.get_scheduler()) __pool_scheduler) :
      __ctx_{__ctx}, __pool_scheduler_{__pool_scheduler} {}

  __exec_system_context_impl* __ctx_;
  decltype(__ctx_->__pool_.get_scheduler()) __pool_scheduler_;

  __exec_system_sender_interface* schedule() override;

  __exec_system_sender_interface* bulk(__exec_system_bulk_shape __shp, __exec_system_bulk_function_object __fn) override;

  stdexec::forward_progress_guarantee get_forward_progress_guarantee() const override {
    return stdexec::forward_progress_guarantee::parallel;
  }

  bool equals(const __exec_system_scheduler_interface* __rhs) const override {
    auto __rhs_impl = dynamic_cast<const __exec_system_scheduler_impl*>(__rhs);
    return __rhs_impl && __rhs_impl->__ctx_ == __ctx_;
  }
};

struct __exec_system_operation_state_impl;
using __exec_pool_sender_t = decltype(stdexec::schedule(std::declval<__exec_system_scheduler_impl>().__pool_scheduler_));

struct __exec_system_pool_receiver {
  using receiver_concept = stdexec::receiver_t;

  friend void tag_invoke(stdexec::set_value_t, __exec_system_pool_receiver&&) noexcept;

  friend void tag_invoke(stdexec::set_stopped_t, __exec_system_pool_receiver&&) noexcept;

  friend void tag_invoke(stdexec::set_error_t, __exec_system_pool_receiver&&, std::exception_ptr) noexcept;

  friend stdexec::empty_env tag_invoke(stdexec::get_env_t, const __exec_system_pool_receiver&) noexcept {
    return {};
  }

  __exec_system_operation_state_impl* __os_ = nullptr;
};

struct __exec_system_operation_state_impl : public __exec_system_operation_state_interface {
  __exec_system_operation_state_impl(
    __exec_pool_sender_t&& __pool_sender,
    __exec_system_receiver&& __recv) :
    __recv_{std::move(__recv)},
    __pool_operation_state_{
      [&](){return stdexec::connect(std::move(__pool_sender), __exec_system_pool_receiver{this});}()} {
  }

  __exec_system_operation_state_impl(const __exec_system_operation_state_impl&) = delete;
  __exec_system_operation_state_impl(__exec_system_operation_state_impl&&) = delete;
  __exec_system_operation_state_impl& operator= (const __exec_system_operation_state_impl&) = delete;
  __exec_system_operation_state_impl& operator= (__exec_system_operation_state_impl&&) = delete;


  void start() noexcept override {
    stdexec::start(__pool_operation_state_);
  }

  __exec_system_receiver __recv_;
  stdexec::connect_result_t<__exec_pool_sender_t, __exec_system_pool_receiver>
    __pool_operation_state_;
};

inline void tag_invoke(stdexec::set_value_t, __exec_system_pool_receiver&& __recv) noexcept {
  __exec_system_receiver &__system_recv = __recv.__os_->__recv_;
  __system_recv.set_value(__system_recv.__cpp_recv_);
}

inline void tag_invoke(stdexec::set_stopped_t, __exec_system_pool_receiver&& __recv) noexcept {
  __exec_system_receiver &__system_recv = __recv.__os_->__recv_;
  __recv.__os_->__recv_.set_stopped(&(__system_recv.__cpp_recv_));
}

inline void tag_invoke(stdexec::set_error_t, __exec_system_pool_receiver&& __recv, std::exception_ptr __ptr) noexcept {
  __exec_system_receiver &__system_recv = __recv.__os_->__recv_;
  __recv.__os_->__recv_.set_error(&(__system_recv.__cpp_recv_), &__ptr);
}


struct __exec_system_sender_impl : public __exec_system_sender_interface {
  __exec_system_sender_impl(
        __exec_system_scheduler_impl* __scheduler, __exec_pool_sender_t&& __pool_sender) :
      __scheduler_{__scheduler}, __pool_sender_(std::move(__pool_sender)) {

  }

  __exec_system_operation_state_interface* connect(__exec_system_receiver __recv) noexcept override {
    return
      new __exec_system_operation_state_impl(std::move(__pool_sender_), std::move(__recv));
  }

  __exec_system_scheduler_interface* get_completion_scheduler() noexcept override {
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

  friend void tag_invoke(stdexec::set_error_t, __exec_system_bulk_pool_receiver&&, std::exception_ptr) noexcept;

  friend stdexec::empty_env tag_invoke(stdexec::get_env_t, const __exec_system_bulk_pool_receiver&) noexcept {
    return {};
  }

  __exec_system_bulk_operation_state_impl* __os_ = nullptr;
};

auto __exec_pool_operation_state(
    __exec_system_bulk_operation_state_impl* __self,
    __exec_pool_sender_t&& __ps,
    __exec_system_bulk_shape __shp,
    __exec_system_bulk_function_object __fn) {
 return stdexec::connect(
          stdexec::bulk(
            std::move(__ps),
            __shp,
            [__fn](long __idx){
              __fn.__fn(__fn.__fn_state, __idx);
            }),
          __exec_system_bulk_pool_receiver{__self});
}

struct __exec_system_bulk_operation_state_impl : public __exec_system_operation_state_interface {
  __exec_system_bulk_operation_state_impl(
    __exec_pool_sender_t&& __pool_sender,
    __exec_system_bulk_shape __bulk_shape,
    __exec_system_bulk_function_object __bulk_function,
    __exec_system_receiver&& __recv) :
    __recv_{std::move(__recv)},
    __bulk_function_{__bulk_function},
    __pool_operation_state_{__exec_pool_operation_state(this, std::move(__pool_sender), __bulk_shape, __bulk_function_)} {
  }

  __exec_system_bulk_operation_state_impl(const __exec_system_bulk_operation_state_impl&) = delete;
  __exec_system_bulk_operation_state_impl(__exec_system_bulk_operation_state_impl&&) = delete;
  __exec_system_bulk_operation_state_impl& operator= (const __exec_system_bulk_operation_state_impl&) = delete;
  __exec_system_bulk_operation_state_impl& operator= (__exec_system_bulk_operation_state_impl&&) = delete;


  void start() noexcept override {
    stdexec::start(__pool_operation_state_);
  }

  __exec_system_receiver __recv_;
  __exec_system_bulk_function_object __bulk_function_;
  stdexec::__result_of<__exec_pool_operation_state, __exec_system_bulk_operation_state_impl*, __exec_pool_sender_t, __exec_system_bulk_shape, __exec_system_bulk_function_object>
    __pool_operation_state_;
};

inline void tag_invoke(stdexec::set_value_t, __exec_system_bulk_pool_receiver&& __recv) noexcept {
  __exec_system_receiver &__system_recv = __recv.__os_->__recv_;
  __system_recv.set_value((__system_recv.__cpp_recv_));
}

inline void tag_invoke(stdexec::set_stopped_t, __exec_system_bulk_pool_receiver&& __recv) noexcept {
  __exec_system_receiver &__system_recv = __recv.__os_->__recv_;
  __recv.__os_->__recv_.set_stopped(&(__system_recv.__cpp_recv_));
}

inline void tag_invoke(stdexec::set_error_t, __exec_system_bulk_pool_receiver&& __recv, std::exception_ptr __ptr) noexcept {
  __exec_system_receiver &__system_recv = __recv.__os_->__recv_;
  __recv.__os_->__recv_.set_error(&(__system_recv.__cpp_recv_), &__ptr);
}

// A bulk sender is just a system sender viewed externally.
// TODO: a bulk operation state is just a system operation state viewed externally
struct __exec_system_bulk_sender_impl : public __exec_system_sender_interface {
  __exec_system_bulk_sender_impl(
        __exec_system_scheduler_impl* __scheduler,
        __exec_system_bulk_shape __bulk_shape,
        __exec_system_bulk_function_object __bulk_function,
        __exec_pool_sender_t&& __pool_sender) :
      __scheduler_{__scheduler},
      __bulk_shape_{__bulk_shape},
      __bulk_function_{__bulk_function},
      __pool_sender_(std::move(__pool_sender)) {

  }

  __exec_system_operation_state_interface* connect(__exec_system_receiver __recv) noexcept override {
    return
      new __exec_system_bulk_operation_state_impl(
        std::move(__pool_sender_), __bulk_shape_, __bulk_function_, std::move(__recv));
  }

  __exec_system_scheduler_interface* get_completion_scheduler() noexcept override {
    return __scheduler_;
  };

  __exec_system_scheduler_impl* __scheduler_;
  __exec_system_bulk_shape __bulk_shape_;
  __exec_system_bulk_function_object __bulk_function_;
  __exec_pool_sender_t __pool_sender_;
};


inline __exec_system_scheduler_interface* __exec_system_context_impl::get_scheduler() noexcept {
  // TODO: ref counting etc
  return new __exec_system_scheduler_impl(this, __pool_.get_scheduler());
}

inline __exec_system_sender_interface* __exec_system_scheduler_impl::schedule() {
  return new __exec_system_sender_impl(this, stdexec::schedule(__pool_scheduler_));
}


inline __exec_system_sender_interface* __exec_system_scheduler_impl::bulk(
    __exec_system_bulk_shape __shp,
    __exec_system_bulk_function_object __fn) {
  // This is bulk off a system_scheduler, so we need to start with schedule.
  // TODO: a version later will key off a bulk *sender* and would behave slightly
  // differently.
  // In both cases pass in the result of schedule, or the predecessor though.
  return new __exec_system_bulk_sender_impl(this, __shp, __fn, stdexec::schedule(__pool_scheduler_));
}



// Phase 1 implementation, single implementation
// TODO: Make a weak symbol and replace in a test
static __exec_system_context_impl* __get_exec_system_context_impl() {
  static __exec_system_context_impl __impl_;

  return &__impl_;
}

// TODO: Move everything above here to a detail header and wrap in a
// namespace to represent extern "C"


namespace exec {
  namespace __system_scheduler {

  } // namespace system_scheduler


  class system_scheduler;
  class system_sender;
  template<stdexec::sender __S, std::integral __Shape, class __Fn>
  struct system_bulk_sender;

  class system_context {
  public:
    system_context() {
      __impl_ = __get_exec_system_context_impl();
      // TODO error handling
    }

    system_context(const system_context&) = delete;
    system_context(system_context&&) = delete;
    system_context& operator=(const system_context&) = delete;
    system_context& operator=(system_context&&) = delete;

    system_scheduler get_scheduler();

    size_t max_concurrency() const noexcept;

  private:
    __exec_system_context_interface* __impl_ = nullptr;

  };

  class system_scheduler {
  public:

    // Pointer that we ref count?
    system_scheduler(__exec_system_scheduler_interface* __scheduler_interface) : __scheduler_interface_(__scheduler_interface) {}

    bool operator==(const system_scheduler& __rhs) const noexcept {
      return __scheduler_interface_->equals(__rhs.__scheduler_interface_);
    }

  private:
    friend system_sender tag_invoke(
      stdexec::schedule_t, const system_scheduler&) noexcept;

    friend stdexec::forward_progress_guarantee tag_invoke(
      stdexec::get_forward_progress_guarantee_t,
      const system_scheduler&) noexcept;

    template <stdexec::sender __S, std::integral __Shape, class __Fn>
    friend system_bulk_sender<__S, __Shape, __Fn> tag_invoke( //
      stdexec::bulk_t,                                        //
      const system_scheduler& __sch,                          //
      __S&& __sndr,                                           //
      __Shape __shape,                                        //
      __Fn __fun)                                             //
      noexcept;

    __exec_system_scheduler_interface* __scheduler_interface_;
    friend class system_context;
  };

  class system_sender {
  public:
    using is_sender = void;
    using completion_signatures =
      stdexec::completion_signatures< stdexec::set_value_t(), stdexec::set_stopped_t(), stdexec::set_error_t(std::exception_ptr) >;

    system_sender(__exec_system_sender_interface* __sender_impl) : __sender_impl_{__sender_impl} {}

  private:
    template <class __S, class __R>
    struct __op {
      template<class __F>
      __op(system_sender&& __snd, __R&& __recv, __F&& __initFunc) :
          __snd_{std::move(__snd)}, __recv_{std::move(__recv)}, __os_{__initFunc(*this)} {
      }
      __op(const __op&) = delete;
      __op(__op&&) = delete;
      __op& operator= (const __op&) = delete;
      __op& operator= (__op&&) = delete;

      friend void tag_invoke(stdexec::start_t, __op& __op_v) noexcept {
        if(auto __os = __op_v.__os_) {
          __os->start();
        }
      }

      __S __snd_;
      __R __recv_;
      __exec_system_operation_state_interface* __os_ = nullptr;
    };

    template <class __R>
    friend auto tag_invoke(stdexec::connect_t, system_sender&& __snd, __R&& __rec) //
      noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<__R>, __R>)
        -> __op<system_sender, std::remove_cvref_t<__R>> {

      return __op<system_sender, std::remove_cvref_t<__R>>{
        std::move(__snd),
        std::move(__rec),
        [](auto& __op){
          __exec_system_receiver __receiver_impl{
            &__op.__recv_,
            [](void* __cpp_recv) noexcept{
              stdexec::set_value(std::move(*static_cast<__R*>(__cpp_recv)));
            },
            [](void* __cpp_recv) noexcept{
              stdexec::set_stopped(std::move(*static_cast<__R*>(__cpp_recv)));
            },
            [](void* __cpp_recv, void* __exception) noexcept{
              stdexec::set_error(
                std::move(*static_cast<__R*>(__cpp_recv)),
                std::move(*reinterpret_cast<std::exception_ptr*>(&__exception)));
            }};

          return __op.__snd_.__sender_impl_->connect(std::move(__receiver_impl));
        }};
    }

    struct __env {
      friend system_scheduler
        tag_invoke(stdexec::get_completion_scheduler_t<stdexec::set_value_t>, const __env& __self) //
        noexcept {
        return {__self.__scheduler_impl_};
      }

      friend system_scheduler
        tag_invoke(stdexec::get_completion_scheduler_t<stdexec::set_stopped_t>, const __env& __self) //
        noexcept {
        return {__self.__scheduler_impl_};
      }

      __exec_system_scheduler_interface* __scheduler_impl_;
    };

    friend __env tag_invoke(stdexec::get_env_t, const system_sender& __snd) noexcept {
      return {__snd.__sender_impl_->get_completion_scheduler()};
    }

    __exec_system_sender_interface* __sender_impl_ = nullptr;
  };

  template <stdexec::sender __Pred, std::integral __Shape, class __Fn, class __R>
  struct __bulk_state {
    system_bulk_sender<__Pred, __Shape, __Fn> __snd_;
    __R __recv_;
    void* __arg_data_ = nullptr;
    __exec_system_operation_state_interface* __os_ = nullptr;
  };

  template <stdexec::sender __Pred, std::integral __Shape, class __Fn, class __R>
  struct __bulk_recv {
    using receiver_concept = stdexec::receiver_t;

    __bulk_state<__Pred, __Shape, __Fn, __R>& __state_;

    template <class... __As>
    friend void tag_invoke(stdexec::set_value_t, __bulk_recv&& __self, __As&&... __as) noexcept {

      // Heap allocate input data in shared state as needed
      std::tuple<__As...> *__inputs = new std::tuple<__As...>{__as...};
      __self.__state_.__arg_data_ = __inputs;

      // Construct bulk operation with type conversions to use C ABI state
      auto __sched = __self.__state_.__snd_.__scheduler_impl_;
      if(__sched) {
        __exec_system_bulk_function_object __fn {
          &__self.__state_,
          [](void* __state_, long __idx){
            __bulk_state<__Pred, __Shape, __Fn, __R>* __state =
              static_cast<__bulk_state<__Pred, __Shape, __Fn, __R>*>(__state_);

            std::apply(
              [&](auto &&... __args) {
                __state->__snd_.__fun_(__idx, __args...);
              },
              *static_cast<std::tuple<__As...> *>(__state->__arg_data_));
          }};

        auto* __sender = __sched->bulk(__self.__state_.__snd_.__shape_, __fn);
        // Connect to a type-erasing receiver to call our receiver on completion
        __self.__state_.__os_ = __sender->connect(
          __exec_system_receiver{
            &__self.__state_.__recv_,
            [](void* __cpp_recv) noexcept{
              stdexec::set_value(std::move(*static_cast<__R*>(__cpp_recv)));
            },
            [](void* __cpp_recv) noexcept{
              stdexec::set_stopped(std::move(*static_cast<__R*>(__cpp_recv)));
            },
            [](void* __cpp_recv, void* __exception) noexcept{
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

    friend void tag_invoke(stdexec::set_error_t, __bulk_recv&& __self, std::exception_ptr __ptr) noexcept {
      stdexec::set_error(std::move(__self.__state_.__recv_), std::move(__ptr));
    }

    friend auto tag_invoke(stdexec::get_env_t, const __bulk_recv& __self) noexcept {
      return stdexec::get_env(__self.__state_.__recv_);
    }
  };

  template <stdexec::sender __Pred, std::integral __Shape, class __Fn, class __R>
  struct __bulk_op {
    using __inner_op_state = stdexec::connect_result_t<__Pred, __bulk_recv<__Pred, __Shape, __Fn, __R>>;

    template<class __InitF>
    __bulk_op(system_bulk_sender<__Pred, __Shape, __Fn>&& __snd, __R&& __recv, __InitF&& __initFunc) :
        __state_{std::move(__snd), std::move(__recv)}, __pred_operation_state_{__initFunc(*this)} {
    }
    __bulk_op(const __bulk_op&) = delete;
    __bulk_op(__bulk_op&&) = delete;
    __bulk_op& operator= (const __bulk_op&) = delete;
    __bulk_op& operator= (__bulk_op&&) = delete;

    friend void tag_invoke(stdexec::start_t, __bulk_op& __op) noexcept {
      if(auto __os = __op.__state_.__os_) {
        __os->start();
      }
      // Start inner operation state
      // Bulk operation will be started when that completes
      stdexec::start(__op.__pred_operation_state_);
    }

    __bulk_state<__Pred, __Shape, __Fn, __R> __state_;
    __inner_op_state __pred_operation_state_;
  };

  template<stdexec::sender __Pred, std::integral __Shape, class __Fn>
  struct system_bulk_sender {
    using __Sender = __Pred;
    using __Fun = __Fn;
    using is_sender = void;
    using completion_signatures =
      stdexec::completion_signatures< stdexec::set_value_t(), stdexec::set_stopped_t(), stdexec::set_error_t(std::exception_ptr) >;
      // TODO: This can complete with different values... should propagate from __Pred

    system_bulk_sender(
      __exec_system_scheduler_interface* __scheduler_impl,
      __Sender __pred,
      __Shape __shape,
      __Fun&& __fun) :
      __scheduler_impl_{__scheduler_impl},
      __pred_{std::move(__pred)},
      __shape_{std::move(__shape)},
      __fun_{std::move(__fun)} {}

    template <class __R>
    friend auto tag_invoke(stdexec::connect_t, system_bulk_sender&& __snd, __R&& __rec) //
      noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<__R>, __R>)
        -> __bulk_op<__Pred, __Shape, __Fn, __R> {

      return __bulk_op<__Pred, __Shape, __Fn, __R>{
        std::move(__snd),
        std::move(__rec),
        [](auto& __op){
          // Connect bulk input receiver with the previous operation and store in the OS
          return stdexec::connect(std::move(__op.__state_.__snd_.__pred_), __bulk_recv<__Pred, __Shape, __Fn, __R>{__op.__state_});
        }};
    }

    struct __env {
      friend system_scheduler
        tag_invoke(stdexec::get_completion_scheduler_t<stdexec::set_value_t>, const __env& __self) //
        noexcept {
        return {__self.__scheduler_impl_};
      }

      friend system_scheduler
        tag_invoke(stdexec::get_completion_scheduler_t<stdexec::set_stopped_t>, const __env& __self) //
        noexcept {
        return {__self.__scheduler_impl_};
      }


      __exec_system_scheduler_interface* __scheduler_impl_;
    };

    friend __env tag_invoke(stdexec::get_env_t, const system_bulk_sender& __snd) noexcept {
      // If we trigger this customization we know what the completion scheduler will be
      return {__snd.__scheduler_impl_};
    }

    __exec_system_scheduler_interface* __scheduler_impl_ = nullptr;
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

  system_sender tag_invoke(
      stdexec::schedule_t, const system_scheduler& sched) noexcept {
    return system_sender(sched.__scheduler_interface_->schedule());
  }

  stdexec::forward_progress_guarantee tag_invoke(
      stdexec::get_forward_progress_guarantee_t,
      const system_scheduler& __sched) noexcept {
    return __sched.__scheduler_interface_->get_forward_progress_guarantee();
  }


  template <stdexec::sender __S, std::integral __Shape, class __Fn>
  system_bulk_sender<__S, __Shape, __Fn> tag_invoke(        //
    stdexec::bulk_t,                                        //
    const system_scheduler& __sch,                          //
    __S&& __pred,                                           //
    __Shape __shape,                                        //
    __Fn __fun)                                             //
    noexcept {
    return system_bulk_sender<__S, __Shape, __Fn>{
      __sch.__scheduler_interface_, (__S&&) __pred, __shape, (__Fn&&) __fun};
  }


} // namespace exec
