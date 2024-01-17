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

#include <iostream>

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
  void* fn_state = nullptr;
  __exec_system_bulk_fn* fn = nullptr;
};


struct __exec_system_scheduler_interface {
  virtual stdexec::forward_progress_guarantee get_forward_progress_guarantee() const = 0;
  virtual __exec_system_sender_interface* schedule() = 0;
  // TODO: Move chaining in here to support chaining after a system_sender or other system_bulk_sender
  // or don't do anything that specific?
  virtual __exec_system_sender_interface* bulk(__exec_system_bulk_shape shp, __exec_system_bulk_function_object fn) = 0;
  virtual bool equals(const __exec_system_scheduler_interface* rhs) const = 0;
};

struct __exec_system_operation_state_interface {
  virtual void start() noexcept = 0;
};

struct __exec_system_receiver {
  void* cpp_recv_ = nullptr;
  void (*set_value)(void* cpp_recv) noexcept;
  void (*set_stopped)(void* cpp_recv) noexcept;
  // Type-erase the exception pointer for extern-c-ness
  void (*set_error)(void* cpp_recv, void* exception) noexcept;

};

struct __exec_system_sender_interface {
  virtual __exec_system_operation_state_interface* connect(__exec_system_receiver recv) noexcept = 0;
  virtual __exec_system_scheduler_interface* get_completion_scheduler() noexcept = 0;
};






// Low-level APIs
// Phase 2 will move to pointers and ref counting ala COM
// Phase 3 will move these to weak symbols and allow replacement in tests
// Default implementation based on static_thread_pool
struct __exec_system_context_impl : public __exec_system_context_interface {
  exec::static_thread_pool pool_;

  __exec_system_scheduler_interface* get_scheduler() noexcept override;
};


struct __exec_system_scheduler_impl : public __exec_system_scheduler_interface {
  __exec_system_scheduler_impl(
      __exec_system_context_impl* ctx, decltype(ctx->pool_.get_scheduler()) pool_scheduler) :
      ctx_{ctx}, pool_scheduler_{pool_scheduler} {}

  __exec_system_context_impl* ctx_;
  decltype(ctx_->pool_.get_scheduler()) pool_scheduler_;

  __exec_system_sender_interface* schedule() override;

  __exec_system_sender_interface* bulk(__exec_system_bulk_shape shp, __exec_system_bulk_function_object fn) override;

  stdexec::forward_progress_guarantee get_forward_progress_guarantee() const override {
    return stdexec::forward_progress_guarantee::parallel;
  }

  bool equals(const __exec_system_scheduler_interface* rhs) const override {
    auto rhs_impl = dynamic_cast<const __exec_system_scheduler_impl*>(rhs);
    return rhs_impl && rhs_impl->ctx_ == ctx_;
  }
};

struct __exec_system_operation_state_impl;
using __exec_pool_sender_t = decltype(stdexec::schedule(std::declval<__exec_system_scheduler_impl>().pool_scheduler_));

struct __exec_system_pool_receiver {
  using receiver_concept = stdexec::receiver_t;

  friend void tag_invoke(stdexec::set_value_t, __exec_system_pool_receiver&&) noexcept;

  friend void tag_invoke(stdexec::set_stopped_t, __exec_system_pool_receiver&&) noexcept;

  friend void tag_invoke(stdexec::set_error_t, __exec_system_pool_receiver&&, std::exception_ptr) noexcept;

  friend stdexec::empty_env tag_invoke(stdexec::get_env_t, const __exec_system_pool_receiver&) noexcept {
    return {};
  }

  __exec_system_operation_state_impl* os_ = nullptr;
};

struct __exec_system_operation_state_impl : public __exec_system_operation_state_interface {
  __exec_system_operation_state_impl(
    __exec_pool_sender_t&& pool_sender,
    __exec_system_receiver&& recv) :
    recv_{std::move(recv)},
    pool_operation_state_{
      [&](){return stdexec::connect(std::move(pool_sender), __exec_system_pool_receiver{this});}()} {
  }

  __exec_system_operation_state_impl(const __exec_system_operation_state_impl&) = delete;
  __exec_system_operation_state_impl(__exec_system_operation_state_impl&&) = delete;
  __exec_system_operation_state_impl& operator= (const __exec_system_operation_state_impl&) = delete;
  __exec_system_operation_state_impl& operator= (__exec_system_operation_state_impl&&) = delete;


  void start() noexcept override {
    stdexec::start(pool_operation_state_);
  }

  __exec_system_receiver recv_;
  stdexec::connect_result_t<__exec_pool_sender_t, __exec_system_pool_receiver>
    pool_operation_state_;
};

inline void tag_invoke(stdexec::set_value_t, __exec_system_pool_receiver&& recv) noexcept {
  __exec_system_receiver &system_recv = recv.os_->recv_;
  system_recv.set_value(&(system_recv.cpp_recv_));
}

inline void tag_invoke(stdexec::set_stopped_t, __exec_system_pool_receiver&& recv) noexcept {
  __exec_system_receiver &system_recv = recv.os_->recv_;
  recv.os_->recv_.set_stopped(&(system_recv.cpp_recv_));
}

inline void tag_invoke(stdexec::set_error_t, __exec_system_pool_receiver&& recv, std::exception_ptr ptr) noexcept {
  __exec_system_receiver &system_recv = recv.os_->recv_;
  recv.os_->recv_.set_error(&(system_recv.cpp_recv_), &ptr);
}


struct __exec_system_sender_impl : public __exec_system_sender_interface {
  __exec_system_sender_impl(
        __exec_system_scheduler_impl* scheduler, __exec_pool_sender_t&& pool_sender) :
      scheduler_{scheduler}, pool_sender_(std::move(pool_sender)) {

  }

  __exec_system_operation_state_interface* connect(__exec_system_receiver recv) noexcept override {
    return
      new __exec_system_operation_state_impl(std::move(pool_sender_), std::move(recv));
  }

  __exec_system_scheduler_interface* get_completion_scheduler() noexcept override {
    return scheduler_;
  };

   __exec_system_scheduler_impl* scheduler_;
   __exec_pool_sender_t pool_sender_;
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

  __exec_system_bulk_operation_state_impl* os_ = nullptr;
};

auto __exec_pool_operation_state(
    __exec_system_bulk_operation_state_impl* self,
    __exec_pool_sender_t&& ps,
    __exec_system_bulk_shape shp,
    __exec_system_bulk_function_object fn) {
 return stdexec::connect(
          stdexec::bulk(
            std::move(ps),
            shp,
            [fn](long idx){
              fn.fn(fn.fn_state, idx);
            }),
          __exec_system_bulk_pool_receiver{self});
}

struct __exec_system_bulk_operation_state_impl : public __exec_system_operation_state_interface {
  __exec_system_bulk_operation_state_impl(
    __exec_pool_sender_t&& pool_sender,
    __exec_system_bulk_shape bulk_shape,
    __exec_system_bulk_function_object bulk_function,
    __exec_system_receiver&& recv) :
    recv_{std::move(recv)},
    bulk_function_{bulk_function},
    pool_operation_state_{__exec_pool_operation_state(this, std::move(pool_sender), bulk_shape, bulk_function_)} {
  }

  __exec_system_bulk_operation_state_impl(const __exec_system_bulk_operation_state_impl&) = delete;
  __exec_system_bulk_operation_state_impl(__exec_system_bulk_operation_state_impl&&) = delete;
  __exec_system_bulk_operation_state_impl& operator= (const __exec_system_bulk_operation_state_impl&) = delete;
  __exec_system_bulk_operation_state_impl& operator= (__exec_system_bulk_operation_state_impl&&) = delete;


  void start() noexcept override {
    stdexec::start(pool_operation_state_);
  }

  __exec_system_receiver recv_;
  __exec_system_bulk_function_object bulk_function_;
stdexec::__result_of<__exec_pool_operation_state, __exec_system_bulk_operation_state_impl*, __exec_pool_sender_t, __exec_system_bulk_shape, __exec_system_bulk_function_object>
    pool_operation_state_;
};

inline void tag_invoke(stdexec::set_value_t, __exec_system_bulk_pool_receiver&& recv) noexcept {
  __exec_system_receiver &system_recv = recv.os_->recv_;
  system_recv.set_value((system_recv.cpp_recv_));
}

inline void tag_invoke(stdexec::set_stopped_t, __exec_system_bulk_pool_receiver&& recv) noexcept {
  __exec_system_receiver &system_recv = recv.os_->recv_;
  recv.os_->recv_.set_stopped(&(system_recv.cpp_recv_));
}

inline void tag_invoke(stdexec::set_error_t, __exec_system_bulk_pool_receiver&& recv, std::exception_ptr ptr) noexcept {
  __exec_system_receiver &system_recv = recv.os_->recv_;
  recv.os_->recv_.set_error(&(system_recv.cpp_recv_), &ptr);
}

// A bulk sender is just a system sender viewed externally.
// TODO: a bulk operation state is just a system operation state viewed externally
struct __exec_system_bulk_sender_impl : public __exec_system_sender_interface {
  __exec_system_bulk_sender_impl(
        __exec_system_scheduler_impl* scheduler,
        __exec_system_bulk_shape bulk_shape,
        __exec_system_bulk_function_object bulk_function,
        __exec_pool_sender_t&& pool_sender) :
      scheduler_{scheduler},
      bulk_shape_{bulk_shape},
      bulk_function_{bulk_function},
      pool_sender_(std::move(pool_sender)) {

  }

  __exec_system_operation_state_interface* connect(__exec_system_receiver recv) noexcept override {
    return
      new __exec_system_bulk_operation_state_impl(
        std::move(pool_sender_), bulk_shape_, bulk_function_, std::move(recv));
  }

  __exec_system_scheduler_interface* get_completion_scheduler() noexcept override {
    return scheduler_;
  };

   __exec_system_scheduler_impl* scheduler_;
   __exec_system_bulk_shape bulk_shape_;
  __exec_system_bulk_function_object bulk_function_;
   __exec_pool_sender_t pool_sender_;
};


inline __exec_system_scheduler_interface* __exec_system_context_impl::get_scheduler() noexcept {
  // TODO: ref counting etc
  return new __exec_system_scheduler_impl(this, pool_.get_scheduler());
}

inline __exec_system_sender_interface* __exec_system_scheduler_impl::schedule() {
  return new __exec_system_sender_impl(this, stdexec::schedule(pool_scheduler_));
}


inline __exec_system_sender_interface* __exec_system_scheduler_impl::bulk(
    __exec_system_bulk_shape shp,
    __exec_system_bulk_function_object fn) {
  // This is bulk off a system_scheduler, so we need to start with schedule.
  // TODO: a version later will key off a bulk *sender* and would behave slightly
  // differently.
  // In both cases pass in the result of schedule, or the predecessor though.
  return new __exec_system_bulk_sender_impl(this, shp, fn, stdexec::schedule(pool_scheduler_));
}



// Phase 1 implementation, single implementation
// TODO: Make a weak symbol and replace in a test
static __exec_system_context_impl* __get_exec_system_context_impl() {
  static __exec_system_context_impl impl_;

  return &impl_;
}

// TODO: Move everything above here to a detail header and wrap in a
// namespace to represent extern "C"


namespace exec {
  namespace __system_scheduler {

  } // namespace system_scheduler


  class system_scheduler;
  class system_sender;
  template<stdexec::sender S, std::integral Shape, class Fn>
  struct system_bulk_sender;

  class system_context {
  public:
    system_context() {
      impl_ = __get_exec_system_context_impl();
      // TODO error handling
    }

    system_context(const system_context&) = delete;
    system_context(system_context&&) = delete;
    system_context& operator=(const system_context&) = delete;
    system_context& operator=(system_context&&) = delete;

    system_scheduler get_scheduler();

    size_t max_concurrency() const noexcept;

  private:
    __exec_system_context_interface* impl_ = nullptr;

  };

  class system_scheduler {
  public:

    // Pointer that we ref count?
    system_scheduler(__exec_system_scheduler_interface* scheduler_interface) : scheduler_interface_(scheduler_interface) {}

    bool operator==(const system_scheduler& rhs) const noexcept {
      return scheduler_interface_->equals(rhs.scheduler_interface_);
    }

  private:
    friend system_sender tag_invoke(
      stdexec::schedule_t, const system_scheduler&) noexcept;

    friend stdexec::forward_progress_guarantee tag_invoke(
      stdexec::get_forward_progress_guarantee_t,
      const system_scheduler&) noexcept;

    template <stdexec::sender S, std::integral Shape, class Fn>
    friend system_bulk_sender<S, Shape, Fn> tag_invoke(       //
      stdexec::bulk_t,                                        //
      const system_scheduler& sch,                            //
      S&& sndr,                                               //
      Shape shape,                                            //
      Fn fun)                                                 //
      noexcept;

    __exec_system_scheduler_interface* scheduler_interface_;
    friend class system_context;
  };

  class system_sender {
  public:
    using is_sender = void;
    using completion_signatures =
      stdexec::completion_signatures< stdexec::set_value_t(), stdexec::set_stopped_t() >;

    system_sender(__exec_system_sender_interface* sender_impl) : sender_impl_{sender_impl} {}

  private:
    template <class S, class R_>
    struct __op {
      using R = R_;

      template<class F>
      __op(system_sender&& snd, R&& recv, F&& initFunc) :
          snd_{std::move(snd)}, recv_{std::move(recv)}, os_{initFunc(*this)} {
      }
      __op(const __op&) = delete;
      __op(__op&&) = delete;
      __op& operator= (const __op&) = delete;
      __op& operator= (__op&&) = delete;

      friend void tag_invoke(stdexec::start_t, __op& op) noexcept {
        if(auto os = op.os_) {
          os->start();
        }
      }

      S snd_;
      R recv_;
      __exec_system_operation_state_interface* os_ = nullptr;
    };

    template <class R>
    friend auto tag_invoke(stdexec::connect_t, system_sender&& snd, R&& rec) //
      noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<R>, R>)
        -> __op<system_sender, std::remove_cvref_t<R>> {

      return __op<system_sender, std::remove_cvref_t<R>>{
        std::move(snd),
        std::move(rec),
        [](auto& op){
          __exec_system_receiver receiver_impl{
            &op.recv_,
            [](void* cpp_recv) noexcept{
              stdexec::set_value(std::move(*static_cast<R*>(cpp_recv)));
            },
            [](void* cpp_recv) noexcept{
              stdexec::set_stopped(std::move(*static_cast<R*>(cpp_recv)));
            }};

          return op.snd_.sender_impl_->connect(std::move(receiver_impl));
        }};
    }

    struct __env {
      friend system_scheduler
        tag_invoke(stdexec::get_completion_scheduler_t<stdexec::set_value_t>, const __env& self) //
        noexcept {
        return {self.scheduler_impl_};
      }

      friend system_scheduler
        tag_invoke(stdexec::get_completion_scheduler_t<stdexec::set_stopped_t>, const __env& self) //
        noexcept {
        return {self.scheduler_impl_};
      }

      __exec_system_scheduler_interface* scheduler_impl_;
    };

    friend __env tag_invoke(stdexec::get_env_t, const system_sender& snd) noexcept {
      return {snd.sender_impl_->get_completion_scheduler()};
    }

    __exec_system_sender_interface* sender_impl_ = nullptr;
  };

  template <stdexec::sender Pred, std::integral Shape, class Fn, class R>
  struct bulk_state {
    system_bulk_sender<Pred, Shape, Fn> snd_;
    R recv_;
    void* arg_data_ = nullptr;
    __exec_system_operation_state_interface* os_ = nullptr;
  };

  template <stdexec::sender Pred, std::integral Shape, class Fn, class R>
  struct bulk_recv {
    using receiver_concept = stdexec::receiver_t;

    bulk_state<Pred, Shape, Fn, R>& state_;

    template <class... As>
    friend void tag_invoke(stdexec::set_value_t, bulk_recv&& self, As&&... as) noexcept {

      // Heap allocate input data in shared state as needed
      std::tuple<As...> *inputs = new std::tuple<As...>{as...};
      self.state_.arg_data_ = inputs;

      // Construct bulk operation with type conversions to use C ABI state
      auto sched = self.state_.snd_.scheduler_impl_;
      if(sched) {
        __exec_system_bulk_function_object fn {
          &self.state_,
          [](void* state_, long idx){
            bulk_state<Pred, Shape, Fn, R>* state =
              static_cast<bulk_state<Pred, Shape, Fn, R>*>(state_);

            std::apply(
              [&](auto &&... args) {
                state->snd_.fun_(idx, args...);
              },
              *static_cast<std::tuple<As...> *>(state->arg_data_));
          }};

        auto* sender = sched->bulk(self.state_.snd_.shape_, fn);
        // Connect to a type-erasing receiver to call our receiver on completion
        self.state_.os_ = sender->connect(
          __exec_system_receiver{
            &self.state_.recv_,
            [](void* cpp_recv) noexcept{
              stdexec::set_value(std::move(*static_cast<R*>(cpp_recv)));
            },
            [](void* cpp_recv) noexcept{
              stdexec::set_stopped(std::move(*static_cast<R*>(cpp_recv)));
            },
            [](void* cpp_recv, void* exception) noexcept{
              stdexec::set_error(
                std::move(*static_cast<R*>(cpp_recv)),
                std::move(*static_cast<std::exception_ptr*>(exception)));
            }});
        self.state_.os_->start();
      }
    }

    friend void tag_invoke(stdexec::set_stopped_t, bulk_recv&& self) noexcept {
      stdexec::set_stopped(std::move(self.state_.recv_));
    }

    friend void tag_invoke(stdexec::set_error_t, bulk_recv&& self, std::exception_ptr ptr) noexcept {
      stdexec::set_error(std::move(self.state_.recv_), std::move(ptr));
    }

    friend auto tag_invoke(stdexec::get_env_t, const bulk_recv& self) noexcept {
      return stdexec::get_env(self.state_.recv_);
    }
  };

  template <stdexec::sender Pred, std::integral Shape, class Fn, class R>
  struct __bulk_op {
    using inner_op_state = stdexec::connect_result_t<Pred, bulk_recv<Pred, Shape, Fn, R>>;

    template<class InitF>
    __bulk_op(system_bulk_sender<Pred, Shape, Fn>&& snd, R&& recv, InitF&& initFunc) :
        state_{std::move(snd), std::move(recv)}, pred_operation_state_{initFunc(*this)} {
    }
    __bulk_op(const __bulk_op&) = delete;
    __bulk_op(__bulk_op&&) = delete;
    __bulk_op& operator= (const __bulk_op&) = delete;
    __bulk_op& operator= (__bulk_op&&) = delete;

    friend void tag_invoke(stdexec::start_t, __bulk_op& op) noexcept {
      if(auto os = op.state_.os_) {
        os->start();
      }
      // Start inner operation state
      // Bulk operation will be started when that completes
      stdexec::start(op.pred_operation_state_);
    }

    bulk_state<Pred, Shape, Fn, R> state_;
    inner_op_state pred_operation_state_;
  };

  template<stdexec::sender Pred, std::integral Shape, class Fn>
  struct system_bulk_sender {
    using Sender = Pred;
    using Fun = Fn;
    using is_sender = void;
    using completion_signatures =
      stdexec::completion_signatures< stdexec::set_value_t(), stdexec::set_stopped_t(), stdexec::set_error_t(std::exception_ptr) >;
      // TODO: This can complete with different values... should propagate from Pred

    system_bulk_sender(
      __exec_system_scheduler_interface* scheduler_impl,
      Sender pred,
      Shape shape,
      Fun&& fun) :
      scheduler_impl_{scheduler_impl},
      pred_{std::move(pred)},
      shape_{std::move(shape)},
      fun_{std::move(fun)} {}

    template <class R>
    friend auto tag_invoke(stdexec::connect_t, system_bulk_sender&& snd, R&& rec) //
      noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<R>, R>)
        -> __bulk_op<Pred, Shape, Fn, R> {

      return __bulk_op<Pred, Shape, Fn, R>{
        std::move(snd),
        std::move(rec),
        [](auto& op){
          // Connect bulk input receiver with the previous operation and store in the OS
          return stdexec::connect(std::move(op.state_.snd_.pred_), bulk_recv<Pred, Shape, Fn, R>{op.state_});
        }};
    }

    struct __env {
      friend system_scheduler
        tag_invoke(stdexec::get_completion_scheduler_t<stdexec::set_value_t>, const __env& self) //
        noexcept {
        return {self.scheduler_impl_};
      }

      friend system_scheduler
        tag_invoke(stdexec::get_completion_scheduler_t<stdexec::set_stopped_t>, const __env& self) //
        noexcept {
        return {self.scheduler_impl_};
      }


      __exec_system_scheduler_interface* scheduler_impl_;
    };

    friend __env tag_invoke(stdexec::get_env_t, const system_bulk_sender& snd) noexcept {
      // If we trigger this customization we know what the completion scheduler will be
      return {snd.scheduler_impl_};
    }

    __exec_system_scheduler_interface* scheduler_impl_ = nullptr;
    Sender pred_;
    Shape shape_;
    Fun fun_;
  };


  inline system_scheduler system_context::get_scheduler() {
    return system_scheduler{impl_->get_scheduler()};
  }

  inline size_t system_context::max_concurrency() const noexcept {
    return std::thread::hardware_concurrency();
  }

  system_sender tag_invoke(
      stdexec::schedule_t, const system_scheduler& sched) noexcept {
    return system_sender(sched.scheduler_interface_->schedule());
  }

  stdexec::forward_progress_guarantee tag_invoke(
      stdexec::get_forward_progress_guarantee_t,
      const system_scheduler& sched) noexcept {
    return sched.scheduler_interface_->get_forward_progress_guarantee();
  }


  template <stdexec::sender S, std::integral Shape, class Fn>
  system_bulk_sender<S, Shape, Fn> tag_invoke(              //
    stdexec::bulk_t,                                        //
    const system_scheduler& sch,                            //
    S&& pred,                                               //
    Shape shape,                                            //
    Fn fun)                                                 //
    noexcept {
    return system_bulk_sender<S, Shape, Fn>{
      sch.scheduler_interface_, (S&&) pred, shape, (Fn&&) fun};
  }


} // namespace exec
