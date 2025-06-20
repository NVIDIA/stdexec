/*
 * Copyright (c) 2021-2024 NVIDIA Corporation
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

#include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "__completion_signatures.hpp"
#include "__env.hpp"
#include "__meta.hpp"
#include "__receivers.hpp"
#include "__utility.hpp"

#include <condition_variable>
#include <exception>
#include <mutex>
#include <utility>

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // run_loop
  namespace __loop {
    class run_loop;

    struct __task : __immovable {
      __task* __next_ = this;

      union {
        __task* __tail_ = nullptr;
        void (*__execute_)(__task*) noexcept;
      };

      void __execute() noexcept {
        (*__execute_)(this);
      }
    };

    template <class _ReceiverId>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t : __task {
        using __id = __operation;

        run_loop* __loop_;
        STDEXEC_ATTRIBUTE(no_unique_address) _Receiver __rcvr_;

        static void __execute_impl(__task* __p) noexcept {
          auto& __rcvr = static_cast<__t*>(__p)->__rcvr_;
          STDEXEC_TRY {
            if (stdexec::get_stop_token(stdexec::get_env(__rcvr)).stop_requested()) {
              stdexec::set_stopped(static_cast<_Receiver&&>(__rcvr));
            } else {
              stdexec::set_value(static_cast<_Receiver&&>(__rcvr));
            }
          }
          STDEXEC_CATCH_ALL {
            stdexec::set_error(static_cast<_Receiver&&>(__rcvr), std::current_exception());
          }
        }

        explicit __t(__task* __tail) noexcept
          : __task{{}, this, __tail} {
        }

        __t(__task* __next, run_loop* __loop, _Receiver __rcvr)
          : __task{{}, __next, {}}
          , __loop_{__loop}
          , __rcvr_{static_cast<_Receiver&&>(__rcvr)} {
          __execute_ = &__execute_impl;
        }

        void start() & noexcept;
      };
    };

    class run_loop {
      template <class>
      friend struct __operation;
     public:
      struct __scheduler {
       private:
        struct __schedule_task {
          using __t = __schedule_task;
          using __id = __schedule_task;
          using sender_concept = sender_t;
          using completion_signatures = stdexec::completion_signatures<
            set_value_t(),
            set_error_t(std::exception_ptr),
            set_stopped_t()
          >;

          template <class _Receiver>
          using __operation = stdexec::__t<__operation<stdexec::__id<_Receiver>>>;

          template <class _Receiver>
          auto connect(_Receiver __rcvr) const -> __operation<_Receiver> {
            return {&__loop_->__head_, __loop_, static_cast<_Receiver&&>(__rcvr)};
          }

         private:
          friend __scheduler;

          struct __env {
            using __t = __env;
            using __id = __env;

            run_loop* __loop_;

            template <class _CPO>
            auto query(get_completion_scheduler_t<_CPO>) const noexcept -> __scheduler {
              return __loop_->get_scheduler();
            }
          };

          explicit __schedule_task(run_loop* __loop) noexcept
            : __loop_(__loop) {
          }

          run_loop* const __loop_;

         public:
          [[nodiscard]]
          auto get_env() const noexcept -> __env {
            return __env{__loop_};
          }
        };

        friend run_loop;

        explicit __scheduler(run_loop* __loop) noexcept
          : __loop_(__loop) {
        }

        run_loop* __loop_;

       public:
        using __t = __scheduler;
        using __id = __scheduler;
        auto operator==(const __scheduler&) const noexcept -> bool = default;

        [[nodiscard]]
        auto schedule() const noexcept -> __schedule_task {
          return __schedule_task{__loop_};
        }

        [[nodiscard]]
        auto query(get_forward_progress_guarantee_t) const noexcept
          -> stdexec::forward_progress_guarantee {
          return stdexec::forward_progress_guarantee::parallel;
        }

        // BUGBUG NOT TO SPEC
        [[nodiscard]]
        auto query(execute_may_block_caller_t) const noexcept -> bool {
          return false;
        }
      };

      auto get_scheduler() noexcept -> __scheduler {
        return __scheduler{this};
      }

      void run();

      void finish();

     private:
      void __push_back_(__task* __task);
      auto __pop_front_() -> __task*;

      std::mutex __mutex_;
      std::condition_variable __cv_;
      __task __head_{{}, &__head_, {&__head_}};
      bool __stop_ = false;
    };

    template <class _ReceiverId>
    inline void __operation<_ReceiverId>::__t::start() & noexcept {
      STDEXEC_TRY {
        __loop_->__push_back_(this);
      }
      STDEXEC_CATCH_ALL {
        stdexec::set_error(static_cast<_Receiver&&>(__rcvr_), std::current_exception());
      }
    }

    inline void run_loop::run() {
      for (__task* __task; (__task = __pop_front_()) != &__head_;) {
        __task->__execute();
      }
    }

    inline void run_loop::finish() {
      std::unique_lock __lock{__mutex_};
      __stop_ = true;
      __cv_.notify_all();
    }

    inline void run_loop::__push_back_(__task* __task) {
      std::unique_lock __lock{__mutex_};
      __task->__next_ = &__head_;
      __head_.__tail_ = __head_.__tail_->__next_ = __task;
      __cv_.notify_one();
    }

    inline auto run_loop::__pop_front_() -> __task* {
      std::unique_lock __lock{__mutex_};
      __cv_.wait(__lock, [this] { return __head_.__next_ != &__head_ || __stop_; });
      if (__head_.__tail_ == __head_.__next_)
        __head_.__tail_ = &__head_;
      return std::exchange(__head_.__next_, __head_.__next_->__next_);
    }
  } // namespace __loop

  // NOT TO SPEC
  using run_loop = __loop::run_loop;
} // namespace stdexec
