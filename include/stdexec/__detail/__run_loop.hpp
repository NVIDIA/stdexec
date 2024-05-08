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
#include "__cpo.hpp"
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
        void (*__execute_)(__task*) noexcept;
        __task* __tail_;
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
        STDEXEC_ATTRIBUTE((no_unique_address))
        _Receiver __rcvr_;

        static void __execute_impl(__task* __p) noexcept {
          auto& __rcvr = static_cast<__t*>(__p)->__rcvr_;
          try {
            if (get_stop_token(get_env(__rcvr)).stop_requested()) {
              set_stopped(static_cast<_Receiver&&>(__rcvr));
            } else {
              set_value(static_cast<_Receiver&&>(__rcvr));
            }
          } catch (...) {
            set_error(static_cast<_Receiver&&>(__rcvr), std::current_exception());
          }
        }

        explicit __t(__task* __tail) noexcept
          : __task{.__tail_ = __tail} {
        }

        __t(__task* __next, run_loop* __loop, _Receiver __rcvr)
          : __task{{}, __next, {&__execute_impl}}
          , __loop_{__loop}
          , __rcvr_{static_cast<_Receiver&&>(__rcvr)} {
        }

        STDEXEC_ATTRIBUTE((always_inline))
        STDEXEC_MEMFN_DECL(
          void start)(this __t& __self) noexcept {
          __self.__start_();
        }

        void __start_() noexcept;
      };
    };

    class run_loop {
      template <class>
      friend struct __operation;
     public:
      struct __scheduler {
        using __t = __scheduler;
        using __id = __scheduler;
        auto operator==(const __scheduler&) const noexcept -> bool = default;

       private:
        struct __schedule_task {
          using __t = __schedule_task;
          using __id = __schedule_task;
          using sender_concept = sender_t;
          using completion_signatures = stdexec::
            completion_signatures<set_value_t(), set_error_t(std::exception_ptr), set_stopped_t()>;

         private:
          friend __scheduler;

          template <class _Receiver>
          using __operation = stdexec::__t<__operation<stdexec::__id<_Receiver>>>;

          template <class _Receiver>
          STDEXEC_MEMFN_DECL(auto connect)(this const __schedule_task& __self, _Receiver __rcvr)
            -> __operation<_Receiver> {
            return __self.__connect_(static_cast<_Receiver&&>(__rcvr));
          }

          template <class _Receiver>
          auto __connect_(_Receiver&& __rcvr) const -> __operation<_Receiver> {
            return {&__loop_->__head_, __loop_, static_cast<_Receiver&&>(__rcvr)};
          }

          struct __env {
            run_loop* __loop_;

            template <class _CPO>
            STDEXEC_MEMFN_DECL(auto query)(this const __env& __self, get_completion_scheduler_t<_CPO>) noexcept
              -> __scheduler {
              return __self.__loop_->get_scheduler();
            }
          };

          STDEXEC_MEMFN_DECL(auto get_env)(this const __schedule_task& __self) noexcept -> __env {
            return __env{__self.__loop_};
          }

          explicit __schedule_task(run_loop* __loop) noexcept
            : __loop_(__loop) {
          }

          run_loop* const __loop_;
        };

        friend run_loop;

        explicit __scheduler(run_loop* __loop) noexcept
          : __loop_(__loop) {
        }

        STDEXEC_MEMFN_DECL(auto schedule)(this const __scheduler& __self) noexcept -> __schedule_task {
          return __self.__schedule();
        }

        STDEXEC_MEMFN_DECL(auto query)(this const __scheduler&, get_forward_progress_guarantee_t) noexcept
          -> stdexec::forward_progress_guarantee {
          return stdexec::forward_progress_guarantee::parallel;
        }

        // BUGBUG NOT TO SPEC
        STDEXEC_MEMFN_DECL(
          auto execute_may_block_caller)(this const __scheduler&) noexcept -> bool {
          return false;
        }

        [[nodiscard]]
        auto __schedule() const noexcept -> __schedule_task {
          return __schedule_task{__loop_};
        }

        run_loop* __loop_;
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
      __task __head_{.__tail_ = &__head_};
      bool __stop_ = false;
    };

    template <class _ReceiverId>
    inline void __operation<_ReceiverId>::__t::__start_() noexcept {
      try {
        __loop_->__push_back_(this);
      } catch (...) {
        set_error(static_cast<_Receiver&&>(__rcvr_), std::current_exception());
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
