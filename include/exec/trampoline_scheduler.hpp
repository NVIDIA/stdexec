/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Copyright (c) 2023 NVIDIA Corporation
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

#include "../stdexec/execution.hpp"

#include <cstddef>
#include <utility>

namespace exec {
  namespace __trampoline {
    using namespace stdexec;

    template <class _Operation>
    struct __trampoline_state {
      static thread_local __trampoline_state* __current_;

      __trampoline_state() noexcept {
        __current_ = this;
      }

      ~__trampoline_state() {
        __current_ = nullptr;
      }

      void __drain() noexcept;

      std::size_t __recursion_depth_ = 1;
      _Operation* __head_ = nullptr;
    };

    class __scheduler {
      std::size_t __max_recursion_depth_;

     public:
      __scheduler() noexcept
        : __max_recursion_depth_(16) {
      }

      explicit __scheduler(std::size_t __max_recursion_depth) noexcept
        : __max_recursion_depth_(__max_recursion_depth) {
      }

     private:
      struct __operation_base {
        using __execute_fn = void(__operation_base*) noexcept;

        explicit __operation_base(__execute_fn* __execute, std::size_t __max_depth) noexcept
          : __execute_(__execute)
          , __max_recursion_depth_(__max_depth) {
        }

        void __execute() noexcept {
          __execute_(this);
        }

        void start() & noexcept {
          auto* __current_state = __trampoline_state<__operation_base>::__current_;
          if (__current_state == nullptr) {
            __trampoline_state<__operation_base> __state;
            __execute();
            __state.__drain();
          } else if (__current_state->__recursion_depth_ < __max_recursion_depth_) {
            ++__current_state->__recursion_depth_;
            __execute();
          } else {
            // Exceeded recursion limit.
            __next_ = std::exchange(__current_state->__head_, static_cast<__operation_base*>(this));
          }
        }

        __operation_base* __next_ = nullptr;
        __execute_fn* __execute_;
        std::size_t __max_recursion_depth_;
      };

      template <class _ReceiverId>
      struct __operation {
        using _Receiver = stdexec::__t<_ReceiverId>;

        struct __t : __operation_base {
          using __id = __operation;
          STDEXEC_ATTRIBUTE(no_unique_address) _Receiver __receiver_;

          explicit __t(_Receiver __rcvr, std::size_t __max_depth)
            noexcept(__nothrow_move_constructible<_Receiver>)
            : __operation_base(&__t::__execute_impl, __max_depth)
            , __receiver_(static_cast<_Receiver&&>(__rcvr)) {
          }

          static void __execute_impl(__operation_base* __op) noexcept {
            auto& __self = *static_cast<__t*>(__op);
            if (stdexec::unstoppable_token<stop_token_of_t<env_of_t<_Receiver&>>>) {
              stdexec::set_value(static_cast<_Receiver&&>(__self.__receiver_));
            } else {
              if (get_stop_token(get_env(__self.__receiver_)).stop_requested()) {
                stdexec::set_stopped(static_cast<_Receiver&&>(__self.__receiver_));
              } else {
                stdexec::set_value(static_cast<_Receiver&&>(__self.__receiver_));
              }
            }
          }
        };
      };

      struct __schedule_sender;
      friend __schedule_sender;

      template <class _Receiver>
      using __operation_t = stdexec::__t<__operation<__id<__decay_t<_Receiver>>>>;

      struct __schedule_sender {
        using sender_concept = stdexec::sender_t;
        using completion_signatures =
          stdexec::completion_signatures<set_value_t(), set_stopped_t()>;

        explicit __schedule_sender(std::size_t __max_depth) noexcept
          : __max_recursion_depth_(__max_depth) {
        }

        template <receiver_of<completion_signatures> _Receiver>
        auto connect(_Receiver __rcvr) const noexcept(__nothrow_move_constructible<_Receiver>)
          -> __operation_t<_Receiver> {
          return __operation_t<_Receiver>{static_cast<_Receiver&&>(__rcvr), __max_recursion_depth_};
        }

        [[nodiscard]]
        auto query(get_completion_scheduler_t<set_value_t>) const noexcept -> __scheduler {
          return __scheduler{__max_recursion_depth_};
        }

        [[nodiscard]]
        auto get_env() const noexcept -> const __schedule_sender& {
          return *this;
        }

        std::size_t __max_recursion_depth_;
      };

     public:
      [[nodiscard]]
      auto schedule() const noexcept -> __schedule_sender {
        return __schedule_sender{__max_recursion_depth_};
      }

      auto operator==(const __scheduler&) const noexcept -> bool = default;
    };

    template <class _Operation>
    thread_local __trampoline_state<_Operation>* __trampoline_state<_Operation>::__current_ =
      nullptr;

    template <class _Operation>
    void __trampoline_state<_Operation>::__drain() noexcept {
      while (__head_ != nullptr) {
        _Operation* __op = std::exchange(__head_, __head_->__next_);
        __recursion_depth_ = 1;
        __op->__execute();
      }
    }
  } // namespace __trampoline

  using trampoline_scheduler = __trampoline::__scheduler;

} // namespace exec
