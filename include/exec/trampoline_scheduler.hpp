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

      __trampoline_state(std::size_t __max_recursion_depth, std::size_t __max_recursion_size) noexcept
        : __max_recursion_size_(__max_recursion_size)
        , __max_recursion_depth_(__max_recursion_depth) {
        __current_ = this;
      }

      ~__trampoline_state() {
        __current_ = nullptr;
      }

      void __drain() noexcept;

      // these origin schedule frame limits will apply to all
      // nested trampoline instances on this thread
      const std::size_t __max_recursion_size_;
      const std::size_t __max_recursion_depth_;

      // track state of origin schedule frame
      std::intptr_t __recursion_origin_ = 0;
      std::size_t __recursion_depth_ = 1;
      _Operation* __head_ = nullptr;
      _Operation* __tail_ = nullptr;
    };

    class __scheduler {
      const std::size_t __max_recursion_size_;
      const std::size_t __max_recursion_depth_;

     public:
      __scheduler() noexcept
        : __max_recursion_size_(4096)
        , __max_recursion_depth_(16) {
      }

      explicit __scheduler(std::size_t __max_recursion_depth) noexcept
        : __max_recursion_size_(4096)
        , __max_recursion_depth_(__max_recursion_depth) {
      }

      explicit __scheduler(std::size_t __max_recursion_depth, std::size_t __max_recursion_size) noexcept
        : __max_recursion_size_(__max_recursion_size)
        , __max_recursion_depth_(__max_recursion_depth) {
      }

     private:
      struct __operation_base {
        using __execute_fn = void(__operation_base*) noexcept;

        explicit __operation_base(__execute_fn* __execute, std::size_t __max_size, std::size_t __max_depth) noexcept
          : __execute_(__execute)
          , __max_recursion_size_(__max_size)
          , __max_recursion_depth_(__max_depth) {
        }

        void __execute() noexcept {
          __execute_(this);
        }

        void start() & noexcept {
          auto* __current_state = __trampoline_state<__operation_base>::__current_;

          if (__current_state == nullptr) {
            // origin schedule frame on this thread
            __trampoline_state<__operation_base> __state{__max_recursion_depth_, __max_recursion_size_};
            __execute();
            __state.__drain();
          } else {
            // recursive schedule frame on this thread

            // calculate stack consumption for this schedule
            std::size_t __current_size = std::abs(reinterpret_cast<std::intptr_t>(&__current_state)
                                            - __current_state->__recursion_origin_);

            if (__current_size < __current_state->__max_recursion_size_
              && __current_state->__recursion_depth_ < __current_state->__max_recursion_depth_) {
              // inline this recursive schedule
              ++__current_state->__recursion_depth_;
              __execute();
            } else {
              // Exceeded recursion limit.

              // push this recursive schedule to list tail
              __prev_ = std::exchange(__current_state->__tail_, static_cast<__operation_base*>(this));
              if (__prev_ != nullptr) {
                // was not empty
                std::exchange(__prev_->__next_, static_cast<__operation_base*>(this));
              } else {
                // was empty
                std::exchange(__current_state->__head_, static_cast<__operation_base*>(this));
              }
            }
          }
        }

        __operation_base* __prev_ = nullptr;
        __operation_base* __next_ = nullptr;
        __execute_fn* __execute_;
        const std::size_t __max_recursion_size_;
        const std::size_t __max_recursion_depth_;
      };

      template <class _ReceiverId>
      struct __operation {
        using _Receiver = stdexec::__t<_ReceiverId>;

        struct __t : __operation_base {
          using __id = __operation;
          STDEXEC_ATTRIBUTE(no_unique_address) _Receiver __receiver_;

          explicit __t(_Receiver __rcvr, std::size_t __max_size, std::size_t __max_depth)
            noexcept(__nothrow_move_constructible<_Receiver>)
            : __operation_base(&__t::__execute_impl, __max_size, __max_depth)
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

        explicit __schedule_sender(std::size_t __max_size, std::size_t __max_depth) noexcept
          : __max_recursion_size_(__max_size)
          , __max_recursion_depth_(__max_depth) {
        }

        template <receiver_of<completion_signatures> _Receiver>
        auto connect(_Receiver __rcvr) const noexcept(__nothrow_move_constructible<_Receiver>)
          -> __operation_t<_Receiver> {
          return __operation_t<_Receiver>{static_cast<_Receiver&&>(__rcvr), __max_recursion_size_, __max_recursion_depth_};
        }

        [[nodiscard]]
        auto query(get_completion_scheduler_t<set_value_t>) const noexcept -> __scheduler {
          return __scheduler{__max_recursion_depth_};
        }

        [[nodiscard]]
        auto get_env() const noexcept -> const __schedule_sender& {
          return *this;
        }

        const std::size_t __max_recursion_size_;
        const std::size_t __max_recursion_depth_;
      };

     public:
      [[nodiscard]]
      auto schedule() const noexcept -> __schedule_sender {
        return __schedule_sender{__max_recursion_size_, __max_recursion_depth_};
      }

      auto operator==(const __scheduler&) const noexcept -> bool = default;
    };

    template <class _Operation>
    thread_local __trampoline_state<_Operation>* __trampoline_state<_Operation>::__current_ =
      nullptr;

    template <class _Operation>
    void __trampoline_state<_Operation>::__drain() noexcept {
      while (__head_ != nullptr) {
        // pop the head of the list
        _Operation* __op = std::exchange(__head_, __head_->__next_);
        __op->__next_ = nullptr;
        __op->__prev_ = nullptr;
        if (__head_ != nullptr) {
          // is not empty
          __head_->__prev_ = nullptr;
        } else {
          // is empty
          __tail_ = nullptr;
        }

        // reset the origin schedule frame state
        __recursion_origin_ = reinterpret_cast<std::intptr_t>(&__op);
        __recursion_depth_ = 1;

        __op->__execute();
      }
    }
  } // namespace __trampoline

  using trampoline_scheduler = __trampoline::__scheduler;

} // namespace exec
