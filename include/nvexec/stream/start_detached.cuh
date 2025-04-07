/*
 * Copyright (c) 2022 NVIDIA Corporation
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

// clang-format Language: Cpp

#pragma once

#include "../../stdexec/execution.hpp"
#include "../../exec/scope.hpp"

#include <exception>
#include <memory>

#include "common.cuh"

namespace nvexec::_strm {
  namespace _start_detached {
    template <class SenderId, class EnvId>
    struct operation {
      using Sender = __cvref_t<SenderId>;
      using Env = __t<EnvId>;

      explicit operation(Sender&& sndr, Env env)
        : env_(static_cast<Env&&>(env))
        , op_state_(connect(static_cast<Sender&&>(sndr), receiver{{}, this})) {
      }

      // If the operation state was allocated with a user-provided allocator, then we must
      // use the allocator stored within the operation state to destroy the operation
      // state. This is a good time to use C++20's destroying delete operation.
      static void destroy_delete(operation* self) noexcept {
        if constexpr (__callable<get_allocator_t, Env>) {
          auto alloc = stdexec::get_allocator(self->env_);
          using Alloc = decltype(alloc);
          using OpAlloc = typename std::allocator_traits<Alloc>::template rebind_alloc<operation>;
          OpAlloc op_alloc{alloc};
          std::allocator_traits<OpAlloc>::destroy(op_alloc, self);
          std::allocator_traits<OpAlloc>::deallocate(op_alloc, self, 1);
        } else {
          delete self;
        }
      }

      // The start_detached receiver deletes the operation state.
      struct receiver : stream_receiver_base {
        using receiver_concept = receiver_t;
        using t = receiver;
        using id = receiver;

        template <class... As>
        void set_value(As&&...) noexcept {
          operation::destroy_delete(op_); // NB: invalidates *this
        }

        template <class Error>
        [[noreturn]]
        void set_error(Error&&) noexcept {
          // A detached operation failed. There is noplace for the error to go.
          // This is unrecoverable, so we terminate.
          std::terminate();
        }

        void set_stopped() noexcept {
          operation::destroy_delete(op_); // NB: invalidates *this
        }

        auto get_env() const noexcept -> const Env& {
          return op_->env_;
        }

        operation* op_;
      };

      STDEXEC_ATTRIBUTE((no_unique_address)) Env env_;
      connect_result_t<Sender, receiver> op_state_;
    };
  } // namespace _start_detached

  template <>
  struct apply_sender_for<start_detached_t> {
    template <class Sender, class Env = __root_env>
    void operator()(Sender&& sndr, Env&& env = {}) const noexcept(false) {
      using Op = _start_detached::operation<__cvref_id<Sender>, __id<__decay_t<Env>>>;
      // Use the provided allocator, if any, to allocate the operation state.
      if constexpr (__callable<get_allocator_t, Env>) {
        auto alloc = get_allocator(env);
        using Alloc = decltype(alloc);
        using OpAlloc = typename std::allocator_traits<Alloc>::template rebind_alloc<Op>;
        // We use the allocator to allocate the operation state and also to construct it.
        OpAlloc op_alloc{alloc};
        Op* op = std::allocator_traits<OpAlloc>::allocate(op_alloc, 1);
        exec::scope_guard g{[op, &op_alloc]() noexcept {
          std::allocator_traits<OpAlloc>::deallocate(op_alloc, op, 1);
        }};
        // This can potentially throw. If it does, the scope guard will deallocate the
        // storage automatically.
        std::allocator_traits<OpAlloc>::construct(
          op_alloc, op, static_cast<Sender&&>(sndr), static_cast<Env&&>(env));
        // The operation state is now constructed, dismiss the scope guard.
        g.dismiss();
        // This cannot throw:
        stdexec::start(op->op_state_);
      } else {
        // The caller did not provide an allocator, so we use the default allocator.
        Op* op = new Op{static_cast<Sender&&>(sndr), static_cast<Env&&>(env)};
        start(op->op_state_);
      }
    }
  };
} // namespace nvexec::_strm
