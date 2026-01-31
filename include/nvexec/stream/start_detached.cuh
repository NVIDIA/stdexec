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

#include "../../exec/scope.hpp"
#include "../../stdexec/execution.hpp"

#include <exception>
#include <memory>

#include "common.cuh"

namespace nvexec::_strm {
  namespace _start_detached {
    struct submit_receiver {
      using receiver_concept = receiver_t;

      template <class... Args>
      void set_value(Args&&...) noexcept {
      }

      template <class Error>
      [[noreturn]]
      void set_error(Error&&) noexcept {
        // A detached operation failed. There is noplace for the error to go.
        // This is unrecoverable, so we terminate.
        std::terminate();
      }

      void set_stopped() noexcept {
      }

      [[nodiscard]]
      auto get_env() const noexcept -> __root_env {
        return {};
      }
    };

    template <class CvSender, class Env>
    struct opstate;

    template <class CvSender, class Env>
    struct receiver : stream_receiver_base {
      using opstate_t = opstate<CvSender, Env>;

      explicit receiver(opstate_t* op) noexcept
        : op_(op) {
      }

      template <class... Args>
      void set_value(Args&&...) noexcept {
        receiver::destroy_delete(op_); // NB: invalidates *this
      }

      template <class Error>
      [[noreturn]]
      void set_error(Error&&) noexcept {
        // A detached operation failed. There is noplace for the error to go.
        // This is unrecoverable, so we terminate.
        std::terminate();
      }

      void set_stopped() noexcept {
        receiver::destroy_delete(op_); // NB: invalidates *this
      }

      [[nodiscard]]
      auto get_env() const noexcept -> const Env& {
        return op_->env_;
      }

     private:
      // If the operation state was allocated with a user-provided allocator, then we must
      // use the allocator stored within the operation state to destroy the operation
      // state. This is a good time to use C++20's destroying delete operation.
      static void destroy_delete(opstate_t* self) noexcept {
        if constexpr (__callable<get_allocator_t, Env>) {
          auto alloc = STDEXEC::get_allocator(self->env_);
          using alloc_t = decltype(alloc);
          using op_alloc_t = std::allocator_traits<alloc_t>::template rebind_alloc<opstate>;
          op_alloc_t op_alloc{alloc};
          std::allocator_traits<op_alloc_t>::destroy(op_alloc, self);
          std::allocator_traits<op_alloc_t>::deallocate(op_alloc, self, 1);
        } else {
          delete self;
        }
      }

      opstate_t* op_;
    };

    template <class CvSender, class Env>
    struct opstate {
      using operation_state_concept = STDEXEC::operation_state_t;

      explicit opstate(connect_t, CvSender&& sndr, Env env)
        : env_(static_cast<Env&&>(env))
        , op_data_(static_cast<CvSender&&>(sndr), receiver<CvSender, Env>{this}) {
      }

      explicit opstate(CvSender&& sndr, Env env)
        : opstate(connect, static_cast<CvSender&&>(sndr), static_cast<Env&&>(env)) {
        // If the operation completes synchronously, then the following line will cause
        // the destruction of *this, which is not a problem because we used a delegating
        // constructor, so *this is considered fully constructed.
        op_data_.submit(static_cast<CvSender&&>(sndr), receiver<CvSender, Env>{this});
      }

     private:
      friend struct receiver<CvSender, Env>;

      STDEXEC_ATTRIBUTE(no_unique_address)
      Env env_;
      STDEXEC_ATTRIBUTE(no_unique_address)
      submit_result<CvSender, receiver<CvSender, Env>> op_data_;
    };

    template <class CvSender, class Env>
    concept _use_submit = __submittable<CvSender, submit_receiver> && __same_as<Env, __root_env>
                       && __same_as<void, __submit_result_t<CvSender, submit_receiver>>;
  } // namespace _start_detached

  template <>
  struct apply_sender_for<start_detached_t> {
    template <class CvSender, class Env = __root_env>
    void operator()(CvSender&& sndr, Env&& env = {}) const noexcept(false) {
      using op_t = _start_detached::opstate<CvSender, __decay_t<Env>>;

#if !STDEXEC_APPLE_CLANG() // There seems to be a codegen bug in apple clang that causes
                           // `start_detached` to segfault when the code path below is
                           // taken.
      // BUGBUG NOT TO SPEC: the use of the non-standard `submit` algorithm here is a
      // conforming extension.
      if constexpr (_start_detached::_use_submit<CvSender, Env>) {
        // If submit(sndr, rcvr) returns void, then no state needs to be kept alive
        // for the operation. We can just call submit and return.
        STDEXEC::__submit::__submit(
          static_cast<CvSender&&>(sndr), _start_detached::submit_receiver{});
      } else
#endif
        if constexpr (__callable<get_allocator_t, Env>) {
        // Use the provided allocator to allocate the operation state.
        auto alloc = get_allocator(env);
        using alloc_t = decltype(alloc);
        using op_alloc_t = std::allocator_traits<alloc_t>::template rebind_alloc<op_t>;
        // We use the allocator to allocate the operation state and also to construct it.
        op_alloc_t op_alloc{alloc};
        op_t* op = std::allocator_traits<op_alloc_t>::allocate(op_alloc, 1);
        exec::scope_guard g{[op, &op_alloc]() noexcept {
          std::allocator_traits<op_alloc_t>::deallocate(op_alloc, op, 1);
        }};
        // This can potentially throw. If it does, the scope guard will deallocate the
        // storage automatically.
        std::allocator_traits<op_alloc_t>::construct(
          op_alloc, op, static_cast<CvSender&&>(sndr), static_cast<Env&&>(env));
        // The operation state is now constructed, dismiss the scope guard.
        g.dismiss();
      } else {
        // The caller did not provide an allocator, so we use the default allocator.
        [[maybe_unused]]
        op_t* op = new op_t{static_cast<CvSender&&>(sndr), static_cast<Env&&>(env)};
        // The operation has now started and is responsible for deleting itself when it
        // completes.
      }
    }
  };
} // namespace nvexec::_strm
