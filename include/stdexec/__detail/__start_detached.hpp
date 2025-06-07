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

#include <memory>

#include "__meta.hpp"
#include "__env.hpp"
#include "__receivers.hpp"
#include "__env.hpp"
#include "__scope.hpp"
#include "__submit.hpp"
#include "__transform_sender.hpp"

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.consumer.start_detached]
  namespace __start_detached {
    struct __submit_receiver {
      using receiver_concept = receiver_t;
      using __t = __submit_receiver;
      using __id = __submit_receiver;

      template <class... _As>
      void set_value(_As&&...) noexcept {
      }

      template <class _Error>
      [[noreturn]]
      void set_error(_Error&&) noexcept {
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

    template <class _SenderId, class _EnvId>
    struct __operation : __immovable {
      using _Sender = __cvref_t<_SenderId>;
      using _Env = __t<_EnvId>;

      explicit __operation(connect_t, _Sender&& __sndr, _Env __env)
        : __env_(static_cast<_Env&&>(__env))
        , __op_data_(static_cast<_Sender&&>(__sndr), __receiver{this}) {
      }

      explicit __operation(_Sender&& __sndr, _Env __env)
        : __operation(connect, static_cast<_Sender&&>(__sndr), static_cast<_Env&&>(__env)) {
        // If the operation completes synchronously, then the following line will cause
        // the destruction of *this, which is not a problem because we used a delegating
        // constructor, so *this is considered fully constructed.
        __op_data_.submit(static_cast<_Sender&&>(__sndr), __receiver{this});
      }

      static void __destroy_delete(__operation* __self) noexcept {
        if constexpr (__callable<get_allocator_t, _Env>) {
          auto __alloc = stdexec::get_allocator(__self->__env_);
          using _Alloc = decltype(__alloc);
          using _OpAlloc =
            typename std::allocator_traits<_Alloc>::template rebind_alloc<__operation>;
          _OpAlloc __op_alloc{__alloc};
          std::allocator_traits<_OpAlloc>::destroy(__op_alloc, __self);
          std::allocator_traits<_OpAlloc>::deallocate(__op_alloc, __self, 1);
        } else {
          delete __self;
        }
      }

      // The start_detached receiver deletes the operation state.
      struct __receiver {
        using receiver_concept = receiver_t;
        using __t = __receiver;
        using __id = __receiver;

        template <class... _As>
        void set_value(_As&&...) noexcept {
          __operation::__destroy_delete(__op_); // NB: invalidates *this
        }

        template <class _Error>
        [[noreturn]]
        void set_error(_Error&&) noexcept {
          // A detached operation failed. There is noplace for the error to go.
          // This is unrecoverable, so we terminate.
          std::terminate();
        }

        void set_stopped() noexcept {
          __operation::__destroy_delete(__op_); // NB: invalidates *this
        }

        auto get_env() const noexcept -> const _Env& {
          return __op_->__env_;
        }

        __operation* __op_;
      };

      STDEXEC_ATTRIBUTE(no_unique_address) _Env __env_;
      STDEXEC_ATTRIBUTE(no_unique_address) submit_result<_Sender, __receiver> __op_data_;
    };

    template <class _Sender, class _Env>
    concept __use_submit = __submittable<_Sender, __submit_receiver> && __same_as<_Env, __root_env>
                        && __same_as<void, __submit_result_t<_Sender, __submit_receiver>>;

    struct start_detached_t {
      template <sender_in<__root_env> _Sender>
        requires __callable<
          apply_sender_t,
          __late_domain_of_t<_Sender, __root_env, __early_domain_of_t<_Sender>>,
          start_detached_t,
          _Sender
        >
      void operator()(_Sender&& __sndr) const {
        auto __domain = __get_late_domain(__sndr, __root_env{}, __get_early_domain(__sndr));
        stdexec::apply_sender(__domain, *this, static_cast<_Sender&&>(__sndr));
      }

      template <class _Env, sender_in<__as_root_env_t<_Env>> _Sender>
        requires __callable<
          apply_sender_t,
          __late_domain_of_t<_Sender, __as_root_env_t<_Env>, __early_domain_of_t<_Sender>>,
          start_detached_t,
          _Sender,
          __as_root_env_t<_Env>
        >
      void operator()(_Sender&& __sndr, _Env&& __env) const {
        auto __env2 = __as_root_env(static_cast<_Env&&>(__env));
        auto __domain = __get_late_domain(__sndr, __env2, __get_early_domain(__sndr));
        stdexec::apply_sender(__domain, *this, static_cast<_Sender&&>(__sndr), __env2);
      }

      // Below is the default implementation for `start_detached`.
      template <class _Sender, class _Env = __root_env>
        requires sender_in<_Sender, __as_root_env_t<_Env>>
      void apply_sender(_Sender&& __sndr, _Env&& __env = {}) const noexcept(false) {
        using _Op = __operation<__cvref_id<_Sender>, __id<__decay_t<_Env>>>;

#if !STDEXEC_APPLE_CLANG() // There seems to be a codegen bug in apple clang that causes
                           // `start_detached` to segfault when the code path below is
                           // taken.
        // BUGBUG NOT TO SPEC: the use of the non-standard `submit` algorithm here is a
        // conforming extension.
        if constexpr (__use_submit<_Sender, _Env>) {
          // If submit(sndr, rcvr) returns void, then no state needs to be kept alive
          // for the operation. We can just call submit and return.
          stdexec::__submit::__submit(static_cast<_Sender&&>(__sndr), __submit_receiver{});
        } else
#endif
          if constexpr (__callable<get_allocator_t, _Env>) {
          // Use the provided allocator if any to allocate the operation state.
          auto __alloc = get_allocator(__env);
          using _Alloc = decltype(__alloc);
          using _OpAlloc = typename std::allocator_traits<_Alloc>::template rebind_alloc<_Op>;
          // We use the allocator to allocate the op state and also to construct it.
          _OpAlloc __op_alloc{__alloc};
          _Op* __op = std::allocator_traits<_OpAlloc>::allocate(__op_alloc, 1);
          __scope_guard __g{[__op, &__op_alloc]() noexcept {
            std::allocator_traits<_OpAlloc>::deallocate(__op_alloc, __op, 1);
          }};
          // This can potentially throw. If it does, the scope guard will deallocate the
          // storage automatically.
          std::allocator_traits<_OpAlloc>::construct(
            __op_alloc, __op, static_cast<_Sender&&>(__sndr), static_cast<_Env&&>(__env));
          // The operation state is now constructed, dismiss the scope guard.
          __g.__dismiss();
        } else {
          // The caller did not provide an allocator, so we use the default allocator.
          [[maybe_unused]]
          _Op* __op = new _Op(static_cast<_Sender&&>(__sndr), static_cast<_Env&&>(__env));
          // The operation has now started and is responsible for deleting itself when it
          // completes.
        }
      }
    };
  } // namespace __start_detached

  using __start_detached::start_detached_t;
  inline constexpr start_detached_t start_detached{};
} // namespace stdexec
