/*
 * Copyright (c) 2025 Ian Petersen
 * Copyright (c) 2025 NVIDIA Corporation
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

#include "__env.hpp"
#include "__receivers.hpp"
#include "__scope.hpp"
#include "__scope_concepts.hpp"
#include "__sender_concepts.hpp"
#include "__spawn_common.hpp"
#include "__type_traits.hpp"
#include "__write_env.hpp"

#include <memory>
#include <type_traits>
#include <utility>

namespace STDEXEC {
  /////////////////////////////////////////////////////////////////////////////
  // [exec.spawn]
  namespace __spawn {
    struct __spawn_state_base {
      explicit __spawn_state_base(void (*__complete)(__spawn_state_base*) noexcept) noexcept
        : __complete_(__complete) {
      }

      __spawn_state_base(__spawn_state_base&&) = delete;

      void __complete() noexcept {
        __complete_(this);
      }

     protected:
      ~__spawn_state_base() = default;

     private:
      void (*__complete_)(__spawn_state_base*) noexcept;
    };

    struct __spawn_receiver {
      using receiver_concept = receiver_t;

      __spawn_state_base* __state_;

      void set_value() && noexcept {
        __state_->__complete();
      }

      void set_stopped() && noexcept {
        __state_->__complete();
      }
    };

    template <class _Alloc, scope_token _Token, sender _Sender>
    struct __spawn_state final : __spawn_state_base {
      using __op_t = connect_result_t<_Sender, __spawn_receiver>;

      __spawn_state(_Alloc __alloc, _Sender&& __sndr, _Token __token)
        : __spawn_state_base(__do_complete)
        , __alloc_(std::move(__alloc))
        , __op_(connect(std::move(__sndr), __spawn_receiver(this)))
        , __assoc_(__token.try_associate()) {
      }

      void __run() noexcept {
        if (__assoc_) {
          start(__op_);
        } else {
          __complete();
        }
      }

     private:
      using __assoc_t = std::remove_cvref_t<decltype(__declval<_Token&>().try_associate())>;

      _Alloc __alloc_;
      __op_t __op_;
      __assoc_t __assoc_;

      static void __do_complete(__spawn_state_base* __base) noexcept {
        auto* __self = static_cast<__spawn_state*>(__base);

        [[maybe_unused]]
        auto __assoc = std::move(__self->__assoc_);

        {
          using __traits = std::allocator_traits<_Alloc>::template rebind_traits<__spawn_state>;
          typename __traits::allocator_type __alloc(std::move(__self->__alloc_));
          __traits::destroy(__alloc, __self);
          __traits::deallocate(__alloc, __self, 1);
        }
      }
    };


    struct spawn_t {
      template <sender _Sender, scope_token _Token>
      void operator()(_Sender&& __sndr, _Token&& __tkn) const {
        return (*this)(static_cast<_Sender&&>(__sndr), static_cast<_Token&&>(__tkn), env<>{});
      }

      template <sender _Sender, scope_token _Token, class _Env>
      void operator()(_Sender&& __sndr, _Token&& __tkn, _Env&& __env) const {
        auto __wrapped_sender = __tkn.wrap(static_cast<_Sender&&>(__sndr));
        auto __sndr_env = get_env(__wrapped_sender);

        using __raw_alloc = decltype(__spawn_common::__choose_alloc(__env, __sndr_env));

        auto __sender_with_env =
          write_env(std::move(__wrapped_sender), __spawn_common::__choose_senv(__env, __sndr_env));

        using __spawn_state_t =
          __spawn_state<__raw_alloc, std::remove_cvref_t<_Token>, decltype(__sender_with_env)>;

        using __traits =
          std::allocator_traits<__raw_alloc>::template rebind_traits<__spawn_state_t>;
        typename __traits::allocator_type __alloc(
          __spawn_common::__choose_alloc(__env, __sndr_env));

        auto* __op = __traits::allocate(__alloc, 1);

        __scope_guard __guard{[&]() noexcept { __traits::deallocate(__alloc, __op, 1); }};

        __traits::construct(
          __alloc, __op, __alloc, std::move(__sender_with_env), static_cast<_Token&&>(__tkn));

        __guard.__dismiss();

        __op->__run();
      }
    };
  } // namespace __spawn

  using __spawn::spawn_t;

  inline constexpr spawn_t spawn{};
} // namespace STDEXEC
