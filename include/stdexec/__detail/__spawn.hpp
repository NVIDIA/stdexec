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
      __spawn_state_base() = default;

      __spawn_state_base(__spawn_state_base&&) = delete;

      virtual void __complete() noexcept = 0;

     protected:
      ~__spawn_state_base() = default;
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
        : __alloc_(std::move(__alloc))
        , __op_(connect(std::move(__sndr), __spawn_receiver(this)))
        , __assoc_(__token.try_associate()) {
      }

      void __complete() noexcept override {
        [[maybe_unused]]
        auto assoc = std::move(__assoc_);

        {
          using traits = std::allocator_traits<_Alloc>::template rebind_traits<__spawn_state>;
          typename traits::allocator_type alloc(__alloc_);
          traits::destroy(alloc, this);
          traits::deallocate(alloc, this, 1);
        }
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
    };


    struct spawn_t {
      template <sender _Sender, scope_token _Token>
      void operator()(_Sender&& __sndr, _Token&& __tkn) const {
        return (*this)(static_cast<_Sender&&>(__sndr), static_cast<_Token&&>(__tkn), env<>{});
      }

      template <sender _Sender, scope_token _Token, class _Env>
      void operator()(_Sender&& __sndr, _Token&& __tkn, _Env&& __env) const {
        auto wrappedSender = __tkn.wrap(static_cast<_Sender&&>(__sndr));
        auto sndrEnv = get_env(wrappedSender);

        using raw_alloc = decltype(__spawn_common::__choose_alloc(__env, sndrEnv));

        auto senderWithEnv =
          write_env(std::move(wrappedSender), __spawn_common::__choose_senv(__env, sndrEnv));

        using spawn_state_t =
          __spawn_state<raw_alloc, std::remove_cvref_t<_Token>, decltype(senderWithEnv)>;

        using traits = std::allocator_traits<raw_alloc>::template rebind_traits<spawn_state_t>;
        typename traits::allocator_type alloc(__spawn_common::__choose_alloc(__env, sndrEnv));

        auto* op = traits::allocate(alloc, 1);

        __scope_guard __guard{[&]() noexcept { traits::deallocate(alloc, op, 1); }};

        traits::construct(alloc, op, alloc, std::move(senderWithEnv), static_cast<_Token&&>(__tkn));

        __guard.__dismiss();

        op->__run();
      }
    };
  } // namespace __spawn

  using __spawn::spawn_t;

  inline constexpr spawn_t spawn{};
} // namespace STDEXEC
