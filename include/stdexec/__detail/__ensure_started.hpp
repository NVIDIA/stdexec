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
#include "__basic_sender.hpp"
#include "__concepts.hpp"
#include "__meta.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__senders.hpp"
#include "__shared.hpp"
#include "__transform_sender.hpp"
#include "__type_traits.hpp"

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.ensure_started]
  namespace __ensure_started {
    using namespace __shared;

    struct __ensure_started_t { };

    struct ensure_started_t {
      template <sender _Sender, class _Env = env<>>
        requires sender_in<_Sender, _Env> && __decay_copyable<env_of_t<_Sender>>
      [[nodiscard]]
      auto operator()(_Sender&& __sndr, _Env&& __env = {}) const -> __well_formed_sender auto {
        if constexpr (sender_expr_for<_Sender, __ensure_started_t>) {
          return static_cast<_Sender&&>(__sndr);
        } else {
          return stdexec::transform_sender(
            __make_sexpr<ensure_started_t>(__env, static_cast<_Sender&&>(__sndr)), __env);
        }
      }

      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()() const noexcept -> __binder_back<ensure_started_t> {
        return {{}, {}, {}};
      }

      template <class _CvrefSender, class _Env>
      using __receiver_t = __t<__meval<__receiver, __cvref_id<_CvrefSender>, __id<_Env>>>;

      template <__decay_copyable _Sender, class _Env>
      static auto transform_sender(set_value_t, _Sender&& __sndr, const _Env&) {
        using _Receiver = __receiver_t<__child_of<_Sender>, __decay_t<__data_of<_Sender>>>;
        static_assert(sender_to<__child_of<_Sender>, _Receiver>);

        return __sexpr_apply(
          static_cast<_Sender&&>(__sndr),
          [&]<class _Env2, class _Child>(__ignore, _Env2&& __env, _Child&& __child) {
            // TODO(ericniebler): should we join the env passed to ensure_started with the
            // env of the receiver?
            // The shared state starts life with a ref-count of one.
            auto* __sh_state =
              new __shared_state{static_cast<_Child&&>(__child), static_cast<_Env2&&>(__env)};

            // Eagerly start the work:
            __sh_state->__try_start(); // cannot throw

            return __make_sexpr<__ensure_started_t>(__box{__ensure_started_t(), __sh_state});
          });
      }

      template <class _Sender, class _Env>
      static auto transform_sender(set_value_t, _Sender&&, const _Env&) {
        return __mexception<_SENDER_TYPE_IS_NOT_COPYABLE_, _WITH_SENDER_<_Sender>>();
      }
    };
  } // namespace __ensure_started

  using __ensure_started::ensure_started_t;
  inline constexpr ensure_started_t ensure_started{};

  template <>
  struct __sexpr_impl<__ensure_started::__ensure_started_t>
    : __shared::__shared_impl<__ensure_started::__ensure_started_t> { };

  template <>
  struct __sexpr_impl<ensure_started_t> : __sexpr_defaults {
    static constexpr auto get_completion_signatures =
      []<class _Sender, class... _Env>(_Sender&&, const _Env&...) noexcept {
        // Use the senders decay-copyability as a proxy for whether it is lvalue-connectable.
        if constexpr (__decay_copyable<_Sender>) {
          using __result_t =
            __completion_signatures_of_t<transform_sender_result_t<_Sender, _Env...>, _Env...>;
          return __result_t{};
        } else {
          return __mexception<_SENDER_TYPE_IS_NOT_COPYABLE_, _WITH_SENDER_<_Sender>>();
        }
      };
  };
} // namespace stdexec
