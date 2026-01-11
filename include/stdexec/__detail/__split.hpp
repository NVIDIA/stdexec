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

namespace STDEXEC {
  ////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.split]
  namespace __split {
    using namespace __shared;

    struct __split_t { };

    struct split_t {
      template <sender _Sender, class _Env = env<>>
        requires sender_in<_Sender, _Env> && __decay_copyable<env_of_t<_Sender>>
      auto operator()(_Sender&& __sndr, _Env&& __env = {}) const -> __well_formed_sender auto {
        return STDEXEC::transform_sender(
          __make_sexpr<split_t>(__env, static_cast<_Sender&&>(__sndr)), __env);
      }

      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()() const noexcept {
        return __closure(*this);
      }

      template <class _CvrefSender, class _Env>
      using __receiver_t = __t<__meval<__receiver, __cvref_id<_CvrefSender>, __id<_Env>>>;

      template <__decay_copyable _Sender>
      static auto transform_sender(set_value_t, _Sender&& __sndr, __ignore) {
        using _Receiver = __receiver_t<__child_of<_Sender>, __decay_t<__data_of<_Sender>>>;
        static_assert(sender_to<__child_of<_Sender>, _Receiver>);

        return __apply(
          [&]<class _Env, class _Child>(__ignore, _Env&& __env, _Child&& __child) {
            // The shared state starts life with a ref-count of one.
            auto* __sh_state =
              new __shared_state{static_cast<_Child&&>(__child), static_cast<_Env&&>(__env)};

            return __make_sexpr<__split_t>(__box{__split_t(), __sh_state});
          },
          static_cast<_Sender&&>(__sndr));
      }

      template <class _Sender>
      static auto transform_sender(set_value_t, _Sender&&, __ignore) {
        return __not_a_sender<_SENDER_TYPE_IS_NOT_COPYABLE_, _WITH_SENDER_<_Sender>>();
      }
    };
  } // namespace __split

  using __split::split_t;
  inline constexpr split_t split{};

  template <>
  struct __sexpr_impl<__split::__split_t> : __shared::__shared_impl<__split::__split_t> { };

  template <>
  struct __sexpr_impl<split_t> : __sexpr_defaults {
    template <class _Sender, class... _Env>
    static consteval auto get_completion_signatures() {
      // Use the senders decay-copyability as a proxy for whether it is lvalue-connectable.
      if constexpr (__decay_copyable<_Sender>) {
        using __sndr_t =
          __detail::__transform_sender_result_t<split_t, set_value_t, _Sender, env<>>;
        return STDEXEC::get_completion_signatures<__sndr_t, _Env...>();
      } else {
        return STDEXEC::__invalid_completion_signature<
          _SENDER_TYPE_IS_NOT_COPYABLE_,
          _WITH_SENDER_<_Sender>
        >();
      }
    }
  };
} // namespace STDEXEC
