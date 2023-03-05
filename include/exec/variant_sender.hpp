/*
 * Copyright (c) 2023 Maikel Nadolski
 * Copyright (c) 2021-2023 NVIDIA Corporation
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

#include <variant>
#include <type_traits>

namespace exec {
  namespace __variant {
    using namespace stdexec;

    template <class _ReceiverId, class... _SenderIds>
    struct __operation_state {
      class __t {
        std::variant<connect_result_t<stdexec::__t<_SenderIds>, stdexec::__t<_ReceiverId>>...>
          __variant_;

        friend void tag_invoke(start_t, __t& __self) noexcept {
          std::visit([](auto& __s) { start(__s); }, __self.__variant_);
        }

       public:
        template <class _Sender, class _Receiver>
          requires(
            __one_of<__id<_Sender>, _SenderIds...> && __same_as<__id<_Receiver>, _ReceiverId>
            && __callable<connect_t, _Sender, _Receiver>)
        __t(_Sender&& __sender, _Receiver&& __receiver) noexcept(
          __nothrow_connectable<_Sender, _Receiver>)
          : __variant_{std::in_place_type<connect_result_t<_Sender, _Receiver>>, __conv{[&] {
                         return connect((_Sender&&) __sender, (_Receiver&&) __receiver);
                       }}} {
        }
      };
    };

    template <class... _SenderIds>
    struct __sender {
      template <class _Self, class _Env>
      using __completion_signatures_t = __concat_completion_signatures_t<
        completion_signatures_of_t<__copy_cvref_t<_Self, stdexec::__t<_SenderIds>>, _Env>...>;

      template <class _Self, class _Receiver>
      struct __visitor {
        _Receiver __r;

        template <class _S>
        stdexec::__t< __operation_state<__id<_Receiver>, __copy_cvref_t<_Self, _SenderIds>...>>
          operator()(_S&& __s) const {
          return {(_S&&) __s, (_Receiver&&) __r};
        }
      };

      class __t {
        std::variant<stdexec::__t<_SenderIds>...> __variant_;

        template <__decays_to<__t> _Self, class _Receiver>
          requires receiver_of<_Receiver, __completion_signatures_t<_Self, env_of_t<_Receiver>>>
        friend stdexec::__t<
          __operation_state<__id<_Receiver>, __copy_cvref_t<_Self, _SenderIds>...>>
          tag_invoke(connect_t, _Self&& __self, _Receiver&& __r) noexcept(
            (__nothrow_connectable<__copy_cvref_t<_Self, stdexec::__t<_SenderIds>>, _Receiver>
             && ...)) {
          return std::visit(
            __visitor<_Self, _Receiver>{(_Receiver&&) __r}, ((_Self&&) __self).__variant_);
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
          -> __completion_signatures_t<_Self, _Env>;

       public:
        template <class _Sender>
          requires __one_of<decay_t<_Sender>, stdexec::__t<_SenderIds>...>
        __t(_Sender&& __sender) noexcept(
          std::is_nothrow_constructible_v<std::variant<stdexec::__t<_SenderIds>...>, _Sender>)
          : __variant_{(_Sender&&) __sender} {
        }

        template <class _Sender>
          requires __one_of<decay_t<_Sender>, stdexec::__t<_SenderIds>...>
        __t& operator=(_Sender&& __sender) noexcept(
          std::is_nothrow_assignable_v<std::variant<stdexec::__t<_SenderIds>...>, _Sender>) {
          __variant_ = (_Sender&&) __sender;
          return *this;
        }
      };
    };
  }

  template <class... _Senders>
  using variant_sender =
    stdexec::__t<__variant::__sender<stdexec::__id<stdexec::decay_t<_Senders>>...>>;
}