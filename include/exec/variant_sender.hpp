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

    template <class _ReceiverId, class... _CvrefSenderIds>
    struct __operation_state {
      class __t {
        std::variant<connect_result_t<__cvref_t<_CvrefSenderIds>, stdexec::__t<_ReceiverId>>...>
          __variant_;

        STDEXEC_MEMFN_DECL(void start)(this __t& __self) noexcept {
          std::visit([](auto& __s) { start(__s); }, __self.__variant_);
        }

       public:
        template <class _Sender, class _Receiver>
        __t(_Sender&& __sender, _Receiver&& __receiver) //
          noexcept(__nothrow_connectable<_Sender, _Receiver>)
          : __variant_{std::in_place_type<connect_result_t<_Sender, _Receiver>>, __conv{[&] {
                         return stdexec::connect(
                           static_cast<_Sender&&>(__sender), static_cast<_Receiver&&>(__receiver));
                       }}} {
        }
      };
    };

    template <class... _SenderIds>
    struct __sender {
      template <class _Self, class _Env>
      using __completion_signatures_t = __concat_completion_signatures_t<
        __completion_signatures_of_t<__copy_cvref_t<_Self, stdexec::__t<_SenderIds>>, _Env>...>;

      template <class _Self, class _Receiver>
      struct __visitor {
        _Receiver __r;

        template <class _Sender>
        auto operator()(_Sender&& __s) //
          -> stdexec::__t<__operation_state<__id<_Receiver>, __copy_cvref_t<_Self, _SenderIds>...>> {
          return {static_cast<_Sender&&>(__s), static_cast<_Receiver&&>(__r)};
        }
      };

      class __t : private std::variant<stdexec::__t<_SenderIds>...> {
        using __variant_t = std::variant<stdexec::__t<_SenderIds>...>;

        auto base() && noexcept -> __variant_t&& {
          return std::move(*this);
        }

        auto base() & noexcept -> __variant_t& {
          return *this;
        }

        auto base() const & noexcept -> const __variant_t& {
          return *this;
        }

        template <__decays_to<__t> _Self, receiver _Receiver>
          requires(sender_to<__copy_cvref_t<_Self, stdexec::__t<_SenderIds>>, _Receiver> && ...)
        STDEXEC_MEMFN_DECL(auto connect)(this _Self&& __self, _Receiver __r) noexcept((
          __nothrow_connectable<__copy_cvref_t<_Self, stdexec::__t<_SenderIds>>, _Receiver> && ...))
          -> stdexec::__t<__operation_state<
            stdexec::__id<_Receiver>,
            __cvref_id<_Self, stdexec::__t<_SenderIds>>...>> {
          return std::visit(
            __visitor<_Self, _Receiver>{static_cast<_Receiver&&>(__r)},
            static_cast<_Self&&>(__self).base());
        }

        template <__decays_to<__t> _Self, class _Env>
        STDEXEC_MEMFN_DECL(auto get_completion_signatures)(this _Self&&, _Env&&)
          -> __completion_signatures_t<_Self, _Env> {
          return {};
        }

       public:
        using sender_concept = stdexec::sender_t;
        using __id = __sender;

        __t() = default;

        template <class _Sender>
          requires __one_of<__decay_t<_Sender>, stdexec::__t<_SenderIds>...>
        __t(_Sender&& __sender) //
          noexcept(__nothrow_constructible_from<std::variant<stdexec::__t<_SenderIds>...>, _Sender>)
          : __variant_t{static_cast<_Sender&&>(__sender)} {
        }

        using __variant_t::operator=;
        using __variant_t::index;
        using __variant_t::emplace;
        using __variant_t::swap;
      };
    };
  } // namespace __variant

  template <class... _Senders>
  using variant_sender =
    stdexec::__t<__variant::__sender<stdexec::__id<stdexec::__decay_t<_Senders>>...>>;
} // namespace exec

namespace stdexec::__detail {
  struct __variant_sender_name {
    template <class _Sender>
    using __f = __mapply<
      __transform<__mcompose<__q<__name_of>, __q<__t>>, __q<exec::__variant::__sender>>,
      _Sender>;
  }; // namespace stdexec::__detail

  template <class... _SenderIds>
  extern __variant_sender_name __name_of_v<exec::__variant::__sender<_SenderIds...>>;
} // namespace stdexec::__detail
