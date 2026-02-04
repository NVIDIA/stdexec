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

#include "__basic_sender.hpp"
#include "__completion_behavior.hpp"
#include "__completion_signatures.hpp"
#include "__meta.hpp"
#include "__receivers.hpp"
#include "__sender_introspection.hpp"
#include "__type_traits.hpp"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace STDEXEC {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.factories]
  namespace __just {
    template <class _SetTag>
    struct __attrs {
      static constexpr auto query(get_completion_behavior_t<_SetTag>) noexcept {
        return completion_behavior::inline_completion;
      }
    };

    template <class _SetTag, class _Tuple, class _Receiver>
    struct __opstate {
      constexpr void start() noexcept {
        __apply(_SetTag(), static_cast<_Tuple&&>(__data_), static_cast<_Receiver&&>(__rcvr_));
      }

      _Receiver __rcvr_;
      _Tuple __data_;
    };

    template <class _JustTag>
    struct __impl : __sexpr_defaults {
      using __set_tag_t = _JustTag::__tag_t;

      static constexpr auto get_attrs = [](__ignore, __ignore) noexcept -> __attrs<__set_tag_t> {
        return {};
      };

      template <class _Sender, class... _Env>
      static consteval auto get_completion_signatures() {
        static_assert(sender_expr_for<_Sender, _JustTag>);
        return completion_signatures<__mapply<__qf<__set_tag_t>, __decay_t<__data_of<_Sender>>>>{};
      }

      static constexpr auto connect =
        []<class _Sender, class _Receiver>(_Sender&& __sndr, _Receiver&& __rcvr) noexcept(
          __nothrow_decay_copyable<_Sender>) {
          auto& [__tag, __data] = __sndr;
          return __opstate<__set_tag_t, decltype(__data), _Receiver>{
            static_cast<_Receiver&&>(__rcvr), STDEXEC::__forward_like<_Sender>(__data)};
        };
    };

    struct just_t {
      using __tag_t = set_value_t;

      template <__movable_value... _Ts>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr auto operator()(_Ts&&... __ts) const noexcept(__nothrow_decay_copyable<_Ts...>) {
        return __make_sexpr<just_t>(__tuple{static_cast<_Ts&&>(__ts)...});
      }
    };

    struct just_error_t {
      using __tag_t = set_error_t;

      template <__movable_value _Error>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr auto operator()(_Error&& __err) const noexcept(__nothrow_decay_copyable<_Error>) {
        return __make_sexpr<just_error_t>(__tuple{static_cast<_Error&&>(__err)});
      }
    };

    struct just_stopped_t {
      using __tag_t = set_stopped_t;

      template <class _Tag = just_stopped_t>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr auto operator()() const noexcept {
        return __make_sexpr<_Tag>(__tuple{});
      }
    };
  } // namespace __just

  using __just::just_t;
  using __just::just_error_t;
  using __just::just_stopped_t;

  template <>
  struct __sexpr_impl<just_t> : __just::__impl<just_t> { };

  template <>
  struct __sexpr_impl<just_error_t> : __just::__impl<just_error_t> { };

  template <>
  struct __sexpr_impl<just_stopped_t> : __just::__impl<just_stopped_t> { };

  inline constexpr just_t just{};
  inline constexpr just_error_t just_error{};
  inline constexpr just_stopped_t just_stopped{};
} // namespace STDEXEC

STDEXEC_PRAGMA_POP()
