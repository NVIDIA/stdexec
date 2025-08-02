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
#include "__completion_signatures.hpp"
#include "__meta.hpp"
#include "__receivers.hpp"
#include "__type_traits.hpp"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.factories]
  namespace __just {
    template <class _JustTag>
    struct __impl : __sexpr_defaults {
      using __tag_t = typename _JustTag::__tag_t;

      static constexpr auto get_attrs =
        [](__ignore) noexcept -> cprop<__is_scheduler_affine_t, true> {
        return {};
      };

      static constexpr auto get_completion_signatures =
        []<class _Sender>(_Sender&&, auto&&...) noexcept {
          static_assert(sender_expr_for<_Sender, _JustTag>);
          return completion_signatures<__mapply<__qf<__tag_t>, __decay_t<__data_of<_Sender>>>>{};
        };

      static constexpr auto start =
        []<class _State, class _Receiver>(_State& __state, _Receiver& __rcvr) noexcept -> void {
        __state.apply(
          [&]<class... _Ts>(_Ts&... __ts) noexcept {
            __tag_t()(static_cast<_Receiver&&>(__rcvr), static_cast<_Ts&&>(__ts)...);
          },
          __state);
      };

      static constexpr auto submit =
        []<class _Sender, class _Receiver>(_Sender&& __sndr, _Receiver __rcvr) noexcept -> void {
        static_assert(sender_expr_for<_Sender, _JustTag>);
        auto&& __state = get_state(static_cast<_Sender&&>(__sndr), __rcvr);
        __state.apply(
          [&]<class... _Ts>(_Ts&&... __ts) noexcept {
            __tag_t()(static_cast<_Receiver&&>(__rcvr), static_cast<_Ts&&>(__ts)...);
          },
          static_cast<decltype(__state)>(__state));
      };
    };

    struct just_t {
      using __tag_t = set_value_t;

      template <__movable_value... _Ts>
      STDEXEC_ATTRIBUTE(host, device)
      auto operator()(_Ts&&... __ts) const noexcept((__nothrow_decay_copyable<_Ts> && ...)) {
        return __make_sexpr<just_t>(__tuple{static_cast<_Ts&&>(__ts)...});
      }
    };

    struct just_error_t {
      using __tag_t = set_error_t;

      template <__movable_value _Error>
      STDEXEC_ATTRIBUTE(host, device)
      auto operator()(_Error&& __err) const noexcept(__nothrow_decay_copyable<_Error>) {
        return __make_sexpr<just_error_t>(__tuple{static_cast<_Error&&>(__err)});
      }
    };

    struct just_stopped_t {
      using __tag_t = set_stopped_t;

      template <class _Tag = just_stopped_t>
      STDEXEC_ATTRIBUTE(host, device)
      auto operator()() const noexcept {
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
} // namespace stdexec

STDEXEC_PRAGMA_POP()
