/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *                         Copyright (c) 2025 Robert Leahy. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "../stdexec/execution.hpp"

namespace exec {
  namespace __unless_stop_requested {
    using namespace STDEXEC;

    template <typename _Env>
    inline constexpr bool __unstoppable_env = unstoppable_token<stop_token_of_t<_Env>>;

    template <typename _Receiver>
    inline constexpr bool __unstoppable_receiver = __unstoppable_env<env_of_t<_Receiver>>;

    template <class _Sender, class _Env>
    using __completions = transform_completion_signatures<
      __completion_signatures_of_t<_Sender, _Env>,
      std::conditional_t<
        __unstoppable_env<_Env>,
        completion_signatures<>,
        completion_signatures<set_stopped_t()>
      >
    >;

    struct __connect_fn {
      template <class _Sender, class _Receiver>
        requires __unstoppable_receiver<_Receiver>
      constexpr connect_result_t<__child_of<_Sender>, _Receiver>
        operator()(_Sender&& __sndr, _Receiver __rcvr) const noexcept(
          noexcept(STDEXEC::connect(__declval<__child_of<_Sender>>(), (_Receiver&&) __rcvr))) {
        return __apply(
          [&]<class _Child>(auto, const auto&, _Child&& __child) {
            return STDEXEC::connect(
              static_cast<_Child&&>(__child), static_cast<_Receiver&&>(__rcvr));
          },
          static_cast<_Sender&&>(__sndr));
      }

      template <class _Sender, class _Receiver>
      constexpr __op_state<_Sender, _Receiver> operator()(_Sender&& __sndr, _Receiver __rcvr) const
        noexcept(__nothrow_constructible_from<__op_state<_Sender, _Receiver>, _Sender, _Receiver>) {
        return __op_state<_Sender, _Receiver>{(_Sender&&) __sndr, (_Receiver&&) __rcvr};
      }
    };

    struct unless_stop_requested_t : sender_adaptor_closure<unless_stop_requested_t> {
      constexpr auto operator()() const noexcept {
        return *this;
      }
      template <sender _Sender>
      constexpr __well_formed_sender auto operator()(_Sender&& __sndr) const {
        return __make_sexpr<unless_stop_requested_t>(__(), static_cast<_Sender&&>(__sndr));
      }
    };

    struct __unless_stop_requested_impl : __sexpr_defaults {
      template <class _Self, class _Env>
      static consteval auto get_completion_signatures() {
        static_assert(sender_expr_for<_Self, unless_stop_requested_t>);
        // TODO: port this to use constant evaluation
        return __completions<__child_of<_Self>, _Env>{};
      };

      static constexpr auto start = //
        []<class _State, class _Operation>(_State& __state, _Operation& __child_op) noexcept
        -> void {
        using __receiver_t = _State::__receiver_t;
        static_assert(!__unstoppable_receiver<__receiver_t>);
        if (get_stop_token(STDEXEC::get_env(__state.__rcvr_)).stop_requested()) {
          STDEXEC::set_stopped(static_cast<__receiver_t&&>(__state.__rcvr_));
        } else {
          STDEXEC::start(__child_op);
        }
      };

      static constexpr __connect_fn connect{};
    };
  } // namespace __unless_stop_requested

  using __unless_stop_requested::unless_stop_requested_t;
  inline constexpr __unless_stop_requested::unless_stop_requested_t unless_stop_requested{};
} // namespace exec

namespace STDEXEC {
  template <>
  struct __sexpr_impl<::exec::unless_stop_requested_t>
    : ::exec::__unless_stop_requested::__unless_stop_requested_impl { };
} // namespace STDEXEC
