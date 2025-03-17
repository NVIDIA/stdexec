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
#include "__env.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__utility.hpp"

namespace stdexec {
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __write adaptor
  namespace __write_ {
    struct __write_env_t {
      template <sender _Sender, class _Env>
      auto operator()(_Sender&& __sndr, _Env __env) const {
        return __make_sexpr<__write_env_t>(
          static_cast<_Env&&>(__env), static_cast<_Sender&&>(__sndr));
      }

      template <class _Env>
      STDEXEC_ATTRIBUTE((always_inline)) auto operator()(_Env __env) const -> __binder_back<__write_env_t, _Env> {
        return {{static_cast<_Env&&>(__env)}, {}, {}};
      }

      template <class _Env>
      STDEXEC_ATTRIBUTE((always_inline)) static auto __transform_env_fn(_Env&& __env) noexcept {
        return [&](__ignore, const auto& __state, __ignore) noexcept {
          return __env::__join(__state, static_cast<_Env&&>(__env));
        };
      }

      template <sender_expr_for<__write_env_t> _Self, class _Env>
      static auto transform_env(const _Self& __self, _Env&& __env) noexcept {
        return __sexpr_apply(__self, __transform_env_fn(static_cast<_Env&&>(__env)));
      }
    };

    struct __write_env_impl : __sexpr_defaults {
      static constexpr auto get_attrs = //
        []<class _Child>(__ignore, const _Child& __child) noexcept {
          return __env::__join(
            prop{__is_scheduler_affine_t{}, __mbool<__is_scheduler_affine<_Child>>{}},
            stdexec::get_env(__child));
        };

      static constexpr auto get_env = //
        [](__ignore, const auto& __state, const auto& __rcvr) noexcept {
          return __env::__join(__state, stdexec::get_env(__rcvr));
        };

      static constexpr auto get_completion_signatures = //
        []<class _Self, class... _Env>(_Self&&, _Env&&...) noexcept
        -> __completion_signatures_of_t<
          __child_of<_Self>,
          __meval<__env::__join_t, const __decay_t<__data_of<_Self>>&, _Env...>> {
        static_assert(sender_expr_for<_Self, __write_env_t>);
        return {};
      };
    };
  } // namespace __write_

  using __write_::__write_env_t;
  inline constexpr __write_env_t __write{};
  inline constexpr __write_env_t __write_env{};

  template <>
  struct __sexpr_impl<__write_env_t> : __write_::__write_env_impl { };
} // namespace stdexec
