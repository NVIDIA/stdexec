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
#include "__concepts.hpp"
#include "__diagnostics.hpp"
#include "__domain.hpp"
#include "__env.hpp"
#include "__let.hpp"
#include "__schedulers.hpp"
#include "__senders_core.hpp"
#include "__transform_sender.hpp"
#include "__utility.hpp"

namespace stdexec {
  namespace __detail {
    //! Constant function object always returning `__val_`.
    template <class _Ty, class = __name_of<__decay_t<_Ty>>>
    struct __always {
      _Ty __val_;

      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()() noexcept(__nothrow_constructible_from<_Ty, _Ty>) -> _Ty {
        return static_cast<_Ty&&>(__val_);
      }
    };

    template <class _Ty>
    __always(_Ty) -> __always<_Ty>;
  } // namespace __detail

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.starts_on]
  namespace __starts_on_ns {
    struct starts_on_t {
      template <scheduler _Scheduler, sender _Sender>
      auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const -> __well_formed_sender auto {
        auto __domain = query_or(get_domain, __sched, default_domain());
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<starts_on_t>(
            static_cast<_Scheduler&&>(__sched), static_cast<_Sender&&>(__sndr)));
      }

      template <class _Env>
      STDEXEC_ATTRIBUTE(always_inline)
      static auto __transform_env_fn(_Env&& __env) noexcept {
        return [&](__ignore, auto __sched, __ignore) noexcept {
          return __env::__join(__sched_env{__sched}, static_cast<_Env&&>(__env));
        };
      }

      template <class _Sender, class _Env>
      static auto transform_env(const _Sender& __sndr, _Env&& __env) noexcept {
        return __sexpr_apply(__sndr, __transform_env_fn(static_cast<_Env&&>(__env)));
      }

      template <class _Sender, class _Env>
      static auto transform_sender(_Sender&& __sndr, const _Env&) {
        return __sexpr_apply(
          static_cast<_Sender&&>(__sndr),
          []<class _Data, class _Child>(__ignore, _Data&& __data, _Child&& __child) -> auto {
            // This is the heart of starts_on: It uses `let_value` to schedule `__child` on the given scheduler:
            return let_value(schedule(__data), __detail::__always{static_cast<_Child&&>(__child)});
          });
      }
    };
  } // namespace __starts_on_ns

  using __starts_on_ns::starts_on_t;
  inline constexpr starts_on_t starts_on{};

  using start_on_t = starts_on_t;
  inline constexpr starts_on_t start_on{};

  template <>
  struct __sexpr_impl<starts_on_t> : __sexpr_defaults {
    static constexpr auto get_completion_signatures = []<class _Sender>(_Sender&&) noexcept
      -> __completion_signatures_of_t<transform_sender_result_t<default_domain, _Sender, env<>>> {
      return {};
    };
  };
} // namespace stdexec
