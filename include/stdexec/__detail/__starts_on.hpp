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
#include "__just.hpp"
#include "__let.hpp"
#include "__schedulers.hpp"
#include "__senders_core.hpp"
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
        return __make_sexpr<starts_on_t>(static_cast<_Scheduler&&>(__sched), static_cast<_Sender&&>(__sndr));
      }

      template <__decay_copyable _Sender, class _Env>
      static auto transform_sender(set_value_t, _Sender&& __sndr, const _Env&) {
        return __sexpr_apply(
          static_cast<_Sender&&>(__sndr),
          []<class _Data, class _Child>(__ignore, _Data&& __data, _Child&& __child) -> auto {
            // This is the heart of starts_on: It uses `let_value` to schedule `__child` on the given scheduler:
            return let_value(
              continues_on(just(), __data), __detail::__always{static_cast<_Child&&>(__child)});
          });
      }

      template <class _Sender, class _Env>
      static auto transform_sender(set_value_t, _Sender&&, const _Env&) {
        return _ERROR_<_SENDER_TYPE_IS_NOT_COPYABLE_, _WITH_SENDER_<_Sender>>{};
      }
    };
  } // namespace __starts_on_ns

  using __starts_on_ns::starts_on_t;
  inline constexpr starts_on_t starts_on{};

  using start_on_t = starts_on_t;
  inline constexpr starts_on_t start_on{};

  template <>
  struct __sexpr_impl<starts_on_t> : __sexpr_defaults {
    template <class _Scheduler, class _Child>
    struct __attrs {
      using __t = __attrs;
      using __id = __attrs;

      template <class _Sch, class... _Env>
      static constexpr auto __mk_env2(_Sch __sch, _Env&&... __env) {
        return env(__mk_sch_env(__sch, __env...), static_cast<_Env&&>(__env)...);
      }

      template <class _Sch, class... _Env>
      using __env2_t = decltype(__mk_env2(__declval<_Sch>(), __declval<_Env>()...));

      // Query for completion scheduler
      template <class _SetTag, class... _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto query(get_completion_scheduler_t<_SetTag>, _Env&&...) const noexcept
        -> _Scheduler
        requires(__completes_inline<_SetTag, env_of_t<_Child>, __env2_t<_Scheduler, _Env>...>) {
        // If child completes inline, then starts_on completes on its scheduler
        return __sched_;
      }

      // Query for completion scheduler - delegates to child's env with augmented environment
      template <class _SetTag, class... _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto query(get_completion_scheduler_t<_SetTag> __query, _Env&&... __env) const noexcept
        -> __call_result_t<
             get_completion_scheduler_t<_SetTag>,
             env_of_t<_Child>,
             __env2_t<_Scheduler, _Env>...
          >
        requires(!__completes_inline<_SetTag, env_of_t<_Child>, __env2_t<_Scheduler, _Env>...>) {
        // If child doesn't complete inline, delegate to child's completion scheduler
        return __query(__attr_, __mk_env2(__sched_, static_cast<_Env&&>(__env))...);
      }

      // Query for completion domain - calculate type from child's env with augmented environment
      template <class _SetTag, class... _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto query(get_completion_domain_t<_SetTag>, _Env&&...) const noexcept
        -> __call_result_t<get_completion_domain_t<_SetTag>, env_of_t<_Child>, __env2_t<_Scheduler, _Env>...> {
        return {};
      }

      _Scheduler __sched_;
      env_of_t<_Child> __attr_;
    };

    static constexpr auto get_attrs = []<class _Data, class _Child>(
                                        const _Data& __data,
                                        const _Child& __child) noexcept -> decltype(auto) {
      return __attrs<_Data, _Child>{__data, stdexec::get_env(__child)};
    };

    static constexpr auto get_completion_signatures =
      []<class _Sender, class... _Env>(_Sender&&, const _Env&...) noexcept
      -> __completion_signatures_of_t<transform_sender_result_t<_Sender, _Env...>, _Env...> {
      return {};
    };
  };
} // namespace stdexec
