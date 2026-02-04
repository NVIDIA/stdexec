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
#include "__completion_signatures_of.hpp"
#include "__concepts.hpp"
#include "__diagnostics.hpp"
#include "__domain.hpp"
#include "__env.hpp"
#include "__just.hpp"
#include "__let.hpp"
#include "__schedulers.hpp"
#include "__utility.hpp"

namespace STDEXEC {
  namespace __detail {
    //! Constant function object always returning `__val_`.
    template <class _Ty, class = __demangle_t<__decay_t<_Ty>>>
    struct __always {
      _Ty __val_;

      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()() noexcept(__nothrow_move_constructible<_Ty>) -> _Ty {
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
      constexpr auto
        operator()(_Scheduler&& __sched, _Sender&& __sndr) const -> __well_formed_sender auto {
        return __make_sexpr<starts_on_t>(
          static_cast<_Scheduler&&>(__sched), static_cast<_Sender&&>(__sndr));
      }

      template <__decay_copyable _Sender>
      static constexpr auto transform_sender(set_value_t, _Sender&& __sndr, __ignore) {
        auto& [__tag, __sched, __child] = __sndr;
        return let_value(
          continues_on(just(), __sched),
          __detail::__always{STDEXEC::__forward_like<_Sender>(__child)});
      }

      template <class _Sender>
      static constexpr auto transform_sender(set_value_t, _Sender&&, __ignore) {
        return __not_a_sender<_SENDER_TYPE_IS_NOT_DECAY_COPYABLE_, _WITH_PRETTY_SENDER_<_Sender>>{};
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
      template <class... _Env>
      static constexpr auto __mk_env2(_Scheduler __sch, _Env&&... __env) {
        return env(__mk_sch_env(__sch, __env...), static_cast<_Env&&>(__env)...);
      }

      template <class... _Env>
      using __env2_t = decltype(__mk_env2(__declval<_Scheduler>(), __declval<_Env>()...));

      // Query for completion scheduler
      template <class _SetTag, class... _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto query(get_completion_scheduler_t<_SetTag>, _Env&&...) const noexcept
        -> _Scheduler
        requires(__completes_inline<_SetTag, env_of_t<_Child>, __env2_t<_Env>...>)
      {
        // If child completes inline, then starts_on completes on its scheduler
        return __sched_;
      }

      // Query for completion scheduler - delegates to child's env with augmented environment
      template <class _SetTag, class... _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto query(get_completion_scheduler_t<_SetTag> __query, _Env&&... __env)
        const noexcept
        -> __call_result_t<get_completion_scheduler_t<_SetTag>, env_of_t<_Child>, __env2_t<_Env>...>
        requires(!__completes_inline<_SetTag, env_of_t<_Child>, __env2_t<_Env>...>)
      {
        // If child doesn't complete inline, delegate to child's completion scheduler
        return __query(__attr_, __mk_env2(__sched_, static_cast<_Env&&>(__env))...);
      }

      // Query for completion domain - calculate type from child's env with augmented environment
      template <class _SetTag, class... _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto query(get_completion_domain_t<_SetTag>, _Env&&...) const noexcept
        -> __call_result_t<get_completion_domain_t<_SetTag>, env_of_t<_Child>, __env2_t<_Env>...> {
        return {};
      }

      _Scheduler __sched_;
      env_of_t<_Child> __attr_;
    };

    static constexpr auto get_attrs =
      []<class _Data, class _Child>(__ignore, const _Data& __data, const _Child& __child) noexcept
      -> __attrs<_Data, _Child> {
      return __attrs<_Data, _Child>{__data, STDEXEC::get_env(__child)};
    };

    template <class _Sender, class... _Env>
    static consteval auto get_completion_signatures() {
      using __sndr_t =
        __detail::__transform_sender_result_t<starts_on_t, set_value_t, _Sender, env<>>;
      return STDEXEC::get_completion_signatures<__sndr_t, _Env...>();
    };
  };
} // namespace STDEXEC
