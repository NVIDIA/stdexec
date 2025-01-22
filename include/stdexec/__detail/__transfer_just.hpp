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

#include "__execution_fwd.hpp" // IWYU pragma: keep

// include these after __execution_fwd.hpp
#include "__basic_sender.hpp"
#include "__concepts.hpp"
#include "__continues_on.hpp"
#include "__domain.hpp"
#include "__env.hpp"
#include "__just.hpp"
#include "__meta.hpp"
#include "__schedule_from.hpp"
#include "__schedulers.hpp"
#include "__sender_introspection.hpp"
#include "__tag_invoke.hpp"
#include "__transform_sender.hpp"
#include "__tuple.hpp"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.transfer_just]
  namespace __transfer_just {
    // This is a helper for finding legacy cusutomizations of transfer_just.
    inline auto __transfer_just_tag_invoke() {
      return []<class... _Ts>(_Ts&&... __ts) -> tag_invoke_result_t<transfer_just_t, _Ts...> {
        return tag_invoke(transfer_just, static_cast<_Ts&&>(__ts)...);
      };
    }

    template <class _Env>
    auto __make_transform_fn(const _Env&) {
      return [&]<class _Scheduler, class... _Values>(_Scheduler&& __sched, _Values&&... __vals) {
        return continues_on(
          just(static_cast<_Values&&>(__vals)...), static_cast<_Scheduler&&>(__sched));
      };
    }

    template <class _Env>
    auto __transform_sender_fn(const _Env& __env) {
      return [&]<class _Data>(__ignore, _Data&& __data) {
        return __data.apply(__make_transform_fn(__env), static_cast<_Data&&>(__data));
      };
    }

    struct __legacy_customization_fn {
      template <class _Data>
      auto operator()(_Data&& __data) const
        -> decltype(__data.apply(__transfer_just_tag_invoke(), static_cast<_Data&&>(__data))) {
        return __data.apply(__transfer_just_tag_invoke(), static_cast<_Data&&>(__data));
      }
    };

    struct transfer_just_t {
      using _Data = __0;
      using __legacy_customizations_t = //
        __types<__legacy_customization_fn(_Data)>;

      template <scheduler _Scheduler, __movable_value... _Values>
      auto
        operator()(_Scheduler&& __sched, _Values&&... __vals) const -> __well_formed_sender auto {
        auto __domain = query_or(get_domain, __sched, default_domain());
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<transfer_just_t>(
            __tuple{static_cast<_Scheduler&&>(__sched), static_cast<_Values&&>(__vals)...}));
      }

      template <class _Sender, class _Env>
      static auto transform_sender(_Sender&& __sndr, const _Env& __env) {
        return __sexpr_apply(static_cast<_Sender&&>(__sndr), __transform_sender_fn(__env));
      }
    };

    inline auto __make_env_fn() noexcept {
      return []<class _Scheduler>(const _Scheduler& __sched, const auto&...) noexcept {
        using _Env = __t<__schfr::__environ<__id<_Scheduler>>>;
        return _Env{__sched};
      };
    }

    struct __transfer_just_impl : __sexpr_defaults {
      static constexpr auto get_attrs = //
        []<class _Data>(const _Data& __data) noexcept {
          return __data.apply(__make_env_fn(), __data);
        };

      static constexpr auto get_completion_signatures = //
        []<class _Sender>(_Sender&&) noexcept           //
        -> __completion_signatures_of_t<                //
          transform_sender_result_t<default_domain, _Sender, empty_env>> {
      };
    };
  } // namespace __transfer_just

  using __transfer_just::transfer_just_t;
  inline constexpr transfer_just_t transfer_just{};

  template <>
  struct __sexpr_impl<transfer_just_t> : __transfer_just::__transfer_just_impl { };
} // namespace stdexec

STDEXEC_PRAGMA_POP()
