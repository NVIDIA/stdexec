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
#include "__env.hpp"
#include "__meta.hpp"
#include "__schedule_from.hpp"
#include "__schedulers.hpp"
#include "__sender_introspection.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__senders_core.hpp"
#include "__tag_invoke.hpp"
#include "__transform_sender.hpp"
#include "__type_traits.hpp"

#include <utility>

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.continues_on]
  namespace __continues_on {
    using __schfr::__environ;

    template <class _Env>
    using __scheduler_t = __result_of<get_completion_scheduler<set_value_t>, _Env>;

    template <class _Sender>
    using __lowered_t = //
      __result_of<schedule_from, __scheduler_t<__data_of<_Sender>>, __child_of<_Sender>>;

    struct continues_on_t {
      template <sender _Sender, scheduler _Scheduler>
      auto operator()(_Sender&& __sndr, _Scheduler&& __sched) const -> __well_formed_sender auto {
        auto __domain = __get_early_domain(__sndr);
        using _Env = __t<__environ<__id<__decay_t<_Scheduler>>>>;
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<continues_on_t>(
            _Env{{static_cast<_Scheduler&&>(__sched)}}, static_cast<_Sender&&>(__sndr)));
      }

      template <scheduler _Scheduler>
      STDEXEC_ATTRIBUTE((always_inline)) auto operator()(_Scheduler&& __sched) const
        -> __binder_back<continues_on_t, __decay_t<_Scheduler>> {
        return {{static_cast<_Scheduler&&>(__sched)}, {}, {}};
      }

      //////////////////////////////////////////////////////////////////////////////////////////////
      using _Env = __0;
      using _Sender = __1;
      using __legacy_customizations_t = //
        __types<
          tag_invoke_t(
            continues_on_t,
            get_completion_scheduler_t<set_value_t>(get_env_t(const _Sender&)),
            _Sender,
            get_completion_scheduler_t<set_value_t>(_Env)),
          tag_invoke_t(continues_on_t, _Sender, get_completion_scheduler_t<set_value_t>(_Env))>;

      template <class _Env>
      static auto __transform_sender_fn(const _Env&) {
        return [&]<class _Data, class _Child>(__ignore, _Data&& __data, _Child&& __child) {
          auto __sched = get_completion_scheduler<set_value_t>(__data);
          return schedule_from(std::move(__sched), static_cast<_Child&&>(__child));
        };
      }

      template <class _Sender, class _Env>
      static auto transform_sender(_Sender&& __sndr, const _Env& __env) {
        return __sexpr_apply(static_cast<_Sender&&>(__sndr), __transform_sender_fn(__env));
      }
    };

    struct __continues_on_impl : __sexpr_defaults {
      static constexpr auto get_attrs = //
        []<class _Data, class _Child>(const _Data& __data, const _Child& __child) noexcept
        -> decltype(auto) {
        return __env::__join(__data, stdexec::get_env(__child));
      };

      static constexpr auto get_completion_signatures = //
        []<class _Sender>(_Sender&&) noexcept           //
        -> __completion_signatures_of_t<                //
          transform_sender_result_t<default_domain, _Sender, empty_env>> {
      };
    };
  } // namespace __continues_on

  using __continues_on::continues_on_t;
  inline constexpr continues_on_t continues_on{};

  using transfer_t = continues_on_t;
  inline constexpr continues_on_t transfer{};

  using continue_on_t = continues_on_t;
  inline constexpr continues_on_t continue_on{};

  template <>
  struct __sexpr_impl<continues_on_t> : __continues_on::__continues_on_impl { };
} // namespace stdexec
