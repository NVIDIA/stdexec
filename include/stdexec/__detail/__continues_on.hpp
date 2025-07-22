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
#include "__meta.hpp"
#include "__schedule_from.hpp"
#include "__schedulers.hpp"
#include "__sender_introspection.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__senders_core.hpp"
#include "__transform_sender.hpp"

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.continues_on]
  namespace __continues_on {
    template <class _Env>
    using __scheduler_t = __result_of<get_completion_scheduler<set_value_t>, _Env>;

    template <class _Sender>
    using __lowered_t =
      __result_of<schedule_from, __scheduler_t<__data_of<_Sender>>, __child_of<_Sender>>;

    struct continues_on_t {
      template <sender _Sender, scheduler _Scheduler>
      auto operator()(_Sender&& __sndr, _Scheduler __sched) const -> __well_formed_sender auto {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<continues_on_t>(
            static_cast<_Scheduler&&>(__sched), static_cast<_Sender&&>(__sndr)));
      }

      template <scheduler _Scheduler>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(_Scheduler __sched) const -> __binder_back<continues_on_t, _Scheduler> {
        return {{static_cast<_Scheduler&&>(__sched)}, {}, {}};
      }

      static auto __transform_sender_fn() {
        return [&]<class _Data, class _Child>(__ignore, _Data&& __data, _Child&& __child) {
          return schedule_from(static_cast<_Data&&>(__data), static_cast<_Child&&>(__child));
        };
      }

      template <class _Sender, class _Env>
      static auto transform_sender(_Sender&& __sndr, const _Env&) {
        return __sexpr_apply(static_cast<_Sender&&>(__sndr), __transform_sender_fn());
      }
    };

    struct __continues_on_impl : __sexpr_defaults {
      static constexpr auto get_attrs = []<class _Data, class _Child>(
                                          const _Data& __data,
                                          const _Child& __child) noexcept -> decltype(auto) {
        using __domain_t = __detail::__early_domain_of_t<_Child, __none_such>;
        return __env::__join(
          __sched_attrs{std::cref(__data), __domain_t{}}, stdexec::get_env(__child));
      };

      static constexpr auto get_completion_signatures = []<class _Sender>(_Sender&&) noexcept
        -> __completion_signatures_of_t<transform_sender_result_t<default_domain, _Sender, env<>>> {
        return {};
      };
    };
  } // namespace __continues_on

  using __continues_on::continues_on_t;
  inline constexpr continues_on_t continues_on{};

  // Backward compatibility:
  using transfer_t = continues_on_t;
  inline constexpr continues_on_t transfer{};

  using continue_on_t = continues_on_t;
  inline constexpr continues_on_t continue_on{};

  namespace v2 {
    using continue_on_t
      [[deprecated("use stdexec::continues_on_t instead")]] = stdexec::continues_on_t;
    [[deprecated("use stdexec::continues_on instead")]]
    inline constexpr stdexec::continues_on_t const & continue_on = stdexec::continues_on;
  } // namespace v2

  template <>
  struct __sexpr_impl<continues_on_t> : __continues_on::__continues_on_impl { };
} // namespace stdexec
