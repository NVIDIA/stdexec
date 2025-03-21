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
#include "__cpo.hpp"
#include "__env.hpp"
#include "__receivers.hpp"
#include "__schedulers.hpp"
#include "__utility.hpp"

namespace stdexec {
  namespace __inln {
    struct __schedule_t { };

    struct __scheduler {
      using __t = __scheduler;
      using __id = __scheduler;

      template <class _Tag = __schedule_t>
      STDEXEC_ATTRIBUTE((host, device)) STDEXEC_MEMFN_DECL(auto schedule)(this __scheduler) {
        return __make_sexpr<_Tag>();
      }

      [[nodiscard]]
      auto query(get_forward_progress_guarantee_t) const noexcept -> forward_progress_guarantee {
        return forward_progress_guarantee::weakly_parallel;
      }

      auto operator==(const __scheduler&) const noexcept -> bool = default;
    };

    struct __env {
      static constexpr auto query(__is_scheduler_affine_t) noexcept -> bool {
        return true;
      }

      [[nodiscard]]
      constexpr auto query(get_completion_scheduler_t<set_value_t>) const noexcept -> __scheduler {
        return {};
      }
    };
  } // namespace __inln

  template <>
  struct __sexpr_impl<__inln::__schedule_t> : __sexpr_defaults {
    static constexpr auto get_attrs = //
      [](__ignore) noexcept {
        return __inln::__env();
      };

    static constexpr auto get_completion_signatures = //
      [](__ignore, __ignore = {}) noexcept -> completion_signatures<set_value_t()> {
      return {};
    };

    static constexpr auto start = //
      []<class _Receiver>(__ignore, _Receiver& __rcvr) noexcept -> void {
      stdexec::set_value(static_cast<_Receiver&&>(__rcvr));
    };
  };

  static_assert(__is_scheduler_affine<schedule_result_t<__inln::__scheduler>>);
} // namespace stdexec
