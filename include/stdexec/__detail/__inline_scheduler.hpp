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

#include "__completion_behavior.hpp"
#include "__queries.hpp"
#include "__receivers.hpp"
#include "__schedulers.hpp"

namespace STDEXEC {
  struct __inline_attrs {
    STDEXEC_ATTRIBUTE(nodiscard, host, device)
    constexpr auto query(get_completion_behavior_t<set_value_t>) const noexcept {
      return completion_behavior::inline_completion;
    }
    constexpr auto operator==(const __inline_attrs&) const noexcept -> bool = default;
  };

  struct inline_scheduler : __inline_attrs {
   private:
    template <class _Receiver>
    struct __opstate {
      using operation_state_concept = operation_state_t;

      STDEXEC_ATTRIBUTE(host, device)
      constexpr void start() noexcept {
        STDEXEC::set_value(static_cast<_Receiver&&>(__rcvr_));
      }

      _Receiver __rcvr_;
    };

    struct __sender {
      using sender_concept = sender_t;
      using completion_signatures = STDEXEC::completion_signatures<set_value_t()>;

      template <class _Receiver>
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      static constexpr auto connect(_Receiver __rcvr) noexcept -> __opstate<_Receiver> {
        return {static_cast<_Receiver&&>(__rcvr)};
      }

      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      static constexpr auto get_env() noexcept -> __inline_attrs {
        return {};
      }
    };

   public:
    using scheduler_concept = scheduler_t;

    constexpr inline_scheduler() = default;

    STDEXEC_ATTRIBUTE(nodiscard, host, device)
    static constexpr auto schedule() noexcept -> __sender {
      return {};
    }

    constexpr auto operator==(const inline_scheduler&) const noexcept -> bool = default;
  };

  static_assert(__is_scheduler_affine<schedule_result_t<inline_scheduler>>);
} // namespace STDEXEC
