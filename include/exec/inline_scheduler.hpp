/*
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#include "../stdexec/execution.hpp"
#include <type_traits>
#include <exception>

namespace exec {
  // A simple scheduler that executes its continuation inline, on the
  // thread of the caller of start().
  struct inline_scheduler {
    template <class R_>
      struct __op {
        using R = stdexec::__t<R_>;
        [[no_unique_address]] R rec_;
        friend void tag_invoke(stdexec::start_t, __op& op) noexcept try {
          stdexec::set_value((R&&) op.rec_);
        } catch(...) {
          stdexec::set_error((R&&) op.rec_, std::current_exception());
        }
      };

    struct __sender {
      using completion_signatures =
        stdexec::completion_signatures<
          stdexec::set_value_t(),
          stdexec::set_error_t(std::exception_ptr)>;

      template <class R>
        friend auto tag_invoke(stdexec::connect_t, __sender, R&& rec)
          noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<R>, R>)
          -> __op<stdexec::__x<std::remove_cvref_t<R>>> {
          return {(R&&) rec};
        }

      struct __attrs {
        friend inline_scheduler
        tag_invoke(stdexec::get_completion_scheduler_t<stdexec::set_value_t>, const __attrs&) noexcept {
          return {};
        }
      };

      friend __attrs tag_invoke(stdexec::get_attrs_t, const __sender&) noexcept {
        return {};
      }
    };

    friend __sender tag_invoke(stdexec::schedule_t, const inline_scheduler&) noexcept {
      return {};
    }

    friend stdexec::forward_progress_guarantee tag_invoke(
        stdexec::get_forward_progress_guarantee_t,
        const inline_scheduler&) noexcept {
      return stdexec::forward_progress_guarantee::weakly_parallel;
    }

    bool operator==(const inline_scheduler&) const noexcept = default;
  };
}
