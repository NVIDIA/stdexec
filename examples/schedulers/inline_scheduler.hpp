/*
 * Copyright (c) NVIDIA
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

#include <execution.hpp>
#include <type_traits>
#include <exception>

namespace example {
  // A simple scheduler that executes its continuation inline, on the
  // thread of the caller of start().
  struct inline_scheduler {
    template <class R_>
      struct __op {
        using R = std::__t<R_>;
        [[no_unique_address]] R rec_;
        friend void tag_invoke(std::execution::start_t, __op& op) noexcept try {
          std::execution::set_value((R&&) op.rec_);
        } catch(...) {
          std::execution::set_error((R&&) op.rec_, std::current_exception());
        }
      };

    struct __sender {
      template <template <class...> class Tuple,
                template <class...> class Variant>
        using value_types = Variant<Tuple<>>;
      template <template <class...> class Variant>
        using error_types = Variant<std::exception_ptr>;
      static constexpr bool sends_done = false;

      template <std::execution::receiver_of R>
        friend auto tag_invoke(std::execution::connect_t, __sender, R&& rec)
          noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<R>, R>)
          -> __op<std::__x<std::remove_cvref_t<R>>> {
          return {(R&&) rec};
        }
    };

    friend __sender tag_invoke(std::execution::schedule_t, const inline_scheduler&) noexcept {
      return {};
    }

    bool operator==(const inline_scheduler&) const noexcept = default;
  };
}
