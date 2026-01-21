/*
 * Copyright (c) 2025 Ian Petersen
 * Copyright (c) 2025 NVIDIA Corporation
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

#include <stdexec/execution.hpp>

#include <utility>

namespace {
  struct null_token {
    struct assoc {
      constexpr explicit operator bool() const noexcept {
        return true;
      }

      constexpr assoc try_associate() const noexcept {
        return {};
      }
    };

    template <STDEXEC::sender Sender>
    constexpr Sender&& wrap(Sender&& sndr) const noexcept {
      return std::forward<Sender>(sndr);
    }

    constexpr assoc try_associate() const noexcept {
      return {};
    }
  };
} // namespace
