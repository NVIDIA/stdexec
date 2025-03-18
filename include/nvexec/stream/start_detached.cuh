/*
 * Copyright (c) 2022 NVIDIA Corporation
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

// clang-format Language: Cpp

#pragma once

#include <exception>

#include "common.cuh"

namespace nvexec::_strm::_start_detached {

  struct detached_receiver_t : stream_receiver_base {
    template <class... _Args>
    void set_value(_Args&&...) noexcept {
    }

    template <class _Error>
    [[noreturn]]
    void set_error(_Error&&) noexcept {
      std::terminate();
    }

    void set_stopped() noexcept {
    }
  };

} // namespace nvexec::_strm::_start_detached
