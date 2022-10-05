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
#pragma once

#include <execution.hpp>
#include <type_traits>

#include "common.cuh"

namespace example::cuda::stream::start_detached {

struct detached_receiver_t : receiver_base_t {
  friend void tag_invoke(std::execution::set_value_t, detached_receiver_t&&, auto&&...) noexcept {}

  [[noreturn]]
  friend void tag_invoke(std::execution::set_error_t, detached_receiver_t&&, auto&&) noexcept {
    std::terminate();
  }
  friend void tag_invoke(std::execution::set_stopped_t, detached_receiver_t&&) noexcept {}
  friend stdexec::__empty_env tag_invoke(std::execution::get_env_t, const detached_receiver_t&) noexcept {
    return {};
  }
};

}
