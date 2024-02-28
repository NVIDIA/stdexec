/*
 * Copyright (c) 2023 Maikel Nadolski
 * Copyright (c) 2023 NVIDIA Corporation
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

namespace exec {
  class safe_file_descriptor {
    int __fd_{-1};
   public:
    safe_file_descriptor() = default;

    explicit safe_file_descriptor(int __fd) noexcept;

    safe_file_descriptor(const safe_file_descriptor&) = delete;
    auto operator=(const safe_file_descriptor&) -> safe_file_descriptor& = delete;

    safe_file_descriptor(safe_file_descriptor&& __other) noexcept;

    auto operator=(safe_file_descriptor&& __other) noexcept -> safe_file_descriptor&;

    ~safe_file_descriptor();

    void reset(int __fd = -1) noexcept;

    explicit operator bool() const noexcept;

    operator int() const noexcept;

    [[nodiscard]]
    auto native_handle() const noexcept -> int;
  };
} // namespace exec

#include "__detail/safe_file_descriptor.hpp"