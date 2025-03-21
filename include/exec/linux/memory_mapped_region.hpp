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

#include <cstddef>

namespace exec {
  class memory_mapped_region {
    void* __ptr_{nullptr};
    std::size_t __size_{0};
   public:
    memory_mapped_region() = default;

    memory_mapped_region(void* __ptr, std::size_t __size) noexcept;

    ~memory_mapped_region();

    memory_mapped_region(const memory_mapped_region&) = delete;
    auto operator=(const memory_mapped_region&) -> memory_mapped_region& = delete;

    memory_mapped_region(memory_mapped_region&& __other) noexcept;

    auto operator=(memory_mapped_region&& __other) noexcept -> memory_mapped_region&;

    explicit operator bool() const noexcept;

    [[nodiscard]]
    auto data() const noexcept -> void*;

    [[nodiscard]]
    auto size() const noexcept -> std::size_t;
  };
} // namespace exec

#include "__detail/memory_mapped_region.hpp"