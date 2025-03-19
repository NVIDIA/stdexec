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

#include "../memory_mapped_region.hpp"

#include <utility>
#include <sys/mman.h>

namespace exec {
  inline memory_mapped_region::memory_mapped_region(void* __ptr, std::size_t __size) noexcept
    : __ptr_(__ptr)
    , __size_(__size) {
    if (__ptr_ == MAP_FAILED) {
      __ptr_ = nullptr;
    }
  }

  inline memory_mapped_region::~memory_mapped_region() {
    if (__ptr_) {
      ::munmap(__ptr_, __size_);
    }
  }

  inline memory_mapped_region::memory_mapped_region(memory_mapped_region&& __other) noexcept
    : __ptr_(std::exchange(__other.__ptr_, nullptr))
    , __size_(std::exchange(__other.__size_, 0)) {
  }

  inline auto memory_mapped_region::operator=(memory_mapped_region&& __other) noexcept
    -> memory_mapped_region& {
    if (this != &__other) {
      if (__ptr_) {
        ::munmap(__ptr_, __size_);
      }
      __ptr_ = std::exchange(__other.__ptr_, nullptr);
      __size_ = std::exchange(__other.__size_, 0);
    }
    return *this;
  }

  inline memory_mapped_region::operator bool() const noexcept {
    return __ptr_ != nullptr;
  }

  inline auto memory_mapped_region::data() const noexcept -> void* {
    return __ptr_;
  }

  inline auto memory_mapped_region::size() const noexcept -> std::size_t {
    return __size_;
  }
} // namespace exec