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

#include "../safe_file_descriptor.hpp"

#include <unistd.h>
#include <utility>

namespace exec {
  inline safe_file_descriptor::safe_file_descriptor(int __fd) noexcept
    : __fd_(__fd) {
  }

  inline safe_file_descriptor::safe_file_descriptor(safe_file_descriptor&& __other) noexcept
    : __fd_(std::exchange(__other.__fd_, -1)) {
  }

  inline auto safe_file_descriptor::operator=(safe_file_descriptor&& __other) noexcept
    -> safe_file_descriptor& {
    if (this != &__other) {
      if (__fd_ != -1) {
        ::close(__fd_);
      }
      __fd_ = std::exchange(__other.__fd_, -1);
    }
    return *this;
  }

  inline safe_file_descriptor::~safe_file_descriptor() {
    reset();
  }

  inline void safe_file_descriptor::reset(int __fd) noexcept {
    if (__fd_ != -1) {
      ::close(__fd_);
    }
    __fd_ = __fd;
  }

  inline safe_file_descriptor::operator bool() const noexcept {
    return __fd_ != -1;
  }

  inline safe_file_descriptor::operator int() const noexcept {
    return __fd_;
  }

  inline auto safe_file_descriptor::native_handle() const noexcept -> int {
    return __fd_;
  }
} // namespace exec