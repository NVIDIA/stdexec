/*
 * Copyright (c) 2022 Lucian Radu Teodorescu
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

#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

namespace ex = STDEXEC;

namespace {

  template <class T>
  struct test_allocator {
    using value_type = T;

    explicit test_allocator(size_t* bytes) noexcept
      : bytes_(bytes) {
    }

    template <class Other>
    test_allocator(const test_allocator<Other>& other) noexcept
      : bytes_(other.bytes_) {
    }

    T* allocate(std::size_t n) {
      if (bytes_ != nullptr)
        *bytes_ += n * sizeof(T);
      return static_cast<T*>(::operator new(n * sizeof(T)));
    }

    void deallocate(T* p, std::size_t n) {
      if (bytes_ != nullptr)
        *bytes_ -= n * sizeof(T);
      ::operator delete(p, n * sizeof(T));
    }

    bool operator==(const test_allocator&) const = default;

   private:
    template <class>
    friend struct test_allocator;

    size_t* bytes_ = nullptr;
  };

} // namespace
