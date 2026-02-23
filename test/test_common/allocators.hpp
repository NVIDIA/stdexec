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

#include <new>

namespace ex = STDEXEC;

namespace
{

  template <class T>
  struct test_allocator
  {
    using value_type = T;

    explicit test_allocator(size_t* bytes) noexcept
      : bytes_(bytes)
    {}

    template <class Other>
    test_allocator(test_allocator<Other> const & other) noexcept
      : bytes_(other.bytes_)
    {}

    T* allocate(std::size_t n)
    {
      if (bytes_ != nullptr)
        *bytes_ += n * sizeof(T);
      return static_cast<T*>(::operator new(n * sizeof(T)));
    }

    void deallocate(T* p, std::size_t n)
    {
      if (bytes_ != nullptr)
        *bytes_ -= n * sizeof(T);
#if defined(__cpp_sized_deallocation) && __cpp_sized_deallocation >= 2013'09L
      ::operator delete(p, n * sizeof(T));
#else
      ::operator delete(p);
#endif
    }

    bool operator==(test_allocator const &) const = default;

   private:
    template <class>
    friend struct test_allocator;

    size_t* bytes_ = nullptr;
  };

}  // namespace
