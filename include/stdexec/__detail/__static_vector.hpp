/*
 * Copyright (c) 2026 NVIDIA Corporation
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

#include "__execution_fwd.hpp"

#include "__concepts.hpp"

#include <algorithm>
#include <cstddef>
#include <initializer_list>

namespace STDEXEC
{
  template <class _Tp, std::size_t _Size>
  struct __static_vector
  {
    using value_type     = _Tp;
    using iterator       = value_type *;
    using const_iterator = value_type const *;

    __static_vector() = default;

    constexpr __static_vector(std::initializer_list<value_type> __init)
      noexcept(__nothrow_copy_constructible<value_type>)
    {
      auto const __end = std::copy_n(__init.begin(), (std::min) (__init.size(), _Size), __data_);
      __size_          = __end - __data_;
    }

    [[nodiscard]]
    constexpr auto operator[](std::size_t __i) noexcept -> value_type &
    {
      return __data_[__i];
    }

    [[nodiscard]]
    constexpr auto operator[](std::size_t __i) const noexcept -> value_type const &
    {
      return __data_[__i];
    }

    [[nodiscard]]
    constexpr auto begin() noexcept -> iterator
    {
      return __data_;
    }

    [[nodiscard]]
    constexpr auto begin() const noexcept -> const_iterator
    {
      return __data_;
    }

    [[nodiscard]]
    constexpr auto end() noexcept -> iterator
    {
      return __data_ + __size_;
    }

    [[nodiscard]]
    constexpr auto end() const noexcept -> const_iterator
    {
      return __data_ + __size_;
    }

    [[nodiscard]]
    constexpr auto size() const noexcept -> std::size_t
    {
      return __size_;
    }

    [[nodiscard]]
    static constexpr auto capacity() noexcept -> std::size_t
    {
      return _Size;
    }

    constexpr void resize(std::size_t __new_size) noexcept
    {
      __size_ = __new_size;
    }

    constexpr auto erase(const_iterator __first, const_iterator __last) noexcept -> iterator
    {
      std::move(__last, end(), const_cast<iterator>(__first));
      resize(size() - (__last - __first));
      return end();
    }

    std::size_t __size_ = 0;
    value_type  __data_[_Size];
  };

  template <class _Tp, std::size_t _Size0, std::size_t _Size1>
  [[nodiscard]]
  constexpr auto __concat(__static_vector<_Tp, _Size0> const &__lhs,  //
                          __static_vector<_Tp, _Size1> const &__rhs)
    noexcept(__nothrow_copy_constructible<_Tp>) -> __static_vector<_Tp, _Size0 + _Size1>
  {
    __static_vector<_Tp, _Size0 + _Size1> __result;
    std::copy(__lhs.begin(), __lhs.end(), __result.begin());
    std::copy(__rhs.begin(), __rhs.end(), __result.begin() + __lhs.size());
    __result.resize(__lhs.size() + __rhs.size());
    return __result;
  }
}  // namespace STDEXEC
