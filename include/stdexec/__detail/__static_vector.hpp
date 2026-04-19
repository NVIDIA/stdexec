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
  template <class _Tp, std::size_t _Capacity>
  struct __static_vector;

  namespace __detail
  {
    struct __static_vector_base
    {
      template <class... _What, class _Tp, std::size_t _Capacity>
      [[nodiscard]]
      friend constexpr auto operator+(__mexception<_What...>,                            //
                                      __static_vector<_Tp, _Capacity> const &) noexcept  //
        -> __mexception<_What...>
      {
        return {};
      }

      template <class... _What, class _Tp, std::size_t _Capacity>
      [[nodiscard]]
      friend constexpr auto operator+(__static_vector<_Tp, _Capacity> const &,  //
                                      __mexception<_What...>) noexcept          //
        -> __mexception<_What...>
      {
        return {};
      }

      template <class _Ty, std::size_t _Capacity0, std::size_t _Capacity1>
      [[nodiscard]]
      friend constexpr auto operator+(__static_vector<_Ty, _Capacity0> const &__lhs,  //
                                      __static_vector<_Ty, _Capacity1> const &__rhs)
        noexcept(__nothrow_copy_constructible<_Ty>) -> __static_vector<_Ty, _Capacity0 + _Capacity1>
      {
        __static_vector<_Ty, _Capacity0 + _Capacity1> __result;
        std::copy(__lhs.begin(), __lhs.end(), __result.begin());
        std::copy(__rhs.begin(), __rhs.end(), __result.begin() + __lhs.size());
        __result.resize(__lhs.size() + __rhs.size());
        return __result;
      }
    };
  }  // namespace __detail

  template <class _Tp, std::size_t _Capacity>
  struct __static_vector : __detail::__static_vector_base
  {
    using value_type     = _Tp;
    using iterator       = value_type *;
    using const_iterator = value_type const *;

    __static_vector() = default;

    constexpr __static_vector(std::initializer_list<value_type> __init)
      noexcept(__nothrow_copy_constructible<value_type>)
    {
      auto const __count = (std::min) (__init.size(), _Capacity);
      auto const __end   = std::ranges::copy_n(__init.begin(), __count, __data_).out;
      __size_            = __end - __data_;
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
      return _Capacity;
    }

    constexpr void resize(std::size_t __new_size) noexcept
    {
      __size_ = __new_size;
    }

    constexpr auto erase(const_iterator __first, const_iterator __last) noexcept -> iterator
    {
      std::move(const_cast<iterator>(__last), end(), const_cast<iterator>(__first));
      resize(size() - (__last - __first));
      return end();
    }

    std::size_t __size_ = 0;
    value_type  __data_[_Capacity];
  };

  // Specialization of __static_vector for zero capacity that doesn't require default
  // constructibility of _Tp.
  template <class _Tp>
  struct __static_vector<_Tp, 0> : __detail::__static_vector_base
  {
    using value_type     = _Tp;
    using iterator       = value_type *;
    using const_iterator = value_type const *;

    __static_vector() = default;

    [[nodiscard]]
    constexpr auto begin() noexcept -> iterator
    {
      return nullptr;
    }

    [[nodiscard]]
    constexpr auto begin() const noexcept -> const_iterator
    {
      return nullptr;
    }

    [[nodiscard]]
    constexpr auto end() noexcept -> iterator
    {
      return nullptr;
    }

    [[nodiscard]]
    constexpr auto end() const noexcept -> const_iterator
    {
      return nullptr;
    }

    [[nodiscard]]
    static constexpr auto size() noexcept -> std::size_t
    {
      return 0;
    }

    [[nodiscard]]
    static constexpr auto capacity() noexcept -> std::size_t
    {
      return 0;
    }
  };

  template <class _First, __same_as<_First>... _Rest>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
  __static_vector(_First, _Rest...) -> __static_vector<_First, 1 + sizeof...(_Rest)>;

  template <class _First, __same_as<_First>... _Rest>
  constexpr auto __make_static_vector(_First __first, _Rest... __rest) noexcept
  {
    return __static_vector<_First, 1 + sizeof...(_Rest)>{__first, __rest...};
  }
}  // namespace STDEXEC
