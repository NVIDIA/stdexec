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

#include "__prologue.hpp"

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

#if STDEXEC_CLANG() && STDEXEC_CLANG_VERSION < 2000
    // Before Clang 20, CTAD with initializer lists was broken in constant evaluation.
    template <__same_as<_Tp>... _Values>
      requires(sizeof...(_Values) <= _Capacity)
    constexpr __static_vector(_Values... __values)
      : __size_(sizeof...(_Values))
      , __data_{__values...}
    {}
#else
    constexpr __static_vector(std::initializer_list<value_type> __init)
      : __static_vector(__init.begin(), __init.size(), __make_indices<_Capacity>())
    {}
#endif

    template <std::ranges::forward_range _Range>
    constexpr __static_vector(_Range &&__rng)
      : __static_vector(std::ranges::begin(__rng),
                        std::size_t(std::ranges::distance(__rng)),
                        __make_indices<_Capacity>())
    {}

    [[nodiscard]]
    constexpr auto operator[](std::size_t __i) noexcept -> value_type &
    {
      STDEXEC_ASSERT(__i < size());
      return __data_[__i];
    }

    [[nodiscard]]
    constexpr auto operator[](std::size_t __i) const noexcept -> value_type const &
    {
      STDEXEC_ASSERT(__i < size());
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
    constexpr auto empty() const noexcept -> bool
    {
      return __size_ == 0;
    }

    [[nodiscard]]
    static constexpr auto capacity() noexcept -> std::size_t
    {
      return _Capacity;
    }

    constexpr void resize(std::size_t __new_size) noexcept
    {
      STDEXEC_ASSERT(__new_size <= capacity());
      __size_ = __new_size;
    }

    constexpr auto erase(const_iterator __first, const_iterator __last) noexcept -> iterator
    {
      std::move(const_cast<iterator>(__last), end(), const_cast<iterator>(__first));
      resize(size() - (__last - __first));
      return end();
    }

    // These must be public so that __static_vector is a structual type.
    std::size_t __size_ = 0;
    value_type  __data_[_Capacity];

   private:
    template <class _Iterator, std::size_t... _Is>
    constexpr explicit __static_vector(_Iterator __it, std::size_t const __size, __indices<_Is...>)
      : __size_(std::min(__size, _Capacity))
      , __data_{(_Is < __size_ ? *__it++ : value_type{})...}
    {
      STDEXEC_ASSERT(__size <= _Capacity);
    }
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

    template <std::ranges::forward_range _Range>
      requires __std::constructible_from<value_type, std::ranges::range_reference_t<_Range>>
    constexpr __static_vector(_Range &&__rng)
    {
      STDEXEC_ASSERT(std::ranges::distance(__rng) == 0);
    }

    [[nodiscard]]
    constexpr auto operator[](std::size_t) noexcept -> value_type &
    {
      STDEXEC_ASSERT(false);
    }

    [[nodiscard]]
    constexpr auto operator[](std::size_t) const noexcept -> value_type const &
    {
      STDEXEC_ASSERT(false);
    }

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
    static constexpr auto empty() noexcept -> bool
    {
      return true;
    }

    [[nodiscard]]
    static constexpr auto capacity() noexcept -> std::size_t
    {
      return 0;
    }

    constexpr auto erase(const_iterator __first, const_iterator __last) noexcept -> iterator
    {
      STDEXEC_ASSERT(__first == __last);
      STDEXEC_ASSERT(__first == nullptr);

      return end();
    }
  };

  template <class _First, __same_as<_First>... _Rest>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
  __static_vector(_First, _Rest...) -> __static_vector<_First, 1 + sizeof...(_Rest)>;
}  // namespace STDEXEC

#include "__epilogue.hpp"
