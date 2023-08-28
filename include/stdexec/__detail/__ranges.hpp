/*
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

#include "__config.hpp"
#include "__type_traits.hpp"

#if 0 //STDEXEC_HAS_STD_RANGES()

#include <ranges>

namespace stdexec::ranges {
  using std::ranges::begin;
  using std::ranges::end;

  using std::ranges::range_value_t;
  using std::ranges::range_reference_t;
  using std::ranges::iterator_t;
  using std::ranges::sentinel_t;
}

#else

#include <iterator>

namespace stdexec::ranges {

  namespace __detail {
    void begin();
    void end();

    template <class _Ty>
    concept __has_member_begin = requires(_Ty&& __v) { ((_Ty&&) __v).begin(); };

    template <class _Ty>
    concept __has_free_begin = __has_member_begin<_Ty> || requires(_Ty&& __v) {
      begin(((_Ty&&) __v));
    };

    template <class _Ty>
    concept __has_member_end = requires(_Ty&& __v) { ((_Ty&&) __v).end(); };

    template <class _Ty>
    concept __has_free_end = __has_member_end<_Ty> || requires(_Ty&& __v) { end(((_Ty&&) __v)); };

    struct __begin_t {
      template <class _Range>
        requires __has_member_begin<_Range>
      auto operator()(_Range&& __rng) const noexcept(noexcept(((_Range&&) __rng).begin()))
        -> decltype(((_Range&&) __rng).begin()) {
        return ((_Range&&) __rng).begin();
      }

      template <class _Range>
        requires __has_free_begin<_Range>
      auto operator()(_Range&& __rng) const noexcept(noexcept(begin(((_Range&&) __rng))))
        -> decltype(begin(((_Range&&) __rng))) {
        return begin(((_Range&&) __rng));
      }
    };

    struct __end_t {
      template <class _Range>
        requires __has_member_end<_Range>
      auto operator()(_Range&& __rng) const noexcept(noexcept(((_Range&&) __rng).end()))
        -> decltype(((_Range&&) __rng).end()) {
        return ((_Range&&) __rng).end();
      }

      template <class _Range>
        requires __has_free_end<_Range>
      auto operator()(_Range&& __rng) const noexcept(noexcept(end(((_Range&&) __rng))))
        -> decltype(end(((_Range&&) __rng))) {
        return end(((_Range&&) __rng));
      }
    };
  }

  inline constexpr __detail::__begin_t begin{};
  inline constexpr __detail::__end_t end{};

  template <class _Range>
  using iterator_t = decltype(begin((__declval<_Range>())));

  template <class _Range>
  using sentinel_t = decltype(end((__declval<_Range>())));

  template <class _Range>
  using range_reference_t = decltype(*begin((__declval<_Range>())));

  template <class _Range>
  using range_value_t = typename std::iterator_traits<iterator_t<_Range>>::value_type;

}

#endif
