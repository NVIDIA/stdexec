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

#if __has_include(<bit>)
#include <bit>
#if __cpp_lib_bit_cast >= 201806L
#define STDEXEC_HAS_BIT_CAST
#endif
#endif

#include <cstring>

namespace exec {

#if defined(STDEXEC_HAS_BIT_CAST)
  using std::bit_cast;
#else
  template <class _To, class _From>
  [[nodiscard]] constexpr _To bit_cast(const _From& __from) noexcept {
    static_assert(sizeof(_From) == sizeof(_To), "bit_cast requires sizeof(_From) == sizeof(_To)");
    static_assert(
      std::is_trivially_copyable_v<_From>, "bit_cast requires _From to be trivially copyable");
    static_assert(
      std::is_trivially_copyable_v<_To>, "bit_cast requires _To to be trivially copyable");
    _To __to;
    std::memcpy(&__to, &__from, sizeof(_From));
    return __to;
  }
#endif
}