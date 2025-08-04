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

#include "../../stdexec/__detail/__config.hpp"

#if __has_include(<bit>)
#  include <bit>
#  if __cpp_lib_bit_cast >= 2018'06L
#    define STDEXEC_HAS_BIT_CAST
#  endif
#endif

#include <cstring>

namespace exec {

  template <class _Ty>
  concept __trivially_copyable = STDEXEC_IS_TRIVIALLY_COPYABLE(_Ty);

#if defined(STDEXEC_HAS_BIT_CAST)
  using std::bit_cast;
#else
  template <__trivially_copyable _To, __trivially_copyable _From>
    requires(sizeof(_To) == sizeof(_From))
  [[nodiscard]]
  constexpr _To bit_cast(const _From& __from) noexcept {
#  if STDEXEC_HAS_BUILTIN(__builtin_bit_cast) || (STDEXEC_MSVC() && STDEXEC_MSVC_VERSION >= 19'26)
    return __builtin_bit_cast(_To, __from);
#  else
    _To __to;
    std::memcpy(&__to, &__from, sizeof(_From));
    return __to;
#  endif
  }
#endif
} // namespace exec
