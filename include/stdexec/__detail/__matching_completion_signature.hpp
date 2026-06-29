/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *                         Copyright (c) 2026 Robert Leahy. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "__completion_signatures.hpp"

#include <cstddef>
#include <type_traits>

namespace STDEXEC
{

  namespace __matching_completion_signature
  {

    template <typename>
    struct __canonicalize;

    template <typename _Tag, typename... _Args>
    struct __canonicalize<_Tag(_Args...)>
    {
      using type = _Tag(_Args&&...);
    };

    template <typename _Signature>
    using __canonicalize_t = typename __canonicalize<_Signature>::type;

    template <typename _Needle, typename _Signature>
    inline constexpr bool __is_canonical_match =
      std::is_same_v<__canonicalize_t<_Needle>, __canonicalize_t<_Signature>>;

    template <typename _Needle, typename _Signatures>
    struct __canonical_match_count;

    template <typename _Needle, typename... _Signatures>
    struct __canonical_match_count<_Needle, completion_signatures<_Signatures...>>
      : std::integral_constant<std::size_t, (__is_canonical_match<_Needle, _Signatures> + ... + 0)>
    {};

    template <typename _Needle, typename... _Signatures>
    struct __find_canonical_match;

    template <typename _Needle>
    struct __find_canonical_match<_Needle>
    {};

    template <typename _Needle, typename _Signature, typename... _Rest>
    struct __find_canonical_match<_Needle, _Signature, _Rest...>
      : std::conditional_t<__is_canonical_match<_Needle, _Signature>,
                           std::type_identity<_Signature>,
                           __find_canonical_match<_Needle, _Rest...>>
    {};

    template <typename _Signatures,
              typename _Signature,
              std::size_t _Matches = __canonical_match_count<_Signature, _Signatures>::value>
    struct __matching_completion_signature
    {};

    template <typename _Signature, typename... _Signatures>
    struct __matching_completion_signature<completion_signatures<_Signatures...>, _Signature, 1>
      : __find_canonical_match<_Signature, _Signatures...>
    {};

  }  // namespace __matching_completion_signature

  template <typename _Signatures, typename _Tag, typename... _Args>
  using matching_completion_signature_t =
    typename __matching_completion_signature::__matching_completion_signature<_Signatures,
                                                                              _Tag(_Args...)>::type;

  template <typename _Signatures, typename _Tag, typename... _Args>
  inline constexpr bool has_matching_completion_signature_v = requires {
    typename matching_completion_signature_t<_Signatures, _Tag, _Args...>;
  };

  template <typename _Signatures, typename _Tag, typename... _Args>
  struct matching_completion_signature
  {
    using type = matching_completion_signature_t<_Signatures, _Tag, _Args...>;
  };

  template <typename _Signatures, typename _Tag, typename... _Args>
  struct has_matching_completion_signature
    : std::bool_constant<has_matching_completion_signature_v<_Signatures, _Tag, _Args...>>
  {};

}  // namespace STDEXEC
