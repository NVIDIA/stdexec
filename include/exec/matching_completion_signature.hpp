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

#include "../stdexec/execution.hpp"

#include <cstddef>
#include <type_traits>

namespace experimental::execution {

namespace detail::matching_completion_signature {

template<typename>
struct canonicalize;

template<typename Tag, typename... Args>
struct canonicalize<Tag(Args...)> {
  using type = Tag(Args&&...);
};

template<typename Signature>
using canonicalize_t = typename canonicalize<Signature>::type;

template<typename Needle, typename Signature>
inline constexpr bool is_canonical_match =
  std::is_same_v<canonicalize_t<Needle>, canonicalize_t<Signature>>;

template<typename Needle, typename Signatures>
struct canonical_match_count;

template<typename Needle, typename... Signatures>
struct canonical_match_count<
  Needle,
  ::STDEXEC::completion_signatures<Signatures...>>
  : std::integral_constant<
      std::size_t,
      (is_canonical_match<Needle, Signatures> + ... + 0)>
{};

template<typename Needle, typename... Signatures>
struct find_canonical_match;

template<typename Needle>
struct find_canonical_match<Needle>
{};

template<typename Needle, typename Signature, typename... Rest>
struct find_canonical_match<Needle, Signature, Rest...>
  : std::conditional_t<
      is_canonical_match<Needle, Signature>,
      std::type_identity<Signature>,
      find_canonical_match<Needle, Rest...>>
{};

template<
  typename Signatures,
  typename Signature,
  std::size_t Matches =
    canonical_match_count<Signature, Signatures>::value>
struct matching_completion_signature
{};

template<typename Signature, typename... Signatures>
struct matching_completion_signature<
  ::STDEXEC::completion_signatures<Signatures...>,
  Signature,
  1>
  : find_canonical_match<Signature, Signatures...>
{};

}

template<typename Signatures, typename Tag, typename... Args>
using matching_completion_signature_t =
  typename detail::matching_completion_signature::
    matching_completion_signature<Signatures, Tag(Args...)>::type;

template<typename Signatures, typename Tag, typename... Args>
inline constexpr bool has_matching_completion_signature_v =
  requires {
    typename matching_completion_signature_t<Signatures, Tag, Args...>;
  };

template<typename Signatures, typename Tag, typename... Args>
struct matching_completion_signature {
  using type = matching_completion_signature_t<Signatures, Tag, Args...>;
};

template<typename Signatures, typename Tag, typename... Args>
struct has_matching_completion_signature
  : std::bool_constant<
      has_matching_completion_signature_v<Signatures, Tag, Args...>>
{
};

} // namespace exec
