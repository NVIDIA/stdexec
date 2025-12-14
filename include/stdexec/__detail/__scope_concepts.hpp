/*
 * Copyright (c) 2025 Ian Petersen
 * Copyright (c) 2025 NVIDIA Corporation
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

namespace STDEXEC {
  /////////////////////////////////////////////////////////////////////////////
  // [exec.scope.concepts]
  template <class _Assoc>
  concept scope_association = __std::movable<_Assoc> && __nothrow_move_constructible<_Assoc>
                           && __nothrow_move_assignable<_Assoc>
                           && __std::default_initializable<_Assoc> && requires(const _Assoc assoc) {
                                { static_cast<bool>(assoc) } noexcept;
                                { assoc.try_associate() } -> __std::same_as<_Assoc>;
                              };
} // namespace STDEXEC
