/*
 * Copyright (c) 2022 NVIDIA Corporation
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

#include "../stdexec/execution.hpp" // IWYU pragma: keep

namespace exec {
  /////////////////////////////////////////////////////////////////////////////
  // A scoped version of [execution.senders.adaptors.on]
  using on_t [[deprecated("on_t has been moved to the STDEXEC:: namespace")]] = STDEXEC::on_t;

  [[deprecated("on has been moved to the STDEXEC:: namespace")]]
  inline constexpr STDEXEC::on_t const & on = STDEXEC::on;
} // namespace exec
