/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *                         Copyright (c) 2025 Robert Leahy. All rights reserved.
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

#include <asioexec/asio_config.hpp>
#include <asioexec/executor_with_default.hpp>
#include <type_traits>
#include <utility>

namespace asioexec {

  template <typename CompletionToken, typename IoObject>
  using as_default_on_t =
    std::remove_cvref_t<IoObject>::template rebind_executor<executor_with_default<
      std::remove_cvref_t<decltype(std::declval<IoObject&>().get_executor())>,
      CompletionToken
    >>::other;

  namespace detail::as_default_on {

    template <typename CompletionToken>
    struct t {
      template <typename IoObject>
      constexpr asioexec::as_default_on_t<CompletionToken, IoObject>
        operator()(IoObject&& io) const {
        return asioexec::as_default_on_t<CompletionToken, IoObject>((IoObject&&) io);
      }
    };

  } // namespace detail::as_default_on

  template <typename CompletionToken>
  inline constexpr detail::as_default_on::t<CompletionToken> as_default_on;

} // namespace asioexec
