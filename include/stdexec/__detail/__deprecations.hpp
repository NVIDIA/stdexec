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

#include "__execution_fwd.hpp"

namespace STDEXEC
{
  using execute_may_block_caller_t [[deprecated]] = __execute_may_block_caller_t;

  [[deprecated]]
  inline constexpr __execute_may_block_caller_t const & execute_may_block_caller =
    __execute_may_block_caller;

  [[deprecated("use STDEXEC::__completion_behavior::__weakest instead")]]
  inline constexpr auto const & min = __completion_behavior::__weakest;

  using empty_env [[deprecated("STDEXEC::empty_env is now spelled STDEXEC::env<>")]] = env<>;

  using dependent_completions [[deprecated("use dependent_sender_error instead of "
                                           "dependent_completions")]] = dependent_sender_error;

  using in_place_stop_token [[deprecated("in_place_stop_token has been renamed "
                                         "inplace_stop_token")]] = inplace_stop_token;

  using in_place_stop_source [[deprecated("in_place_stop_token has been renamed "
                                          "inplace_stop_source")]] = inplace_stop_source;

  template <class _Fun>
  using in_place_stop_callback
    [[deprecated("in_place_stop_callback has been renamed "
                 "inplace_stop_callback")]] = inplace_stop_callback<_Fun>;

  using start_on_t [[deprecated("start_on_t has been renamed starts_on_t")]] = starts_on_t;
  [[deprecated("start_on has been renamed starts_on")]]
  inline constexpr starts_on_t const & start_on = starts_on;

  using transfer_t [[deprecated("transfer_t has been renamed continues_on_t")]] = continues_on_t;
  [[deprecated("transfer has been renamed continues_on")]]
  inline constexpr continues_on_t const & transfer = continues_on;

  using transfer_just_t [[deprecated]] = __transfer_just_t;
  [[deprecated]]
  inline constexpr __transfer_just_t const & transfer_just = __transfer_just;

  [[deprecated("read has been renamed to read_env")]]
  inline constexpr __read_env_t const & read = read_env;

  // Moved to namespace experimental::execution:
  using split_t
    [[deprecated("Include <exec/split.hpp> and use exec::split_t instead")]] = exec::split_t;

  using ensure_started_t [[deprecated("Include <exec/ensure_started.hpp> and use "
                                      "exec::ensure_started_t instead")]] = exec::ensure_started_t;

  using start_detached_t [[deprecated("Include <exec/start_detached.hpp> and use "
                                      "exec::start_detached_t instead")]] = exec::start_detached_t;

  using execute_t [[deprecated]] = exec::__execute_t;

  [[deprecated("Include <exec/split.hpp> and use exec::split instead")]]
  inline constexpr exec::split_t const & split = exec::split;

  [[deprecated("Include <exec/ensure_started.hpp> and use exec::ensure_started instead")]]
  inline constexpr exec::ensure_started_t const & ensure_started = exec::ensure_started;

  [[deprecated("Include <exec/start_detached.hpp> and use exec::start_detached instead")]]
  inline constexpr exec::start_detached_t const & start_detached = exec::start_detached;

  [[deprecated]]
  inline constexpr exec::__execute_t const & execute = exec::__execute;
}  // namespace STDEXEC
