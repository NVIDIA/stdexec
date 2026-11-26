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

#include "../stdexec/execution.hpp"

#include "completion_signatures.hpp"
#include "sender_for.hpp"

namespace experimental::execution
{
  struct unless_stop_requested_t;

  namespace _unless_stop_requested
  {
    using namespace STDEXEC;

    template <typename Env>
    concept _unstoppable_env = unstoppable_token<stop_token_of_t<Env>>;

    // unless_stop_requested(sndr): if stop has been requested when the operation starts, it
    // completes with set_stopped_t(), and otherwise it starts the child, whose completions are
    // the operation's. (If the receiver's stop token cannot be stopped, it starts the child.)
    struct _algo : __facade::__defaults
    {
      // (Whether the operation can complete with set_stopped_t() depends on the receiver's
      // stop token.)
      template <class Sender>
      static constexpr bool __depends_on_env = true;

      template <class State, class Children>
      static constexpr auto __start(State& __state, Children& __children) noexcept
      {
        if constexpr (_unstoppable_env<env_of_t<decltype(__state.__rcvr_)>>)
        {
          return __children.__start_all();
        }
        else
        {
          return __facade::__choose(
            STDEXEC::get_stop_token(STDEXEC::get_env(__state.__rcvr_)).stop_requested(),
            [&]() noexcept { return __facade::__send(__state.__rcvr_, set_stopped_t()); },
            [&]() noexcept { return __children.__start_all(); });
        }
      }
    };
  }  // namespace _unless_stop_requested

  struct unless_stop_requested_t : STDEXEC::sender_adaptor_closure<unless_stop_requested_t>
  {
    [[nodiscard]]
    constexpr auto operator()() const noexcept
    {
      return *this;
    }

    template <STDEXEC::sender Sender>
    [[nodiscard]]
    constexpr auto operator()(Sender&& sndr) const -> auto
    {
      return STDEXEC::__check_well_formed(STDEXEC::__facade::__sender{unless_stop_requested_t(),
                                                                      STDEXEC::__(),
                                                                      static_cast<Sender&&>(sndr)});
    }
  };

  inline constexpr unless_stop_requested_t unless_stop_requested{};
}  // namespace experimental::execution

namespace exec = experimental::execution;

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  template <>
  struct __facade::__algorithm<::exec::unless_stop_requested_t>
    : ::exec::_unless_stop_requested::_algo
  {};
}  // namespace STDEXEC
