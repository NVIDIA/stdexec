/*
 * Copyright (c) 2026 NVIDIA Corporation
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
// clang-format Language: Cpp

#pragma once

#include "../../stdexec/execution.hpp"
#include "../../stdexec/functional.hpp"

#include "let_xxx.cuh"

#include "common.cuh"

namespace nv::execution::_strm
{
  template <>
  struct transform_sender_for<STDEXEC::starts_on_t>
  {
    template <class Env, STDEXEC::scheduler Scheduler, STDEXEC::sender Sender>
    auto operator()(Env const &, STDEXEC::starts_on_t, Scheduler&& sched, Sender&& sndr) const
    {
      return STDEXEC::let_value(STDEXEC::schedule(sched),
                                STDEXEC::__always(static_cast<Sender&&>(sndr)));
    }
  };
}  // namespace nv::execution::_strm
