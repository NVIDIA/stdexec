/*
 * Copyright (c) 2021-2024 NVIDIA Corporation
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

#include <stdexec/execution.hpp>

#include <exec/single_thread_context.hpp>
#include <nvexec/stream_context.cuh>

namespace ex = STDEXEC;

auto main() -> int
{
  nvexec::stream_context      stream_ctx{};
  exec::single_thread_context thread_ctx{};

  auto sndr = ex::when_all(ex::schedule(stream_ctx.get_scheduler()),
                           ex::schedule(thread_ctx.get_scheduler()))
            | ex::continues_on(thread_ctx.get_scheduler());

  // build error: ERROR: indeterminate domains: cannot pick an algorithm customization
  ex::sync_wait(std::move(sndr));
}
