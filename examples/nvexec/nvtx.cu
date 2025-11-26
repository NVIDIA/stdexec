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

#include <nvexec/stream_context.cuh>
#include <nvexec/nvtx.cuh>
#include <cstdio>

// clang-format off

namespace ex = stdexec;

auto main() -> int {
  nvexec::stream_context stream_ctx{};
  auto snd = ex::schedule(stream_ctx.get_scheduler())
           | nvexec::nvtx::push("manual push")
           | nvexec::nvtx::scoped("scope", ex::then([] { printf("hello!\n"); })
                                         | ex::then([] { printf("bye!\n"); }))
           | nvexec::nvtx::pop();
  stdexec::sync_wait(std::move(snd));
}
