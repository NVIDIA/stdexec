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

#include <nvexec/stream_context.cuh>
#include <stdexec/execution.hpp>

#include <cstdio>

namespace ex = stdexec;

auto main() -> int {
  using nvexec::is_on_gpu;

  nvexec::stream_context stream_ctx{};
  ex::scheduler auto sch = stream_ctx.get_scheduler();

  auto bulk_fn = [](int lbl) {
    return [=](int i) {
      std::printf("B%d: i = %d\n", lbl, i);
    };
  };

  auto then_fn = [](int lbl) {
    return [=] {
      std::printf("T%d\n", lbl);
    };
  };

  auto fork = ex::schedule(sch) | ex::bulk(ex::par, 4, bulk_fn(0)) | ex::split();

  auto snd = ex::transfer_when_all(
               sch,
               fork | ex::bulk(ex::par, 4, bulk_fn(1)),
               fork | ex::then(then_fn(1)),
               fork | ex::bulk(ex::par, 4, bulk_fn(2)))
           | ex::then(then_fn(2));

  stdexec::sync_wait(std::move(snd));
}
