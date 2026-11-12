/*
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#include <cstdio>
#include <stdexec/execution.hpp>

namespace ex = stdexec;

int main()
{
  auto sched  = ex::get_parallel_scheduler();
  auto square = [](int i)
  {
    return i * i;
  };

  // Build a lazy pipeline: three squares, computed in parallel.
  auto work = ex::when_all(ex::on(sched, ex::just(0) | ex::then(square)),
                           ex::on(sched, ex::just(1) | ex::then(square)),
                           ex::on(sched, ex::just(2) | ex::then(square)));

  // Launch the work and wait for the result.
  auto [i, j, k] = ex::sync_wait(std::move(work)).value();

  std::printf("%d %d %d\n", i, j, k);  // prints "0 1 4"
}
