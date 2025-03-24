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

// Pull in the reference implementation of P2300:
#include <stdexec/execution.hpp>
#include "./algorithms/then.hpp"

#include <cstdio>

///////////////////////////////////////////////////////////////////////////////
// Example code:
auto main() -> int {
  auto x = then(stdexec::just(42), [](int i) {
    std::printf("Got: %d\n", i);
    return i;
  });

  // prints:
  //   Got: 42
  auto [a] = stdexec::sync_wait(std::move(x)).value();
  (void) a;
}
