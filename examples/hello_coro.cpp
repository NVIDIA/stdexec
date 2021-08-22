/*
 * Copyright (c) NVIDIA
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
#include <iostream>

// Pull in the reference implementation of P2300:
#include <execution.hpp>

#include "./task.hpp"

task<int> async_answer() {
  co_return 42;
}

int main() {
  // Awaitables are implicitly senders:
  auto [i] = std::this_thread::sync_wait(async_answer()).value();
  std::cout << "The answer is " << i << '\n';
}
