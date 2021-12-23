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

using namespace std::execution;

template <typed_sender S1, typed_sender S2>
task<int> async_answer(S1 s1, S2 s2) {
  // Senders are implicitly awaitable (int this coroutine type):
  co_await (S2&&) s2;
  co_return co_await (S1&&) s1;
}

template <typed_sender S1, typed_sender S2>
task<std::optional<int>> async_answer2(S1 s1, S2 s2) {
  co_return co_await done_as_optional(async_answer(s1, s2));
}

int main() try {
  // Awaitables are implicitly senders:
  auto [i] = std::this_thread::sync_wait(async_answer2(just(42), just())).value();
  std::cout << "The answer is " << i.value() << '\n';
} catch(std::exception & e) {
  std::cout << e.what() << '\n';
}
