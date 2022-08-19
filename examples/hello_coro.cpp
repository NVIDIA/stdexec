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
#if defined(__GNUC__) && !defined(__clang__)
int main() { return 0; }
#else

#include <iostream>

// Pull in the reference implementation of P2300:
#include <execution.hpp>

#include "./task.hpp"

using namespace std::execution;

template <sender S1, sender S2>
task<int> async_answer(S1 s1, S2 s2) {
  // Senders are implicitly awaitable (in this coroutine type):
  co_await (S2&&) s2;
  co_return co_await (S1&&) s1;
}

template <sender S1, sender S2>
task<std::optional<int>> async_answer2(S1 s1, S2 s2) {
  co_return co_await stopped_as_optional(async_answer(s1, s2));
}

// tasks have an associated stop token
task<std::optional<std::in_place_stop_token>> async_stop_token() {
  co_return co_await stopped_as_optional(get_stop_token());
}

int main() try {
  // Awaitables are implicitly senders:
  auto [i] = _P2300::this_thread::sync_wait(async_answer2(just(42), just())).value();
  std::cout << "The answer is " << i.value() << '\n';
} catch(std::exception & e) {
  std::cout << e.what() << '\n';
}
#endif
