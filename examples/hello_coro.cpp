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
#include <iostream>

// Pull in the reference implementation of P2300:
#include <stdexec/execution.hpp>

#if !STDEXEC_NO_STD_COROUTINES() && !STDEXEC_NVHPC()
#  include <exec/task.hpp>

using namespace stdexec;

template <sender S1, sender S2>
auto async_answer(S1 s1, S2 s2) -> exec::task<int> {
  // Senders are implicitly awaitable (in this coroutine type):
  co_await static_cast<S2&&>(s2);
  co_return co_await static_cast<S1&&>(s1);
}

template <sender S1, sender S2>
auto async_answer2(S1 s1, S2 s2) -> exec::task<std::optional<int>> {
  co_return co_await stopped_as_optional(async_answer(s1, s2));
}

// tasks have an associated stop token
auto async_stop_token() -> exec::task<std::optional<stdexec::inplace_stop_token>> {
  co_return co_await stopped_as_optional(get_stop_token());
}

auto main() -> int {
  STDEXEC_TRY {
    // Awaitables are implicitly senders:
    auto [i] = stdexec::sync_wait(async_answer2(just(42), just())).value();
    std::cout << "The answer is " << i.value() << '\n';
  }
  STDEXEC_CATCH(std::exception & e) {
    std::cerr << "error: " << e.what() << '\n';
  }
  STDEXEC_CATCH_ALL {
    std::cerr << "unknown error\n";
  }
}
#else
int main() {
}
#endif
