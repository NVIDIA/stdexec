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
#include "./algorithms/retry.hpp"
#include <stdexec/execution.hpp>

#include <cstdio>

///////////////////////////////////////////////////////////////////////////////
// Example code:
struct fail_some {
  using sender_concept = stdexec::sender_t;
  using completion_signatures = stdexec::completion_signatures<
    stdexec::set_value_t(int),
    stdexec::set_error_t(std::exception_ptr)
  >;

  template <class R>
  struct op {
    R r_;

    void start() & noexcept {
      static int i = 0;
      if (++i < 3) {
        std::printf("fail!\n");
        stdexec::set_error(std::move(r_), std::exception_ptr{});
      } else {
        std::printf("success!\n");
        stdexec::set_value(std::move(r_), 42);
      }
    }
  };

  template <class R>
  auto connect(R r) const -> op<R> {
    return {std::move(r)};
  }
};

auto main() -> int {
  auto x = retry(fail_some{});
  // prints:
  //   fail!
  //   fail!
  //   success!
  auto [a] = stdexec::sync_wait(std::move(x)).value();
  (void) a;
}
