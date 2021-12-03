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

#include "./schedulers/static_thread_pool.hpp"

using namespace std::execution;
using std::this_thread::sync_wait;

int main() {
  example::static_thread_pool ctx{8};
  scheduler auto sch = ctx.get_scheduler();                               // 1

  sender auto begin = schedule(sch);                                      // 2
  sender auto hi_again = then(begin, [] {                                 // 3
    std::cout << "Hello world! Have an int.\n";                           // 3
    return 13;                                                            // 3
  });                                                                     // 3

  sender auto add_42 = then(hi_again, [](int arg) { return arg + 42; });  // 4

  auto [i] = sync_wait(std::move(add_42)).value();                        // 5
  std::cout << "Result: " << i << std::endl;
}
