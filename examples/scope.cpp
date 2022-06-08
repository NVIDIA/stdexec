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

// Pull in the reference implementation of P2300:
#include <execution.hpp>

#include "./schedulers/static_thread_pool.hpp"

#include <cstdio>

///////////////////////////////////////////////////////////////////////////////
// Example code:
using namespace std::execution;
using std::this_thread::sync_wait;

int main() {
  example::static_thread_pool ctx{1};
  async_scope scope;

  scheduler auto sch = ctx.get_scheduler();                               // 1

  sender auto begin = schedule(sch);                                      // 2

  sender auto printVoid = then(begin, 
    []()noexcept { printf("void\n"); });                                  // 3

  sender auto printEmpty = then(on(sch, scope.empty()), 
    []()noexcept{ printf("scope is empty\n"); });                         // 4


  printf("\n"
    "spawn void\n"
    "==========\n");

  scope.spawn(printVoid);                                                 // 5

  sync_wait(printEmpty);


  printf("\n"
    "spawn void and 42\n"
    "=================\n");

  sender auto fortyTwo = then(begin, []()noexcept {return 42;});          // 6

  scope.spawn(printVoid);                                                 // 7

  sender auto fortyTwoFuture = scope.spawn_future(fortyTwo);              // 8

  sender auto printFortyTwo = then(std::move(fortyTwoFuture), 
    [](int fortyTwo)noexcept{ printf("%d\n", fortyTwo); });               // 9

  sender auto allDone = then(
    when_all(printEmpty, std::move(printFortyTwo)), 
    [](auto&&...)noexcept{printf("\nall done\n");});                      // 10

  sync_wait(std::move(allDone));
}
