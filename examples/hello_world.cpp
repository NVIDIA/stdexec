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

#include "exec/static_thread_pool.hpp"

using namespace stdexec;
using stdexec::sync_wait;

struct sink {
  using receiver_concept = receiver_t;
  static void set_value(auto&&...) noexcept {}
  static void set_error(auto&&) noexcept {}
  static void set_stopped() noexcept {}
  static empty_env get_env() noexcept { return {}; }
};

struct operation {
  int data;
  using receiver_concept = receiver_t;
  void set_value(auto&&...) noexcept {}
  void set_error(auto&&) noexcept {}
  void set_stopped() noexcept {}
  empty_env get_env() noexcept { return {}; }
  void start() noexcept {}
};

int main() {
  operation op = {42};
  auto op2 = connect(just(42), &op);
  start(op2);

  auto op3 = connect(just(42), static_cast<sink*>(nullptr));
  start(op3);

                                                                         //





  // exec::static_thread_pool ctx{8};
  // scheduler auto sch = ctx.get_scheduler();                              // 1
  //                                                                        //
  // sender auto begin = schedule(sch);                                     // 2
  // sender auto hi_again = then(                                           // 3
  //   begin,                                                               // 3
  //   [] {                                                                 // 3
  //     std::cout << "Hello world! Have an int.\n";                        // 3
  //     return 13;                                                         // 3
  //   });                                                                  // 3
  //                                                                        //
  // sender auto add_42 = then(hi_again, [](int arg) { return arg + 42; }); // 4

  // auto [i] = sync_wait(std::move(add_42)).value();                       // 5
  // std::cout << "Result: " << i << std::endl;

  // // Sync_wait provides a run_loop scheduler
  // std::tuple<run_loop::__scheduler> t = sync_wait(get_scheduler()).value();
  // (void) t;

  // auto y = let_value(get_scheduler(), [](auto sched) {
  //   return on(sched, then(just(), [] {
  //               std::cout << "from run_loop\n";
  //               return 42;
  //             }));
  // });
  // sync_wait(std::move(y));

  // sync_wait(when_all(just(42), get_scheduler(), get_stop_token()));
}
