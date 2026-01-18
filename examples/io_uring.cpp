/*
 * Copyright (c) 2023 Maikel Nadolski
 * Copyright (c) 2023 NVIDIA Corporation
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

#include "exec/linux/io_uring_context.hpp"

#include "exec/finally.hpp"
#include "exec/when_any.hpp"

#include "stdexec/execution.hpp"

#include <chrono>
#include <thread>

#include <iostream>

auto main() -> int {
  exec::io_uring_context context;
  exec::io_uring_context context2;
  std::thread io_thread{[&] { context.run_until_stopped(); }};
  std::thread io_thread2{[&] { context2.run_until_stopped(); }};
  auto scheduler = context.get_scheduler();
  auto scheduler2 = context2.get_scheduler();
  using namespace std::chrono_literals;

  stdexec::sync_wait(
    exec::when_any(
      exec::schedule_after(scheduler, 1s) | stdexec::then([] { std::cout << "Hello, 1!\n"; }),
      exec::schedule_after(scheduler2, 2s) | stdexec::then([] { std::cout << "Hello, 2!\n"; })
        | stdexec::upon_stopped([] { std::cout << "Hello, 2, stopped.\n"; })));

  stdexec::sync_wait(
    exec::when_any(
      exec::schedule_after(scheduler, 1s) | stdexec::then([] { std::cout << "Hello, 1!\n"; })
        | stdexec::upon_stopped([] { std::cout << "Hello, 1, stopped.\n"; }),
      exec::schedule_after(scheduler2, 500ms) | stdexec::then([] { std::cout << "Hello, 2!\n"; })
        | stdexec::upon_stopped([] { std::cout << "Hello, 2, stopped.\n"; })));

  stdexec::sync_wait(
    stdexec::when_all(
      stdexec::schedule(scheduler) | stdexec::then([] { std::cout << "Hello, 0!\n"; }),
      exec::schedule_after(scheduler, 1s) | stdexec::then([] { std::cout << "Hello, 1!\n"; }),
      exec::schedule_after(scheduler2, 2s) | stdexec::then([] { std::cout << "Hello, 2!\n"; }),
      exec::schedule_after(scheduler, 3s) | stdexec::then([] { std::cout << "Stop it!\n"; }),
      exec::finally(exec::schedule_after(scheduler2, 4s), stdexec::just() | stdexec::then([&] {
                                                            context.request_stop();
                                                          })),
      exec::finally(exec::schedule_after(scheduler, 4s), stdexec::just() | stdexec::then([&] {
                                                           context2.request_stop();
                                                         })),
      exec::schedule_after(scheduler, 10s) | stdexec::then([] { std::cout << "Hello, world!\n"; })
        | stdexec::upon_stopped([] { std::cout << "Hello, stopped.\n"; }),
      exec::schedule_after(scheduler2, 10s) | stdexec::then([] { std::cout << "Hello, world!\n"; })
        | stdexec::upon_stopped([] { std::cout << "Hello, stopped.\n"; })));
  io_thread.join();
  io_thread2.join();

  stdexec::sync_wait(
    stdexec::schedule(scheduler)
    | stdexec::then([] { std::cout << "This should not print, because the context is stopped.\n"; })
    | stdexec::upon_stopped([] { std::cout << "The context is stopped!\n"; }));

  stdexec::sync_wait(
    stdexec::schedule(scheduler2)
    | stdexec::then([] { std::cout << "This should not print, because the context is stopped.\n"; })
    | stdexec::upon_stopped([] { std::cout << "The context is stopped!\n"; }));

  context.reset();
  io_thread = std::thread{[&] { context.run_until_stopped(); }};

  while (!context.is_running())
    ;
  stdexec::sync_wait(
    exec::when_any(
      exec::schedule_after(scheduler, 1s) | stdexec::then([] { std::cout << "Hello, 1!\n"; }),
      exec::schedule_after(scheduler, 500ms) | stdexec::then([] { std::cout << "Hello, 2!\n"; })));

  auto time_point = std::chrono::steady_clock::now() + 1s;
  stdexec::sync_wait(exec::schedule_at(scheduler, time_point) | stdexec::then([] {
                       std::cout << "Hello, schedule_at!\n";
                     }));

  static_assert(exec::timed_scheduler<exec::io_uring_scheduler>);

  context.request_stop();
  io_thread.join();
}
