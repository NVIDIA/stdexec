/*
 * Copyright (c) 2023 Maikel Nadolski
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

#include <stdexec/execution.hpp>

#if !_STD_NO_COROUTINES_ && !STDEXEC_NVHPC()
#include <exec/task.hpp>

#include <exec/single_thread_context.hpp>

#include <thread>
#include <iostream>

std::mutex cout_mutex{};

void say_hello() {
  std::scoped_lock lock{cout_mutex};
  std::cout << std::this_thread::get_id() << ": Hello, world!" << std::endl;
}

using namespace exec;
using namespace stdexec;

task<void> hello_world2(scheduler auto scheduler1, scheduler auto scheduler2, auto id1, auto id2);

void check_thread_id(std::thread::id expected) {
  if (std::this_thread::get_id() != expected)
    throw std::runtime_error("unexpected thread id");
  std::cout << std::this_thread::get_id() << ": check_thread_id(" << expected << ")" << std::endl;
}

task<void> hello_world(auto scheduler1, auto scheduler2, auto id1, auto id2) {
  check_thread_id(id1);
  co_await schedule(scheduler2);
  check_thread_id(id1);
  co_await (schedule(scheduler1) | then([&] { check_thread_id(id1); }));
  co_await (schedule(scheduler2) | then([&] { check_thread_id(id2); }));
  check_thread_id(id1);
  co_await complete_inline(scheduler2);
  check_thread_id(id2);
  co_await hello_world2(scheduler1, scheduler2, id1, id2);
  check_thread_id(id2);
}

task<void> hello_world_init(auto scheduler1, auto scheduler2, auto id1, auto id2) {
  co_await complete_inline(scheduler1);
  check_thread_id(id1);
  co_await hello_world(scheduler1, scheduler2, id1, id2);
}

task<void> hello_world2(scheduler auto scheduler1, scheduler auto scheduler2, auto id1, auto id2) {
  check_thread_id(id2);
  co_await schedule(scheduler1);
  check_thread_id(id2);
  co_await complete_inline(scheduler1);
  check_thread_id(id1);
}

int main() {
  single_thread_context context1;
  single_thread_context context2;
  scheduler auto scheduler1 = context1.get_scheduler();
  scheduler auto scheduler2 = context2.get_scheduler();
  auto id1 = context1.get_thread_id();
  auto id2 = context2.get_thread_id();
  sync_wait(
    hello_world_init(scheduler1, scheduler2, id1, id2) | then([&] { check_thread_id(id1); }));
}
#else
int main() {
}
#endif