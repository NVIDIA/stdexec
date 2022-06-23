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
#include <exec/sequence.hpp>

#include <exec/static_thread_pool.hpp>
#include <test/test_common/type_helpers.hpp>

#include <cstdio>

///////////////////////////////////////////////////////////////////////////////
// Example code:
using namespace stdexec;
using namespace _P0TBD::execution;
using std::this_thread::sync_wait;

int main() {
  auto print_each = iotas(1, 10)
  | then_each([](int v){ printf("%d, ", v); })
  | ignore_all()
  | ex::then([](){ printf("\n"); });

  check_val_types<type_array<type_array<>>>(print_each);
  check_err_types<type_array<std::exception_ptr>>(print_each);
  check_sends_stopped<false>(print_each);

  stdexec::sync_wait(std::move(print_each));
}
