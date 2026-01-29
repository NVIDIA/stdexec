
/*
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

#include <exec/repeat_n.hpp>
#include <stdexec/execution.hpp>

namespace ex = STDEXEC;

auto main() -> int {
  ex::sender auto snd = ex::just(42) | exec::repeat_n(10);
  // build error: _THE_INPUT_SENDER_MUST_HAVE_VOID_VALUE_COMPLETION_
  STDEXEC::sync_wait(std::move(snd));
}
