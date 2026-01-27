/*
 * Copyright (c) 2026 NVIDIA Corporation
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

#include <exec/repeat_until.hpp>
#include <stdexec/execution.hpp>

namespace ex = STDEXEC;

struct not_a_bool { };

auto main() -> int {
  ex::sender auto snd = ex::just(not_a_bool()) | exec::repeat_until();
  // build error: _EXPECTING_A_SENDER_OF_ONE_VALUE_THAT_IS_CONVERTIBLE_TO_BOOL_
  STDEXEC::sync_wait(std::move(snd));
}
