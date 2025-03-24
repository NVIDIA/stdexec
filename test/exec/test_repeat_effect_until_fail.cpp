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

#include <stdexec/execution.hpp>
#include <exec/repeat_effect_until.hpp>

namespace ex = stdexec;

struct not_a_bool { };

auto main() -> int {
  ex::sender auto snd = ex::just(not_a_bool()) | exec::repeat_effect_until();
  // build error: _INVALID_ARGUMENT_TO_REPEAT_EFFECT_UNTIL_
  stdexec::sync_wait(std::move(snd));
}
