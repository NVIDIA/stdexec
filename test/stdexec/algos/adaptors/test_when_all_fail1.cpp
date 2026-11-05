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

namespace ex = STDEXEC;

template <class... Sigs>
struct a_sender_of
{
  using sender_concept        = ex::sender_tag;
  using completion_signatures = ex::completion_signatures<Sigs...>;

  struct operation
  {
    void start() & noexcept {}
  };

  template <class Receiver>
  auto connect(Receiver) const noexcept
  {
    return operation{};
  }
};

auto main() -> int
{
  using invalid_arg_t = a_sender_of<ex::set_value_t(), ex::set_value_t(int)>;
  ex::sender auto snd = ex::when_all(ex::just(42), invalid_arg_t());
  // build error: _THE_GIVEN_SENDER_CAN_COMPLETE_SUCCESSFULLY_IN_MORE_THAN_ONE_WAY_
  STDEXEC::sync_wait(std::move(snd));
}
