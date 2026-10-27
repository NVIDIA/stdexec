/*
 * Copyright (c) 2026 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <stdexec/execution.hpp>

namespace test_common
{
  namespace ex = STDEXEC;

  struct non_completion_query_t : ex::__query<non_completion_query_t>, ex::forwarding_query_t
  {
    using ex::__query<non_completion_query_t>::operator();
  };

  inline constexpr non_completion_query_t non_completion_query{};

  struct non_completion_query_env
  {
    constexpr auto query(non_completion_query_t, int value) const noexcept -> int
    {
      return value + 1;
    }
  };

  struct non_completion_query_sender
  {
    using sender_concept = ex::sender_tag;
    using completion_signatures = ex::completion_signatures<ex::set_value_t()>;

    template <ex::receiver Receiver>
    struct operation
    {
      using operation_state_concept = ex::operation_state_tag;

      void start() & noexcept
      {
        ex::set_value(static_cast<Receiver&&>(receiver));
      }

      Receiver receiver;
    };

    template <ex::receiver Receiver>
    auto connect(Receiver&& receiver) const -> operation<std::remove_cvref_t<Receiver>>
    {
      return {static_cast<Receiver&&>(receiver)};
    }

    auto get_env() const noexcept -> non_completion_query_env
    {
      return {};
    }
  };
}  // namespace test_common
