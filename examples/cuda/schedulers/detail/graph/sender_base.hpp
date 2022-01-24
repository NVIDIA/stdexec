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
#pragma once

#include <execution.hpp>
#include <schedulers/detail/graph/concepts.hpp>

namespace example::cuda::graph::detail
{

template <class Sender>
concept expose_requirements =
     graph_sender<Sender>
  && requires(const Sender &sndr)
{
  sndr.storage_requirements();
};

template <class Derived, class Base>
struct sender_base_t
{
  Base sender_;

  template <std::__decays_to<Derived> Self, class Receiver>
  friend auto tag_invoke(std::execution::connect_t, Self&& self, Receiver&& rcvr)
  noexcept(noexcept(((Self&&) self).connect((Receiver&&) rcvr)))
  -> decltype(((Self&&) self).connect((Receiver&&) rcvr)) {
    return ((Self&&) self).connect((Receiver&&) rcvr);
  }

  template <std::__none_of<std::execution::connect_t> Tag, class... Ts>
  requires std::__callable<Tag, const Base&, Ts...> friend constexpr decltype(auto)
  tag_invoke(Tag tag, const Derived &s, Ts &&...ts) noexcept
  {
    return tag(s.sender_, std::forward<Ts>(ts)...);
  }

  friend constexpr auto tag_invoke(cuda::storage_requirements_t,
                                   const Derived &s) noexcept
  {
    if constexpr(expose_requirements<Derived>)
    {
      return s.storage_requirements();
    }

    return cuda::storage_requirements(s.sender_);
  }

  static constexpr bool is_cuda_graph_api = true;
};

} // namespace graph
