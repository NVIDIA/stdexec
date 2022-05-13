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
concept expose_requirements = requires(const Sender &sndr)
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

  template <std::execution::__sender_queries::__sender_query _Tag, class... _As>
    requires std::__callable<_Tag, const Base&, _As...>
  friend auto tag_invoke(_Tag __tag, const Derived& __self, _As&&... __as)
    noexcept(std::__nothrow_callable<_Tag, const Base&, _As...>)
    -> std::__call_result_if_t<std::execution::__sender_queries::__sender_query<_Tag>, _Tag, const Base&, _As...> {
    return ((_Tag&&) __tag)(__self.sender_, (_As&&) __as...);
  }

  template <class _CPO>
  friend auto tag_invoke(std::execution::get_completion_scheduler_t<_CPO>,
                         const sender_base_t &self) noexcept
  {
    return std::execution::get_completion_scheduler<_CPO>(self.sender_);
  }
};

template <std::execution::sender Sender, class Derived, class Base, class Env>
struct operation_state_base_t
{
  Base op_state_;
  cudaGraphNode_t node_;

  friend auto tag_invoke(get_predecessors_t, const operation_state_base_t & self) noexcept
  {
    return std::span(&self.node_, &self.node_ + 1);
  }

  friend constexpr auto tag_invoke(cuda::storage_requirements_t,
                                   const Derived &os) noexcept
  {
    if constexpr(expose_requirements<Derived>)
    {
      return os.storage_requirements();
    }

    auto predecessor_requirements = cuda::storage_requirements(os.op_state_);
    using storage_type = storage_type_for_t<Sender, Env>;
    using self_requirement_t = cuda::static_storage_from<storage_type>;

    std::size_t alignment = std::max(self_requirement_t::alignment,
                                     predecessor_requirements.alignment);

    std::size_t size = std::max(self_requirement_t::size,
                                predecessor_requirements.size);

    return storage_description_t{ alignment, size };
  }
};

} // namespace graph
