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
#include <schedulers/detail/distributed/concepts.hpp>
#include <schedulers/detail/variant.hpp>

namespace example::cuda::distributed::detail
{

template <class Sender>
concept expose_requirements =
     distributed_sender<Sender>
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

  friend constexpr auto tag_invoke(cuda::storage_requirements_t,
                                   const Derived &s) noexcept
  {
    if constexpr(expose_requirements<Derived>)
    {
      return s.storage_requirements();
    }

    auto predecessor_requirements = cuda::storage_requirements(s.sender_);
    using value_t = value_of_t<Derived>;
    using self_requirement_t = cuda::static_storage_from<value_t>;
    cuda::storage_description_t self_requirement{};

    if (!std::is_same_v<value_t, cuda::variant<cuda::tuple<>>>)
    {
      self_requirement.alignment = self_requirement_t::alignment;
      self_requirement.size = self_requirement_t::size;
    }

    const std::size_t alignment = std::max(self_requirement.alignment,
                                           predecessor_requirements.alignment);

    const std::size_t size = std::max(self_requirement.size,
                                      predecessor_requirements.size);

    return cuda::storage_description_t{alignment, size};
  }

  static constexpr bool is_cuda_distributed_api = true;
};

} // namespace graph
