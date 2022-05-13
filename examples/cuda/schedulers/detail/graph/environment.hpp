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

#include <schedulers/detail/graph/concepts.hpp>
#include <schedulers/detail/graph/consumer.hpp>
#include <schedulers/detail/graph/graph_instance.hpp>
#include <schedulers/detail/graph/sender_base.hpp>
#include <schedulers/detail/helpers.hpp>
#include <schedulers/detail/storage.hpp>
#include <schedulers/detail/tuple.hpp>

#include <execution.hpp>
#include <span>

#include <type_traits>

namespace example::cuda::graph::detail
{


template <class BaseEnvT>
class env_t
{
  BaseEnvT base_;

  std::byte *storage_{};
  graph_info_t graph_{};


public:
  explicit env_t(BaseEnvT base,
                 std::byte *storage,
                 graph_info_t graph)
      : base_(std::forward<BaseEnvT>(base))
      , storage_(storage)
      , graph_(graph)
  {}

  [[nodiscard]] friend std::byte *tag_invoke(cuda::get_storage_t,
                                             const env_t &self) noexcept
  {
    if (self.storage_)
    {
      return self.storage_;
    }

    return cuda::get_storage(self.base_);
  }

  [[nodiscard]] friend graph_info_t tag_invoke(get_graph_t,
                                               const env_t &self) noexcept
  {
    return {self.graph_};
  }

  template <std::__none_of<cuda::get_storage_t, get_graph_t> Tag, class... As>
  requires std::__callable<Tag, const BaseEnvT&, As...>
  friend auto tag_invoke(Tag tag, const env_t& self, As&&... as) noexcept
  -> std::__call_result_t<Tag, const BaseEnvT&, As...> {
    return (std::forward<Tag>(tag))(self.base_, std::forward<As>(as)...);
  }
};

}
