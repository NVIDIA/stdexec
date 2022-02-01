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

#include <schedulers/detail/helpers.hpp>
#include <schedulers/detail/storage.hpp>
#include <schedulers/detail/tuple.hpp>

#include <execution.hpp>
#include <vector>
#include <span>

#include <type_traits>

namespace example::cuda::distributed::detail
{

struct context_t
{
  context_t() = delete;
  explicit context_t(
    int rank,
    int size,
    cudaStream_t stream)
    : rank_(rank)
    , size_(size)
    , stream_(stream)
  {}

  [[nodiscard]] int size() const
  {
    return size_;
  }

  [[nodiscard]] int rank() const
  {
    return rank_;
  }

  [[nodiscard]] cudaStream_t stream() const
  {
    return stream_;
  }


  // MPI
  int rank_{0};
  int size_{1};

  // CUDA
  cudaStream_t stream_{};
};


template <class BaseEnvT>
class env_t
{
  BaseEnvT base_;

  const context_t &context_;
  std::byte *storage_{};


public:
  explicit env_t(BaseEnvT base,
                 const context_t &context,
                 std::byte *storage)
      : base_(std::forward<BaseEnvT>(base))
      , context_(context)
      , storage_(storage)
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

  [[nodiscard]] cudaStream_t stream() const
  {
    return context_.stream_;
  }

  [[nodiscard]] const context_t& context() const
  {
    return context_;
  }

  template <std::__none_of<cuda::get_storage_t> Tag, class... As>
  requires std::__callable<Tag, const BaseEnvT &, As...> friend auto
  tag_invoke(Tag tag, const env_t &self, As &&...as) noexcept
    -> std::__call_result_t<Tag, const BaseEnvT &, As...>
  {
    return (std::forward<Tag>(tag))(self.base_, std::forward<As>(as)...);
  }
};


}
