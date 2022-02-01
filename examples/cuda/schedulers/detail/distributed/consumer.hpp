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

#include <cstdint>

#include <schedulers/detail/tuple.hpp>
#include <schedulers/detail/variant.hpp>

namespace example::cuda::distributed::detail
{

struct id_t
{
  unsigned int id_{};

  [[nodiscard]] __host__ __device__ bool is_first() const { return id_ == 0; }
};

struct thread_id_t : id_t
{
  explicit __host__ __device__ thread_id_t(unsigned int id = 0) : id_t{id} {}
};
struct block_id_t : id_t
{
  explicit __host__ __device__ block_id_t(unsigned int id = 0) : id_t{id} {}
};

struct consumer_t
{
  std::byte *storage_;

  template <class... Ts>
  __host__ __device__ void operator()(thread_id_t tid,
                                      block_id_t bid,
                                      Ts&&... ts) const
  {
    if (tid.is_first() && bid.is_first())
    {
      using storage_t = cuda::variant<cuda::tuple<std::decay_t<Ts>...>>;
      new (storage_) storage_t{std::forward<Ts>(ts)...};
    }
  }

  __host__ __device__ void operator()(thread_id_t, block_id_t) const {}
};

struct sink_consumer_t
{
  template <class SrcT, class /* DstT */>
  __host__ __device__ void operator()(thread_id_t, block_id_t, SrcT &&) const
  {}

  __host__ __device__ void operator()(thread_id_t, block_id_t) const {}
};

} // namespace example::cuda::graph::detail
