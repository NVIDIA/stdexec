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

namespace example::cuda::graph::detail
{

struct id_t
{
  unsigned int id_{};

  [[nodiscard]] __host__ __device__ bool is_first() const { return id_ == 0; }
};

struct thread_id_t : id_t
{};
struct block_id_t : id_t
{};

struct consumer_t
{
  std::byte *storage_;

  template <class SrcT, class DstT = std::decay_t<SrcT>>
  __host__ __device__ void operator()(thread_id_t tid,
                                      block_id_t bid,
                                      SrcT &&result) const
  {
    if (tid.is_first() && bid.is_first())
    {
      *reinterpret_cast<DstT *>(storage_) = result;
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

template <class F, class... Ts>
__host__ __device__ void invoke(F f, cuda::variant<Ts...> &storage)
{
  if constexpr (cuda::variant<Ts...>::empty)
  {
    f();
  }
  else
  {
    cuda::apply(
      [f](auto &&tpl) {
        cuda::apply(f, std::forward<decltype(tpl)>(tpl));
      }, storage);
  }
}

} // namespace example::cuda::graph::detail
