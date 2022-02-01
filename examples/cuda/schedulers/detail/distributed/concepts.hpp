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

#include <schedulers/detail/storage.hpp>
#include <schedulers/detail/graph/graph_instance.hpp>

namespace example::cuda::distributed
{

template <class S>
concept distributed_api = requires
{
  requires std::decay_t<S>::is_cuda_distributed_api;
};

template <class S>
concept distributed_sender = std::execution::sender<S> && distributed_api<S>;

template <class R>
concept distributed_receiver = std::execution::receiver<R> && distributed_api<R>;

template <class E>
concept distributed_env = requires(E &e)
{
  { cuda::get_storage(e) } -> std::convertible_to<std::byte *>;
};

template <distributed_sender S>
using value_of_t = typename std::decay_t<S>::value_t;

} // namespace example::cuda::graph
