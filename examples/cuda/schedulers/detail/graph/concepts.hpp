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

namespace example::cuda::graph
{

template <class S>
concept graph_api = requires(S &&s)
{
  requires S::is_cuda_graph_api;
};

template <class S>
concept graph_sender = std::execution::sender<S> &&graph_api<S>;

template <class R>
concept graph_receiver = graph_api<R> &&requires(R &&r)
{
  { cuda::get_storage(r) } -> std::convertible_to<std::byte *>;
};

template <graph_sender S>
using value_of_t = typename std::decay_t<S>::value_t;

} // namespace example::cuda::graph
