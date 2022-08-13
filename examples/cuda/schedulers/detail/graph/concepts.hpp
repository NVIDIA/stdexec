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

namespace example::cuda::graph
{

template <class E>
concept graph_env = requires(const E &e)
{
  { detail::get_graph(e) } -> std::convertible_to<detail::graph_info_t>;
  { cuda::get_storage_ptr(e) } -> std::convertible_to<std::byte **>;
};

template<typename ...Args>
using decayed_tuple = tuple<std::decay_t<Args>...>;

template<typename Signature>
struct transform_signature_impl;

template<typename Tag, typename ...Args>
struct transform_signature_impl<Tag(Args...)>
{
    using type = decayed_tuple<Tag, Args...>;
};

template<typename Signature>
using transform_signature_t = typename transform_signature_impl<Signature>::type;

template<typename Signatures>
struct storage_type_for_impl;

template<typename ...Signatures>
struct storage_type_for_impl<std::execution::completion_signatures<Signatures...>>
{
  using type = variant<transform_signature_t<Signatures>...>;
};

template<typename Sender, typename Env>
using storage_type_for_t = typename storage_type_for_impl<std::execution::completion_signatures_of_t<Sender, Env>>::type;

} // namespace example::cuda::graph
