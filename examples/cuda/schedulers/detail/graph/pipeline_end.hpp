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
#include <schedulers/detail/graph/environment.hpp>
#include <schedulers/detail/graph/graph_instance.hpp>
#include <schedulers/detail/graph/sender_base.hpp>
#include <schedulers/detail/helpers.hpp>
#include <schedulers/detail/storage.hpp>
#include <schedulers/detail/tuple.hpp>

#include <execution.hpp>
#include <span>

#include <type_traits>

namespace example::cuda::graph::detail::pipeline_end
{
struct trampoline_thread_t {
    std::thread thread;
    std::execution::run_loop loop;

    trampoline_thread_t() : thread([&]{ loop.run(); })
    {
    }

    ~trampoline_thread_t() {
        loop.finish();
        thread.join();
    }
};
inline auto get_trampoline_scheduler() {
    static trampoline_thread_t trampoline_state;
    return trampoline_state.loop.get_scheduler();
}

template <class Receiver>
struct indirect_receiver
{
public:
    indirect_receiver(Receiver * r) : base_(r)
    {
    }

    template <class Tag, class ...Ts>
    __host__ __device__
    friend auto tag_invoke(Tag, indirect_receiver && self, Ts &&... ts)
      noexcept(noexcept(tag_invoke(Tag{}, std::move(*self.base_), std::forward<Ts>(ts)...)))
      -> decltype(tag_invoke(Tag{}, std::declval<Receiver>(), std::forward<Ts>(ts)...))
    {
      return tag_invoke(Tag{}, std::move(*self.base_), std::forward<Ts>(ts)...);
    }

    template <class Tag, class ...Ts>
    __host__ __device__
    friend auto tag_invoke(Tag, indirect_receiver & self, Ts &&... ts)
      noexcept(noexcept(tag_invoke(Tag{}, *self.base_, std::forward<Ts>(ts)...)))
      -> decltype(tag_invoke(Tag{}, std::declval<Receiver &>(), std::forward<Ts>(ts)...))
    {
      return tag_invoke(Tag{}, *self.base_, std::forward<Ts>(ts)...);
    }

    template <class Tag, class ...Ts>
    __host__ __device__
    friend auto tag_invoke(Tag, const indirect_receiver & self, Ts &&... ts)
      noexcept(noexcept(tag_invoke(Tag{}, *static_cast<const Receiver *>(self.base_), std::forward<Ts>(ts)...)))
      -> decltype(tag_invoke(Tag{}, std::declval<const Receiver &>(), std::forward<Ts>(ts)...))
    {
      return tag_invoke(Tag{}, *static_cast<const Receiver *>(self.base_), std::forward<Ts>(ts)...);
    }

private:
    Receiver *base_;
};

template <class Receiver>
using trampoline_op_state_t = std::execution::connect_result_t<
  decltype(std::execution::schedule(get_trampoline_scheduler())),
  Receiver>;

template <std::execution::sender S>
struct sender_t;

template <class Receiver>
struct host_callback_data_t {
  Receiver receiver_;
  std::optional<trampoline_op_state_t<Receiver>> trampoline_op_state_;
};

template <class Receiver>
void host_callback(void * data_v)
{
  auto * data = reinterpret_cast<host_callback_data_t<Receiver> *>(data_v);

  data->trampoline_op_state_.emplace(
      std::execution::connect(std::execution::schedule(get_trampoline_scheduler()), std::move(data->receiver_)));
  std::execution::start(*data->trampoline_op_state_);
}

template <std::execution::sender Sender, std::execution::operation_state OpState,
         class Env, class InventedReceiver, class Receiver>
struct operation_state_t
{
private:
  std::byte **storage_ptr_;
  OpState *op_state_;

  graph_t graph_;
  cuda::pipeline_storage_t storage_;

  host_callback_data_t<Receiver> host_callback_data_;

public:
  template <std::execution::sender S>
  friend struct sender_t;

  operation_state_t(std::byte ** storage_ptr, OpState * state, graph_t graph, pipeline_storage_t storage,
          Receiver && receiver, std::span<const cudaGraphNode_t> predecessors)
      : storage_ptr_(storage_ptr)
      , op_state_(state)
      , graph_(std::move(graph))
      , storage_(std::move(storage))
      , host_callback_data_{std::forward<Receiver>(receiver)}
  {
    cudaHostNodeParams host_node_params{};
    host_node_params.fn = &host_callback<Receiver>;
    host_node_params.userData = &host_callback_data_;

    cudaGraphNode_t node;
    check(cudaGraphAddHostNode(&node, graph_.graph(), predecessors.data(), predecessors.size(), &host_node_params));
  }

  ~operation_state_t()
  {
    op_state_->~OpState();
    check(cudaFree(op_state_));
    check(cudaFree(storage_ptr_));
  }

  friend void tag_invoke(std::execution::start_t, operation_state_t & self) noexcept
  {
    std::execution::start(*self.op_state_);
  }
};

template <std::execution::sender S>
struct sender_t
{
  S sender_;
  cudaStream_t stream_{};

  template <std::__decays_to<sender_t> Self, class Receiver>
  friend auto tag_invoke(std::execution::connect_t,
                         Self &&self,
                         Receiver &&receiver) noexcept
  {
    using graph_env_t = detail::env_t<std::execution::env_of_t<Receiver>>;
    using storage_type = storage_type_for_t<S, graph_env_t>;

    using invented_receiver_t = consumer_receiver_t<storage_type, graph_env_t>;
    using nested_op_state_t = std::execution::connect_result_t<S, invented_receiver_t>;

    std::byte ** storage_ptr;
    check(cudaMallocManaged(&storage_ptr, sizeof(std::byte *)));
    new (storage_ptr) std::byte*(nullptr);

    auto graph = graph_t(self.stream_);
    auto invented_env = detail::env_t<std::execution::env_of_t<Receiver>>{
      std::execution::get_env(receiver),
      storage_ptr,
      graph_info_t(graph.graph(), self.stream_)};

    nested_op_state_t * op_state_storage;
    check(cudaMallocManaged(&op_state_storage, sizeof(nested_op_state_t)));
    new (op_state_storage) auto(std::execution::connect(std::move(self.sender_),
        invented_receiver_t(storage_ptr, std::move(invented_env))));

    auto dependencies = get_predecessors(*op_state_storage);

    auto storage_requirement = cuda::storage_requirements(*op_state_storage);
    auto pipeline_storage = pipeline_storage_t(storage_requirement);
    *storage_ptr = pipeline_storage.get();

    return operation_state_t<S, nested_op_state_t, graph_env_t, invented_receiver_t, Receiver>(
        storage_ptr, op_state_storage, std::move(graph), std::move(pipeline_storage),
        std::forward<Receiver>(receiver), dependencies);
  }

  template <std::execution::__sender_queries::__sender_query _Tag, class... _As>
  requires std::__callable<_Tag, const S&, _As...>
  friend auto tag_invoke(_Tag __tag, const S& __self, _As&&... __as)
  noexcept(std::__nothrow_callable<_Tag, const S&, _As...>)
  -> std::__call_result_if_t<std::execution::__sender_queries::__sender_query<_Tag>, _Tag, const S&, _As...> {
    return ((_Tag&&) __tag)(__self.sender_, (_As&&) __as...);
  }

  template <std::__decays_to<sender_t> Self, class Env>
  friend auto tag_invoke(std::execution::get_completion_signatures_t,
                         Self &&,
                         Env)
    -> std::execution::completion_signatures_of_t<std::__member_t<Self, S>, Env>;
};

} // namespace example::cuda::graph::detail::pipeline_end
