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

namespace example::cuda::graph::detail::pipeline_end
{

template <class OutTuple, class Receiver>
struct receiver_t
{
  Receiver receiver_;
  cuda::storage_description_t storage_requirement_;

  graph_t graph_;
  mutable cuda::pipeline_storage_t storage_;

  receiver_t(cudaStream_t stream,
             Receiver &&receiver,
             cuda::storage_description_t storage_requirement)
      : receiver_(std::move(receiver))
      , storage_requirement_(storage_requirement)
      , graph_(stream)
  {}

  template <class... As>
  friend void tag_invoke(std::execution::set_value_t,
                         receiver_t &&self,
                         As &&...as) noexcept
  {}

  friend void tag_invoke(std::execution::set_value_t,
                         receiver_t &&self,
                         std::span<cudaGraphNode_t>) noexcept
  {
    self.graph_.instantiate().launch();

    cudaStreamSynchronize(self.graph_.stream_);

    OutTuple &res = *reinterpret_cast<OutTuple *>(self.get_storage());

    cuda::apply(
      [&](auto &&...args) {
        std::execution::set_value(std::move(self.receiver_),
                                  std::forward<decltype(args)>(args)...);
      },
      res);
  }

  friend void tag_invoke(std::execution::set_error_t,
                         receiver_t &&r,
                         std::exception_ptr ex_ptr) noexcept
  {
    std::execution::set_error(std::move(r.receiver_), ex_ptr);
  }

  friend void tag_invoke(std::execution::set_stopped_t, receiver_t &&r) noexcept
  {
    std::execution::set_stopped(std::move(r.receiver_));
  }

  [[nodiscard]] graph_info_t graph() noexcept { return {graph_.graph()}; }

  [[nodiscard]] consumer_t get_consumer() const noexcept
  {
    return {get_storage()};
  }

  [[nodiscard]] std::byte *get_storage() const noexcept
  {
    if (std::byte *storage = storage_.get(); storage != nullptr)
    {
      return storage;
    }

    storage_ = cuda::pipeline_storage_t(storage_requirement_,
                                        cuda::get_storage(receiver_));

    return storage_.get();
  }

  friend std::byte *tag_invoke(cuda::get_storage_t,
                               const receiver_t &self) noexcept
  {
    return self.get_storage();
  }

  template <
    std::__none_of<std::execution::set_value_t, cuda::get_storage_t> Tag,
    class... As>
  requires std::invocable<Tag, const Receiver &, As...> friend decltype(auto)
  tag_invoke(Tag tag, const receiver_t &self, As &&...as) noexcept
  {
    return ((Tag &&) tag)(std::as_const(self.receiver_), (As &&) as...);
  }

  friend auto tag_invoke(std::execution::get_env_t, const receiver_t &self)
    -> std::execution::env_of_t<Receiver>
  {
    return std::execution::get_env(self.receiver_);
  }

  static constexpr bool is_cuda_graph_api = true;
};

template <graph_sender S>
struct sender_t
{
  S sender_;
  cudaStream_t stream_{};

  template <std::__decays_to<sender_t> Self, class Receiver>
  friend auto tag_invoke(std::execution::connect_t,
                         Self &&self,
                         Receiver &&receiver) noexcept
  {
    auto storage_requirement = cuda::storage_requirements(self.sender_);

    using env_t = std::execution::env_of_t<Receiver>;
    using value_tuple =
      std::execution::value_types_of_t<Self, env_t, cuda::tuple, std::__single_t>;

    return std::execution::connect(
      std::move(self.sender_),
      receiver_t<value_tuple, Receiver>{self.stream_,
                                        std::forward<Receiver>(receiver),
                                        storage_requirement});
  }

  template <std::__decays_to<sender_t> Self, class _Env>
  friend auto tag_invoke(std::execution::get_completion_signatures_t,
                         Self &&,
                         _Env)
    -> std::execution::completion_signatures_of_t<std::__member_t<Self, S>,
                                                  _Env>;

  static constexpr bool is_cuda_graph_api = true;
};

} // namespace example::cuda::graph::detail::pipeline_end
