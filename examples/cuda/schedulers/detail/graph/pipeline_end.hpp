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

template <class S, class Receiver>
struct receiver_t
    : std::execution::receiver_adaptor<receiver_t<S, Receiver>, Receiver>
{
  using super_t =
    std::execution::receiver_adaptor<receiver_t<S, Receiver>, Receiver>;

  friend super_t;

  cuda::storage_description_t storage_requirement_;

  graph_t graph_;
  mutable cuda::pipeline_storage_t storage_;

  receiver_t(cudaStream_t stream,
             Receiver &&receiver,
             cuda::storage_description_t storage_requirement)
      : super_t(std::move(receiver))
      , storage_requirement_(storage_requirement)
      , graph_(stream)
  {}

  template <class... Ts>
  void set_value(Ts&&... ts) &&noexcept
  try
  {
    if constexpr(graph_sender<std::decay_t<S>>)
    {
      graph_.instantiate().launch();

      check(cudaStreamSynchronize(graph_.stream_));

      using value_t = value_of_t<S>;
      value_t &res = *reinterpret_cast<value_t*>(get_storage());

      cuda::graph::detail::invoke(
        [&](auto&&... args) {
          std::execution::set_value(std::move(this->base()),
                                    std::forward<decltype(args)>(args)...);
        }, res);
    }
    else
    {
      std::execution::set_value(std::move(this->base()),
                                std::forward<Ts>(ts)...);
    }
  } catch(...) {
    std::execution::set_error(std::move(this->base()),
                              std::current_exception());
  }

  void set_error(std::exception_ptr ex_ptr) && noexcept
  {
    std::execution::set_error(std::move(this->base()), ex_ptr);
  }

  void set_stopped() noexcept
  {
    std::execution::set_stopped(std::move(this->base()));
  }

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

    auto env = std::execution::get_env(this->base());
    storage_ = cuda::pipeline_storage_t(storage_requirement_,
                                        cuda::get_storage(env));

    return storage_.get();
  }

  friend auto tag_invoke(std::execution::get_env_t, const receiver_t &self)
    -> detail::env_t<std::execution::env_of_t<Receiver>>
  {
    return detail::env_t<std::execution::env_of_t<Receiver>>{
      std::execution::get_env(self.base()),
      self.get_storage(),
      self.graph_.graph()};
  }

  static constexpr bool is_cuda_graph_api = true;
};

template <class S>
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

    return std::execution::connect(
      std::move(self.sender_),
      receiver_t<S, Receiver>{self.stream_,
                              std::forward<Receiver>(receiver),
                              storage_requirement});
  }

  template <std::__decays_to<sender_t> Self, class Env>
  friend auto tag_invoke(std::execution::get_completion_signatures_t,
                         Self &&,
                         Env)
    -> std::execution::completion_signatures_of_t<std::__member_t<Self, S>, Env>;
};

} // namespace example::cuda::graph::detail::pipeline_end
