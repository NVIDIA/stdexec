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
#include <schedulers/detail/graph/environment.hpp>
#include <schedulers/detail/graph/sender_base.hpp>
#include <schedulers/detail/helpers.hpp>
#include <schedulers/detail/storage.hpp>
#include <schedulers/detail/tuple.hpp>
#include <schedulers/detail/variant.hpp>

#include <atomic>
#include <execution.hpp>
#include <variant>
#include <span>

namespace example::cuda::graph::detail::ensure_started
{

template <graph_sender S>
struct shared_state_t;

template <graph_sender S>
struct receiver_t : std::execution::receiver_adaptor<receiver_t<S>>
{
  using super_t = std::execution::receiver_adaptor<receiver_t>;
  using sh_state_ptr_t = std::shared_ptr<shared_state_t<S>>;
  friend super_t;

  sh_state_ptr_t sh_state_;

  explicit receiver_t(sh_state_ptr_t sh_state)
    : super_t()
    , sh_state_(std::move(sh_state))
  {}

  void set_value(std::span<cudaGraphNode_t>) &&noexcept
  try
  {
    sh_state_->graph_.instantiate().launch();
    sh_state_->graph_.sync();
    sh_state_->notify(std::execution::set_value_t{});
    sh_state_.reset();
  }
  catch (...)
  {
    sh_state_->notify(std::current_exception());
    sh_state_.reset();
  }

  void set_error(std::exception_ptr ex_ptr) && noexcept
  {
    sh_state_->notify(ex_ptr);
    sh_state_.reset();
  }

  void set_stopped() && noexcept
  {
    sh_state_->notify(std::execution::set_stopped_t{});
    sh_state_.reset();
  }

  [[nodiscard]] consumer_t get_consumer() const noexcept
  {
    return {sh_state_->storage_.get()};
  }

  friend auto tag_invoke(std::execution::get_env_t, const receiver_t &self)
    -> detail::env_t<std::execution::no_env>
  {
    return detail::env_t<std::execution::no_env>{
      {},
      self.sh_state_->storage_.get(),
      self.sh_state_->graph_.graph()};
  }

  static constexpr bool is_cuda_graph_api = true;
};

template <graph_sender S>
struct shared_state_t
{
  explicit shared_state_t(cudaStream_t stream, S sender)
    : graph_(stream)
    , storage_{cuda::storage_requirements(sender)}
  {}

  void start(std::execution::connect_result_t<S, receiver_t<S>> &&op_state)
  {
    op_state_.emplace(std::move(op_state));
    std::execution::start(op_state_.value());
  }

  template <class SignalT>
  void notify(SignalT signal)
  {
    data_.emplace<SignalT>(signal);
    flag_.test_and_set(std::memory_order_release);
    flag_.notify_all();
  }

  void wait()
  {
    flag_.wait(false, std::memory_order_acquire);
  }

  std::variant<
    std::execution::set_value_t,
    std::exception_ptr,
    std::execution::set_stopped_t> data_;

  graph_t graph_;
  cuda::pipeline_storage_t storage_{};
  std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
  std::optional<std::execution::connect_result_t<S, receiver_t<S>>> op_state_;
};

template <graph_sender S,
          graph_receiver Receiver>
struct operation_state_t
{
  using value_t = value_of_t<S>;

  std::shared_ptr<shared_state_t<S>> sh_state_;
  Receiver receiver_;

  friend auto
  tag_invoke(std::execution::start_t, operation_state_t &self) noexcept try
  {
    self.sh_state_->wait();

    switch(self.sh_state_->data_.index())
    {
      case 0: {
        auto consumer = self.receiver_.get_consumer();

        if constexpr (std::is_same_v<value_t, cuda::variant<cuda::tuple<>>>)
        {
          consumer(thread_id_t{}, block_id_t{});
        }
        else
        {
          value_t &res = *reinterpret_cast<value_t*>(self.sh_state_->storage_.get());

          cuda::invoke(
            [&](auto &&...args) {
              consumer(thread_id_t{},
                       block_id_t{},
                       std::forward<decltype(args)>(args)...);
            },
            res);
        }

        std::execution::set_value(
          std::move(self.receiver_),
          std::span<cudaGraphNode_t>{});

        break;
      };

      case 1: {
        std::execution::set_error(std::move(self.receiver_),
                                  std::get<1>(self.sh_state_->data_));
        break;
      }

      case 2: {
        std::execution::set_stopped(std::move(self.receiver_));
        break;
      }
    }
  } catch (...) {
    std::execution::set_error(std::move(self.receiver_),
                              std::current_exception());
  }
};

template <graph_sender S>
struct sender_t : sender_base_t<sender_t<S>, S>
{
  using value_t = value_of_t<S>;

  template <graph_receiver Receiver>
  auto connect(Receiver &&receiver) && noexcept
  {
    return operation_state_t<S, Receiver>{
      std::move(sh_state_), std::forward<Receiver>(receiver)};
  }

  template <std::__decays_to<sender_t> Self, class _Env>
  friend auto tag_invoke(std::execution::get_completion_signatures_t,
                         Self &&,
                         _Env)
    -> std::execution::completion_signatures_of_t<std::__member_t<Self, S>,
                                                  _Env>;

  explicit sender_t(cudaStream_t stream, S sender)
      : sender_base_t<sender_t<S>, S>{sender}
      , stream_(stream)
      , sh_state_(new shared_state_t<S>{stream_, sender})
  {
    sh_state_->start(
      std::execution::connect(std::move(sender), receiver_t<S>{sh_state_}));
  }

  cudaStream_t stream_{};
  std::shared_ptr<shared_state_t<S>> sh_state_;
};
} // namespace example::cuda::graph::detail::ensure_started
