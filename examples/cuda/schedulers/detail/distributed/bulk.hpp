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

#include <schedulers/detail/distributed/concepts.hpp>
#include <schedulers/detail/distributed/even_share.hpp>
#include <schedulers/detail/distributed/environment.hpp>
#include <schedulers/detail/distributed/sender_base.hpp>
#include <schedulers/detail/helpers.hpp>
#include <schedulers/detail/storage.hpp>
#include <schedulers/detail/tuple.hpp>
#include <schedulers/detail/variant.hpp>

#include <execution.hpp>
#include <span>

namespace example::cuda::distributed::detail::bulk
{

template <unsigned int BlockThreads, class Shape, class F>
__global__ __launch_bounds__(BlockThreads) 
void bulk_kernel(Shape begin, Shape end, F f)
{
  const Shape i = begin + static_cast<Shape>(blockIdx.x * BlockThreads + threadIdx.x);

  if (i < end)
  {
    f(i);
  }
}

template <class InTuple, class Receiver, class Shape, class F>
class receiver_t
    : std::execution::receiver_adaptor<receiver_t<InTuple, Receiver, Shape, F>,
                                       Receiver>
{
  using super_t =
    std::execution::receiver_adaptor<receiver_t<InTuple, Receiver, Shape, F>, Receiver>;
  friend super_t;

  Shape shape_;
  F function_;

  void set_value() && noexcept try
  {
    const auto &ctx = std::execution::get_env(this->base()).context();

    const auto rank = ctx.rank();
    const auto size = ctx.size();
    auto [begin, end] = even_share(shape_, rank, size);

    constexpr Shape block_size = 256;
    const Shape grid_size = (end - begin + block_size - 1) / block_size;

    bulk_kernel<block_size><<<grid_size, block_size, 0, ctx.stream()>>>(
        begin, end, function_);

    std::execution::set_value(std::move(this->base()));
  } catch(...) {
    std::execution::set_error(std::move(this->base()),
                              std::current_exception());
  }

  void set_error(std::exception_ptr ex_ptr) && noexcept
  {
    std::execution::set_error(std::move(this->base()), ex_ptr);
  }

  void set_stopped() && noexcept
  {
    std::execution::set_stopped(std::move(this->base()));
  }

public:
  explicit receiver_t(Receiver receiver, Shape shape, F function)
    : super_t(std::move(receiver))
    , shape_(shape)
    , function_(function)
  {}

  static constexpr bool is_cuda_distributed_api = true;
};

template <distributed_sender S, std::integral Shape, class F>
struct sender_t : sender_base_t<sender_t<S, Shape, F>, S>
{
  using value_t = value_of_t<S>;
  using super_t = sender_base_t<sender_t<S, Shape, F>, S>;
  friend super_t;

  Shape shape_;
  F function_;

  template <distributed_receiver Receiver>
  auto connect(Receiver &&receiver) && noexcept
  {
    return std::execution::connect(
      std::move(this->sender_),
      receiver_t<value_t, Receiver, Shape, std::decay_t<F>>{
        std::forward<Receiver>(receiver),
        shape_,
        function_});
  }

  template <std::__decays_to<sender_t> Self, class _Env>
  friend auto tag_invoke(std::execution::get_completion_signatures_t,
                         Self &&,
                         _Env)
    -> std::execution::completion_signatures_of_t<std::__member_t<Self, S>,
                                                  _Env>;

  explicit sender_t(S sender, Shape shape, F function)
    : super_t{std::forward<S>(sender)}
    , shape_{shape}
    , function_{function}
  {}
};

} // namespace example::cuda::distributed::detail::then

