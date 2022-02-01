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
#include <schedulers/detail/distributed/environment.hpp>
#include <schedulers/detail/distributed/sender_base.hpp>
#include <schedulers/detail/distributed/consumer.hpp>
#include <schedulers/detail/helpers.hpp>
#include <schedulers/detail/storage.hpp>
#include <schedulers/detail/tuple.hpp>
#include <schedulers/detail/variant.hpp>

#include <execution.hpp>
#include <span>

namespace example::cuda::distributed::detail::then
{

template <class F, class ConsumerT>
__global__ __launch_bounds__(1) void then_kernel(F f, ConsumerT consumer)
{
  if constexpr (std::is_void_v<std::invoke_result_t<F>>)
  {
    f();
    consumer(thread_id_t{}, block_id_t{});
  }
  else
  {
    consumer(thread_id_t{}, block_id_t{}, f());
  }
}

template <class InTuple, class Receiver, class F>
class receiver_t
    : std::execution::receiver_adaptor<receiver_t<InTuple, Receiver, F>,
                                       Receiver>
{
  using super_t =
    std::execution::receiver_adaptor<receiver_t<InTuple, Receiver, F>, Receiver>;
  friend super_t;

  F function_;

  void set_value() && noexcept try
  {
    distributed_env auto env = std::execution::get_env(this->base());

    // if (env.context().is_main_device_holder())
    {
      auto consumer = consumer_t{cuda::get_storage(env)};
      then_kernel<<<1, 1, 0, env.stream()>>>(function_, consumer);
    }

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
  explicit receiver_t(Receiver receiver, F function)
    : super_t(std::move(receiver))
    , function_(function)
  {}

  static constexpr bool is_cuda_distributed_api = true;
};

template <distributed_sender S, class F>
struct sender_t : sender_base_t<sender_t<S, F>, S>
{
  using arguments_t = value_of_t<S>;
  using value_t = cuda::apply_t<F, arguments_t>;
  using super_t = sender_base_t<sender_t<S, F>, S>;
  friend super_t;

  F function_;

  template <distributed_receiver Receiver>
  auto connect(Receiver &&receiver) && noexcept
  {
    return std::execution::connect(
      std::move(this->sender_),
      receiver_t<arguments_t, Receiver, std::decay_t<F>>{std::forward<Receiver>(
                                                           receiver),
                                                         function_});
  }

  template <class Result>
  using set_value_ = std::__minvoke1<
    std::__uncurry<std::__qf<std::execution::set_value_t>>,
    std::__if<std::is_void<Result>, std::__types<>, std::__types<Result>>>;
  template <class... Args>
  requires std::invocable<F, Args...> using result =
    set_value_<std::invoke_result_t<F, Args...>>;

  template <class EnvT>
  friend auto tag_invoke(std::execution::get_completion_signatures_t,
                         const sender_t &,
                         EnvT)
    -> std::execution::make_completion_signatures<
      S, EnvT, std::execution::__with_exception_ptr, result>;

  explicit sender_t(S sender, F function)
    : super_t{std::forward<S>(sender)}
    , function_{function}
  {}
};
} // namespace example::cuda::distributed::detail::then
