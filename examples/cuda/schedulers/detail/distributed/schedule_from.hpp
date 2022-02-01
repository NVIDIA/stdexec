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
#include <type_traits>
#include <span>

namespace example::cuda::distributed::detail::schedule_from
{

template <class R>
struct receiver_t
{
  R receiver_;

  template <class... Ts>
  friend void
  tag_invoke(std::execution::set_value_t, receiver_t &&self, Ts&&... ts) noexcept
  {
    std::execution::set_value(std::move(self.receiver_),
                              std::forward<Ts>(ts)...);
  }

  friend void tag_invoke(
    std::execution::set_error_t,
    receiver_t &&r,
    std::exception_ptr ex_ptr) noexcept
  {
    std::execution::set_error(std::move(r.receiver_), ex_ptr);
  }

  friend void tag_invoke(std::execution::set_stopped_t, receiver_t &&r) noexcept
  {
    std::execution::set_stopped(std::move(r.receiver_));
  }

  template <std::__none_of<std::execution::set_value_t> Tag, class... Ts>
  requires std::invocable<Tag, const R &, Ts...> friend decltype(auto)
  tag_invoke(Tag tag, const receiver_t &self, Ts &&...ts) noexcept
  {
    return ((Tag &&) tag)(std::as_const(self.receiver_), (Ts &&) ts...);
  }

  static constexpr bool is_cuda_distributed_api = true;
};

template <class Scheduler, class S>
struct sender_t
{
  S sender_;
  Scheduler scheduler_;

  template <std::__decays_to<sender_t> Self, class Receiver>
  friend auto tag_invoke(std::execution::connect_t, Self&& self, Receiver&& rcvr) noexcept
  {
    return std::execution::connect(
      std::move(self.sender_),
      receiver_t<Receiver>{std::forward<Receiver>(rcvr)});
  }

  template <std::execution::__sender_queries::__sender_query _Tag, class... _As>
  requires std::__callable<_Tag, const S&, _As...>
  friend auto tag_invoke(_Tag __tag, const sender_t& __self, _As&&... __as)
  noexcept(std::__nothrow_callable<_Tag, const S&, _As...>)
  -> std::__call_result_if_t<std::execution::__sender_queries::__sender_query<_Tag>, _Tag, const S&, _As...> {
    return ((_Tag&&) __tag)(__self.sender_, (_As&&) __as...);
  }

  friend constexpr auto tag_invoke(cuda::storage_requirements_t,
                                   const sender_t &s) noexcept
  {
    return cuda::storage_requirements(s.sender_);
  }

  template <class CPO>
  friend Scheduler tag_invoke(
    std::execution::get_completion_scheduler_t<CPO>,
    const sender_t &self) noexcept
  {
    return self.scheduler_;
  }

  template <std::__decays_to<sender_t> Self, class _Env>
  friend auto
  tag_invoke(std::execution::get_completion_signatures_t, Self &&, _Env)
  -> std::execution::completion_signatures_of_t<std::__member_t<Self, S>, _Env>;

  using value_t = std::execution::
  value_types_of_t<S, std::execution::no_env, cuda::tuple, cuda::variant>;

  static constexpr bool is_cuda_distributed_api = true;
};

} // namespace distributed::schedule_from
