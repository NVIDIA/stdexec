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

#include <cstdint>

#include <schedulers/detail/tuple.hpp>
#include <schedulers/detail/variant.hpp>
#include <schedulers/detail/graph/concepts.hpp>

namespace example::cuda::graph::detail
{

template <class ArgsOut, class Env>
class consumer_receiver_t
    : std::execution::receiver_adaptor<consumer_receiver_t<ArgsOut, Env>>
{
  using super_t =
    std::execution::receiver_adaptor<consumer_receiver_t<ArgsOut, Env>>;
  friend super_t;

  template <class... Ts>
  __host__ __device__
  void set_value(Ts &&... ts) && noexcept
  {
    new (*storage_ptr_) ArgsOut(decayed_tuple<std::execution::set_value_t, Ts...>(std::execution::set_value, std::forward<Ts>(ts)...));
  }

  template <class Err>
  __host__ __device__
  void set_error(Err && err) && noexcept
  {
    new (*storage_ptr_) ArgsOut(decayed_tuple<std::execution::set_error_t, Err>(std::execution::set_error, std::forward<Err>(err)));
  }

  __host__ __device__
  void set_stopped() && noexcept
  {
    new (*storage_ptr_) ArgsOut(decayed_tuple<std::execution::set_stopped_t>(std::execution::set_stopped));
  }

  std::byte ** storage_ptr_;
  std::optional<Env> env_;

public:
  consumer_receiver_t(std::byte ** storage_ptr, std::optional<Env> env) : storage_ptr_(storage_ptr), env_(std::move(env))
  {
  }

  friend Env tag_invoke(std::execution::get_env_t, const consumer_receiver_t &self)
  {
    assert(self.env_);
    return *self.env_;
  }
};

template <class ArgsOut, class Receiver>
class reader_receiver_t : std::execution::receiver_adaptor<reader_receiver_t<ArgsOut, Receiver>>
{
  using super_t =
    std::execution::receiver_adaptor<consumer_receiver_t<ArgsOut, Receiver>>;
  friend super_t;

  void set_value() && noexcept
  {
    cuda::visit([&](auto signature) {
      storage_ptr_->~ArgsOut();

      cuda::apply([&](auto tag, auto &&... args) {
        tag(std::move(receiver_), std::forward<decltype(args)>(args)...);
      }, std::move(signature));
    }, std::move(**storage_ptr_));
  }

  template <class Err>
  void set_error(Err && err) noexcept
  {
    storage_ptr_->~ArgsOut();
    std::execution::set_error(std::move(receiver_), std::forward<Err>(err));
  }

  void set_stopped() noexcept
  {
    storage_ptr_->~ArgsOut();
    std::execution::set_stopped(std::move(receiver_));
  }

  ArgsOut ** storage_ptr_;
  Receiver receiver_;

public:
  reader_receiver_t(std::byte ** storage_ptr, Receiver receiver)
      : storage_ptr_(reinterpret_cast<ArgsOut **>(storage_ptr)), receiver_(std::move(receiver))
  {
  }

  friend auto tag_invoke(std::execution::get_env_t, const reader_receiver_t & self)
  {
    return std::execution::get_env(self.receiver_);
  }
};

} // namespace example::cuda::graph::detail
