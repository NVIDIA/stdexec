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

#include <algorithm>
#include <cstddef>
#include <execution.hpp>

namespace example::cuda
{

template <std::size_t Alignment, std::size_t Size>
struct alignas(Alignment) static_storage_t
{
  constexpr static bool empty = false;
  constexpr static std::size_t alignment = Alignment;
  constexpr static std::size_t size = Size;

  std::byte data[size];
};

template <std::size_t Alignment>
struct alignas(Alignment) static_storage_t<Alignment, 0>
{
  constexpr static bool empty = true;
  constexpr static std::size_t alignment = Alignment;
  constexpr static std::size_t size = 0;

  constexpr static std::byte *data = nullptr;
};

template <typename T>
using static_storage_from =
  static_storage_t<std::alignment_of_v<std::decay_t<T>>,
                   sizeof(std::decay_t<T>)>;

template <class Storage1, class Storage2>
using static_storage_union =
  static_storage_t<std::max(Storage1::alignment, Storage2::alignment),
                   std::max(Storage1::size, Storage2::size)>;

struct storage_description_t
{
  std::size_t alignment{1};
  std::size_t size{0};
};

struct pipeline_storage_t
{
  std::byte *aligned_storage_{};
  std::byte *storage_{};

  pipeline_storage_t() = default;
  pipeline_storage_t(pipeline_storage_t &&other) noexcept
      : aligned_storage_(other.aligned_storage_)
      , storage_(other.storage_)
  {
    other.storage_ = nullptr;
  }

  pipeline_storage_t &operator=(pipeline_storage_t &&other) noexcept
  {
    storage_ = other.storage_;
    aligned_storage_ = other.aligned_storage_;

    other.storage_ = nullptr;
    other.aligned_storage_ = nullptr;
    return *this;
  }

  ~pipeline_storage_t()
  {
    if (storage_)
    {
      cudaFree(storage_);
      storage_ = nullptr;
    }
  }

  explicit pipeline_storage_t(storage_description_t storage_requirement,
                              std::byte *forward_storage = nullptr)
      : aligned_storage_(forward_storage)
  {
    if (aligned_storage_ == nullptr)
    {
      if (storage_requirement.size > 0)
      {
        const std::size_t to_allocate = storage_requirement.alignment +
                                        storage_requirement.size - 1;

        cudaMallocManaged(&storage_, to_allocate);

        const std::size_t mask = ~(storage_requirement.alignment - 1);
        aligned_storage_ = reinterpret_cast<std::byte *>(
          reinterpret_cast<std::uintptr_t>(storage_ +
                                           storage_requirement.alignment - 1) &
          mask);
      }
    }
  }

  [[nodiscard]] std::byte *get() const noexcept { return aligned_storage_; }
};

inline constexpr struct storage_requirements_t
{
  friend constexpr bool
  tag_invoke(std::execution::forwarding_sender_query_t, storage_requirements_t)
  {
    return true;
  }

  template <std::execution::sender Sender>
    requires std::tag_invocable<storage_requirements_t, Sender>
  constexpr auto operator()(const Sender &sndr) const noexcept
  {
    return std::tag_invoke(storage_requirements_t{}, sndr);
  }

  template <std::execution::sender Sender>
    requires(!std::tag_invocable<storage_requirements_t, Sender>)
  constexpr auto operator()(const Sender &) const noexcept
  {
    return storage_description_t{};
  }
} storage_requirements{};

inline constexpr struct get_storage_t
{
  template <class EnvT>
    requires std::tag_invocable<get_storage_t, EnvT>
  constexpr auto operator()(EnvT &&env) const noexcept
  {
    return std::tag_invoke(get_storage_t{}, std::forward<EnvT>(env));
  }

  template <class EnvT>
    requires(!std::tag_invocable<get_storage_t, EnvT>)
  constexpr std::byte * operator()(EnvT&&) const noexcept
  {
    return nullptr;
  }
} get_storage{};

} // namespace example::cuda::detail
