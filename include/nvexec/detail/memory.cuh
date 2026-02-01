/*
 * Copyright (c) 2022 NVIDIA Corporation
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

// clang-format Language: Cpp

#pragma once

#include "../../stdexec/execution.hpp"

#include <cuda/std/bit>

#include <cstddef>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <new>
#include <set>
#include <vector>

#include "config.cuh"
#include "throw_on_cuda_error.cuh"

namespace nvexec::_strm {

  struct device_deleter {
    template <class Type>
    void operator()(Type* ptr) {
      // ptr->~Type();
      STDEXEC_ASSERT_CUDA_API(::cudaFree(ptr));
    }
  };

  template <class Type>
  using device_ptr_t = std::unique_ptr<Type, device_deleter>;

  template <class Type, class... Args>
  auto device_allocate(cudaError_t& status, Args&&... args) -> device_ptr_t<Type> {
    static_assert(STDEXEC_IS_TRIVIALLY_COPYABLE(Type));

    if (status == cudaSuccess) {
      Type* ptr = nullptr;
      if (status = STDEXEC_LOG_CUDA_API(::cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(Type)));
          status == cudaSuccess) {
        STDEXEC_TRY {
          Type h(static_cast<Args&&>(args)...);
          status = STDEXEC_LOG_CUDA_API(::cudaMemcpy(ptr, &h, sizeof(Type), cudaMemcpyHostToDevice));
          if (status == cudaSuccess) {
            return device_ptr_t<Type>(ptr);
          }
        }
        STDEXEC_CATCH_ALL {
          status = cudaErrorUnknown;
          STDEXEC_ASSERT_CUDA_API(::cudaFree(ptr));
        }
      }
    }

    return device_ptr_t<Type>();
  }

  template <int = 0, class Type>
  auto device_allocate(cudaError_t& status, Type&& value) -> device_ptr_t<__decay_t<Type>> {
    return device_allocate<__decay_t<Type>>(status, static_cast<Type&&>(value));
  }

  template <class Type>
  struct host_deleter {
    std::pmr::memory_resource* resource_{nullptr};

    void operator()(Type* ptr) const {
      if (ptr) {
        ptr->~Type();
        resource_->deallocate(ptr, sizeof(Type), alignof(Type));
      }
    }
  };

  template <class Type>
  using host_ptr_t = std::unique_ptr<Type, host_deleter<Type>>;

  template <class Type, class... Args>
  auto
    host_allocate(cudaError_t& status, std::pmr::memory_resource* resource, Args&&... args) -> host_ptr_t<Type> {
    Type* ptr = nullptr;

    if (status == cudaSuccess) {
      STDEXEC_TRY {
        ptr = static_cast<Type*>(resource->allocate(sizeof(Type), alignof(Type)));
        ::new (static_cast<void*>(ptr)) Type(static_cast<Args&&>(args)...);
        return host_ptr_t<Type>(ptr, {resource});
      }
      STDEXEC_CATCH_ALL {
        if (ptr) {
          status = cudaErrorUnknown;
          resource->deallocate(ptr, sizeof(Type), alignof(Type));
        } else {
          status = cudaErrorMemoryAllocation;
        }
      }
    }

    return host_ptr_t<Type>();
  }

  template <int = 0, class Type>
  auto host_allocate(cudaError_t& status, std::pmr::memory_resource* resource, Type&& value)
    -> host_ptr_t<__decay_t<Type>> {
    return host_allocate<__decay_t<Type>>(status, resource, static_cast<Type&&>(value));
  }

  struct pinned_resource : public std::pmr::memory_resource {
    pinned_resource() noexcept = default;

    auto do_allocate(const std::size_t bytes, const std::size_t /* alignment */) -> void* override {
      void* ret;

      if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaMallocHost(&ret, bytes));
          status != cudaSuccess) {
        throw std::bad_alloc();
      }

      return ret;
    }

    void do_deallocate(void* ptr, const std::size_t /* bytes */, const std::size_t /* alignment */)
      override {
      STDEXEC_ASSERT_CUDA_API(cudaFreeHost(ptr));
    }

    [[nodiscard]]
    auto do_is_equal(const std::pmr::memory_resource& other) const noexcept -> bool override {
      return this == &other;
    }
  };

  struct gpu_resource : public std::pmr::memory_resource {
    gpu_resource() noexcept = default;

    auto do_allocate(const std::size_t bytes, const std::size_t /* alignment */) -> void* override {
      void* ret;

      if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaMalloc(&ret, bytes));
          status != cudaSuccess) {
        throw std::bad_alloc();
      }

      return ret;
    }

    void do_deallocate(void* ptr, const std::size_t /* bytes */, const std::size_t /* alignment */)
      override {
      STDEXEC_ASSERT_CUDA_API(cudaFree(ptr));
    }

    [[nodiscard]]
    auto do_is_equal(const std::pmr::memory_resource& other) const noexcept -> bool override {
      return this == &other;
    }
  };

  struct managed_resource : public std::pmr::memory_resource {
    auto do_allocate(const std::size_t bytes, const std::size_t /* alignment */) -> void* override {
      void* ret;

      if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaMallocManaged(&ret, bytes));
          status != cudaSuccess) {
        throw std::bad_alloc();
      }

      return ret;
    }

    void do_deallocate(void* ptr, const std::size_t /* bytes */, const std::size_t /* alignment */)
      override {
      STDEXEC_ASSERT_CUDA_API(cudaFree(ptr));
    }

    [[nodiscard]]
    auto do_is_equal(const std::pmr::memory_resource& other) const noexcept -> bool override {
      return this == &other;
    }
  };

  struct monotonic_buffer_resource : std::pmr::memory_resource {
    static constexpr std::size_t block_alignment = 256;

    struct block_descriptor {
      void* ptr{};
      std::size_t total{};
    };

    std::pmr::memory_resource* upstream;
    std::vector<block_descriptor> allocated_blocks;

    std::size_t space{};
    void* current_ptr{};

    monotonic_buffer_resource(std::size_t bytes, std::pmr::memory_resource* upstream)
      : upstream(upstream)
      , space(STDEXEC::__umax({bytes, std::size_t{2}})) {
      block_descriptor first_block{
        .ptr = upstream->allocate(space, block_alignment), .total = space};
      current_ptr = first_block.ptr;
      allocated_blocks.push_back(first_block);
    }

    ~monotonic_buffer_resource() override {
      for (block_descriptor& block: allocated_blocks) {
        upstream->deallocate(block.ptr, block.total, block_alignment);
      }
    }

    auto get_current_block() -> block_descriptor {
      return allocated_blocks.back();
    }

    auto get_next_space() -> std::size_t {
      const std::size_t last_block_size = get_current_block().total;
      return last_block_size + last_block_size / 2;
    }

    auto do_allocate(const std::size_t bytes, const std::size_t alignment) -> void* override {
      STDEXEC_ASSERT(alignment <= block_alignment);
      void* ptr = std::align(alignment, bytes, current_ptr, space);

      if (ptr == nullptr) {
        space = STDEXEC::__umax({bytes, get_next_space()});
        ptr = current_ptr = upstream->allocate(space, block_alignment);
        allocated_blocks.push_back(block_descriptor{.ptr = current_ptr, .total = space});
      }

      current_ptr = static_cast<char*>(ptr) + bytes;
      space -= bytes;

      return ptr;
    }

    void do_deallocate(
      void* /* ptr */,
      const std::size_t /* bytes */,
      const std::size_t /* alignment */) override {
    }

    [[nodiscard]]
    auto do_is_equal(const std::pmr::memory_resource& other) const noexcept -> bool override {
      return this == &other;
    }
  };

  struct synchronized_pool_resource : std::pmr::memory_resource {
    static constexpr std::size_t block_alignment = 256;

    struct block_descriptor {
      static constexpr unsigned int min_bin = 3;

      void* ptr{};
      unsigned int bin{};
      std::size_t bytes{};

      explicit block_descriptor(std::size_t bytes)
        : bin(cuda::std::bit_width(bytes))
        , bytes(1ull << bin) {
        if (bin < min_bin) {
          bin = min_bin;
          bytes = 1ull << bin;
        }
      }

      explicit block_descriptor(void* ptr)
        : ptr(ptr) {
      }
    };

    struct ptr_comparator {
      auto operator()(const block_descriptor& a, const block_descriptor& b) const -> bool {
        return a.ptr < b.ptr;
      }
    };

    struct size_comparator {
      auto operator()(const block_descriptor& a, const block_descriptor& b) const -> bool {
        return a.bytes < b.bytes;
      }
    };

    using cached_blocks_t = std::multiset<block_descriptor, size_comparator>;
    using busy_blocks_t = std::set<block_descriptor, ptr_comparator>;

    std::mutex mutex;

    std::pmr::memory_resource* upstream;
    cached_blocks_t cached_blocks;
    busy_blocks_t live_blocks;

    synchronized_pool_resource(std::pmr::memory_resource* upstream)
      : upstream(upstream)
      , cached_blocks(size_comparator{})
      , live_blocks(ptr_comparator{}) {
    }

    auto do_allocate(const std::size_t bytes, [[maybe_unused]] const std::size_t alignment)
      -> void* override {
      STDEXEC_ASSERT(alignment <= block_alignment);

      std::lock_guard<std::mutex> lock(mutex);

      block_descriptor search_key{bytes};
      auto block_itr = cached_blocks.lower_bound(search_key);

      if ((block_itr != cached_blocks.end()) && (block_itr->bin == search_key.bin)) {
        search_key = *block_itr;
        live_blocks.insert(search_key);
        cached_blocks.erase(block_itr);
        return search_key.ptr;
      }

      search_key.ptr = upstream->allocate(search_key.bytes, block_alignment);
      live_blocks.insert(search_key);
      return search_key.ptr;
    }

    void do_deallocate(void* ptr, std::size_t /* bytes */, std::size_t /* alignment */) override {
      std::lock_guard<std::mutex> lock(mutex);

      block_descriptor search_key{ptr};
      auto block_itr = live_blocks.find(search_key);

      if (block_itr != live_blocks.end()) {
        search_key = *block_itr;
        live_blocks.erase(block_itr);
        cached_blocks.insert(search_key);
      }
    }

    [[nodiscard]]
    auto do_is_equal(const std::pmr::memory_resource& other) const noexcept -> bool override {
      return this == &other;
    }

    ~synchronized_pool_resource() override {
      STDEXEC_ASSERT(live_blocks.empty());

      while (!cached_blocks.empty()) {
        auto begin = cached_blocks.begin();
        upstream->deallocate(begin->ptr, begin->bytes, block_alignment);
        cached_blocks.erase(begin);
      }
    }
  };

  template <class UnderlyingResource>
  class resource_storage {
    UnderlyingResource underlying_resource_{};
    monotonic_buffer_resource monotonic_resource_;
    synchronized_pool_resource resource_;

   public:
    resource_storage()
      : monotonic_resource_{512 * 1024, &underlying_resource_}
      , resource_{&monotonic_resource_} {
    }

    auto get() -> std::pmr::memory_resource* {
      return &resource_;
    }
  };
} // namespace nvexec::_strm
