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
#pragma once

#include "../../stdexec/execution.hpp"

#include <cuda/std/bit>

#include <algorithm>
#include <memory_resource>
#include <new>
#include <type_traits>
#include <mutex>
#include <set>

#include "config.cuh"
#include "throw_on_cuda_error.cuh"

namespace nvexec::STDEXEC_STREAM_DETAIL_NS {

  struct device_deleter {
    template <class T>
    void operator()(T* ptr) {
      // ptr->~T();
      STDEXEC_DBG_ERR(::cudaFree(ptr));
    }
  };

  template <class T>
  using device_ptr = std::unique_ptr<T, device_deleter>;

  template <class T, class... As>
  device_ptr<T> make_device(cudaError_t& status, As&&... as) {
    static_assert(STDEXEC_IS_TRIVIALLY_COPYABLE(T));

    if (status == cudaSuccess) {
      T* ptr = nullptr;
      if (status = STDEXEC_DBG_ERR(::cudaMalloc(&ptr, sizeof(T))); status == cudaSuccess) {
        try {
          T h((As&&) as...);
          status = STDEXEC_DBG_ERR(::cudaMemcpy(ptr, &h, sizeof(T), cudaMemcpyHostToDevice));
          if (status == cudaSuccess) {
            return device_ptr<T>(ptr);
          }
        } catch (...) {
          status = cudaErrorUnknown;
          STDEXEC_DBG_ERR(::cudaFree(ptr));
        }
      }
    }

    return device_ptr<T>();
  }

  template <class T = void, class A>
    requires same_as<T, void>
  device_ptr<__decay_t<A>> make_device(cudaError_t& status, A&& t) {
    return make_device<__decay_t<A>>(status, (A&&) t);
  }

  template <class T>
  struct host_deleter {
    std::pmr::memory_resource* resource_{nullptr};

    void operator()(T* ptr) const {
      if (ptr) {
        ptr->~T();
        resource_->deallocate(ptr, sizeof(T), alignof(T));
      }
    }
  };

  template <class T>
  using host_ptr = std::unique_ptr<T, host_deleter<T>>;

  template <class T, class... As>
  host_ptr<T> make_host(cudaError_t& status, std::pmr::memory_resource* resource, As&&... as) {
    T* ptr = nullptr;

    if (status == cudaSuccess) {
      try {
        ptr = static_cast<T*>(resource->allocate(sizeof(T), alignof(T)));
        ::new (static_cast<void*>(ptr)) T((As&&) as...);
        return host_ptr<T>(ptr, {resource});
      } catch (...) {
        if (ptr) {
          status = cudaErrorUnknown;
          resource->deallocate(ptr, sizeof(T), alignof(T));
        } else {
          status = cudaErrorMemoryAllocation;
        }
      }
    }

    return host_ptr<T>();
  }

  template <class T = void, class A>
    requires same_as<T, void>
  host_ptr<__decay_t<A>>
    make_host(cudaError_t& status, std::pmr::memory_resource* resource, A&& t) {
    return make_host<__decay_t<A>>(status, resource, (A&&) t);
  }

  struct pinned_resource : public std::pmr::memory_resource {
    pinned_resource() noexcept {
    }

    void* do_allocate(const std::size_t bytes, const std::size_t /* alignment */) override {
      void* ret;

      if (cudaError_t status = STDEXEC_DBG_ERR(cudaMallocHost(&ret, bytes));
          status != cudaSuccess) {
        throw std::bad_alloc();
      }

      return ret;
    }

    void do_deallocate(void* ptr, const std::size_t /* bytes */, const std::size_t /* alignment */)
      override {
      STDEXEC_DBG_ERR(cudaFreeHost(ptr));
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
      return this == &other;
    }
  };

  struct gpu_resource : public std::pmr::memory_resource {
    gpu_resource() noexcept {
    }

    void* do_allocate(const std::size_t bytes, const std::size_t /* alignment */) override {
      void* ret;

      if (cudaError_t status = STDEXEC_DBG_ERR(cudaMalloc(&ret, bytes)); status != cudaSuccess) {
        throw std::bad_alloc();
      }

      return ret;
    }

    void do_deallocate(void* ptr, const std::size_t /* bytes */, const std::size_t /* alignment */)
      override {
      STDEXEC_DBG_ERR(cudaFree(ptr));
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
      return this == &other;
    }
  };

  struct managed_resource : public std::pmr::memory_resource {
    void* do_allocate(const std::size_t bytes, const std::size_t /* alignment */) override {
      void* ret;

      if (cudaError_t status = STDEXEC_DBG_ERR(cudaMallocManaged(&ret, bytes));
          status != cudaSuccess) {
        throw std::bad_alloc();
      }

      return ret;
    }

    void do_deallocate(void* ptr, const std::size_t /* bytes */, const std::size_t /* alignment */)
      override {
      STDEXEC_DBG_ERR(cudaFree(ptr));
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
      return this == &other;
    }
  };

  struct monotonic_buffer_resource : std::pmr::memory_resource {
    static constexpr std::size_t block_alignment = 256;

    struct block_descriptor_t {
      void* ptr{};
      std::size_t total{};
    };

    std::pmr::memory_resource* upstream;
    std::vector<block_descriptor_t> allocated_blocks;

    std::size_t space{};
    void* current_ptr{};

    monotonic_buffer_resource(std::size_t bytes, std::pmr::memory_resource* upstream)
      : upstream(upstream)
      , space(std::max(bytes, std::size_t{2})) {
      block_descriptor_t first_block{upstream->allocate(space, block_alignment), space};
      current_ptr = first_block.ptr;
      allocated_blocks.push_back(first_block);
    }

    ~monotonic_buffer_resource() {
      for (block_descriptor_t& block: allocated_blocks) {
        upstream->deallocate(block.ptr, block.total, block_alignment);
      }
    }

    block_descriptor_t get_current_block() {
      return allocated_blocks.back();
    }

    std::size_t get_next_space() {
      const std::size_t last_block_size = get_current_block().total;
      return last_block_size + last_block_size / 2;
    }

    void* do_allocate(const std::size_t bytes, const std::size_t alignment) override {
      assert(alignment <= block_alignment);
      void* ptr = std::align(alignment, bytes, current_ptr, space);

      if (ptr == nullptr) {
        space = std::max(bytes, get_next_space());
        ptr = current_ptr = upstream->allocate(space, block_alignment);
        allocated_blocks.push_back(block_descriptor_t{current_ptr, space});
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

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
      return this == &other;
    }
  };

  struct synchronized_pool_resource : std::pmr::memory_resource {
    constexpr static std::size_t block_alignment = 256;

    struct block_descriptor_t {
      static constexpr unsigned int min_bin = 3;

      void* ptr{};
      unsigned int bin{};
      std::size_t bytes{};

      explicit block_descriptor_t(std::size_t bytes)
        : bin(cuda::std::bit_width(bytes))
        , bytes(1ull << bin) {
        if (bin < min_bin) {
          bin = min_bin;
          bytes = 1ull << bin;
        }
      }

      explicit block_descriptor_t(void* ptr)
        : ptr(ptr) {
      }
    };

    struct ptr_comparator_t {
      bool operator()(const block_descriptor_t& a, const block_descriptor_t& b) const {
        return a.ptr < b.ptr;
      }
    };

    struct size_comparator_t {
      bool operator()(const block_descriptor_t& a, const block_descriptor_t& b) const {
        return a.bytes < b.bytes;
      }
    };

    using cached_blocks_t = std::multiset<block_descriptor_t, size_comparator_t>;
    using busy_blocks_t = std::set<block_descriptor_t, ptr_comparator_t>;

    std::mutex mutex;

    std::pmr::memory_resource* upstream;
    cached_blocks_t cached_blocks;
    busy_blocks_t live_blocks;

    synchronized_pool_resource(std::pmr::memory_resource* upstream)
      : upstream(upstream)
      , cached_blocks(size_comparator_t{})
      , live_blocks(ptr_comparator_t{}) {
    }

    void* do_allocate(const std::size_t bytes, const std::size_t alignment) override {
      assert(alignment <= block_alignment);

      std::lock_guard<std::mutex> lock(mutex);

      block_descriptor_t search_key{bytes};
      cached_blocks_t::iterator block_itr = cached_blocks.lower_bound(search_key);

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

      block_descriptor_t search_key{ptr};
      busy_blocks_t::iterator block_itr = live_blocks.find(search_key);

      if (block_itr != live_blocks.end()) {
        search_key = *block_itr;
        live_blocks.erase(block_itr);
        cached_blocks.insert(search_key);
      }
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
      return this == &other;
    }

    ~synchronized_pool_resource() {
      STDEXEC_ASSERT(live_blocks.empty());

      while (!cached_blocks.empty()) {
        cached_blocks_t::iterator begin = cached_blocks.begin();
        upstream->deallocate(begin->ptr, begin->bytes, block_alignment);
        cached_blocks.erase(begin);
      }
    }
  };

  template <class UnderlyingResource>
  class resource_storage {
    UnderlyingResource underlying_resource_;
    monotonic_buffer_resource monotonic_resource_;
    synchronized_pool_resource resource_;

   public:
    resource_storage()
      : underlying_resource_{}
      , monotonic_resource_{512 * 1024, &underlying_resource_}
      , resource_{&monotonic_resource_} {
    }

    std::pmr::memory_resource* get() {
      return &resource_;
    }
  };
}
