/*
 * Copyright (c) 2021-2022 NVIDIA Corporation
 * Copyright (c) 2023 Maikel Nadolski
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

#include "../../stdexec/__detail/__config.hpp"
#include "../scope.hpp"

#include <algorithm>
#include <cstddef>
#include <thread>

namespace exec {
  struct numa_policy {
    virtual std::size_t num_nodes() = 0;
    virtual std::size_t num_cpus(int node) = 0;
    virtual int bind_to_node(int node) = 0;
    virtual std::size_t thread_index_to_node(std::size_t index) = 0;
  };

  class no_numa_policy : public numa_policy {
   public:
    no_numa_policy() noexcept = default;
    std::size_t num_nodes() override { return 1; }
    std::size_t num_cpus(int node) override { return std::thread::hardware_concurrency(); }
    int bind_to_node(int node) override { return 0; }
    std::size_t thread_index_to_node(std::size_t index) override { return 0; }
  };
}

#if STDEXEC_ENABLE_NUMA
#include <numa.h>
namespace exec {
  struct default_numa_policy : numa_policy {
    default_numa_policy() : node_to_thread_index_(::numa_num_task_nodes()) {
      std::size_t total_cpus = 0;
      std::size_t n_nodes = num_nodes();
      for (std::size_t node = 0; node < n_nodes; ++node) {
        total_cpus += this->num_cpus(node);
        node_to_thread_index_[node] = total_cpus;
      }
    }

    std::size_t num_nodes() override { return ::numa_num_task_nodes(); }
    
    std::size_t num_cpus(int node) override { 
      struct ::bitmask* cpus = ::numa_allocate_cpumask();
      if (!cpus) {
        return 0;
      }
      scope_guard sg{[&]() noexcept { ::numa_free_cpumask(cpus); }};
      int rc = ::numa_node_to_cpus(node, cpus);
      if (rc < 0) {
        return 0;
      }
      std::size_t num_cpus = ::numa_bitmask_weight(cpus);
      return num_cpus;
    }

    int bind_to_node(int node) override { 
      struct ::bitmask* nodes = ::numa_allocate_nodemask();
      if (!nodes) {
        return -1;
      }
      scope_guard sg{[&]() noexcept { ::numa_free_nodemask(nodes); }};
      ::numa_bitmask_setbit(nodes, node);
      ::numa_bind(nodes);
      return 0;
    }

    std::size_t thread_index_to_node(std::size_t index) override {
      index %= node_to_thread_index_.back();
      auto it = std::upper_bound(node_to_thread_index_.begin(), node_to_thread_index_.end(), index);
      STDEXEC_ASSERT(it != node_to_thread_index_.end());
      return std::distance(node_to_thread_index_.begin(), it);
    }

    std::vector<int> node_to_thread_index_{};
  };

  inline numa_policy* get_numa_policy() noexcept {
    thread_local default_numa_policy g_default_numa_policy{};
    thread_local no_numa_policy g_no_numa_policy{};
    if (::numa_available() < 0) {
      return &g_no_numa_policy;
    }
    return &g_default_numa_policy;
  }

  template <class T>
  struct numa_allocator {
    using pointer = T*;
    using const_pointer = const T*;
    using value_type = T;

    explicit numa_allocator(int node) noexcept : node_(node) {}

    template <class U>
    explicit numa_allocator(const numa_allocator<U>& other) noexcept : node_(other.node_) {}

    int node_;

    void* do_allocate(std::size_t n) {
      return ::numa_alloc_onnode(n, node_);
    }

    void do_deallocate(void* p, std::size_t n) {
      ::numa_free(p, n);
    }

    T* allocate(std::size_t n) {
      return static_cast<T*>(do_allocate(n * sizeof(T)));
    }

    void deallocate(T* p, std::size_t n) {
      do_deallocate(p, n * sizeof(T));
    }

    friend bool operator==(const numa_allocator&, const numa_allocator&) noexcept = default;
  };
}
#else
namespace exec {
  using default_numa_policy = no_numa_policy;

  inline numa_policy* get_numa_policy() noexcept {
    thread_local default_numa_policy g_default_numa_policy{};
    return &g_default_numa_policy;
  }

  template <class T>
  struct numa_allocator {
    using pointer = T*;
    using const_pointer = const T*;
    using value_type = T;

    explicit numa_allocator(int) noexcept {}

    template <class U>
    explicit numa_allocator(const numa_allocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
      std::allocator<T> alloc{};
      return alloc.allocate(n);
    }

    void deallocate(T* p, std::size_t n) {
      std::allocator<T> alloc{};
      alloc.deallocate(p, n);
    }

    friend bool operator==(const numa_allocator&, const numa_allocator&) noexcept = default;
  };
}
#endif