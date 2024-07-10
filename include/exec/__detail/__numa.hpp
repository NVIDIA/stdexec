/*
 * Copyright (c) 2021-2024 NVIDIA Corporation
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
    numa_policy() = default;
    virtual ~numa_policy() = default;

    virtual auto num_nodes() -> std::size_t = 0;
    virtual auto num_cpus(int node) -> std::size_t = 0;
    virtual auto bind_to_node(int node) -> int = 0;
    virtual auto thread_index_to_node(std::size_t index) -> int = 0;
  };

  struct no_numa_policy : numa_policy {
    no_numa_policy() = default;

    auto num_nodes() -> std::size_t override {
      return 1;
    }

    auto num_cpus(int) -> std::size_t override {
      return std::thread::hardware_concurrency();
    }

    auto bind_to_node(int) -> int override {
      return 0;
    }

    auto thread_index_to_node(std::size_t) -> int override {
      return 0;
    }
  };
} // namespace exec

#if STDEXEC_ENABLE_NUMA
#  include <numa.h>

namespace exec {
  inline std::size_t _get_numa_num_cpus(int node) {
    struct ::bitmask* cpus = ::numa_allocate_cpumask();
    if (!cpus) {
      return 0;
    }
    scope_guard sg{[&]() noexcept {
      ::numa_free_cpumask(cpus);
    }};
    int rc = ::numa_node_to_cpus(node, cpus);
    if (rc < 0) {
      return 0;
    }
    std::size_t num_cpus = ::numa_bitmask_weight(cpus);
    return num_cpus;
  }

  struct _node_to_thread_index {
    static const std::vector<int>& get() noexcept {
      // This leaks one memory block at shutdown, but it's fine. Clang's and gcc's leak
      // sanitizer do not report it.
      static const stdexec::__indestructible<std::vector<int>> g_node_to_thread_index{[] {
        std::vector<int> index(::numa_num_task_nodes());
        for (std::size_t node = 0, total_cpus = 0; node < index.size(); ++node) {
          total_cpus += exec::_get_numa_num_cpus(static_cast<int>(node));
          index[node] = static_cast<int>(total_cpus);
        }
        return index;
      }()};
      return g_node_to_thread_index.get();
    }
  };

  struct default_numa_policy : numa_policy {
    default_numa_policy() = default;

    std::size_t num_nodes() override {
      return _node_to_thread_index::get().size();
    }

    std::size_t num_cpus(int node) override {
      return exec::_get_numa_num_cpus(node);
    }

    int bind_to_node(int node) override {
      struct ::bitmask* nodes = ::numa_allocate_nodemask();
      if (!nodes) {
        return -1;
      }
      scope_guard sg{[&]() noexcept {
        ::numa_free_nodemask(nodes);
      }};
      ::numa_bitmask_setbit(nodes, node);
      ::numa_bind(nodes);
      return 0;
    }

    int thread_index_to_node(std::size_t idx) override {
      const auto& node_to_thread_index = _node_to_thread_index::get();
      int index = static_cast<int>(idx) % node_to_thread_index.back();
      auto it = std::upper_bound(node_to_thread_index.begin(), node_to_thread_index.end(), index);
      STDEXEC_ASSERT(it != node_to_thread_index.end());
      return static_cast<int>(std::distance(node_to_thread_index.begin(), it));
    }
  };

  inline numa_policy* get_numa_policy() noexcept {
    thread_local stdexec::__indestructible<default_numa_policy> g_default_numa_policy{};
    thread_local stdexec::__indestructible<no_numa_policy> g_no_numa_policy{};
    if (::numa_available() < 0) {
      return &g_no_numa_policy.get();
    }
    return &g_default_numa_policy.get();
  }

  template <class T>
  struct numa_allocator {
    using pointer = T*;
    using const_pointer = const T*;
    using value_type = T;

    explicit numa_allocator(int node) noexcept
      : node_(node) {
    }

    template <class U>
    explicit numa_allocator(const numa_allocator<U>& other) noexcept
      : node_(other.node_) {
    }

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

  class nodemask {
    static nodemask make_any() noexcept {
      nodemask mask;
      ::copy_bitmask_to_nodemask(::numa_all_nodes_ptr, &mask.mask_);
      return mask;
    }

   public:
    nodemask() noexcept
      : mask_{} {
      ::copy_bitmask_to_nodemask(::numa_no_nodes_ptr, &mask_);
    }

    static const nodemask& any() noexcept {
      static const stdexec::__indestructible<nodemask> mask{make_any()};
      return mask.get();
    }

    bool operator[](std::size_t nodemask) const noexcept {
      ::bitmask mask;
      mask.maskp = const_cast<unsigned long*>(mask_.n);
      mask.size = sizeof(nodemask_t);
      return ::numa_bitmask_isbitset(&mask, static_cast<unsigned int>(nodemask));
    }

    void set(std::size_t nodemask) noexcept {
      ::bitmask mask;
      mask.maskp = const_cast<unsigned long*>(mask_.n);
      mask.size = sizeof(nodemask_t);
      ::numa_bitmask_setbit(&mask, static_cast<unsigned int>(nodemask));
    }

    bool get(std::size_t nodemask) const noexcept {
      ::bitmask mask;
      mask.maskp = const_cast<unsigned long*>(mask_.n);
      mask.size = sizeof(nodemask_t);
      return ::numa_bitmask_isbitset(&mask, static_cast<unsigned int>(nodemask));
    }

    friend bool operator==(const nodemask& lhs, const nodemask& rhs) noexcept {
      ::bitmask lhs_mask;
      ::bitmask rhs_mask;
      lhs_mask.maskp = const_cast<unsigned long*>(lhs.mask_.n);
      lhs_mask.size = sizeof(nodemask_t);
      rhs_mask.maskp = const_cast<unsigned long*>(rhs.mask_.n);
      rhs_mask.size = sizeof(nodemask_t);
      return ::numa_bitmask_equal(&lhs_mask, &rhs_mask);
    }

   private:
    ::nodemask_t mask_;
  };
} // namespace exec
#else
namespace exec {
  using default_numa_policy = no_numa_policy;

  inline auto get_numa_policy() noexcept -> numa_policy* {
    thread_local stdexec::__indestructible<default_numa_policy> g_default_numa_policy{};
    return &g_default_numa_policy.get();
  }

  template <class T>
  struct numa_allocator {
    using pointer = T*;
    using const_pointer = const T*;
    using value_type = T;

    explicit numa_allocator(int) noexcept {
    }

    template <class U>
    explicit numa_allocator(const numa_allocator<U>&) noexcept {
    }

    auto allocate(std::size_t n) -> T* {
      std::allocator<T> alloc{};
      return alloc.allocate(n);
    }

    void deallocate(T* p, std::size_t n) {
      std::allocator<T> alloc{};
      alloc.deallocate(p, n);
    }

    friend auto operator==(const numa_allocator&, const numa_allocator&) noexcept -> bool = default;
  };

  class nodemask {
    static auto make_any() noexcept -> nodemask {
      nodemask mask;
      mask.mask_ = true;
      return mask;
    }

   public:
    nodemask() noexcept = default;

    static auto any() noexcept -> const nodemask& {
      static nodemask mask = make_any();
      return mask;
    }

    auto operator[](std::size_t nodemask) const noexcept -> bool {
      return mask_ && nodemask == 0;
    }

    void set(std::size_t nodemask) noexcept {
      mask_ |= nodemask == 0;
    }

    friend auto operator==(const nodemask& lhs, const nodemask& rhs) noexcept -> bool {
      return lhs.mask_ == rhs.mask_;
    }

   private:
    bool mask_{false};
  };
} // namespace exec
#endif