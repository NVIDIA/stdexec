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
#include "../scope.hpp" // IWYU pragma: keep

#include <algorithm> // IWYU pragma: keep
#include <cstddef>
#include <memory>
#include <new> // IWYU pragma: keep
#include <thread>
#include <utility>

// Work around a bug in the NVHPC compilers prior to version 24.3
#if STDEXEC_NVHPC() && STDEXEC_NVHPC_VERSION < 24'03
#  define STDEXEC_NUMA_VTABLE_INLINE
#else
#  define STDEXEC_NUMA_VTABLE_INLINE inline
#endif

namespace exec {
  namespace _numa {
    using small = void* [1];

    template <class T>
    using _is_small = stdexec::__mbool<sizeof(T) <= sizeof(small)>;

    union _storage {
      _storage() noexcept = default;

      template <stdexec::__not_decays_to<_storage> Ty>
      explicit _storage(Ty&& value)
        : ptr{new stdexec::__decay_t<Ty>{static_cast<Ty&&>(value)}} {
      }

      template <stdexec::__not_decays_to<_storage> Ty>
        requires(_is_small<stdexec::__decay_t<Ty>>::value)
      explicit _storage(Ty&& value) noexcept(stdexec::__nothrow_decay_copyable<Ty>)
        : buf{} {
        ::new (static_cast<void*>(buf)) stdexec::__decay_t<Ty>{static_cast<Ty&&>(value)};
      }

      void* ptr{};
      char buf[sizeof(small)];
    };

    struct _vtable {
      auto (*move)(_storage*, _storage*) noexcept -> void;
      auto (*copy)(_storage*, const _storage*) -> void;
      auto (*destroy)(_storage*) noexcept -> void;
      auto (*num_nodes)(const _storage*) noexcept -> std::size_t;
      auto (*num_cpus)(const _storage*, int) noexcept -> std::size_t;
      auto (*bind_to_node)(const _storage*, int) noexcept -> int;
      auto (*thread_index_to_node)(const _storage*, std::size_t) noexcept -> int;
    };

    template <class T>
    struct _vtable_for {
      // move
      static auto _move(_storage* self, _storage* other) noexcept -> void {
        if constexpr (!_is_small<T>::value) {
          self->ptr = std::exchange(other->ptr, nullptr);
        } else {
          ::new (static_cast<void*>(self->buf))
            T{static_cast<T&&>(*reinterpret_cast<T*>(other->buf))};
        }
      }

      // copy
      static auto _copy(_storage* self, const _storage* other) noexcept -> void {
        if constexpr (!_is_small<T>::value) {
          self->ptr = new T{*static_cast<const T*>(other->ptr)};
        } else {
          ::new (static_cast<void*>(self->buf)) T{*reinterpret_cast<const T*>(other->buf)};
        }
      }

      // destroy
      static auto _destroy(_storage* self) noexcept -> void {
        if constexpr (!_is_small<T>::value) {
          delete static_cast<T*>(self->ptr);
        } else {
          std::destroy_at(reinterpret_cast<T*>(self->buf));
        }
      }

      // num_nodes
      static auto _num_nodes(const _storage* self) noexcept -> std::size_t {
        if constexpr (!_is_small<T>::value) {
          return static_cast<const T*>(self->ptr)->num_nodes();
        } else {
          return reinterpret_cast<const T*>(self->buf)->num_nodes();
        }
      }

      // num_cpus
      static auto _num_cpus(const _storage* self, int node) noexcept -> std::size_t {
        if constexpr (!_is_small<T>::value) {
          return static_cast<const T*>(self->ptr)->num_cpus(node);
        } else {
          return reinterpret_cast<const T*>(self->buf)->num_cpus(node);
        }
      }

      // bind_to_node
      static auto _bind_to_node(const _storage* self, int node) noexcept -> int {
        if constexpr (!_is_small<T>::value) {
          return static_cast<const T*>(self->ptr)->bind_to_node(node);
        } else {
          return reinterpret_cast<const T*>(self->buf)->bind_to_node(node);
        }
      }

      // thread_index_to_node
      static auto _thread_index_to_node(const _storage* self, std::size_t index) noexcept -> int {
        if constexpr (!_is_small<T>::value) {
          return static_cast<const T*>(self->ptr)->thread_index_to_node(index);
        } else {
          return reinterpret_cast<const T*>(self->buf)->thread_index_to_node(index);
        }
      }
    };

    template <class NumaPolicy>
    STDEXEC_NUMA_VTABLE_INLINE constexpr _vtable _vtable_for_v = {
      .move = _vtable_for<NumaPolicy>::_move,
      .copy = _vtable_for<NumaPolicy>::_copy,
      .destroy = _vtable_for<NumaPolicy>::_destroy,
      .num_nodes = _vtable_for<NumaPolicy>::_num_nodes,
      .num_cpus = _vtable_for<NumaPolicy>::_num_cpus,
      .bind_to_node = _vtable_for<NumaPolicy>::_bind_to_node,
      .thread_index_to_node = _vtable_for<NumaPolicy>::_thread_index_to_node};
  } // namespace _numa

  struct numa_policy {
   private:
    const _numa::_vtable* vtable_;
    _numa::_storage storage_;

   public:
    template <stdexec::__not_decays_to<numa_policy> NumaPolicy>
    numa_policy(NumaPolicy&& policy)
      : vtable_(&_numa::_vtable_for_v<stdexec::__decay_t<NumaPolicy>>)
      , storage_(static_cast<NumaPolicy&&>(policy)) {
    }

    numa_policy(numa_policy&& other) noexcept
      : vtable_(other.vtable_)
      , storage_{} {
      vtable_->move(&storage_, &other.storage_);
    }

    numa_policy(const numa_policy& other)
      : vtable_(other.vtable_)
      , storage_{} {
      vtable_->copy(&storage_, &other.storage_);
    }

    ~numa_policy() {
      vtable_->destroy(&storage_);
    }

    [[nodiscard]]
    auto num_nodes() const noexcept -> std::size_t {
      return vtable_->num_nodes(&storage_);
    }

    [[nodiscard]]
    auto num_cpus(int node) const noexcept -> std::size_t {
      return vtable_->num_cpus(&storage_, node);
    }

    auto bind_to_node(int node) const noexcept -> int { // NOLINT(modernize-use-nodiscard)
      return vtable_->bind_to_node(&storage_, node);
    }

    [[nodiscard]]
    auto thread_index_to_node(std::size_t index) const noexcept -> int {
      return vtable_->thread_index_to_node(&storage_, index);
    }
  };

  struct no_numa_policy {
    [[nodiscard]]
    auto num_nodes() const noexcept -> std::size_t {
      return 1;
    }

    [[nodiscard]]
    auto num_cpus(int) const noexcept -> std::size_t {
      return std::thread::hardware_concurrency();
    }

    auto bind_to_node(int) const noexcept -> int { // NOLINT(modernize-use-nodiscard)
      return 0;
    }

    [[nodiscard]]
    auto thread_index_to_node(std::size_t) const noexcept -> int {
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
    scope_guard sg{[&]() noexcept { ::numa_free_cpumask(cpus); }};
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

  struct default_numa_policy {
    std::size_t num_nodes() const noexcept {
      return _node_to_thread_index::get().size();
    }

    std::size_t num_cpus(int node) const noexcept {
      return exec::_get_numa_num_cpus(node);
    }

    int bind_to_node(int node) const noexcept {
      struct ::bitmask* nodes = ::numa_allocate_nodemask();
      if (!nodes) {
        return -1;
      }
      scope_guard sg{[&]() noexcept { ::numa_free_nodemask(nodes); }};
      ::numa_bitmask_setbit(nodes, node);
      ::numa_bind(nodes);
      return 0;
    }

    int thread_index_to_node(std::size_t idx) const noexcept {
      const auto& node_to_thread_index = _node_to_thread_index::get();
      int index = static_cast<int>(idx) % node_to_thread_index.back();
      auto it = std::upper_bound(node_to_thread_index.begin(), node_to_thread_index.end(), index);
      STDEXEC_ASSERT(it != node_to_thread_index.end());
      return static_cast<int>(std::distance(node_to_thread_index.begin(), it));
    }
  };

  inline numa_policy get_numa_policy() noexcept {
    if (::numa_available() < 0) {
      return numa_policy{no_numa_policy{}};
    }
    return numa_policy{default_numa_policy{}};
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

  inline auto get_numa_policy() noexcept -> numa_policy {
    return numa_policy{default_numa_policy{}};
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