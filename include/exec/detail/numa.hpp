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

#include "../../stdexec/__detail/__any.hpp"
#include "../../stdexec/__detail/__config.hpp"
#include "../scope.hpp"  // IWYU pragma: keep

#include <cstddef>
#include <thread>
#include <vector>

// Work around a bug in the NVHPC compilers prior to version 24.3
#if STDEXEC_NVHPC() && STDEXEC_NVHPC_VERSION < 24'03
#  define STDEXEC_NUMA_VTABLE_INLINE
#else
#  define STDEXEC_NUMA_VTABLE_INLINE inline
#endif

namespace experimental::execution
{
  namespace _numa
  {
    // NOLINTBEGIN(modernize-use-override)
    template <class Base>
    struct _ipolicy
      : STDEXEC::__any::interface<_ipolicy,
                                  Base,
                                  STDEXEC::__any::__extends<STDEXEC::__any::__icopyable>>
    {
      using _ipolicy::interface::interface;

      [[nodiscard]]
      constexpr virtual auto num_nodes() const noexcept -> std::size_t
      {
        return STDEXEC::__any::__value(*this).num_nodes();
      }

      [[nodiscard]]
      constexpr virtual auto num_cpus(int node) const noexcept -> std::size_t
      {
        return STDEXEC::__any::__value(*this).num_cpus(node);
      }

      // NOLINTNEXTLINE(modernize-use-nodiscard)
      constexpr virtual auto bind_to_node(int node) const noexcept -> int
      {
        return STDEXEC::__any::__value(*this).bind_to_node(node);
      }

      [[nodiscard]]
      constexpr virtual auto thread_index_to_node(std::size_t index) const noexcept -> int
      {
        return STDEXEC::__any::__value(*this).thread_index_to_node(index);
      }
    };
    // NOLINTEND(modernize-use-override)
  }  // namespace _numa

  struct numa_policy final : STDEXEC::__any::__any<exec::_numa::_ipolicy>
  {
    using numa_policy::__any::__any;
  };

  struct no_numa_policy
  {
    [[nodiscard]]
    constexpr auto num_nodes() const noexcept -> std::size_t
    {
      return 1;
    }

    [[nodiscard]]
    auto num_cpus(int) const noexcept -> std::size_t
    {
      return std::thread::hardware_concurrency();
    }

    constexpr auto bind_to_node(int) const noexcept -> int  // NOLINT(modernize-use-nodiscard)
    {
      return 0;
    }

    [[nodiscard]]
    constexpr auto thread_index_to_node(std::size_t) const noexcept -> int
    {
      return 0;
    }
  };
}  // namespace experimental::execution

namespace exec = experimental::execution;

#if STDEXEC_ENABLE_NUMA
#  include <numa.h>

namespace experimental::execution
{
  inline std::size_t _get_numa_num_cpus(int node)
  {
    struct ::bitmask *cpus = ::numa_allocate_cpumask();
    if (!cpus)
    {
      return 0;
    }
    scope_guard sg{[&]() noexcept { ::numa_free_cpumask(cpus); }};
    int         rc = ::numa_node_to_cpus(node, cpus);
    if (rc < 0)
    {
      return 0;
    }
    std::size_t num_cpus = ::numa_bitmask_weight(cpus);
    return num_cpus;
  }

  struct _node_to_thread_index
  {
    static std::vector<int> const &get() noexcept
    {
      // This leaks one memory block at shutdown, but it's fine. Clang's and gcc's leak
      // sanitizer do not report it.
      static STDEXEC::__indestructible<std::vector<int>> const g_node_to_thread_index{
        []
        {
          std::vector<int> index(::numa_num_task_nodes());
          for (std::size_t node = 0, total_cpus = 0; node < index.size(); ++node)
          {
            total_cpus += exec::_get_numa_num_cpus(static_cast<int>(node));
            index[node] = static_cast<int>(total_cpus);
          }
          return index;
        }()};
      return g_node_to_thread_index.get();
    }
  };

  struct default_numa_policy
  {
    [[nodiscard]]
    std::size_t num_nodes() const noexcept
    {
      return _node_to_thread_index::get().size();
    }

    [[nodiscard]]
    std::size_t num_cpus(int node) const noexcept
    {
      return exec::_get_numa_num_cpus(node);
    }

    int bind_to_node(int node) const noexcept  // NOLINT(modernize-use-nodiscard)
    {
      struct ::bitmask *nodes = ::numa_allocate_nodemask();
      if (!nodes)
      {
        return -1;
      }
      scope_guard sg{[&]() noexcept { ::numa_free_nodemask(nodes); }};
      ::numa_bitmask_setbit(nodes, node);
      ::numa_bind(nodes);
      return 0;
    }

    [[nodiscard]]
    int thread_index_to_node(std::size_t idx) const noexcept
    {
      auto const &node_to_thread_index = _node_to_thread_index::get();
      int         index                = static_cast<int>(idx) % node_to_thread_index.back();
      auto it = std::upper_bound(node_to_thread_index.begin(), node_to_thread_index.end(), index);
      STDEXEC_ASSERT(it != node_to_thread_index.end());
      return static_cast<int>(std::distance(node_to_thread_index.begin(), it));
    }
  };

  inline numa_policy get_numa_policy() noexcept
  {
    if (::numa_available() < 0)
    {
      return numa_policy{no_numa_policy{}};
    }
    return numa_policy{default_numa_policy{}};
  }

  template <class T>
  struct numa_allocator
  {
    using pointer       = T *;
    using const_pointer = T const *;
    using value_type    = T;

    explicit numa_allocator(int node) noexcept
      : node_(node)
    {}

    template <class U>
    explicit numa_allocator(numa_allocator<U> const &other) noexcept
      : node_(other.node_)
    {}

    void *do_allocate(std::size_t n)
    {
      return ::numa_alloc_onnode(n, node_);
    }

    void do_deallocate(void *p, std::size_t n)
    {
      ::numa_free(p, n);
    }

    T *allocate(std::size_t n)
    {
      return static_cast<T *>(do_allocate(n * sizeof(T)));
    }

    void deallocate(T *p, std::size_t n)
    {
      do_deallocate(p, n * sizeof(T));
    }

    friend bool operator==(numa_allocator const &, numa_allocator const &) noexcept = default;

   private:
    template <class>
    friend struct numa_allocator;

    int node_;
  };

  class nodemask
  {
    static nodemask make_any() noexcept
    {
      nodemask mask;
      ::copy_bitmask_to_nodemask(::numa_all_nodes_ptr, &mask.mask_);
      return mask;
    }

   public:
    nodemask() noexcept
    {
      ::copy_bitmask_to_nodemask(::numa_no_nodes_ptr, &mask_);
    }

    static nodemask const &any() noexcept
    {
      static STDEXEC::__indestructible<nodemask> const mask{make_any()};
      return mask.get();
    }

    bool operator[](std::size_t nodemask) const noexcept
    {
      ::bitmask mask;
      mask.maskp = const_cast<unsigned long *>(mask_.n);
      mask.size  = sizeof(nodemask_t);
      return ::numa_bitmask_isbitset(&mask, static_cast<unsigned int>(nodemask));
    }

    void set(std::size_t nodemask) noexcept
    {
      ::bitmask mask;
      mask.maskp = const_cast<unsigned long *>(mask_.n);
      mask.size  = sizeof(nodemask_t);
      ::numa_bitmask_setbit(&mask, static_cast<unsigned int>(nodemask));
    }

    bool get(std::size_t nodemask) const noexcept
    {
      ::bitmask mask;
      mask.maskp = const_cast<unsigned long *>(mask_.n);
      mask.size  = sizeof(nodemask_t);
      return ::numa_bitmask_isbitset(&mask, static_cast<unsigned int>(nodemask));
    }

    friend bool operator==(nodemask const &lhs, nodemask const &rhs) noexcept
    {
      ::bitmask lhs_mask;
      ::bitmask rhs_mask;
      lhs_mask.maskp = const_cast<unsigned long *>(lhs.mask_.n);
      lhs_mask.size  = sizeof(nodemask_t);
      rhs_mask.maskp = const_cast<unsigned long *>(rhs.mask_.n);
      rhs_mask.size  = sizeof(nodemask_t);
      return ::numa_bitmask_equal(&lhs_mask, &rhs_mask);
    }

   private:
    ::nodemask_t mask_{};
  };
}  // namespace experimental::execution

namespace exec = experimental::execution;

#else

namespace experimental::execution
{
  using default_numa_policy = no_numa_policy;

  inline auto get_numa_policy() noexcept -> numa_policy
  {
    return numa_policy{default_numa_policy{}};
  }

  template <class T>
  struct numa_allocator : std::allocator<T>
  {
    numa_allocator() = default;

    template <STDEXEC::__not_same_as<T> U>
    numa_allocator(numa_allocator<U> const &) noexcept
    {}
  };

  class nodemask
  {
    static auto make_any() noexcept -> nodemask
    {
      nodemask mask;
      mask.mask_ = true;
      return mask;
    }

   public:
    nodemask() noexcept = default;

    static auto any() noexcept -> nodemask const &
    {
      static nodemask mask = make_any();
      return mask;
    }

    auto operator[](std::size_t nodemask) const noexcept -> bool
    {
      return mask_ && nodemask == 0;
    }

    void set(std::size_t nodemask) noexcept
    {
      mask_ |= nodemask == 0;
    }

    friend auto operator==(nodemask const &lhs, nodemask const &rhs) noexcept -> bool
    {
      return lhs.mask_ == rhs.mask_;
    }

   private:
    bool mask_{false};
  };
}  // namespace experimental::execution

namespace exec = experimental::execution;

#endif
