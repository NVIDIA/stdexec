#include "../../stdexec/__detail/__config.hpp"
#include "../scope.hpp"

#include <cstddef>
#include <thread>

namespace exec {
  struct numa_policy {
    virtual std::size_t num_nodes() = 0;
    virtual std::size_t num_cpus(int node) = 0;
    virtual int bind_to_node(int node) = 0;
  };

  class no_numa_policy : public numa_policy {
   public:
    no_numa_policy() noexcept;
    std::size_t num_nodes() override { return 1; }
    std::size_t num_cpus(int node) override { return std::thread::hardware_concurrency(); }
    int bind_to_node(int node) override { return 0; }
  };
}

#if __has_include(<numa.h>)
#include <numa.h>
namespace exec {
  struct numa_policy_impl : numa_policy {
    numa_policy_impl() noexcept = default;

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
  };

  inline numa_policy* get_numa_policy() noexcept {
    thread_local numa_policy_impl g_default_numa_policy{};
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
  inline numa_policy* get_numa_policy() noexcept {
    thread_local no_numa_policy g_default_numa_policy{};
    return &g_default_numa_policy;
  }
}
#endif