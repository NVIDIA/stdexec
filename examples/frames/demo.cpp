/*
 * Exploration: LIFO "automatic storage" for asynchronous operations.
 *
 * Each section below runs a small sender under `with_automatic_storage`,
 * which owns a chain of slabs and makes the chain's storage available to
 * the work through the environment. `indirect` stands in for anything that
 * must allocate its child's operation state (type erasure, for example):
 * it pushes the child's operation state onto the automatic storage it is
 * given and gives its child the storage "after" it, the way a function call
 * pushes a frame onto a thread's stack.
 *
 * Nothing in stdexec is changed: `then`, `let_value` and `when_all` forward
 * the storage query as they forward any other forwarding query.
 */

#include "automatic_storage.hpp"

#include <stdexec/execution.hpp>

#include <cstdio>
#include <memory_resource>

namespace ex = stdexec;
using namespace frames;

namespace
{
  // Counts the slabs taken from the system.
  struct counting_resource : std::pmr::memory_resource
  {
    int allocations   = 0;
    int deallocations = 0;

   private:
    auto do_allocate(std::size_t bytes, std::size_t align) -> void* override
    {
      ++allocations;
      void* p = std::pmr::new_delete_resource()->allocate(bytes, align);
      std::printf("    [slab %p, %zu bytes]\n", p, bytes);
      return p;
    }

    void do_deallocate(void* p, std::size_t bytes, std::size_t align) override
    {
      ++deallocations;
      std::pmr::new_delete_resource()->deallocate(p, bytes, align);
    }

    [[nodiscard]]
    auto do_is_equal(memory_resource const & other) const noexcept -> bool override
    {
      return this == &other;
    }
  };

  void section(char const * title)
  {
    std::printf("\n== %s\n", title);
  }

  template <class Sndr>
  auto run(Sndr&& sndr, counting_resource& mr, std::size_t slab_size = 512)
  {
    auto result = ex::sync_wait(with_automatic_storage(static_cast<Sndr&&>(sndr), &mr, slab_size));
    std::printf("    slabs: %d allocated, %d freed\n", mr.allocations, mr.deallocations);
    return result;
  }
}  // namespace

int main()
{
  {
    section("1. Sequential phases reuse the same storage");
    // Each phase's frame is pushed from the same storage, after the previous
    // phase's operation state has been destroyed, so all three land at the
    // same address: the asynchronous analogue of calling three functions in
    // a row.
    counting_resource mr;
    auto [v] = run(indirect(ex::just(1), "phase 1")
                     | ex::let_value([](int i) { return indirect(ex::just(i + 1), "phase 2"); })
                     | ex::let_value([](int i) { return indirect(ex::just(i + 1), "phase 3"); }),
                   mr)
                 .value();
    std::printf("    result: %d\n", v);
  }

  {
    section("2. Nested frames stack up, spill into a new slab, and reuse it");
    // Nesting pushes each frame after its parent's. With small slabs the
    // third frame no longer fits and spills into a second slab. The second
    // phase repeats the same nesting and reuses both slabs: no new slab is
    // allocated.
    counting_resource mr;
    auto              nest = [](char const * a, char const * b, char const * c)
    {
      return indirect(indirect(indirect(ex::just(), c), b), a);
    };
    run(nest("1: outer", "1: middle", "1: inner")
          | ex::let_value([=] { return nest("2: outer", "2: middle", "2: inner"); }),
        mr,
        /*slab_size=*/160);
  }

  {
    section("3. when_all needs a fork: each child gets its own strand");
    // when_all forwards its environment to both children. If both pushed
    // from the same storage they would overlap while running concurrently
    // (build with -DFRAMES_SHOW_OVERLAP to see the addresses collide).
    // `strand` gives its child a new chain from the same slab source, so the
    // children's frames are disjoint; one child could also keep the parent's
    // storage, as an optimization.
    counting_resource mr;
#if defined(FRAMES_SHOW_OVERLAP)
    // Undefined behaviour: the second frame is constructed over the first.
    auto r = run(ex::when_all(indirect(ex::just(1), "left"), indirect(ex::just(2), "right")), mr);
#else
    auto r = run(ex::when_all(strand(indirect(ex::just(1), "left")),
                              strand(indirect(ex::just(2), "right"))),
                 mr);
#endif
    auto [a, b] = r.value();
    std::printf("    result: %d %d\n", a, b);
  }

  {
    section("4. reserve_async_stack: inline room first, the strand's slabs after");
    // The reserved buffer lives in reserve_async_stack's own operation
    // state. Frames that fit are pushed into it; the third doesn't, and
    // falls back to the storage reserve_async_stack was given.
    counting_resource mr;
    run(reserve_async_stack<128>(
          indirect(indirect(indirect(ex::just(), "inner"), "middle"), "outer")),
        mr);
  }

  {
    section("5. Without automatic storage, indirect falls back to the heap");
    ex::sync_wait(indirect(ex::just(), "no storage"));
  }
}
