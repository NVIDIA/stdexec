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
#include "typestate.hpp"

#include <exec/variant_sender.hpp>
#include <stdexec/execution.hpp>

#include <array>
#include <cstdio>
#include <memory_resource>
#include <stdexcept>
#include <string>

namespace ex = stdexec;
namespace ts = frames::typestate;
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

  // A receiver that ignores its completions, for measuring operation states.
  // It's only ever named in unevaluated contexts.
  struct sink
  {
    using receiver_concept = ex::receiver_t;

    template <class... As>
    [[maybe_unused]]
    void set_value(As&&...) && noexcept
    {}

    template <class E>
    [[maybe_unused]]
    void set_error(E&&) && noexcept
    {}

    void set_stopped() && noexcept {}
  };

  template <class Sndr>
  constexpr auto opstate_size = sizeof(ex::connect_result_t<Sndr, sink>);

  struct big
  {
    std::array<char, 256> bytes{};
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

  // The remaining sections use frames::typestate's just, then, let_value and
  // sequence, whose operation states hold one phase at a time.

  {
    section("6. Typestate algorithms");
    auto [v] = ex::sync_wait(ts::sequence(ts::then(ts::just(20), [](int i) { return i + 1; }),
                                          ts::let_value(ts::just(std::string("answer")),
                                                        [](std::string& s)
                                                        { return ts::just(s.size() * 2 + 30); })))
                 .value();
    std::printf("    result: %zu\n", v);
  }

  {
    section("7. let_value picks its second phase by the child's value type");
    // The child completes with either an int or a string, so let_value has
    // two possible second phases sharing one block of storage. Nothing
    // records which one is live: each completes through its own receiver
    // type.
    using either  = exec::variant_sender<decltype(ex::just(0)), decltype(ex::just(std::string()))>;
    auto describe = [](bool text)
    {
      either child = text ? either(ex::just(std::string("forty-two"))) : either(ex::just(42));
      return ts::let_value(std::move(child),
                           [](auto& value)
                           {
                             if constexpr (std::is_same_v<std::decay_t<decltype(value)>, int>)
                             {
                               return ts::just(std::string("an int: ") + std::to_string(value));
                             }
                             else
                             {
                               return ts::just("a string: " + value);
                             }
                           });
    };
    std::printf("    %s\n", std::get<0>(ex::sync_wait(describe(false)).value()).c_str());
    std::printf("    %s\n", std::get<0>(ex::sync_wait(describe(true)).value()).c_str());

    // Errors and stopped pass through the first phase untouched.
    auto outcome = [](auto child)
    {
      return ts::let_value(std::move(child), [](auto&&...) { return ts::just(std::string()); })
           | ex::upon_error([](auto) { return std::string("error"); })
           | ex::upon_stopped([] { return std::string("stopped"); });
    };
    std::printf("    failed child: %s\n",
                std::get<0>(ex::sync_wait(outcome(ex::just_error(42))).value()).c_str());
    std::printf("    stopped child: %s\n",
                std::get<0>(ex::sync_wait(outcome(ex::just_stopped())).value()).c_str());
  }

  {
    section("8. Typestate phases reuse automatic storage too");
    // let_value's two phases share one block of its operation state, and the
    // frames their children push land at the same address.
    counting_resource mr;
    auto [v] = run(ts::let_value(indirect(ts::just(1), "phase 1"),
                                 [](int i) { return indirect(ts::just(i + 1), "phase 2"); }),
                   mr)
                 .value();
    std::printf("    result: %d\n", v);
  }

  {
    section("9. Operation-state sizes: stdexec vs typestate");
    // stdexec's let_value keeps the values it passes to fn apart from the
    // storage it shares between its two children. The typestate version
    // keeps them in the second phase, which reuses the first phase's bytes.
    auto takes_big = [](big&)
    {
      return ex::just();
    };
    auto takes_big_ts = [](big&)
    {
      return ts::just();
    };
    std::printf("    let_value(just(big), fn)       stdexec %4zu  typestate %4zu\n",
                opstate_size<decltype(ex::just(big{}) | ex::let_value(takes_big))>,
                opstate_size<decltype(ts::let_value(ts::just(big{}), takes_big_ts))>);

    // stdexec has no sequence; compare let_value with a continuation that
    // makes the second sender. No saving here: the typestate sequence holds
    // the second sender, with its big value, in the first phase until it's
    // needed, while stdexec's continuation makes it on demand.
    std::printf(
      "    sequence(just(big), just(big)) stdexec %4zu  typestate %4zu\n",
      opstate_size<decltype(ex::just(big{}) | ex::let_value([](big&) { return ex::just(big{}); }))>,
      opstate_size<decltype(ts::sequence(ts::just(big{}), ts::just(big{})))>);

    // The typestate version pays for its one bit of state (was it started?).
    auto plus_one = [](int i)
    {
      return i + 1;
    };
    std::printf("    then(just(1), fn)              stdexec %4zu  typestate %4zu\n",
                opstate_size<decltype(ex::just(1) | ex::then(plus_one))>,
                opstate_size<decltype(ts::then(ts::just(1), plus_one))>);
  }

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  {
    section("10. Exceptions end the live phase and complete with set_error");
    try
    {
      ex::sync_wait(ts::then(ts::just(), []() -> int { throw std::runtime_error("boom"); }));
    }
    catch (std::runtime_error const & e)
    {
      std::printf("    caught: %s\n", e.what());
    }
  }
#endif
}
