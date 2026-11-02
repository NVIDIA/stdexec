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
#include "just.hpp"
#include "let_value.hpp"
#include "repeat_n.hpp"
#include "sequence.hpp"
#include "then.hpp"

#include <exec/repeat_n.hpp>
#include <exec/sequence.hpp>
#include <exec/variant_sender.hpp>
#include <stdexec/execution.hpp>

#include <array>
#include <cstdio>
#include <memory>
#include <memory_resource>
#include <stdexcept>
#include <string>
#include <utility>

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

  // Counts its copies and moves.
  struct counted
  {
    inline static int copies = 0;
    inline static int moves  = 0;

    counted() = default;

    counted(counted const &)
    {
      ++copies;
    }

    counted(counted&&) noexcept
    {
      ++moves;
    }
  };

  // Says when the last object holding its value is destroyed.
  struct traced
  {
    traced() = default;

    traced(traced&& other) noexcept
      : live(std::exchange(other.live, false))
    {}

    ~traced()
    {
      if (live)
      {
        std::printf(" values destroyed;");
      }
    }

    bool live = true;
  };

  // Moves and copies while running `sndr` under sync_wait, as "m (c)".
  template <class Sndr>
  auto count(Sndr sndr) -> std::string
  {
    if constexpr (std::is_null_pointer_v<Sndr>)
    {
      return "";
    }
    else
    {
      counted::copies = counted::moves = 0;
      ex::sync_wait(std::move(sndr));
      return std::to_string(counted::moves) + " (" + std::to_string(counted::copies) + ")";
    }
  }

  // The typestate algorithms under one ending and one entry.
  template <class Ending, class Entry>
  struct policy
  {
    template <class... Vs>
    static auto just(Vs&&... vs)
    {
      return ts::just<Ending, Entry>(static_cast<Vs&&>(vs)...);
    }

    template <class Sndr, class Fn>
    static auto then(Sndr&& sndr, Fn&& fn)
    {
      return ts::then<Ending, Entry>(static_cast<Sndr&&>(sndr), static_cast<Fn&&>(fn));
    }

    template <class Layout = ts::carried_layout, class Sndr, class Fn>
    static auto let_value(Sndr&& sndr, Fn&& fn)
    {
      return ts::let_value<Ending, Layout, Entry>(static_cast<Sndr&&>(sndr), static_cast<Fn&&>(fn));
    }

    template <class Sndr1, class Sndr2>
    static auto sequence(Sndr1&& sndr1, Sndr2&& sndr2)
    {
      return ts::sequence<Ending, Entry>(static_cast<Sndr1&&>(sndr1), static_cast<Sndr2&&>(sndr2));
    }

    template <class Sndr>
    static auto repeat_n(Sndr&& sndr, std::size_t count)
    {
      return ts::repeat_n<Ending, Entry>(static_cast<Sndr&&>(sndr), count);
    }
  };

  using eager       = policy<ts::eager_teardown, ts::connect_entry>;
  using forward     = policy<ts::forward_results, ts::connect_entry>;
  using eager_start = policy<ts::eager_teardown, ts::start_entry>;

  // The heading of a table of costs: each policy over a pair of columns,
  // operation-state bytes and moves (copies).
  void cost_heading()
  {
    std::printf("    %24s      stdexec          eager          forward        eager+start\n"
                "    %24s   -------------   -------------   -------------   -------------\n"
                "    %24s   bytes   moves   bytes   moves   bytes   moves   bytes   moves\n",
                "",
                "",
                "");
  }

  // A row of costs: the size of the operation state that make_sized's
  // pipeline connects to, and the moves (copies) of a value while running
  // make_counted's. A stdexec size of 0 leaves stdexec's columns blank.
  template <class MakeSized, class MakeCounted>
  void cost_row(char const *        label,
                std::size_t         stdexec_size,
                std::string const & stdexec_count,
                MakeSized           make_sized,
                MakeCounted         make_counted)
  {
    std::printf("    %-24s", label);
    if (stdexec_size == 0)
    {
      std::printf("   %13s", "");
    }
    else
    {
      std::printf("   %5zu  %6s", stdexec_size, stdexec_count.c_str());
    }
    [&]<class... Ps>(Ps...)
    {
      ((std::printf("   %5zu  %6s",
                    opstate_size<decltype(make_sized(Ps()))>,
                    count(make_counted(Ps())).c_str())),
       ...);
    }(eager(), forward(), eager_start());
    std::printf("\n");
  }

  // A row of move counts.
  template <class Make>
  void count_row(char const * label, std::string const & stdexec_count, Make make)
  {
    std::printf("    %-44s %10s", label, stdexec_count.c_str());
    [&]<class... Ps>(Ps...)
    {
      ((std::printf(" %10s", count(make(Ps())).c_str())), ...);
    }(eager(), forward(), eager_start());
    std::printf("\n");
  }

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  // A value whose countdown-th move throws.
  struct fragile
  {
    inline static int countdown = 0;

    fragile() = default;

    fragile(fragile&& other)
      : value(std::move(other.value))
    {
      if (countdown > 0 && --countdown == 0)
      {
        throw std::runtime_error("move");
      }
    }

    std::unique_ptr<int> value = std::make_unique<int>(42);
  };

  // A sender whose connect throws.
  struct throwing_connect
  {
    using sender_concept        = ex::sender_t;
    using completion_signatures = ex::completion_signatures<ex::set_value_t()>;

    struct opstate
    {
      void start() & noexcept {}
    };

    template <class Rcvr>
    auto connect(Rcvr) && -> opstate
    {
      throw std::runtime_error("connect");
    }
  };

  // Records whether it was completed with an error.
  struct error_probe
  {
    using receiver_concept = ex::receiver_t;

    void set_value() && noexcept {}

    template <class E>
    void set_error(E&&) && noexcept
    {
      *failed = true;
    }

    void set_stopped() && noexcept {}

    bool* failed;
  };
#endif

#if defined(FRAMES_SHOW_USE_AFTER_FINISH)
  // A sender whose operation, wrongly, touches its own state after
  // completing.
  struct sloppy_sender
  {
    using sender_concept        = ex::sender_t;
    using completion_signatures = ex::completion_signatures<ex::set_value_t(int)>;

    template <class Rcvr>
    struct opstate
    {
      void start() & noexcept
      {
        ex::set_value(std::move(rcvr), 42);
        ++completions;  // the bug: the parent may already have destroyed us
      }

      Rcvr rcvr;
      int  completions = 0;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) && -> opstate<Rcvr>
    {
      return {std::move(rcvr)};
    }
  };
#endif

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
    section("9. Costs against stdexec");
    auto takes_big = [](big&)
    {
      return ex::just();
    };
    auto takes_big_ts = [](big&)
    {
      return ts::just();
    };
    auto plus_one = [](int i)
    {
      return i + 1;
    };

    // Each row builds one pipeline under every typestate policy (one ending
    // and one entry throughout), next to stdexec's version, twice: with a
    // big value, to measure its operation state, and with a counted one, to
    // count its moves and copies from connect to sync_wait's result.
    auto let_fn = [](counted&)
    {
      return ex::just();
    };
    auto let_fn_ts = [](counted&)
    {
      return ts::just();
    };
    auto then_fn = [](counted c)
    {
      return c;
    };
    // exec::sequence requires every sender but the last to send no values,
    // so to measure size, the first sender holds the big value and drops it.
    auto drop = [](big const &) {};

    std::printf("    v is big for bytes (an int for then), a counted value for moves;\n"
                "    s is then(just(big), drop) for bytes, just() for moves\n\n");
    cost_heading();
    cost_row(
      "let_value(just(v), fn)",
      opstate_size<decltype(ex::just(big{}) | ex::let_value(takes_big))>,
      count(ex::just(counted()) | ex::let_value(let_fn)),
      [&]<class P>(P) { return P::let_value(P::just(big{}), takes_big_ts); },
      [&]<class P>(P) { return P::let_value(P::just(counted()), let_fn_ts); });
    cost_row(
      "  ... overlapped_layout",
      0,
      "",
      [&]<class P>(P)
      { return P::template let_value<ts::overlapped_layout>(P::just(big{}), takes_big_ts); },
      [&]<class P>(P)
      { return P::template let_value<ts::overlapped_layout>(P::just(counted()), let_fn_ts); });
    cost_row(
      "  ... scoped_layout",
      0,
      "",
      [&]<class P>(P)
      { return P::template let_value<ts::scoped_layout>(P::just(big{}), takes_big_ts); },
      [&]<class P>(P)
      { return P::template let_value<ts::scoped_layout>(P::just(counted()), let_fn_ts); });
    cost_row(
      "sequence(s, just(v))",
      opstate_size<decltype(exec::sequence(ex::just(big{}) | ex::then(drop), ex::just(big{})))>,
      count(exec::sequence(ex::just(), ex::just(counted()))),
      [&]<class P>(P) { return P::sequence(P::then(P::just(big{}), drop), P::just(big{})); },
      []<class P>(P) { return P::sequence(P::just(), P::just(counted())); });
    cost_row(
      "then(just(v), fn)",
      opstate_size<decltype(ex::just(1) | ex::then(plus_one))>,
      count(ex::just(counted()) | ex::then(then_fn)),
      [&]<class P>(P) { return P::then(P::just(1), plus_one); },
      [&]<class P>(P) { return P::then(P::just(counted()), then_fn); });
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

    // Make the k-th move of a value throw, for every k until the pipeline
    // succeeds. Each failure lands somewhere different (connect, a
    // transition, a finish) and must end the live phase without leaking or
    // destroying anything twice; run it under ASan to check.
    auto failure_points = []<class P, class Layout = ts::carried_layout>(P, Layout = {})
    {
      int failures = 0;
      for (int k = 1;; ++k)
      {
        auto pipeline      = P::then(P::template let_value<Layout>(P::just(fragile()),
                                                              [](fragile& f)
                                                              { return P::just(std::move(f)); }),
                                [](fragile f) { return *f.value; });
        fragile::countdown = k;
        try
        {
          ex::sync_wait(std::move(pipeline));
          break;
        }
        catch (std::runtime_error const &)
        {
          ++failures;
        }
      }
      fragile::countdown = 0;
      return failures;
    };
    std::printf("    eager:       %2d failure points, each surfacing from sync_wait\n",
                failure_points(eager()));
    std::printf("    forward:     %2d failure points, each surfacing from sync_wait\n",
                failure_points(forward()));
    std::printf("    eager+start: %2d failure points, each surfacing from sync_wait\n",
                failure_points(eager_start()));
    // The same, with let_value's values kept under the other layouts.
    std::printf("    with overlapped_layout: %2d, %2d and %2d failure points\n",
                failure_points(eager(), ts::overlapped_layout()),
                failure_points(forward(), ts::overlapped_layout()),
                failure_points(eager_start(), ts::overlapped_layout()));
    std::printf("    with scoped_layout:     %2d, %2d and %2d failure points\n",
                failure_points(eager(), ts::scoped_layout()),
                failure_points(forward(), ts::scoped_layout()),
                failure_points(eager_start(), ts::scoped_layout()));

    // Under start_entry, connect doesn't enter the first phase, so a child
    // whose connect throws makes start complete with set_error instead of
    // making connect throw.
    auto where = []<class P>(P) -> char const *
    {
      bool failed = false;
      try
      {
        auto op = ex::connect(P::then(throwing_connect(), [] {}), error_probe{&failed});
        ex::start(op);
      }
      catch (std::runtime_error const &)
      {
        return "connect threw";
      }
      return failed ? "start completed with set_error" : "no error";
    };
    std::printf("    a throwing child connect: connect_entry: %s; start_entry: %s\n",
                where(eager()),
                where(eager_start()));
  }
#endif

  {
    section("11. Bound operations among stdexec senders");
    // Typestate senders bind to the sender stored once at the root; a stdexec
    // child, which can't bind, is moved into connect instead.
    using bound = eager_start;
    auto [a]    = ex::sync_wait(bound::then(ex::just(20), [](int i) { return i + 1; })).value();
    auto [b]    = ex::sync_wait(bound::just(20) | ex::then([](int i) { return i + 2; })).value();
    std::printf("    bound then over stdexec just: %d; stdexec then over bound just: %d\n", a, b);

    counting_resource mr;
    auto [c] = run(bound::let_value(indirect(bound::just(1), "phase 1"),
                                    [](int i) { return indirect(bound::just(i + 1), "phase 2"); }),
                   mr)
                 .value();
    std::printf("    result: %d\n", c);
  }

  {
    section("12. repeat_n binds its stored child again without consuming it");
    // Each iteration re-enters one phase, which binds the stored child.
    int  runs       = 0;
    auto count_runs = [p = &runs]
    {
      ++*p;
    };
    ex::sync_wait(eager_start::repeat_n(eager_start::then(eager_start::just(), count_runs), 5));
    ex::sync_wait(eager_start::repeat_n(eager_start::then(eager_start::just(), count_runs), 0));
    std::printf("    repeat_n(..., 5) then repeat_n(..., 0): %d runs\n", runs);

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
    runs = 0;
    try
    {
      ex::sync_wait(eager_start::repeat_n(eager_start::then(eager_start::just(),
                                                            [p = &runs]
                                                            {
                                                              if (++*p == 3)
                                                              {
                                                                throw std::runtime_error("third");
                                                              }
                                                            }),
                                          5));
    }
    catch (std::runtime_error const & e)
    {
      std::printf("    a failing third run ends the repeat: %d runs, then \"%s\"\n",
                  runs,
                  e.what());
    }
#endif

    // Copies and moves over 100 iterations. stdexec's repeat_n reconnects
    // a copy of its child every time; repeat_n reads its child in place, so
    // what the child only reads (here, a capture) is never copied. Values
    // the child sends are copied each time either way.
    std::printf("\n    %-44s %10s %10s %10s %10s\n",
                "moves (copies), 100 iterations",
                "stdexec",
                "eager",
                "forward",
                "eag+start");
    count_row("repeat_n(then(just(), fn capturing c))",
              count(exec::repeat_n(ex::just() | ex::then([c = counted()] {}), 100)),
              []<class P>(P) { return P::repeat_n(P::then(P::just(), [c = counted()] {}), 100); });
    count_row("repeat_n(then(just(c), drop))",
              count(exec::repeat_n(ex::just(counted()) | ex::then([](counted const &) {}), 100)),
              []<class P>(P)
              { return P::repeat_n(P::then(P::just(counted()), [](counted const &) {}), 100); });
  }

  {
    section("13. Touching a phase after it has ended");
#if defined(FRAMES_SHOW_USE_AFTER_FINISH)
    // Under AddressSanitizer, eager teardown poisons the child's bytes before
    // then completes, so the child's late write is reported.
    ex::sync_wait(ts::then<ts::eager_teardown>(sloppy_sender(), [](int i) { return i; }));
#else
    std::printf("    build with -fsanitize=address -DFRAMES_SHOW_USE_AFTER_FINISH to see\n"
                "    a child that touches its own state after completing reported\n");
#endif
  }

  {
    section("14. How long let_value's values live");
    // Carried values live as long as the operation state. Overlapped or
    // scoped, they're part of the second phase, so they end with it: before
    // the operation completes under eager teardown, with the operation state
    // otherwise.
    auto trace = []<class P, class Layout>(P, Layout, char const * label)
    {
      std::printf("    %-20s", label);
      ex::sync_wait(
        P::template let_value<Layout>(P::just(traced()), [](traced&) { return P::just(); })
        | ex::then([] { std::printf(" completed;"); }));
      std::printf(" sync_wait returned\n");
    };
    trace(eager(), ts::carried_layout(), "eager, carried:");
    trace(eager(), ts::overlapped_layout(), "eager, overlapped:");
    trace(eager(), ts::scoped_layout(), "eager, scoped:");
    trace(forward(), ts::carried_layout(), "forward, carried:");
    trace(forward(), ts::overlapped_layout(), "forward, overlapped:");
    trace(forward(), ts::scoped_layout(), "forward, scoped:");
  }
}
