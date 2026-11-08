/*
 * Exploration: operation states as sequences of phases.
 *
 * Each section below runs frames::typestate's just, then, let_value,
 * sequence, repeat_n or when_all, whose operation states hold one phase at a
 * time, alone or among stdexec's senders, and some compare them with
 * stdexec's own algorithms. The policies they run under come from the
 * receiver's environment.
 */

#include "associate.hpp"
#include "finally.hpp"
#include "just.hpp"
#include "let_value.hpp"
#include "repeat_n.hpp"
#include "sequence.hpp"
#include "then.hpp"
#include "when_all.hpp"

#include <exec/finally.hpp>
#include <exec/repeat_n.hpp>
#include <exec/sequence.hpp>
#include <exec/single_thread_context.hpp>
#include <exec/variant_sender.hpp>
#include <stdexec/execution.hpp>

#include <array>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ex = stdexec;
namespace ts = frames::typestate;

namespace
{
  // A receiver that ignores its completions, with Env as its environment,
  // for measuring operation states and counting moves.
  template <class Env = ex::env<>>
  struct sink
  {
    using receiver_concept = ex::receiver_t;

    auto get_env() const noexcept -> Env
    {
      return {};
    }

    template <class... As>
    void set_value(As&&...) && noexcept
    {}

    template <class E>
    void set_error(E&&) && noexcept
    {}

    void set_stopped() && noexcept {}
  };

  template <class Sndr, class Env = ex::env<>>
  constexpr auto opstate_size = sizeof(ex::connect_result_t<Sndr, sink<Env>>);

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

  // Moves and copies while connecting `sndr` to a sink whose environment is
  // Env and starting it, as "m (c)". Every pipeline counted here completes
  // inline, so it has finished when start returns.
  template <class Env = ex::env<>, class Sndr>
  auto count(Sndr sndr) -> std::string
  {
    if constexpr (std::is_null_pointer_v<Sndr>)
    {
      return "";
    }
    else
    {
      counted::copies = counted::moves = 0;
      auto op                          = ex::connect(std::move(sndr), sink<Env>());
      ex::start(op);
      return std::to_string(counted::moves) + " (" + std::to_string(counted::copies) + ")";
    }
  }

  // Moves and copies while connecting the lvalue `sndr`, which stays
  // usable, to a sink whose environment is Env and starting it, as "m (c)".
  template <class Env = ex::env<>, class Sndr>
  auto count_lvalue(Sndr const & sndr) -> std::string
  {
    counted::copies = counted::moves = 0;
    auto op                          = ex::connect(sndr, sink<Env>());
    ex::start(op);
    return std::to_string(counted::moves) + " (" + std::to_string(counted::copies) + ")";
  }

  // A column of the cost tables: one ending, one entry and one way of keeping
  // the sender's data, with whichever layout the row asks for.
  template <class Ending, class Entry, class Data = ts::stored_sender>
  struct column
  {
    template <class Layout = ts::carried_layout>
    using policies = ts::policies<Ending, Layout, Entry, Data>;
  };

  using eager       = column<ts::eager_teardown, ts::connect_entry>;
  using forward     = column<ts::forward_results, ts::connect_entry>;
  using eager_start = column<ts::eager_teardown, ts::start_entry>;
  using eager_split = column<ts::eager_teardown, ts::connect_entry, ts::split_sender>;

  // The heading of a table of costs: each policy over a pair of columns,
  // operation-state bytes and moves (copies).
  void cost_heading()
  {
    std::printf("    %24s      stdexec          eager          forward        eager+start"
                "      eager+split\n"
                "    %24s   -------------   -------------   -------------   -------------"
                "   -------------\n"
                "    %24s   bytes   moves   bytes   moves   bytes   moves   bytes   moves"
                "   bytes   moves\n",
                "",
                "",
                "");
  }

  // A row of costs, under each column's policies with Layout: the size of
  // the operation state that make_sized's pipeline connects to, and the
  // moves (copies) of a value while running make_counted's. A stdexec size of
  // 0 leaves stdexec's columns blank.
  template <class Layout, class MakeSized, class MakeCounted>
  void cost_row(char const *        label,
                std::size_t         stdexec_size,
                std::string const & stdexec_count,
                Layout,
                MakeSized   make_sized,
                MakeCounted make_counted)
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
    [&]<class... Columns>(Columns...)
    {
      ((std::printf("   %5zu  %6s",
                    opstate_size<decltype(make_sized()),
                                 ts::policies_env<typename Columns::template policies<Layout>>>,
                    count<ts::policies_env<typename Columns::template policies<Layout>>>(
                      make_counted())
                      .c_str())),
       ...);
    }(eager(), forward(), eager_start(), eager_split());
    std::printf("\n");
  }

  // A row of move counts for connecting the lvalue sndr, under each
  // column's policies.
  template <class Sndr>
  void lvalue_row(char const * label, std::string const & stdexec_count, Sndr const & sndr)
  {
    std::printf("    %-44s %10s", label, stdexec_count.c_str());
    [&]<class... Columns>(Columns...)
    {
      ((std::printf(
         " %10s",
         count_lvalue<ts::policies_env<typename Columns::template policies<>>>(sndr).c_str())),
       ...);
    }(eager(), forward(), eager_start(), eager_split());
    std::printf("\n");
  }

  // A row of move counts, under each column's policies.
  template <class Make>
  void count_row(char const * label, std::string const & stdexec_count, Make make)
  {
    std::printf("    %-44s %10s", label, stdexec_count.c_str());
    [&]<class... Columns>(Columns...)
    {
      ((std::printf(
         " %10s",
         count<ts::policies_env<typename Columns::template policies<>>>(make()).c_str())),
       ...);
    }(eager(), forward(), eager_start(), eager_split());
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

  // Records whether it was completed with an error; Env is its environment.
  template <class Env = ex::env<>>
  struct error_probe
  {
    using receiver_concept = ex::receiver_t;

    auto get_env() const noexcept -> Env
    {
      return {};
    }

    template <class... As>
    void set_value(As&&...) && noexcept
    {}

    template <class E>
    void set_error(E&&) && noexcept
    {
      *failed = true;
    }

    void set_stopped() && noexcept {}

    bool* failed;
  };
#endif

  // Owns the operation it's connected to, and destroys it from inside its
  // own completion, after copying what it was sent: what stdexec's
  // receivers may do, and what a parent's eager teardown does.
  template <class Env>
  struct destroying_receiver
  {
    using receiver_concept = ex::receiver_t;

    struct state
    {
      std::string                            result;
      std::unique_ptr<void, void (*)(void*)> op{nullptr, [](void*) {}};
    };

    auto get_env() const noexcept -> Env
    {
      return {};
    }

    void set_value(std::string&& s) && noexcept
    {
      done("value " + s);
    }

    void set_error(int e) && noexcept
    {
      done("error " + std::to_string(e));
    }

    void set_error(std::exception_ptr) && noexcept
    {
      done("exception");
    }

    void set_stopped() && noexcept
    {
      done("stopped");
    }

    void done(std::string result) noexcept
    {
      state* s  = self;
      s->result = std::move(result);
      s->op.reset();  // destroys this receiver too
    }

    state* self;
  };

  // Connects sndr to a destroying_receiver on the heap, starts it, and
  // returns what it completed with; the operation has destroyed itself.
  template <class Env, class Sndr>
  auto run_destroying(Sndr&& sndr) -> std::string
  {
    using rcvr_t = destroying_receiver<Env>;
    using op_t   = ex::connect_result_t<Sndr, rcvr_t>;
    typename rcvr_t::state state;
    auto*                  op = new op_t(ex::connect(static_cast<Sndr&&>(sndr), rcvr_t{&state}));
    state.op                  = {op, [](void* p) { delete static_cast<op_t*>(p); }};
    ex::start(*op);
    return state.result;
  }

  // A sender whose operation sends an lvalue referring to a string it holds.
  struct lvalue_sender
  {
    using sender_concept        = ex::sender_t;
    using completion_signatures = ex::completion_signatures<ex::set_value_t(std::string&)>;

    template <class Rcvr>
    struct opstate
    {
      void start() & noexcept
      {
        ex::set_value(std::move(rcvr), text);
      }

      Rcvr        rcvr;
      std::string text = "a string the child holds";
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) const -> opstate<Rcvr>
    {
      return {std::move(rcvr)};
    }
  };

  // Says which overload ran: a view of the string an lvalue refers to, or
  // that it was given an rvalue.
  struct which_overload
  {
    auto operator()(std::string& s) const -> std::string_view
    {
      return s;
    }

    auto operator()(std::string&&) const -> std::string_view
    {
      return "an rvalue";
    }
  };

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

  // A sender that says when it's connected, and completes with set_value().
  struct announced
  {
    using sender_concept        = ex::sender_t;
    using completion_signatures = ex::completion_signatures<ex::set_value_t()>;

    template <class Rcvr>
    struct opstate
    {
      void start() & noexcept
      {
        ex::set_value(std::move(rcvr));
      }

      Rcvr rcvr;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) && -> opstate<Rcvr>
    {
      std::printf(" %s connected;", name);
      return {std::move(rcvr)};
    }

    char const * name;
  };

  // An async scope that counts the associations it has given out, and
  // refuses new ones once closed.
  struct counting_scope
  {
    class association
    {
     public:
      association() = default;

      explicit association(counting_scope* scope) noexcept
        : scope_(scope)
      {
        if (scope_ != nullptr)
        {
          ++scope_->live;
        }
      }

      association(association&& other) noexcept
        : scope_(std::exchange(other.scope_, nullptr))
      {}

      auto operator=(association other) noexcept -> association&
      {
        std::swap(scope_, other.scope_);
        return *this;
      }

      ~association()
      {
        if (scope_ != nullptr)
        {
          --scope_->live;
        }
      }

      explicit operator bool() const noexcept
      {
        return scope_ != nullptr;
      }

      [[nodiscard]]
      auto try_associate() const noexcept -> association
      {
        return association(scope_ != nullptr && !scope_->closed ? scope_ : nullptr);
      }

     private:
      counting_scope* scope_ = nullptr;
    };

    struct token
    {
      [[nodiscard]]
      auto try_associate() const noexcept -> association
      {
        return association(scope->closed ? nullptr : scope);
      }

      template <class Sndr>
      auto wrap(Sndr&& sndr) const noexcept -> Sndr&&
      {
        return static_cast<Sndr&&>(sndr);
      }

      counting_scope* scope;
    };

    auto get_token() noexcept -> token
    {
      return {this};
    }

    int  live   = 0;
    bool closed = false;
  };

  // Says how many associations its scope has given out when it's
  // completed, and how; Env is its environment.
  template <class Env>
  struct association_probe
  {
    using receiver_concept = ex::receiver_t;

    auto get_env() const noexcept -> Env
    {
      return {};
    }

    void set_value() && noexcept
    {
      std::printf(" completed with %d;", scope->live);
    }

    void set_stopped() && noexcept
    {
      std::printf(" stopped with %d;", scope->live);
    }

    template <class E>
    void set_error(E&&) && noexcept
    {
      std::printf(" error;");
    }

    counting_scope* scope;
  };

  void section(char const * title)
  {
    std::printf("\n== %s\n", title);
  }
}  // namespace

int main()
{
  {
    section("1. Typestate algorithms");
    auto [v] = ex::sync_wait(ts::sequence(ts::then(ts::just(20), [](int i) { return i + 1; }),
                                          ts::let_value(ts::just(std::string("answer")),
                                                        [](std::string& s)
                                                        { return ts::just(s.size() * 2 + 30); })))
                 .value();
    std::printf("    result: %zu\n", v);
  }

  {
    section("2. let_value picks its second phase by the child's value type");
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
    section("3. Costs against stdexec");
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
      ts::carried_layout(),
      [&] { return ts::let_value(ts::just(big{}), takes_big_ts); },
      [&] { return ts::let_value(ts::just(counted()), let_fn_ts); });
    cost_row(
      "  ... overlapped_layout",
      0,
      "",
      ts::overlapped_layout(),
      [&] { return ts::let_value(ts::just(big{}), takes_big_ts); },
      [&] { return ts::let_value(ts::just(counted()), let_fn_ts); });
    cost_row(
      "  ... scoped_layout",
      0,
      "",
      ts::scoped_layout(),
      [&] { return ts::let_value(ts::just(big{}), takes_big_ts); },
      [&] { return ts::let_value(ts::just(counted()), let_fn_ts); });
    cost_row(
      "sequence(s, just(v))",
      opstate_size<decltype(exec::sequence(ex::just(big{}) | ex::then(drop), ex::just(big{})))>,
      count(exec::sequence(ex::just(), ex::just(counted()))),
      ts::carried_layout(),
      [&] { return ts::sequence(ts::then(ts::just(big{}), drop), ts::just(big{})); },
      [] { return ts::sequence(ts::just(), ts::just(counted())); });
    cost_row(
      "then(just(v), fn)",
      opstate_size<decltype(ex::just(1) | ex::then(plus_one))>,
      count(ex::just(counted()) | ex::then(then_fn)),
      ts::carried_layout(),
      [&] { return ts::then(ts::just(1), plus_one); },
      [&] { return ts::then(ts::just(counted()), then_fn); });
    cost_row(
      "when_all of two just(v)",
      opstate_size<decltype(ex::when_all(ex::just(big{}), ex::just(big{})))>,
      count(ex::when_all(ex::just(counted()), ex::just(counted()))),
      ts::carried_layout(),
      [] { return ts::when_all(ts::just(big{}), ts::just(big{})); },
      [] { return ts::when_all(ts::just(counted()), ts::just(counted())); });
    cost_row(
      "finally(just(v), just())",
      opstate_size<decltype(exec::finally(ex::just(big{}), ex::just()))>,
      count(exec::finally(ex::just(counted()), ex::just())),
      ts::carried_layout(),
      [] { return ts::finally(ts::just(big{}), ts::just()); },
      [] { return ts::finally(ts::just(counted()), ts::just()); });
    cost_row(
      "  ... overlapped_layout",
      0,
      "",
      ts::overlapped_layout(),
      [] { return ts::finally(ts::just(big{}), ts::just()); },
      [] { return ts::finally(ts::just(counted()), ts::just()); });
  }

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  {
    section("4. Exceptions end the live phase and complete with set_error");
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
    auto failure_points = []<class C, class Layout = ts::carried_layout>(C, Layout = {})
    {
      int failures = 0;
      for (int k = 1;; ++k)
      {
        using env_t        = ts::policies_env<typename C::template policies<Layout>>;
        auto pipeline      = ts::then(ts::let_value(ts::just(fragile()),
                                               [](fragile& f) { return ts::just(std::move(f)); }),
                                 [](fragile f) { return *f.value; });
        fragile::countdown = k;
        bool failed        = false;
        try
        {
          auto op = ex::connect(std::move(pipeline), error_probe<env_t>{&failed});
          ex::start(op);
        }
        catch (std::runtime_error const &)
        {
          failed = true;
        }
        if (!failed)
        {
          break;
        }
        ++failures;
      }
      fragile::countdown = 0;
      return failures;
    };
    std::printf("    eager:       %2d failure points, each an exception or set_error\n",
                failure_points(eager()));
    std::printf("    forward:     %2d failure points, each an exception or set_error\n",
                failure_points(forward()));
    std::printf("    eager+start: %2d failure points, each an exception or set_error\n",
                failure_points(eager_start()));
    std::printf("    eager+split: %2d failure points, each an exception or set_error\n",
                failure_points(eager_split()));
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
    auto where = []<class C>(C) -> char const *
    {
      bool failed = false;
      try
      {
        auto op = ex::connect(ts::then(throwing_connect(), [] {}),
                              error_probe<ts::policies_env<typename C::template policies<>>>{
                                &failed});
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
    section("5. Bound operations among stdexec senders");
    // Typestate senders bind to the sender stored once at the root; a stdexec
    // child, which can't bind, is moved into connect instead.
    using bound = eager_start::policies<>;
    auto [a]    = ex::sync_wait(
                 ts::with_policies<bound>(ts::then(ex::just(20), [](int i) { return i + 1; })))
                 .value();
    auto [b] = ex::sync_wait(ts::with_policies<bound>(ts::just(20))
                             | ex::then([](int i) { return i + 2; }))
                 .value();
    std::printf("    bound then over stdexec just: %d; stdexec then over bound just: %d\n", a, b);
  }

  {
    section("6. repeat_n binds its stored child again without consuming it");
    // Each iteration re-enters one phase, which binds the stored child.
    using bound     = eager_start::policies<>;
    int  runs       = 0;
    auto count_runs = [p = &runs]
    {
      ++*p;
    };
    ex::sync_wait(ts::with_policies<bound>(ts::repeat_n(ts::then(ts::just(), count_runs), 5)));
    ex::sync_wait(ts::with_policies<bound>(ts::repeat_n(ts::then(ts::just(), count_runs), 0)));
    std::printf("    repeat_n(..., 5) then repeat_n(..., 0): %d runs\n", runs);

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
    runs = 0;
    try
    {
      ex::sync_wait(ts::with_policies<bound>(ts::repeat_n(ts::then(ts::just(),
                                                                   [p = &runs]
                                                                   {
                                                                     if (++*p == 3)
                                                                     {
                                                                       throw std::runtime_error(
                                                                         "third");
                                                                     }
                                                                   }),
                                                          5)));
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
    std::printf("\n    %-44s %10s %10s %10s %10s %10s\n",
                "moves (copies), 100 iterations",
                "stdexec",
                "eager",
                "forward",
                "eag+start",
                "eag+split");
    count_row("repeat_n(then(just(), fn capturing c))",
              count(exec::repeat_n(ex::just() | ex::then([c = counted()] {}), 100)),
              [] { return ts::repeat_n(ts::then(ts::just(), [c = counted()] {}), 100); });
    count_row("repeat_n(then(just(c), drop))",
              count(exec::repeat_n(ex::just(counted()) | ex::then([](counted const &) {}), 100)),
              []
              { return ts::repeat_n(ts::then(ts::just(counted()), [](counted const &) {}), 100); });
  }

  {
    section("7. Touching a phase after it has ended");
#if defined(FRAMES_SHOW_USE_AFTER_FINISH)
    // Under AddressSanitizer, eager teardown poisons the child's bytes before
    // then completes, so the child's late write is reported.
    ex::sync_wait(ts::then(sloppy_sender(), [](int i) { return i; }));
#else
    std::printf("    build with -fsanitize=address -DFRAMES_SHOW_USE_AFTER_FINISH to see\n"
                "    a child that touches its own state after completing reported\n");
#endif
  }

  {
    section("8. How long let_value's values live");
    // Carried values live as long as the operation state. Overlapped or
    // scoped, they're part of the second phase, so they end with it: before
    // the operation completes under eager teardown, with the operation state
    // otherwise.
    auto trace = []<class C, class Layout>(C, Layout, char const * label)
    {
      std::printf("    %-20s", label);
      ex::sync_wait(ts::with_policies<typename C::template policies<Layout>>(
                      ts::let_value(ts::just(traced()), [](traced&) { return ts::just(); }))
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

  {
    section("9. when_all stops its children together");
    // when_all's phase holds a stop source of its own, and its children see
    // its token in place of the receiver's, with the rest of the receiver's
    // environment (the policies, say) carried over. A child's error stops
    // the others, and so does a stop request from the receiver, which
    // reaches the phase's source once when_all starts. Each child here
    // reads its token as it starts.
    auto child = [](char const * name)
    {
      return ex::read_env(ex::get_stop_token)
           | ex::then(
               [name](auto token)
               { std::printf(" %s %s;", name, token.stop_requested() ? "stopped" : "running"); });
    };
    std::printf("    a child's error:      ");
    ex::sync_wait(ts::when_all(child("first"), ex::just_error(1), child("third"))
                  | ex::upon_error(
                    []<class E>(E const & e)
                    {
                      if constexpr (std::is_same_v<E, int>)
                      {
                        std::printf(" error %d\n", e);
                      }
                      else
                      {
                        std::printf(" exception\n");
                      }
                    }));

    ex::inplace_stop_source source;
    source.request_stop();
    std::printf("    the receiver's stop:  ");
    ex::sync_wait(ex::write_env(ts::when_all(child("first"), child("second")),
                                ex::prop{ex::get_stop_token, source.get_token()})
                  | ex::then([] { std::printf(" values\n"); }));
  }

  {
    section("10. then's function sees values as its child sent them");
    // Under eager teardown, then calls its function after the child's phase
    // has ended, with copies of the values on the stack, passed as the child
    // sent them. So the same overload runs under every ending, and a view of
    // an argument stays valid while the operation completes.
    auto run = []<class C>(C, char const * label)
    {
      std::printf("    %-13s", label);
      ex::sync_wait(
        ts::with_policies<typename C::template policies<>>(
          ts::then(lvalue_sender(), which_overload()))
        | ex::then([](std::string_view v) { std::printf(" %.*s\n", int(v.size()), v.data()); }));
    };
    run(eager(), "eager:");
    run(forward(), "forward:");
    run(eager_start(), "eager+start:");
    run(eager_split(), "eager+split:");
  }

  {
    section("11. finally runs final, then completes as initial did");
    // finally's second phase receives initial's completion, whichever it
    // was, runs final, and replays it if final succeeds. Here the receiver
    // destroys the operation from inside that completion, under every
    // policy and layout: nothing touches the operation once it completes.
    auto show = []<class C, class Layout>(C, Layout, char const * label)
    {
      using env_t = ts::policies_env<typename C::template policies<Layout>>;
      auto final  = []
      {
        return ts::just();
      };
      std::printf(
        "    %-22s %s; %s; %s; %s\n",
        label,
        run_destroying<env_t>(ts::finally(ts::just(std::string("initial")), final())).c_str(),
        run_destroying<env_t>(
          ts::finally(ex::just_error(42) | ex::then([] { return std::string(); }), final()))
          .c_str(),
        run_destroying<env_t>(
          ts::finally(ex::just_stopped() | ex::then([] { return std::string(); }), final()))
          .c_str(),
        run_destroying<env_t>(ts::finally(ts::just(std::string("initial")), ex::just_error(7)))
          .c_str());
    };
    show(eager(), ts::carried_layout(), "eager:");
    show(eager(), ts::overlapped_layout(), "eager, overlapped:");
    show(eager(), ts::scoped_layout(), "eager, scoped:");
    show(forward(), ts::carried_layout(), "forward:");
    show(forward(), ts::overlapped_layout(), "forward, overlapped:");
    show(forward(), ts::scoped_layout(), "forward, scoped:");
    show(eager_start(), ts::carried_layout(), "eager+start:");
    show(eager_split(), ts::carried_layout(), "eager+split:");

    // final is connected with initial, just before it, as the first phase is
    // entered: by connect or by start.
    auto order = []<class C>(C, char const * label)
    {
      std::printf("    %-13s", label);
      auto op = ex::connect(ts::finally(announced{"initial"}, announced{"final"}),
                            sink<ts::policies_env<typename C::template policies<>>>());
      std::printf(" connect returned;");
      ex::start(op);
      std::printf(" started\n");
    };
    order(eager(), "eager:");
    order(eager_start(), "eager+start:");

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
    // So if final can't be connected, initial isn't either, and neither runs.
    auto failed_final = []<class C>(C) -> char const *
    {
      bool failed = false;
      try
      {
        auto op = ex::connect(ts::finally(announced{"initial"}, throwing_connect()),
                              error_probe<ts::policies_env<typename C::template policies<>>>{
                                &failed});
        ex::start(op);
      }
      catch (std::runtime_error const &)
      {
        return "connect threw";
      }
      return failed ? "start completed with set_error" : "no error";
    };
    std::printf("    a throwing final connect: connect_entry: %s;", failed_final(eager()));
    std::printf(" start_entry: %s\n", failed_final(eager_start()));
#endif

    // final starts where initial completed, and its environment says so.
    exec::single_thread_context context;
    auto                        where_final_starts = [&](auto sndr)
    {
      ex::sync_wait(std::move(sndr));
    };
    auto says = [sch = context.get_scheduler()](char const * label)
    {
      return ex::read_env(ex::get_start_scheduler)
           | ex::then(
               [sch, label](auto start)
               {
                 std::printf("    %s final's start scheduler is %s\n",
                             label,
                             start == sch ? "initial's completion scheduler" : "another");
               });
    };
    where_final_starts(ts::finally(ex::schedule(context.get_scheduler()), says("finally:     ")));
    where_final_starts(
      exec::finally(ex::schedule(context.get_scheduler()), says("exec::finally:")));
  }

  {
    section("12. associate holds its association until the operation is destroyed");
    // [exec.associate] keeps the association in the operation state, and
    // nothing releases it when the operation completes, so a scope can't
    // finish joining until the operation is destroyed. associate keeps it in
    // the carried region, so it ends there under every policy, as it does
    // with stdexec::associate. Each line counts the scope's associations as
    // the sender is made, connected, completed and destroyed.
    counting_scope scope;
    auto           show = [&]<class Env>(char const * label, auto make, Env)
    {
      std::printf("    %-17s", label);
      {
        auto sndr = make(scope.get_token());
        std::printf(" made with %d;", scope.live);
        auto op = ex::connect(std::move(sndr), association_probe<Env>{&scope});
        std::printf(" connected with %d;", scope.live);
        ex::start(op);
      }
      std::printf(" destroyed with %d\n", scope.live);
    };
    auto ours = [](auto token)
    {
      return ts::associate(ts::just(), std::move(token));
    };
    auto theirs = [](auto token)
    {
      return ex::associate(ex::just(), std::move(token));
    };
    show("eager:", ours, ts::policies_env<eager::policies<>>());
    show("forward:", ours, ts::policies_env<forward::policies<>>());
    show("eager+start:", ours, ts::policies_env<eager_start::policies<>>());
    show("eager+split:", ours, ts::policies_env<eager_split::policies<>>());
    show("stdexec:", theirs, ex::env<>());

    // A closed scope refuses the association, and the operation completes
    // with set_stopped() without connecting the sender.
    scope.closed = true;
    show("closed, eager:", ours, ts::policies_env<eager::policies<>>());
    show("closed, stdexec:", theirs, ex::env<>());
  }

  {
    section("13. Connecting an lvalue copies the sender, which stays usable");
    // Connected as an lvalue, an operation copies what it uses of the
    // sender, into the root under stored_sender or into itself under
    // split_sender, and then only reads the copy, as when it's bound without
    // consuming. The sender is unchanged, so it can be connected again.
    auto greet = ts::then(ts::just(std::string("hello")),
                          [](std::string const & s) { return s + ", again"; });
    for (int i = 0; i != 2; ++i)
    {
      auto [s] = ex::sync_wait(greet).value();
      std::printf("    connected %s: %s\n", i == 0 ? "once " : "twice", s.c_str());
    }

    // Copies are made once, as the operation copies the sender; a value
    // the operation sends is then copied again, since the operation can't
    // move from what it only reads.
    auto drop = [](auto&&...) noexcept {};
    auto c = counted();
    std::printf("\n    %-44s %10s %10s %10s %10s %10s\n",
                "moves (copies)",
                "stdexec",
                "eager",
                "forward",
                "eag+start",
                "eag+split");
    lvalue_row("then(just(c), drop)",
               count_lvalue(ex::then(ex::just(c), drop)),
               ts::then(ts::just(c), drop));
    lvalue_row("then(just(), fn capturing c)",
               count_lvalue(ex::then(ex::just(), [c]() noexcept {})),
               ts::then(ts::just(), [c]() noexcept {}));
    lvalue_row("let_value(just(c), fn)",
               count_lvalue(
                 ex::let_value(ex::just(c), [](counted&) noexcept { return ex::just(); })),
               ts::let_value(ts::just(c), [](counted&) noexcept { return ts::just(); }));
  }
}
