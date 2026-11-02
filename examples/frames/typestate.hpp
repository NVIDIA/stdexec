/*
 * Exploration: operation states as explicit sequences of phases.
 *
 * An algorithm's operation state holds one phase at a time in storage sized
 * for the largest phase. Each phase's child completes through a receiver
 * whose *type* names the phase, so every completion handler knows statically
 * what the storage holds. (One byte records the live phase too, for the
 * destructor and for exception paths, which have no receiver to tell them.)
 *
 * A transition happens inside the live phase's completion:
 *
 *   1. gather what the next phase needs while the live phase is still alive,
 *   2. destroy the live phase,
 *   3. construct the next phase in the same storage and start it.
 *
 * This relies on an operation never touching its own state after it has
 * signalled completion, which stdexec's own let_value relies on too.
 *
 * How an operation ends is a policy:
 *
 *   - eager_teardown destroys the last phase before completing the parent
 *     receiver, so an operation holds nothing live once it has completed.
 *     Results may refer into the last phase, so they're moved onto the stack
 *     first.
 *   - forward_results completes the parent receiver from inside the last
 *     phase, with the results as they came, as stdexec's algorithms do. The
 *     phase is destroyed with the operation state.
 *
 * Prototype simplifications: senders are connected as rvalues only, every
 * algorithm except just conservatively declares
 * set_error_t(std::exception_ptr), and let_value doesn't add a completion
 * scheduler to its second phase's environment.
 */
#pragma once

#include <stdexec/execution.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <new>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace frames::typestate
{
  namespace ex = stdexec;

  // How an operation ends; see above.
  struct eager_teardown
  {};

  struct forward_results
  {};

  // Where state that crosses a transition is kept: let_value's values, or
  // sequence's second sender.
  struct carried_layout  // beside the phases, for the whole operation, as stdexec does
  {};

  struct overlapped_layout  // in the phases, moving across on the stack
  {};

  //////////////////////////////////////////////////////////////////////////////
  // Completion-signature bookkeeping.

  namespace detail
  {
    template <class Sig>
    struct decay_sig;

    template <class Tag, class... As>
    struct decay_sig<Tag(As...)>
    {
      using type = Tag(std::decay_t<As>...);
    };

    // Completions an operation sends after eager teardown are decayed: it
    // completes with copies on the stack.
    template <class Ending>
    struct sent_as
    {
      template <class Sig>
      using f = typename decay_sig<Sig>::type;
    };

    template <>
    struct sent_as<forward_results>
    {
      template <class Sig>
      using f = Sig;
    };

    template <class Sigs, class Sig>
    struct add_sig;

    template <class... Sigs, class Sig>
    struct add_sig<ex::completion_signatures<Sigs...>, Sig>
    {
      using type = std::conditional_t<(std::is_same_v<Sigs, Sig> || ...),
                                      ex::completion_signatures<Sigs...>,
                                      ex::completion_signatures<Sigs..., Sig>>;
    };

    template <class Sigs, class... More>
    struct add_all
    {
      using type = Sigs;
    };

    template <class Sigs, class First, class... More>
    struct add_all<Sigs, First, More...> : add_all<typename add_sig<Sigs, First>::type, More...>
    {};

    // The union of several signature sets, each signature passed through
    // Xform::f.
    template <class Xform, class... Sets>
    struct join
    {
      using type = ex::completion_signatures<>;
    };

    template <class Xform, class... Sigs, class... Rest>
    struct join<Xform, ex::completion_signatures<Sigs...>, Rest...>
      : add_all<typename join<Xform, Rest...>::type, typename Xform::template f<Sigs>...>
    {};

    template <class Xform, class... Sets>
    using join_t = typename join<Xform, Sets...>::type;

    // Replaces each set_value_t(As...) in a set by Fn::f<As...>.
    template <class Fn, class Sig>
    struct map_one
    {
      using type = ex::completion_signatures<Sig>;
    };

    template <class Fn, class... As>
    struct map_one<Fn, ex::set_value_t(As...)>
    {
      using type = typename Fn::template f<As...>;
    };

    template <class Ending, class Sigs, class Fn>
    struct map_values;

    template <class Ending, class... Sigs, class Fn>
    struct map_values<Ending, ex::completion_signatures<Sigs...>, Fn>
    {
      using type = join_t<sent_as<Ending>,
                          typename map_one<Fn, Sigs>::type...,
                          ex::completion_signatures<ex::set_error_t(std::exception_ptr)>>;
    };

    template <class Ending, class Sigs, class Fn>
    using map_values_t = typename map_values<Ending, Sigs, Fn>::type;

    template <class R>
    struct value_of
    {
      using type = ex::completion_signatures<ex::set_value_t(std::decay_t<R>)>;
    };

    template <>
    struct value_of<void>
    {
      using type = ex::completion_signatures<ex::set_value_t()>;
    };

    template <class... Ts>
    struct type_list
    {};

    template <class T, class... Ts>
    consteval auto index_of() -> unsigned char
    {
      constexpr bool same[] = {std::is_same_v<T, Ts>...};
      for (unsigned char i = 0; i < sizeof...(Ts); ++i)
      {
        if (same[i])
        {
          return i;
        }
      }
      return 0xff;
    }

    template <class List, class T>
    struct add_type;

    template <class... Ts, class T>
    struct add_type<type_list<Ts...>, T>
    {
      using type =
        std::conditional_t<(std::is_same_v<Ts, T> || ...), type_list<Ts...>, type_list<Ts..., T>>;
    };

    // type_list<type_list<decay_t<As>...>...>, one per distinct decayed
    // set_value_t(As...).
    template <class List, class... Sigs>
    struct value_lists_of
    {
      using type = List;
    };

    template <class List, class Sig, class... Rest>
    struct value_lists_of<List, Sig, Rest...> : value_lists_of<List, Rest...>
    {};

    template <class List, class... As, class... Rest>
    struct value_lists_of<List, ex::set_value_t(As...), Rest...>
      : value_lists_of<typename add_type<List, type_list<std::decay_t<As>...>>::type, Rest...>
    {};

    template <class Sigs>
    struct value_lists;

    template <class... Sigs>
    struct value_lists<ex::completion_signatures<Sigs...>> : value_lists_of<type_list<>, Sigs...>
    {};

    // std::variant<std::monostate, std::tuple<Ts...>...>.
    template <class Lists>
    struct args_variant;

    template <class... Lists>
    struct args_variant<type_list<Lists...>>
    {
      template <class List>
      struct tuple_of;

      template <class... Ts>
      struct tuple_of<type_list<Ts...>>
      {
        using type = std::tuple<Ts...>;
      };

      using type = std::variant<std::monostate, typename tuple_of<Lists>::type...>;
    };
  }  // namespace detail

  //////////////////////////////////////////////////////////////////////////////
  // phased: an operation's receiver, a carried region that lives as long as
  // the operation, and storage for whichever phase is live.
  //
  // Algorithms never construct or destroy phases themselves. They call:
  //
  //   transition<From, To>(prepare, extra...)
  //       from inside From's completion. `prepare(From&)` runs while From and
  //       the completion's arguments are still alive; it may store into the
  //       carried region, and returns a tuple of what To's constructor needs.
  //       Then From is destroyed, To is constructed from that tuple and
  //       `extra...`, and To's `child` is started. `extra...` must not refer
  //       into From.
  //
  //   finish<Phase>(tag, as...)
  //       from inside Phase's completion: completes the operation with
  //       tag(as...).
  //
  //   finish_with<Phase>(fn, as...)
  //       from inside Phase's set_value: completes the operation with
  //       set_value(fn(as...)), or set_value() if fn returns void.
  //
  // An exception from any of these completes the operation with
  // set_error(std::current_exception()); under eager_teardown it ends the
  // live phase first.

  template <class Ending, class Rcvr, class Carried, class First, class... Rest>
  class phased
  {
    static constexpr unsigned char none  = 0xff;
    static constexpr bool          eager = std::is_same_v<Ending, eager_teardown>;

    template <class Phase>
    static constexpr unsigned char index_of = detail::index_of<Phase, First, Rest...>();

   public:
    template <class CarriedInit, class... Args>
    phased(Rcvr rcvr, CarriedInit&& carried, Args&&... first)
      : rcvr_(static_cast<Rcvr&&>(rcvr))
      , carried_(static_cast<CarriedInit&&>(carried))
    {
      enter<First>(static_cast<Args&&>(first)...);
    }

    phased(phased&&) = delete;

    ~phased()
    {
      destroy_live();
    }

    auto first() noexcept -> First&
    {
      return get<First>();
    }

    auto carried() noexcept -> Carried&
    {
      return carried_;
    }

    auto receiver() noexcept -> Rcvr&
    {
      return rcvr_;
    }

    template <class From, class To, class Prepare, class... Extra>
    void transition(Prepare prepare, Extra... extra) noexcept
    {
      STDEXEC_TRY
      {
        auto bundle = prepare(get<From>());
        leave<From>();
        To& to = std::apply([&](auto&... parts) -> To&
                            { return enter<To>(std::move(parts)..., extra...); },
                            bundle);
        ex::start(to.child);
      }
      STDEXEC_CATCH_ALL
      {
        fail();
      }
    }

    template <class Phase, class Tag, class... As>
    void finish(Tag, As&&... as) noexcept
    {
      if constexpr (eager)
      {
        STDEXEC_TRY
        {
          std::tuple<std::decay_t<As>...> results(static_cast<As&&>(as)...);
          leave<Phase>();
          std::apply([this](auto&... rs) { Tag()(static_cast<Rcvr&&>(rcvr_), std::move(rs)...); },
                     results);
        }
        STDEXEC_CATCH_ALL
        {
          fail();
        }
      }
      else
      {
        Tag()(static_cast<Rcvr&&>(rcvr_), static_cast<As&&>(as)...);
      }
    }

    template <class Phase, class Fn, class... As>
    void finish_with(Fn& fn, As&&... as) noexcept
    {
      STDEXEC_TRY
      {
        if constexpr (eager)
        {
          std::tuple<std::decay_t<As>...> args(static_cast<As&&>(as)...);
          leave<Phase>();
          complete_with([&]() -> decltype(auto) { return std::apply(fn, std::move(args)); });
        }
        else
        {
          complete_with([&]() -> decltype(auto) { return fn(static_cast<As&&>(as)...); });
        }
      }
      STDEXEC_CATCH_ALL
      {
        fail();
      }
    }

   private:
    template <class Call>
    void complete_with(Call call)
    {
      if constexpr (std::is_void_v<decltype(call())>)
      {
        call();
        ex::set_value(static_cast<Rcvr&&>(rcvr_));
      }
      else
      {
        std::decay_t<decltype(call())> result = call();
        ex::set_value(static_cast<Rcvr&&>(rcvr_), std::move(result));
      }
    }

    template <class Phase>
    auto get() noexcept -> Phase&
    {
      return *std::launder(reinterpret_cast<Phase*>(storage_));
    }

    template <class Phase, class... Args>
    auto enter(Args&&... args) -> Phase&
    {
      static_assert(index_of<Phase> != none, "not one of this operation's phases");
      Phase& phase = *::new (static_cast<void*>(storage_)) Phase(static_cast<Args&&>(args)...);
      live_        = index_of<Phase>;
      return phase;
    }

    template <class Phase>
    void leave() noexcept
    {
      std::destroy_at(&get<Phase>());
      live_ = none;
    }

    void destroy_live() noexcept
    {
      [this]<std::size_t... Is>(std::index_sequence<Is...>)
      {
        ((live_ == Is ? leave<std::tuple_element_t<Is, std::tuple<First, Rest...>>>() : void()),
         ...);
      }(std::index_sequence_for<First, Rest...>());
    }

    void fail() noexcept
    {
      if constexpr (eager)
      {
        destroy_live();
      }
      ex::set_error(static_cast<Rcvr&&>(rcvr_), std::current_exception());
    }

    static constexpr std::size_t storage_size  = std::max({sizeof(First), sizeof(Rest)...});
    static constexpr std::size_t storage_align = std::max({alignof(First), alignof(Rest)...});

    Rcvr rcvr_;
    STDEXEC_ATTRIBUTE(no_unique_address) Carried carried_;
    unsigned char live_ = none;
    alignas(storage_align) std::byte storage_[storage_size];
  };

  // phased<Ending, Rcvr, Carried, First, Next<Ts...>...>, one Next per
  // type_list<Ts...>.
  template <class Ending,
            class Rcvr,
            class Carried,
            class First,
            template <class...> class Next,
            class Lists>
  struct phased_with;

  template <class Ending,
            class Rcvr,
            class Carried,
            class First,
            template <class...> class Next,
            class... Lists>
  struct phased_with<Ending, Rcvr, Carried, First, Next, detail::type_list<Lists...>>
  {
    template <class List>
    struct apply;

    template <class... Ts>
    struct apply<detail::type_list<Ts...>>
    {
      using type = Next<Ts...>;
    };

    using type = phased<Ending, Rcvr, Carried, First, typename apply<Lists>::type...>;
  };

  // The receiver a phase's child completes through. `Tag` names the phase.
  template <class Op, class Rcvr, class Tag>
  struct phase_receiver
  {
    using receiver_concept = ex::receiver_t;

    template <class... As>
    void set_value(As&&... as) && noexcept
    {
      op->complete(Tag(), ex::set_value, static_cast<As&&>(as)...);
    }

    template <class E>
    void set_error(E&& e) && noexcept
    {
      op->complete(Tag(), ex::set_error, static_cast<E&&>(e));
    }

    void set_stopped() && noexcept
    {
      op->complete(Tag(), ex::set_stopped);
    }

    [[nodiscard]]
    auto get_env() const noexcept -> ex::env_of_t<Rcvr>
    {
      return ex::get_env(op->phases_.receiver());
    }

    Op* op;
  };

  // Nothing to carry.
  struct nothing
  {};

  //////////////////////////////////////////////////////////////////////////////
  // just(vs...): one phase, holding the values.

  template <class Ending, class... Vs>
  struct just_sender
  {
    using sender_concept = ex::sender_t;

    static constexpr bool may_throw = std::is_same_v<Ending, eager_teardown>
                                   && !(std::is_nothrow_move_constructible_v<Vs> && ...);

    using completion_signatures = std::conditional_t<
      may_throw,
      ex::completion_signatures<ex::set_value_t(Vs...), ex::set_error_t(std::exception_ptr)>,
      ex::completion_signatures<ex::set_value_t(Vs...)>>;

    template <class Rcvr>
    struct opstate
    {
      struct ready
      {
        explicit ready(std::tuple<Vs...>&& vs)
          : values(std::move(vs))
        {}

        std::tuple<Vs...> values;
      };

      opstate(std::tuple<Vs...>&& vs, Rcvr rcvr)
        : phases_(static_cast<Rcvr&&>(rcvr), nothing(), std::move(vs))
      {}

      void start() & noexcept
      {
        std::apply([this](Vs&... vs)
                   { phases_.template finish<ready>(ex::set_value, std::move(vs)...); },
                   phases_.first().values);
      }

      phased<Ending, Rcvr, nothing, ready> phases_;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) && -> opstate<Rcvr>
    {
      return {std::move(values_), static_cast<Rcvr&&>(rcvr)};
    }

    std::tuple<Vs...> values_;
  };

  template <class Ending = eager_teardown, class... Vs>
  auto just(Vs&&... vs)
  {
    return just_sender<Ending, std::decay_t<Vs>...>{{static_cast<Vs&&>(vs)...}};
  }

  //////////////////////////////////////////////////////////////////////////////
  // then(sndr, fn): one phase, running the child; fn is carried.

  template <class Ending, class Sndr, class Fn>
  struct then_sender
  {
    using sender_concept = ex::sender_t;

    // After eager teardown, fn is called with rvalue copies of the values.
    struct values_fn
    {
      template <class... As>
      using result_t = typename std::conditional_t<std::is_same_v<Ending, eager_teardown>,
                                                   std::invoke_result<Fn&, std::decay_t<As>...>,
                                                   std::invoke_result<Fn&, As...>>::type;

      template <class... As>
      using f = typename detail::value_of<result_t<As...>>::type;
    };

    template <class Self, class... Env>
    static consteval auto get_completion_signatures()
    {
      if constexpr (sizeof...(Env) == 0)
      {
        return STDEXEC::__throw_dependent_sender_error<Self>();
      }
      else
      {
        return detail::map_values_t<Ending,
                                    ex::completion_signatures_of_t<Sndr, Env...>,
                                    values_fn>();
      }
    }

    template <class Rcvr>
    struct opstate
    {
      struct running_tag
      {};

      using rcvr_t = phase_receiver<opstate, Rcvr, running_tag>;

      struct carried
      {
        Fn fn;
      };

      struct running
      {
        running(Sndr&& sndr, opstate* op)
          : child(ex::connect(static_cast<Sndr&&>(sndr), rcvr_t{op}))
        {}

        ex::connect_result_t<Sndr, rcvr_t> child;
      };

      opstate(Sndr&& sndr, Fn&& fn, Rcvr rcvr)
        : phases_(static_cast<Rcvr&&>(rcvr),
                  carried{static_cast<Fn&&>(fn)},
                  static_cast<Sndr&&>(sndr),
                  this)
      {}

      void start() & noexcept
      {
        ex::start(phases_.first().child);
      }

      template <class... As>
      void complete(running_tag, ex::set_value_t, As&&... as) noexcept
      {
        phases_.template finish_with<running>(phases_.carried().fn, static_cast<As&&>(as)...);
      }

      template <class Tag, class... As>
      void complete(running_tag, Tag, As&&... as) noexcept
      {
        phases_.template finish<running>(Tag(), static_cast<As&&>(as)...);
      }

      phased<Ending, Rcvr, carried, running> phases_;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) && -> opstate<Rcvr>
    {
      return {static_cast<Sndr&&>(sndr_), static_cast<Fn&&>(fn_), static_cast<Rcvr&&>(rcvr)};
    }

    Sndr sndr_;
    Fn   fn_;
  };

  template <class Ending = eager_teardown, class Sndr, class Fn>
  auto then(Sndr&& sndr, Fn&& fn)
  {
    return then_sender<Ending, std::decay_t<Sndr>, std::decay_t<Fn>>{static_cast<Sndr&&>(sndr),
                                                                     static_cast<Fn&&>(fn)};
  }

  //////////////////////////////////////////////////////////////////////////////
  // let_value(sndr, fn): a first phase running the child, then one possible
  // second phase per value signature of the child, running the sender fn
  // returns. fn is carried: the sender it returns may refer to its captures.
  // The values fn is called with are carried too (carried_layout), or live
  // in the second phase, reusing the first phase's bytes (overlapped_layout). Each second phase has its own
  // receiver type, so its completion knows which one is live.

  template <class Ending, class Layout, class Sndr, class Fn>
  struct let_value_sender
  {
    using sender_concept = ex::sender_t;

    static constexpr bool overlapped = std::is_same_v<Layout, overlapped_layout>;

    template <class Env>
    struct values_fn
    {
      template <class... As>
      using f =
        ex::completion_signatures_of_t<std::invoke_result_t<Fn&, std::decay_t<As>&...>, Env>;
    };

    template <class Self, class... Env>
    static consteval auto get_completion_signatures()
    {
      if constexpr (sizeof...(Env) == 0)
      {
        return STDEXEC::__throw_dependent_sender_error<Self>();
      }
      else
      {
        return detail::map_values_t<Ending,
                                    ex::completion_signatures_of_t<Sndr, Env...>,
                                    values_fn<Env...>>();
      }
    }

    template <class Rcvr>
    struct opstate
    {
      struct running_tag
      {};

      template <class... Ts>
      struct continuing_tag
      {};

      using first_rcvr_t = phase_receiver<opstate, Rcvr, running_tag>;
      using values_t     = typename detail::value_lists<
            ex::completion_signatures_of_t<Sndr, ex::env_of_t<Rcvr>>>::type;

      struct carried
      {
        Fn fn;
        STDEXEC_ATTRIBUTE(no_unique_address)
        std::conditional_t<overlapped, nothing, typename detail::args_variant<values_t>::type> args;
      };

      struct running
      {
        running(Sndr&& sndr, opstate* op)
          : child(ex::connect(static_cast<Sndr&&>(sndr), first_rcvr_t{op}))
        {}

        ex::connect_result_t<Sndr, first_rcvr_t> child;
      };

      template <class... Ts>
      struct continuing
      {
        using rcvr_t   = phase_receiver<opstate, Rcvr, continuing_tag<Ts...>>;
        using args_t   = std::tuple<Ts...>;
        using stored_t = std::conditional_t<overlapped, args_t, nothing>;
        using sndr_t   = std::invoke_result_t<Fn&, Ts&...>;

        continuing(stored_t&& stored, opstate* op)
          : stored(std::move(stored))
          , child(ex::connect(std::apply(op->phases_.carried().fn, args(op)), rcvr_t{op}))
        {}

        auto args(opstate* op) noexcept -> args_t&
        {
          if constexpr (overlapped)
          {
            return stored;
          }
          else
          {
            return std::get<args_t>(op->phases_.carried().args);
          }
        }

        STDEXEC_ATTRIBUTE(no_unique_address) stored_t stored;
        ex::connect_result_t<sndr_t, rcvr_t> child;
      };

      opstate(Sndr&& sndr, Fn&& fn, Rcvr rcvr)
        : phases_(static_cast<Rcvr&&>(rcvr),
                  carried{static_cast<Fn&&>(fn), {}},
                  static_cast<Sndr&&>(sndr),
                  this)
      {}

      void start() & noexcept
      {
        ex::start(phases_.first().child);
      }

      template <class... As>
      void complete(running_tag, ex::set_value_t, As&&... as) noexcept
      {
        using next_t = continuing<std::decay_t<As>...>;
        using args_t = typename next_t::args_t;

        phases_.template transition<running, next_t>(
          [&](running&)
          {
            if constexpr (overlapped)
            {
              return std::tuple<args_t>(args_t(static_cast<As&&>(as)...));
            }
            else
            {
              phases_.carried().args.template emplace<args_t>(static_cast<As&&>(as)...);
              return std::tuple<nothing>();
            }
          },
          this);
      }

      template <class Tag, class... As>
      void complete(running_tag, Tag, As&&... as) noexcept
      {
        phases_.template finish<running>(Tag(), static_cast<As&&>(as)...);
      }

      template <class... Ts, class Tag, class... As>
      void complete(continuing_tag<Ts...>, Tag, As&&... as) noexcept
      {
        phases_.template finish<continuing<Ts...>>(Tag(), static_cast<As&&>(as)...);
      }

      typename phased_with<Ending, Rcvr, carried, running, continuing, values_t>::type phases_;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) && -> opstate<Rcvr>
    {
      return {static_cast<Sndr&&>(sndr_), static_cast<Fn&&>(fn_), static_cast<Rcvr&&>(rcvr)};
    }

    Sndr sndr_;
    Fn   fn_;
  };

  template <class Ending = eager_teardown, class Layout = carried_layout, class Sndr, class Fn>
  auto let_value(Sndr&& sndr, Fn&& fn)
  {
    return let_value_sender<Ending, Layout, std::decay_t<Sndr>, std::decay_t<Fn>>{
      static_cast<Sndr&&>(sndr),
      static_cast<Fn&&>(fn)};
  }

  //////////////////////////////////////////////////////////////////////////////
  // sequence(first, second): runs first, discards its values, then runs
  // second. The second sender is carried until it's connected
  // (carried_layout), or waits in the first phase and crosses to the second
  // on the stack (overlapped_layout).

  template <class Ending, class Layout, class Sndr1, class Sndr2>
  struct sequence_sender
  {
    using sender_concept = ex::sender_t;

    static constexpr bool overlapped = std::is_same_v<Layout, overlapped_layout>;

    template <class Env>
    struct values_fn
    {
      template <class...>
      using f = ex::completion_signatures_of_t<Sndr2, Env>;
    };

    template <class Self, class... Env>
    static consteval auto get_completion_signatures()
    {
      if constexpr (sizeof...(Env) == 0)
      {
        return STDEXEC::__throw_dependent_sender_error<Self>();
      }
      else
      {
        return detail::map_values_t<Ending,
                                    ex::completion_signatures_of_t<Sndr1, Env...>,
                                    values_fn<Env...>>();
      }
    }

    template <class Rcvr>
    struct opstate
    {
      struct first_tag
      {};

      struct second_tag
      {};

      using first_rcvr_t  = phase_receiver<opstate, Rcvr, first_tag>;
      using second_rcvr_t = phase_receiver<opstate, Rcvr, second_tag>;
      using next_t        = std::conditional_t<overlapped, Sndr2, nothing>;

      struct carried
      {
        STDEXEC_ATTRIBUTE(no_unique_address) std::conditional_t<overlapped, nothing, Sndr2> next;
      };

      struct first
      {
        first(Sndr1&& sndr1, next_t&& next, opstate* op)
          : next(static_cast<next_t&&>(next))
          , child(ex::connect(static_cast<Sndr1&&>(sndr1), first_rcvr_t{op}))
        {}

        STDEXEC_ATTRIBUTE(no_unique_address) next_t next;
        ex::connect_result_t<Sndr1, first_rcvr_t> child;
      };

      struct second
      {
        second(next_t&& next, opstate* op)
          : child(ex::connect(static_cast<Sndr2&&>(sender(next, op)), second_rcvr_t{op}))
        {}

        static auto sender(next_t& next, opstate* op) noexcept -> Sndr2&
        {
          if constexpr (overlapped)
          {
            return next;
          }
          else
          {
            return op->phases_.carried().next;
          }
        }

        ex::connect_result_t<Sndr2, second_rcvr_t> child;
      };

      opstate(Sndr1&& sndr1, Sndr2&& sndr2, Rcvr rcvr)
        : phases_(static_cast<Rcvr&&>(rcvr),
                  make_carried(sndr2),
                  static_cast<Sndr1&&>(sndr1),
                  make_next(sndr2),
                  this)
      {}

      static auto make_carried(Sndr2& sndr2) -> carried
      {
        if constexpr (overlapped)
        {
          return {};
        }
        else
        {
          return {static_cast<Sndr2&&>(sndr2)};
        }
      }

      static auto make_next(Sndr2& sndr2) -> next_t
      {
        if constexpr (overlapped)
        {
          return static_cast<Sndr2&&>(sndr2);
        }
        else
        {
          return {};
        }
      }

      void start() & noexcept
      {
        ex::start(phases_.first().child);
      }

      template <class... As>
      void complete(first_tag, ex::set_value_t, As&&...) noexcept
      {
        phases_.template transition<first, second>(
          [](first& f) { return std::tuple<next_t>(static_cast<next_t&&>(f.next)); },
          this);
      }

      template <class Tag, class... As>
      void complete(first_tag, Tag, As&&... as) noexcept
      {
        phases_.template finish<first>(Tag(), static_cast<As&&>(as)...);
      }

      template <class Tag, class... As>
      void complete(second_tag, Tag, As&&... as) noexcept
      {
        phases_.template finish<second>(Tag(), static_cast<As&&>(as)...);
      }

      phased<Ending, Rcvr, carried, first, second> phases_;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) && -> opstate<Rcvr>
    {
      return {static_cast<Sndr1&&>(sndr1_),
              static_cast<Sndr2&&>(sndr2_),
              static_cast<Rcvr&&>(rcvr)};
    }

    Sndr1 sndr1_;
    Sndr2 sndr2_;
  };

  template <class Ending = eager_teardown, class Layout = carried_layout, class Sndr1, class Sndr2>
  auto sequence(Sndr1&& sndr1, Sndr2&& sndr2)
  {
    return sequence_sender<Ending, Layout, std::decay_t<Sndr1>, std::decay_t<Sndr2>>{
      static_cast<Sndr1&&>(sndr1),
      static_cast<Sndr2&&>(sndr2)};
  }
}  // namespace frames::typestate
