/*
 * Exploration: operation states as explicit sequences of phases.
 *
 * An algorithm's operation state holds one phase at a time in storage sized
 * for the largest phase. Which phase is live isn't stored anywhere: each
 * phase's child completes through a receiver whose *type* names the phase,
 * so every completion handler knows statically what its storage holds.
 *
 * A transition happens inside the live phase's completion:
 *
 *   1. move what the next phase needs onto the stack,
 *   2. destroy the live phase,
 *   3. construct the next phase in the same storage and start it.
 *
 * This relies on an operation never touching its own state after it has
 * signalled completion, which stdexec's own let_value relies on too.
 *
 * The last phase is torn down the same way before the parent receiver is
 * completed ("eager teardown"), so the operation state's destructor only has
 * one thing to know: whether the operation was ever started.
 *
 * Prototype simplifications: senders are connected as rvalues only, every
 * algorithm conservatively declares set_error_t(std::exception_ptr), results
 * are always decay-copied, and let_value doesn't add a completion scheduler to
 * its second phase's environment.
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

namespace frames::typestate
{
  namespace ex = stdexec;

  //////////////////////////////////////////////////////////////////////////////
  // Completion-signature bookkeeping. Every set this produces is decayed and
  // includes set_error_t(std::exception_ptr).

  namespace detail
  {
    template <class Sig>
    struct decay_sig;

    template <class Tag, class... As>
    struct decay_sig<Tag(As...)>
    {
      using type = Tag(std::decay_t<As>...);
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

    // The union of several signature sets, decayed.
    template <class... Sets>
    struct join
    {
      using type = ex::completion_signatures<>;
    };

    template <class... Sigs, class... Rest>
    struct join<ex::completion_signatures<Sigs...>, Rest...>
      : add_all<typename join<Rest...>::type, typename decay_sig<Sigs>::type...>
    {};

    template <class... Sets>
    using join_t = typename join<Sets...>::type;

    // Replaces each set_value_t(As...) in a set by Fn::f<decay_t<As>...>.
    template <class Fn, class Sig>
    struct map_one
    {
      using type = ex::completion_signatures<Sig>;
    };

    template <class Fn, class... As>
    struct map_one<Fn, ex::set_value_t(As...)>
    {
      using type = typename Fn::template f<std::decay_t<As>...>;
    };

    template <class Sigs, class Fn>
    struct map_values;

    template <class... Sigs, class Fn>
    struct map_values<ex::completion_signatures<Sigs...>, Fn>
    {
      using type = join_t<typename map_one<Fn, Sigs>::type...,
                          ex::completion_signatures<ex::set_error_t(std::exception_ptr)>>;
    };

    template <class Sigs, class Fn>
    using map_values_t = typename map_values<Sigs, Fn>::type;

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

    // type_list<type_list<decay_t<As>...>...>, one per set_value_t(As...).
    template <class Sigs>
    struct value_lists;

    template <class... Sigs>
    struct value_lists<ex::completion_signatures<Sigs...>>
    {
      template <class Sig>
      struct one
      {
        using type = type_list<>;
      };

      template <class... As>
      struct one<ex::set_value_t(As...)>
      {
        using type = type_list<type_list<std::decay_t<As>...>>;
      };

      template <class... Lists>
      struct cat
      {
        using type = type_list<>;
      };

      template <class... As, class... Bs, class... Rest>
      struct cat<type_list<As...>, type_list<Bs...>, Rest...>
      {
        using type = typename cat<type_list<As..., Bs...>, Rest...>::type;
      };

      template <class... As>
      struct cat<type_list<As...>>
      {
        using type = type_list<As...>;
      };

      using type = typename cat<typename one<Sigs>::type...>::type;
    };
  }  // namespace detail

  //////////////////////////////////////////////////////////////////////////////
  // phased: the receiver of the whole operation, plus storage for whichever
  // phase is live.

  template <class Rcvr, class First, class... Rest>
  class phased
  {
   public:
    explicit phased(Rcvr rcvr) noexcept(std::is_nothrow_move_constructible_v<Rcvr>)
      : rcvr_(static_cast<Rcvr&&>(rcvr))
    {}

    phased(phased&&) = delete;

    // An operation that was started has torn down its last phase before
    // completing; one that wasn't still holds its first.
    ~phased()
    {
      if (unstarted_)
      {
        leave<First>();
      }
    }

    template <class... Args>
    auto enter_first(Args&&... args) -> First&
    {
      First& first = enter<First>(static_cast<Args&&>(args)...);
      unstarted_   = true;
      return first;
    }

    auto start() noexcept -> First&
    {
      unstarted_ = false;
      return get<First>();
    }

    template <class Phase, class... Args>
    auto enter(Args&&... args) -> Phase&
    {
      static_assert((std::is_same_v<Phase, First> || ... || std::is_same_v<Phase, Rest>) );
      return *::new (static_cast<void*>(storage_)) Phase(static_cast<Args&&>(args)...);
    }

    template <class Phase>
    auto get() noexcept -> Phase&
    {
      return *std::launder(reinterpret_cast<Phase*>(storage_));
    }

    template <class Phase>
    void leave() noexcept
    {
      std::destroy_at(&get<Phase>());
    }

    // Completes the operation from inside `Phase`'s completion: moves the
    // results onto the stack (they may refer into the phase), destroys the
    // phase, then completes the receiver.
    template <class Phase, class Tag, class... As>
    void finish(Tag, As&&... as) noexcept
    {
      [[maybe_unused]]
      bool live = true;
      STDEXEC_TRY
      {
        std::tuple<std::decay_t<As>...> results(static_cast<As&&>(as)...);
        leave<Phase>();
        live = false;
        std::apply([this](auto&... rs) { Tag()(static_cast<Rcvr&&>(rcvr_), std::move(rs)...); },
                   results);
      }
      STDEXEC_CATCH_ALL
      {
        if (live)
        {
          leave<Phase>();
        }
        fail();
      }
    }

    // Completes with the current exception. No phase may be live.
    void fail() noexcept
    {
      ex::set_error(static_cast<Rcvr&&>(rcvr_), std::current_exception());
    }

    auto receiver() noexcept -> Rcvr&
    {
      return rcvr_;
    }

    static constexpr std::size_t storage_size  = std::max({sizeof(First), sizeof(Rest)...});
    static constexpr std::size_t storage_align = std::max({alignof(First), alignof(Rest)...});

   private:
    Rcvr rcvr_;
    bool unstarted_ = false;
    alignas(storage_align) std::byte storage_[storage_size];
  };

  // phased<Rcvr, First, Next<Ts...>...>, one Next per type_list<Ts...>.
  template <class Rcvr, class First, template <class...> class Next, class Lists>
  struct phased_with;

  template <class Rcvr, class First, template <class...> class Next, class... Lists>
  struct phased_with<Rcvr, First, Next, detail::type_list<Lists...>>
  {
    template <class List>
    struct apply;

    template <class... Ts>
    struct apply<detail::type_list<Ts...>>
    {
      using type = Next<Ts...>;
    };

    using type = phased<Rcvr, First, typename apply<Lists>::type...>;
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

  //////////////////////////////////////////////////////////////////////////////
  // just(vs...): one phase, holding the values. Starting moves them onto the
  // stack and ends the phase before completing.

  template <class... Vs>
  struct just_sender
  {
    using sender_concept        = ex::sender_t;
    using completion_signatures = detail::join_t<
      ex::completion_signatures<ex::set_value_t(Vs...)>,
      std::conditional_t<(std::is_nothrow_move_constructible_v<Vs> && ...),
                         ex::completion_signatures<>,
                         ex::completion_signatures<ex::set_error_t(std::exception_ptr)>>>;

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
        : phases_(static_cast<Rcvr&&>(rcvr))
      {
        phases_.enter_first(std::move(vs));
      }

      void start() & noexcept
      {
        std::apply([this](Vs&... vs)
                   { phases_.template finish<ready>(ex::set_value, std::move(vs)...); },
                   phases_.start().values);
      }

      phased<Rcvr, ready> phases_;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) && -> opstate<Rcvr>
    {
      return {std::move(values_), static_cast<Rcvr&&>(rcvr)};
    }

    std::tuple<Vs...> values_;
  };

  template <class... Vs>
  auto just(Vs&&... vs)
  {
    return just_sender<std::decay_t<Vs>...>{{static_cast<Vs&&>(vs)...}};
  }

  //////////////////////////////////////////////////////////////////////////////
  // then(sndr, fn): one phase, running the child. On a value, fn and the
  // values move onto the stack, the phase ends, and fn runs.

  template <class Sndr, class Fn>
  struct then_sender
  {
    using sender_concept = ex::sender_t;

    struct values_fn
    {
      template <class... As>
      using f = typename detail::value_of<std::invoke_result_t<Fn&, As...>>::type;
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
        return detail::map_values_t<ex::completion_signatures_of_t<Sndr, Env...>, values_fn>();
      }
    }

    template <class Rcvr>
    struct opstate
    {
      struct running_tag
      {};

      using rcvr_t = phase_receiver<opstate, Rcvr, running_tag>;

      struct running
      {
        running(Sndr&& sndr, Fn&& fn, opstate* op)
          : fn(static_cast<Fn&&>(fn))
          , child(ex::connect(static_cast<Sndr&&>(sndr), rcvr_t{op}))
        {}

        Fn                                 fn;
        ex::connect_result_t<Sndr, rcvr_t> child;
      };

      opstate(Sndr&& sndr, Fn&& fn, Rcvr rcvr)
        : phases_(static_cast<Rcvr&&>(rcvr))
      {
        phases_.enter_first(static_cast<Sndr&&>(sndr), static_cast<Fn&&>(fn), this);
      }

      void start() & noexcept
      {
        ex::start(phases_.start().child);
      }

      template <class... As>
      void complete(running_tag, ex::set_value_t, As&&... as) noexcept
      {
        [[maybe_unused]]
        bool live = true;
        STDEXEC_TRY
        {
          Fn fn = static_cast<Fn&&>(phases_.template get<running>().fn);
          std::tuple<std::decay_t<As>...> args(static_cast<As&&>(as)...);
          phases_.template leave<running>();
          live = false;

          using result_t = std::invoke_result_t<Fn&, std::decay_t<As>...>;
          if constexpr (std::is_void_v<result_t>)
          {
            std::apply(fn, std::move(args));
            ex::set_value(static_cast<Rcvr&&>(phases_.receiver()));
          }
          else
          {
            std::decay_t<result_t> result = std::apply(fn, std::move(args));
            ex::set_value(static_cast<Rcvr&&>(phases_.receiver()), std::move(result));
          }
        }
        STDEXEC_CATCH_ALL
        {
          if (live)
          {
            phases_.template leave<running>();
          }
          phases_.fail();
        }
      }

      template <class Tag, class... As>
      void complete(running_tag, Tag, As&&... as) noexcept
      {
        phases_.template finish<running>(Tag(), static_cast<As&&>(as)...);
      }

      phased<Rcvr, running> phases_;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) && -> opstate<Rcvr>
    {
      return {static_cast<Sndr&&>(sndr_), static_cast<Fn&&>(fn_), static_cast<Rcvr&&>(rcvr)};
    }

    Sndr sndr_;
    Fn   fn_;
  };

  template <class Sndr, class Fn>
  auto then(Sndr&& sndr, Fn&& fn)
  {
    return then_sender<std::decay_t<Sndr>, std::decay_t<Fn>>{static_cast<Sndr&&>(sndr),
                                                             static_cast<Fn&&>(fn)};
  }

  //////////////////////////////////////////////////////////////////////////////
  // let_value(sndr, fn): a first phase running the child, then one possible
  // second phase per value signature of the child, each holding fn, the
  // values and the operation of the sender fn returns. Each second phase has
  // its own receiver type, so its completion knows which one is live.

  template <class Sndr, class Fn>
  struct let_value_sender
  {
    using sender_concept = ex::sender_t;

    template <class Env>
    struct values_fn
    {
      template <class... As>
      using f = ex::completion_signatures_of_t<std::invoke_result_t<Fn&, As&...>, Env>;
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
        return detail::map_values_t<ex::completion_signatures_of_t<Sndr, Env...>,
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

      struct running
      {
        running(Sndr&& sndr, Fn&& fn, opstate* op)
          : fn(static_cast<Fn&&>(fn))
          , child(ex::connect(static_cast<Sndr&&>(sndr), first_rcvr_t{op}))
        {}

        Fn                                       fn;
        ex::connect_result_t<Sndr, first_rcvr_t> child;
      };

      // fn moves here too: the sender it returns may refer to its captures.
      template <class... Ts>
      struct continuing
      {
        using rcvr_t = phase_receiver<opstate, Rcvr, continuing_tag<Ts...>>;
        using sndr_t = std::invoke_result_t<Fn&, Ts&...>;

        continuing(Fn&& fn, std::tuple<Ts...>&& args, opstate* op)
          : fn(static_cast<Fn&&>(fn))
          , args(std::move(args))
          , child(ex::connect(std::apply(this->fn, this->args), rcvr_t{op}))
        {}

        Fn                                   fn;
        std::tuple<Ts...>                    args;
        ex::connect_result_t<sndr_t, rcvr_t> child;
      };

      using values_t = typename detail::value_lists<
        ex::completion_signatures_of_t<Sndr, ex::env_of_t<Rcvr>>>::type;

      opstate(Sndr&& sndr, Fn&& fn, Rcvr rcvr)
        : phases_(static_cast<Rcvr&&>(rcvr))
      {
        phases_.enter_first(static_cast<Sndr&&>(sndr), static_cast<Fn&&>(fn), this);
      }

      void start() & noexcept
      {
        ex::start(phases_.start().child);
      }

      template <class... As>
      void complete(running_tag, ex::set_value_t, As&&... as) noexcept
      {
        using next_t = continuing<std::decay_t<As>...>;

        [[maybe_unused]]
        bool live = true;
        STDEXEC_TRY
        {
          Fn fn = static_cast<Fn&&>(phases_.template get<running>().fn);
          std::tuple<std::decay_t<As>...> args(static_cast<As&&>(as)...);
          phases_.template leave<running>();
          live = false;

          next_t& next = phases_.template enter<next_t>(std::move(fn), std::move(args), this);
          ex::start(next.child);
        }
        STDEXEC_CATCH_ALL
        {
          if (live)
          {
            phases_.template leave<running>();
          }
          phases_.fail();
        }
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

      typename phased_with<Rcvr, running, continuing, values_t>::type phases_;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) && -> opstate<Rcvr>
    {
      return {static_cast<Sndr&&>(sndr_), static_cast<Fn&&>(fn_), static_cast<Rcvr&&>(rcvr)};
    }

    Sndr sndr_;
    Fn   fn_;
  };

  template <class Sndr, class Fn>
  auto let_value(Sndr&& sndr, Fn&& fn)
  {
    return let_value_sender<std::decay_t<Sndr>, std::decay_t<Fn>>{static_cast<Sndr&&>(sndr),
                                                                  static_cast<Fn&&>(fn)};
  }

  //////////////////////////////////////////////////////////////////////////////
  // sequence(first, second): runs first, discards its values, then runs
  // second. The second sender waits inside the first phase and crosses to
  // the second phase on the stack.

  template <class Sndr1, class Sndr2>
  struct sequence_sender
  {
    using sender_concept = ex::sender_t;

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
        return detail::map_values_t<ex::completion_signatures_of_t<Sndr1, Env...>,
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

      struct first
      {
        first(Sndr1&& sndr1, Sndr2&& sndr2, opstate* op)
          : next(static_cast<Sndr2&&>(sndr2))
          , child(ex::connect(static_cast<Sndr1&&>(sndr1), first_rcvr_t{op}))
        {}

        Sndr2                                     next;
        ex::connect_result_t<Sndr1, first_rcvr_t> child;
      };

      struct second
      {
        second(Sndr2&& sndr2, opstate* op)
          : child(ex::connect(static_cast<Sndr2&&>(sndr2), second_rcvr_t{op}))
        {}

        ex::connect_result_t<Sndr2, second_rcvr_t> child;
      };

      opstate(Sndr1&& sndr1, Sndr2&& sndr2, Rcvr rcvr)
        : phases_(static_cast<Rcvr&&>(rcvr))
      {
        phases_.enter_first(static_cast<Sndr1&&>(sndr1), static_cast<Sndr2&&>(sndr2), this);
      }

      void start() & noexcept
      {
        ex::start(phases_.start().child);
      }

      template <class... As>
      void complete(first_tag, ex::set_value_t, As&&...) noexcept
      {
        [[maybe_unused]]
        bool live = true;
        STDEXEC_TRY
        {
          Sndr2 next = static_cast<Sndr2&&>(phases_.template get<first>().next);
          phases_.template leave<first>();
          live = false;

          ex::start(phases_.template enter<second>(std::move(next), this).child);
        }
        STDEXEC_CATCH_ALL
        {
          if (live)
          {
            phases_.template leave<first>();
          }
          phases_.fail();
        }
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

      phased<Rcvr, first, second> phases_;
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

  template <class Sndr1, class Sndr2>
  auto sequence(Sndr1&& sndr1, Sndr2&& sndr2)
  {
    return sequence_sender<std::decay_t<Sndr1>, std::decay_t<Sndr2>>{static_cast<Sndr1&&>(sndr1),
                                                                     static_cast<Sndr2&&>(sndr2)};
  }
}  // namespace frames::typestate
