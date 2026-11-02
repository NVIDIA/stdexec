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

#if defined(__SANITIZE_ADDRESS__)
#  define FRAMES_ASAN 1
#elif defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define FRAMES_ASAN 1
#  endif
#endif
#if !defined(FRAMES_ASAN)
#  define FRAMES_ASAN 0
#endif
#if FRAMES_ASAN
#  include <sanitizer/asan_interface.h>
#endif

namespace frames::typestate
{
  namespace ex = stdexec;

  // How an operation ends; see above.
  struct eager_teardown
  {};

  struct forward_results
  {};

  // When an operation's first phase is entered: by connect, or by start.
  struct connect_entry
  {};

  struct start_entry
  {};

  // Where state that crosses a transition is kept: let_value's values, or
  // sequence's second sender.
  struct carried_layout  // beside the phases, for the whole operation, as stdexec does
  {};

  struct overlapped_layout  // in the phases, moving across on the stack
  {};

  struct scoped_layout  // beside the phases, for as long as the phase that uses it
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
    // Empty; constructible from anything, which it ignores.
    struct empty
    {
      empty() = default;

      template <class... Args>
      constexpr explicit empty(Args&&...) noexcept
      {}
    };

    // Bytes that AddressSanitizer reports any access to until they're
    // unpoisoned, and that are unpoisoned again when this is destroyed.
    template <std::size_t Size, std::size_t Align>
    struct guarded_bytes
    {
      guarded_bytes() noexcept
      {
        poison(Size);
      }

      guarded_bytes(guarded_bytes&&) = delete;

      ~guarded_bytes()
      {
        unpoison(Size);
      }

      void poison([[maybe_unused]] std::size_t size) noexcept
      {
#if FRAMES_ASAN
        __asan_poison_memory_region(bytes, size);
#endif
      }

      void unpoison([[maybe_unused]] std::size_t size) noexcept
      {
#if FRAMES_ASAN
        __asan_unpoison_memory_region(bytes, size);
#endif
      }

      alignas(Align) std::byte bytes[Size];
    };

    struct no_scoped_state
    {};

    // A phase's scoped state: Phase::scoped_t if it names one other than
    // void, otherwise void.
    template <class Phase>
    struct scoped_of
    {
      using type = void;
    };

    template <class Phase>
      requires requires { typename Phase::scoped_t; }
    struct scoped_of<Phase>
    {
      using type = typename Phase::scoped_t;
    };

    template <class T>
    inline constexpr std::size_t size_of = sizeof(T);

    template <>
    inline constexpr std::size_t size_of<void> = 0;

    template <class T>
    inline constexpr std::size_t align_of = alignof(T);

    template <>
    inline constexpr std::size_t align_of<void> = 1;
  }  // namespace detail

  // What prepare returns when the phase it prepares has scoped state: the
  // arguments for that state, which may refer into the dying phase, and the
  // arguments for the phase itself.
  template <class ScopedArgs, class Args>
  struct with_scoped
  {
    ScopedArgs scoped;
    Args       args;
  };

  template <class ScopedArgs, class Args>
  with_scoped(ScopedArgs, Args) -> with_scoped<ScopedArgs, Args>;

  //////////////////////////////////////////////////////////////////////////////
  // phased: an operation's receiver, a carried region that lives as long as
  // the operation, and storage for whichever phase is live.
  //
  // Algorithms never construct or destroy phases themselves. They call:
  //
  //   begin(then, args...)
  //       from start: under start_entry, enters the first phase, constructed
  //       from args...; then calls then(first phase), e.g. start_child.
  //       Under connect_entry the first phase was entered by the constructor
  //       and args... are ignored.
  //
  //   transition<From, To>(prepare, extra...)
  //       from inside From's completion. `prepare(From&)` runs while From and
  //       the completion's arguments are still alive; it may store into the
  //       carried region, and returns a tuple of what To's constructor needs.
  //       Then From is destroyed, To is constructed from that tuple and
  //       `extra...`, and To's `child` is started. `extra...` must not refer
  //       into From. If To has scoped state (below), prepare returns a
  //       with_scoped instead, and the scoped state is constructed from its
  //       first tuple before From is destroyed.
  //
  //   finish<Phase>(tag, as...)
  //       from inside Phase's completion: completes the operation with
  //       tag(as...).
  //
  //   finish_copies<Phase>(tag, as...)
  //       like finish, but sends copies of as..., which the operation may
  //       only read (it was bound without consuming).
  //
  //   finish_with<Phase>(fn, as...)
  //       from inside Phase's set_value: completes the operation with
  //       set_value(fn(as...)), or set_value() if fn returns void.
  //
  // An exception from any of these completes the operation with
  // set_error(std::current_exception()); under eager_teardown it ends the
  // live phase first. Each exception path knows statically which phase, if
  // any, is live.
  //
  // Under start_entry the first phase is entered by start. Combined with
  // eager_teardown, no phase is ever live when the operation state is
  // destroyed, so phased keeps no record of the live phase and its
  // destructor destroys only the receiver and the carried region. Otherwise
  // one byte records the live phase for the destructor.
  //
  // A phase may own state that has to be constructed while the phase before
  // it is still alive, because it's built from that phase's results (which
  // may refer into it): let_value's values under scoped_layout. The phase
  // names its type as `scoped_t`. That state can't share the phases' bytes,
  // so it lives in a second region beside them, entered just before the
  // phase and ended just after it. One region holds one phase's scoped
  // state at a time, so two consecutive phases can't both have any.
  //
  // Under AddressSanitizer, storage no phase occupies is poisoned, so
  // touching a phase after it has ended (a child that touches its own state
  // after completing, say) is reported. Bytes the next phase reuses are
  // live again, so this catches what eager teardown leaves dead. The same
  // goes for the scoped region.

  template <class Ending, class Entry, class Rcvr, class Carried, class First, class... Rest>
  class phased
  {
    static constexpr unsigned char none        = 0xff;
    static constexpr bool          eager       = std::is_same_v<Ending, eager_teardown>;
    static constexpr bool          start_entry = std::is_same_v<Entry, typestate::start_entry>;
    static constexpr bool          track_live  = !(eager && start_entry);

    template <class Phase>
    static constexpr unsigned char index_of = detail::index_of<Phase, First, Rest...>();

    template <class Phase>
    using scoped_t = typename detail::scoped_of<Phase>::type;

    template <class Phase>
    static constexpr bool has_scoped = !std::is_void_v<scoped_t<Phase>>;

    static_assert(!has_scoped<First>,
                  "a first phase follows no phase, so it can hold its own state");

    using live_t = std::conditional_t<track_live, unsigned char, detail::empty>;

   public:
    // Under connect_entry, enters the first phase, constructed from
    // first...; under start_entry there must be no first....
    template <class CarriedInit, class... Args>
    phased(Rcvr rcvr, CarriedInit&& carried, Args&&... first)
      : rcvr_(static_cast<Rcvr&&>(rcvr))
      , carried_(static_cast<CarriedInit&&>(carried))
    {
      if constexpr (start_entry)
      {
        static_assert(sizeof...(Args) == 0, "start_entry enters the first phase in begin");
      }
      else
      {
        enter<First>(static_cast<Args&&>(first)...);
      }
    }

    phased(phased&&) = delete;

    ~phased()
    {
      if constexpr (track_live)
      {
        destroy_live();
      }
    }

    auto carried() noexcept -> Carried&
    {
      return carried_;
    }

    auto receiver() noexcept -> Rcvr&
    {
      return rcvr_;
    }

    // Phase's scoped state, from just before Phase is entered until just
    // after it ends.
    template <class Phase>
      requires has_scoped<Phase>
    auto scoped() noexcept -> scoped_t<Phase>&
    {
      return *std::launder(reinterpret_cast<scoped_t<Phase>*>(scoped_.bytes));
    }

    template <class Then, class... Args>
    void begin(Then then, [[maybe_unused]] Args&&... args) noexcept
    {
      if constexpr (start_entry)
      {
        STDEXEC_TRY
        {
          First& first = enter<First>(static_cast<Args&&>(args)...);
          then(first);
        }
        STDEXEC_CATCH_ALL
        {
          fail();  // no phase was entered
        }
      }
      else
      {
        then(get<First>());
      }
    }

    template <class From, class To, class Prepare, class... Extra>
    void transition(Prepare prepare, Extra... extra) noexcept
    {
      static_assert(!(has_scoped<From> && has_scoped<To>),
                    "consecutive phases can't both have scoped state");
      STDEXEC_TRY
      {
        auto bundle = prepare(get<From>());
        if constexpr (has_scoped<To>)
        {
          enter_scoped<To>(bundle.scoped);
        }
        leave<From>();
        STDEXEC_TRY
        {
          auto& args = [&]() -> auto&
          {
            if constexpr (has_scoped<To>)
            {
              return bundle.args;
            }
            else
            {
              return bundle;
            }
          }();
          To& to = std::apply([&](auto&... parts) -> To&
                              { return enter<To>(std::move(parts)..., extra...); },
                              args);
          ex::start(to.child);
        }
        STDEXEC_CATCH_ALL
        {
          if constexpr (has_scoped<To>)
          {
            leave_scoped<To>();  // To's constructor threw
          }
          fail();  // no phase is live
        }
      }
      STDEXEC_CATCH_ALL
      {
        end<From>();  // prepare or To's scoped state threw: From is live
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
          leave<Phase>();  // only copying the results can throw
          fail();
        }
      }
      else
      {
        Tag()(static_cast<Rcvr&&>(rcvr_), static_cast<As&&>(as)...);
      }
    }

    template <class Phase, class Tag, class... As>
    void finish_copies(Tag, As const &... as) noexcept
    {
      if constexpr (eager)
      {
        finish<Phase>(Tag(), as...);  // copies anyway
      }
      else
      {
        STDEXEC_TRY
        {
          std::tuple<As...> copies(as...);
          std::apply([this](auto&... cs) { Tag()(static_cast<Rcvr&&>(rcvr_), std::move(cs)...); },
                     copies);
        }
        STDEXEC_CATCH_ALL
        {
          fail();  // Phase stays live until the destructor
        }
      }
    }

    template <class Phase, class Fn, class... As>
    void finish_with(Fn& fn, As&&... as) noexcept
    {
      if constexpr (eager)
      {
        STDEXEC_TRY
        {
          std::tuple<std::decay_t<As>...> args(static_cast<As&&>(as)...);
          leave<Phase>();
          STDEXEC_TRY
          {
            complete_with([&]() -> decltype(auto) { return std::apply(fn, std::move(args)); });
          }
          STDEXEC_CATCH_ALL
          {
            fail();  // no phase is live
          }
        }
        STDEXEC_CATCH_ALL
        {
          leave<Phase>();  // copying the arguments threw: Phase is live
          fail();
        }
      }
      else
      {
        STDEXEC_TRY
        {
          complete_with([&]() -> decltype(auto) { return fn(static_cast<As&&>(as)...); });
        }
        STDEXEC_CATCH_ALL
        {
          fail();  // Phase stays live until the destructor
        }
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
      return *std::launder(reinterpret_cast<Phase*>(storage_.bytes));
    }

    template <class Phase, class... Args>
    auto enter(Args&&... args) -> Phase&
    {
      static_assert(index_of<Phase> != none, "not one of this operation's phases");
      storage_.unpoison(sizeof(Phase));
      Phase& phase = *::new (static_cast<void*>(storage_.bytes))
                       Phase(static_cast<Args&&>(args)...);
      if constexpr (track_live)
      {
        live_ = index_of<Phase>;
      }
      return phase;
    }

    template <class Phase>
    void leave() noexcept
    {
      std::destroy_at(&get<Phase>());
      if constexpr (track_live)
      {
        live_ = none;
      }
      storage_.poison(sizeof(Phase));
      if constexpr (has_scoped<Phase>)
      {
        leave_scoped<Phase>();
      }
    }

    template <class Phase, class Args>
    void enter_scoped(Args& args)
    {
      using state_t = scoped_t<Phase>;
      scoped_.unpoison(sizeof(state_t));
      STDEXEC_TRY
      {
        std::apply(
          [this](auto&&... as)
          {
            ::new (static_cast<void*>(scoped_.bytes)) state_t(static_cast<decltype(as)&&>(as)...);
          },
          std::move(args));
      }
      STDEXEC_CATCH_ALL
      {
        scoped_.poison(sizeof(state_t));
        STDEXEC_THROW();
      }
    }

    template <class Phase>
    void leave_scoped() noexcept
    {
      std::destroy_at(&scoped<Phase>());
      scoped_.poison(sizeof(scoped_t<Phase>));
    }

    // On a failure inside Phase: eager teardown ends it now; otherwise the
    // destructor will.
    template <class Phase>
    void end() noexcept
    {
      if constexpr (eager)
      {
        leave<Phase>();
      }
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
      ex::set_error(static_cast<Rcvr&&>(rcvr_), std::current_exception());
    }

    static constexpr std::size_t storage_size  = std::max({sizeof(First), sizeof(Rest)...});
    static constexpr std::size_t storage_align = std::max({alignof(First), alignof(Rest)...});

    static constexpr std::size_t scoped_size = std::max(
      {detail::size_of<scoped_t<Rest>>..., std::size_t(0)});
    static constexpr std::size_t scoped_align = std::max(
      {detail::align_of<scoped_t<Rest>>..., std::size_t(1)});

    // Not detail::empty: two empty members of one type can't share an address.
    using scoped_storage_t = std::conditional_t<scoped_size == 0,
                                                detail::no_scoped_state,
                                                detail::guarded_bytes<scoped_size, scoped_align>>;

    Rcvr rcvr_;
    STDEXEC_ATTRIBUTE(no_unique_address) Carried carried_;
    STDEXEC_ATTRIBUTE(no_unique_address) live_t live_ { none_or_empty() };
    STDEXEC_ATTRIBUTE(no_unique_address) scoped_storage_t scoped_;
    detail::guarded_bytes<storage_size, storage_align> storage_;

    static constexpr auto none_or_empty() noexcept -> live_t
    {
      if constexpr (track_live)
      {
        return none;
      }
      else
      {
        return {};
      }
    }
  };

  // phased<Ending, Entry, Rcvr, Carried, First, Next<Ts...>...>, one Next
  // per type_list<Ts...>.
  template <class Ending,
            class Entry,
            class Rcvr,
            class Carried,
            class First,
            template <class...> class Next,
            class Lists>
  struct phased_with;

  template <class Ending,
            class Entry,
            class Rcvr,
            class Carried,
            class First,
            template <class...> class Next,
            class... Lists>
  struct phased_with<Ending, Entry, Rcvr, Carried, First, Next, detail::type_list<Lists...>>
  {
    template <class List>
    struct apply;

    template <class... Ts>
    struct apply<detail::type_list<Ts...>>
    {
      using type = Next<Ts...>;
    };

    using type = phased<Ending, Entry, Rcvr, Carried, First, typename apply<Lists>::type...>;
  };

  // The receiver a phase's child completes through. `Tag` names the phase;
  // `Child`, if not void, is the child sender the phase binds (const if it
  // binds without consuming).
  template <class Op, class Rcvr, class Tag, class Child = void>
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

    // The child sender this phase binds, for a child that binds rather than
    // connects: the parent knows which of its child senders each phase uses.
    [[nodiscard]]
    auto bound_sender() const noexcept -> std::add_lvalue_reference_t<Child>
      requires(!std::is_void_v<Child>)
    {
      return op->bound_sender(Tag());
    }

    Op* op;
  };

  // Nothing to carry.
  struct nothing
  {};

  // T, const if Const.
  template <bool Const, class T>
  using maybe_const = std::conditional_t<Const, T const, T>;

  // How a sender's member is used when the sender is used as Self: as an
  // rvalue, or as a const lvalue when Self is const (binding without
  // consuming).
  template <class Self, class T>
  using member_t = std::conditional_t<std::is_const_v<std::remove_reference_t<Self>>, T const &, T>;

  // Starts a phase's child: what most algorithms do on entering a phase.
  inline constexpr auto start_child = [](auto& phase) noexcept
  {
    ex::start(phase.child);
  };

  //////////////////////////////////////////////////////////////////////////////
  // bind: connects a sender that the caller keeps alive, unmoved, until the
  // operation is destroyed, so the operation may refer into it instead of
  // copying from it. Through a non-const reference the operation may consume
  // the sender's data (move from it), as connecting an rvalue may.
  //
  // A sender opts in with a member bind(rcvr) &; for any other sender, bind
  // moves it into connect.

  template <class Sndr, class Rcvr>
  auto bind(Sndr& sndr, Rcvr rcvr)
  {
    if constexpr (requires { sndr.bind(static_cast<Rcvr&&>(rcvr)); })
    {
      return sndr.bind(static_cast<Rcvr&&>(rcvr));
    }
    else
    {
      return ex::connect(static_cast<Sndr&&>(sndr), static_cast<Rcvr&&>(rcvr));
    }
  }

  // Binding through a const reference doesn't consume: the operation only
  // reads the sender, which stays usable for another bind. A sender opts in
  // with a member bind(rcvr) const &; any other sender is connected as a const
  // lvalue, which copies what it needs, or else copied and moved into
  // connect.
  template <class Sndr, class Rcvr>
  auto bind(Sndr const & sndr, Rcvr rcvr)
  {
    if constexpr (requires { sndr.bind(static_cast<Rcvr&&>(rcvr)); })
    {
      return sndr.bind(static_cast<Rcvr&&>(rcvr));
    }
    else if constexpr (requires { ex::connect(sndr, static_cast<Rcvr&&>(rcvr)); })
    {
      return ex::connect(sndr, static_cast<Rcvr&&>(rcvr));
    }
    else
    {
      return ex::connect(Sndr(sndr), static_cast<Rcvr&&>(rcvr));
    }
  }

  // Sndr may be const, to name the non-consuming bind.
  template <class Sndr, class Rcvr>
  using bind_result_t = decltype(typestate::bind(std::declval<Sndr&>(), std::declval<Rcvr>()));

  // The receiver the root binds its sender with: the caller's receiver, plus
  // where the stored sender is. It's the only pointer to a sender in the
  // tree; every bound operation below finds its own sender through its
  // receiver.
  template <class Sndr, class Rcvr>
  struct root_receiver
  {
    using receiver_concept = ex::receiver_t;

    template <class... As>
    void set_value(As&&... as) && noexcept
    {
      ex::set_value(static_cast<Rcvr&&>(rcvr), static_cast<As&&>(as)...);
    }

    template <class E>
    void set_error(E&& e) && noexcept
    {
      ex::set_error(static_cast<Rcvr&&>(rcvr), static_cast<E&&>(e));
    }

    void set_stopped() && noexcept
    {
      ex::set_stopped(static_cast<Rcvr&&>(rcvr));
    }

    [[nodiscard]]
    auto get_env() const noexcept -> ex::env_of_t<Rcvr>
    {
      return ex::get_env(rcvr);
    }

    [[nodiscard]]
    auto bound_sender() const noexcept -> Sndr&
    {
      return *sndr;
    }

    STDEXEC_ATTRIBUTE(no_unique_address) Rcvr rcvr;
    Sndr* sndr;
  };

  // What connect returns for a sender that binds: the sender, stored once,
  // and the operation bound to it. The stored sender is the operation's
  // carried state, all the way down; nothing below it copies the data again.
  template <class Sndr, class Rcvr>
  struct root
  {
    using rcvr_t = root_receiver<Sndr, Rcvr>;

    root(Sndr&& sndr, Rcvr rcvr)
      : stored(static_cast<Sndr&&>(sndr))
      , op(stored.bind(rcvr_t{static_cast<Rcvr&&>(rcvr), &stored}))
    {}

    void start() & noexcept
    {
      ex::start(op);
    }

    Sndr                                                         stored;
    decltype(std::declval<Sndr&>().bind(std::declval<rcvr_t>())) op;
  };

  //////////////////////////////////////////////////////////////////////////////
  // just(vs...): one phase, holding the values, or (when bound) referring to
  // the sender's.

  template <class Ending, class Entry, class... Vs>
  struct just_sender
  {
    using sender_concept = ex::sender_t;

    static constexpr bool bound = std::is_same_v<Entry, start_entry>;

    // Moving the values may throw if they cross the stack (eager_teardown);
    // bound without consuming (a const Self), the values are copied.
    template <class Self>
    static constexpr bool may_throw = std::is_const_v<std::remove_reference_t<Self>>
                                      ? !(std::is_nothrow_copy_constructible_v<Vs> && ...)
                                      : std::is_same_v<Ending, eager_teardown>
                                          && !(std::is_nothrow_move_constructible_v<Vs> && ...);

    template <class Self, class... Env>
    static consteval auto get_completion_signatures()
    {
      if constexpr (may_throw<Self>)
      {
        return ex::completion_signatures<ex::set_value_t(Vs...),
                                         ex::set_error_t(std::exception_ptr)>();
      }
      else
      {
        return ex::completion_signatures<ex::set_value_t(Vs...)>();
      }
    }

    template <class Rcvr, bool Const = false>
    struct opstate
    {
      struct owned_ready
      {
        explicit owned_ready(std::tuple<Vs...>&& vs)
          : values(std::move(vs))
        {}

        std::tuple<Vs...> values;
      };

      struct bound_ready
      {
        explicit bound_ready(opstate*) noexcept {}
      };

      using ready = std::conditional_t<bound, bound_ready, owned_ready>;

      opstate(std::tuple<Vs...>&& vs, Rcvr rcvr)
        requires(!bound)
        : phases_(static_cast<Rcvr&&>(rcvr), nothing(), std::move(vs))
      {}

      explicit opstate(Rcvr rcvr)
        requires bound
        : phases_(static_cast<Rcvr&&>(rcvr), nothing())
      {}

      void start() & noexcept
      {
        phases_.begin(
          [this](ready& r) noexcept
          {
            if constexpr (Const)
            {
              std::apply([this](Vs const &... vs)
                         { phases_.template finish_copies<ready>(ex::set_value, vs...); },
                         values(r));
            }
            else
            {
              std::apply([this](Vs&... vs)
                         { phases_.template finish<ready>(ex::set_value, std::move(vs)...); },
                         values(r));
            }
          },
          this);
      }

      auto values(ready& r) noexcept -> maybe_const<Const, std::tuple<Vs...>>&
      {
        if constexpr (bound)
        {
          return self().values_;
        }
        else
        {
          return r.values;
        }
      }

      // The sender this operation is bound to, found through the receiver
      // rather than stored.
      auto self() noexcept -> maybe_const<Const, just_sender>&
      {
        return phases_.receiver().bound_sender();
      }

      phased<Ending, Entry, Rcvr, nothing, ready> phases_;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) &&
    {
      if constexpr (bound)
      {
        return root<just_sender, Rcvr>(std::move(*this), static_cast<Rcvr&&>(rcvr));
      }
      else
      {
        return opstate<Rcvr>(std::move(values_), static_cast<Rcvr&&>(rcvr));
      }
    }

    template <class Rcvr>
      requires bound && requires(Rcvr const & r) { r.bound_sender(); }
    auto bind(Rcvr rcvr) & -> opstate<Rcvr>
    {
      STDEXEC_ASSERT(&rcvr.bound_sender() == this);
      return opstate<Rcvr>(static_cast<Rcvr&&>(rcvr));
    }

    template <class Rcvr>
      requires bound && requires(Rcvr const & r) { r.bound_sender(); }
    auto bind(Rcvr rcvr) const & -> opstate<Rcvr, true>
    {
      STDEXEC_ASSERT(&rcvr.bound_sender() == this);
      return opstate<Rcvr, true>(static_cast<Rcvr&&>(rcvr));
    }

    std::tuple<Vs...> values_;
  };

  template <class Ending = eager_teardown, class Entry = connect_entry, class... Vs>
  auto just(Vs&&... vs)
  {
    return just_sender<Ending, Entry, std::decay_t<Vs>...>{{static_cast<Vs&&>(vs)...}};
  }

  //////////////////////////////////////////////////////////////////////////////
  // then(sndr, fn): one phase, running the child. fn is carried, or (when
  // bound) used in place in the sender.

  template <class Ending, class Entry, class Sndr, class Fn>
  struct then_sender
  {
    using sender_concept = ex::sender_t;

    static constexpr bool bound = std::is_same_v<Entry, start_entry>;

    // After eager teardown, fn is called with rvalue copies of the values.
    // Bound without consuming, fn is called as const.
    template <class FnRef>
    struct values_fn
    {
      template <class... As>
      using result_t = typename std::conditional_t<std::is_same_v<Ending, eager_teardown>,
                                                   std::invoke_result<FnRef, std::decay_t<As>...>,
                                                   std::invoke_result<FnRef, As...>>::type;

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
        using fn_ref =
          std::conditional_t<std::is_const_v<std::remove_reference_t<Self>>, Fn const &, Fn&>;
        return detail::map_values_t<Ending,
                                    ex::completion_signatures_of_t<member_t<Self, Sndr>, Env...>,
                                    values_fn<fn_ref>>();
      }
    }

    template <class Rcvr, bool Const = false>
    struct opstate
    {
      struct running_tag
      {};

      using rcvr_t  = phase_receiver<opstate, Rcvr, running_tag, maybe_const<Const, Sndr>>;
      using child_t = std::conditional_t<bound,
                                         bind_result_t<maybe_const<Const, Sndr>, rcvr_t>,
                                         ex::connect_result_t<Sndr, rcvr_t>>;

      struct carried
      {
        Fn fn;
      };

      using carried_t = std::conditional_t<bound, nothing, carried>;

      struct running
      {
        running(Sndr&& sndr, opstate* op)
          : child(ex::connect(static_cast<Sndr&&>(sndr), rcvr_t{op}))
        {}

        explicit running(opstate* op)
          : child(typestate::bind(op->self().sndr_, rcvr_t{op}))
        {}

        child_t child;
      };

      opstate(Sndr&& sndr, Fn&& fn, Rcvr rcvr)
        requires(!bound)
        : phases_(static_cast<Rcvr&&>(rcvr),
                  carried{static_cast<Fn&&>(fn)},
                  static_cast<Sndr&&>(sndr),
                  this)
      {}

      explicit opstate(Rcvr rcvr)
        requires bound
        : phases_(static_cast<Rcvr&&>(rcvr), nothing())
      {}

      void start() & noexcept
      {
        phases_.begin(start_child, this);
      }

      auto fn() noexcept -> maybe_const<Const, Fn>&
      {
        if constexpr (bound)
        {
          return self().fn_;
        }
        else
        {
          return phases_.carried().fn;
        }
      }

      template <class... As>
      void complete(running_tag, ex::set_value_t, As&&... as) noexcept
      {
        phases_.template finish_with<running>(fn(), static_cast<As&&>(as)...);
      }

      template <class Tag, class... As>
      void complete(running_tag, Tag, As&&... as) noexcept
      {
        phases_.template finish<running>(Tag(), static_cast<As&&>(as)...);
      }

      // Which child sender a phase binds.
      auto bound_sender(running_tag) noexcept -> maybe_const<Const, Sndr>&
      {
        return self().sndr_;
      }

      // The sender this operation is bound to, found through the receiver
      // rather than stored.
      auto self() noexcept -> maybe_const<Const, then_sender>&
      {
        return phases_.receiver().bound_sender();
      }

      phased<Ending, Entry, Rcvr, carried_t, running> phases_;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) &&
    {
      if constexpr (bound)
      {
        return root<then_sender, Rcvr>(std::move(*this), static_cast<Rcvr&&>(rcvr));
      }
      else
      {
        return opstate<Rcvr>(static_cast<Sndr&&>(sndr_),
                             static_cast<Fn&&>(fn_),
                             static_cast<Rcvr&&>(rcvr));
      }
    }

    template <class Rcvr>
      requires bound && requires(Rcvr const & r) { r.bound_sender(); }
    auto bind(Rcvr rcvr) & -> opstate<Rcvr>
    {
      STDEXEC_ASSERT(&rcvr.bound_sender() == this);
      return opstate<Rcvr>(static_cast<Rcvr&&>(rcvr));
    }

    template <class Rcvr>
      requires bound && requires(Rcvr const & r) { r.bound_sender(); }
    auto bind(Rcvr rcvr) const & -> opstate<Rcvr, true>
    {
      STDEXEC_ASSERT(&rcvr.bound_sender() == this);
      return opstate<Rcvr, true>(static_cast<Rcvr&&>(rcvr));
    }

    Sndr sndr_;
    Fn   fn_;
  };

  template <class Ending = eager_teardown, class Entry = connect_entry, class Sndr, class Fn>
  auto then(Sndr&& sndr, Fn&& fn)
  {
    return then_sender<Ending, Entry, std::decay_t<Sndr>, std::decay_t<Fn>>{static_cast<Sndr&&>(
                                                                              sndr),
                                                                            static_cast<Fn&&>(fn)};
  }

  //////////////////////////////////////////////////////////////////////////////
  // let_value(sndr, fn): a first phase running the child, then one possible
  // second phase per value signature of the child, running the sender fn
  // returns. fn is carried, or (when bound) used in place in the sender: the
  // sender it returns may refer to its captures. The values fn is called with
  // are carried too (carried_layout), or live in the second phase, reusing
  // the first phase's bytes (overlapped_layout), or are the second phase's
  // scoped state, beside the phases but ending with that phase
  // (scoped_layout). Each second phase has its own receiver type, so its
  // completion knows which one is live. The sender fn returns is created at
  // run time, so its phase owns it.

  template <class Ending, class Layout, class Entry, class Sndr, class Fn>
  struct let_value_sender
  {
    using sender_concept = ex::sender_t;

    static constexpr bool overlapped = std::is_same_v<Layout, overlapped_layout>;
    static constexpr bool scoped     = std::is_same_v<Layout, scoped_layout>;
    static constexpr bool bound      = std::is_same_v<Entry, start_entry>;

    template <class FnRef, class Env>
    struct values_fn
    {
      template <class... As>
      using f =
        ex::completion_signatures_of_t<std::invoke_result_t<FnRef, std::decay_t<As>&...>, Env>;
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
        using fn_ref =
          std::conditional_t<std::is_const_v<std::remove_reference_t<Self>>, Fn const &, Fn&>;
        return detail::map_values_t<Ending,
                                    ex::completion_signatures_of_t<member_t<Self, Sndr>, Env...>,
                                    values_fn<fn_ref, Env...>>();
      }
    }

    template <class Rcvr, bool Const = false>
    struct opstate
    {
      struct running_tag
      {};

      template <class... Ts>
      struct continuing_tag
      {};

      using first_rcvr_t = phase_receiver<opstate, Rcvr, running_tag, maybe_const<Const, Sndr>>;
      using child_t      = std::conditional_t<bound,
                                              bind_result_t<maybe_const<Const, Sndr>, first_rcvr_t>,
                                              ex::connect_result_t<Sndr, first_rcvr_t>>;
      using values_t     = typename detail::value_lists<
            ex::completion_signatures_of_t<std::conditional_t<Const, Sndr const &, Sndr>,
                                           ex::env_of_t<Rcvr>>>::type;

      struct carried
      {
        STDEXEC_ATTRIBUTE(no_unique_address) std::conditional_t<bound, nothing, Fn> fn;
        STDEXEC_ATTRIBUTE(no_unique_address)
        std::conditional_t<overlapped || scoped,
                           nothing,
                           typename detail::args_variant<values_t>::type>
          args;
      };

      struct running
      {
        running(Sndr&& sndr, opstate* op)
          : child(ex::connect(static_cast<Sndr&&>(sndr), first_rcvr_t{op}))
        {}

        explicit running(opstate* op)
          : child(typestate::bind(op->self().sndr_, first_rcvr_t{op}))
        {}

        child_t child;
      };

      template <class... Ts>
      struct continuing
      {
        using rcvr_t   = phase_receiver<opstate, Rcvr, continuing_tag<Ts...>>;
        using args_t   = std::tuple<Ts...>;
        using stored_t = std::conditional_t<overlapped, args_t, nothing>;
        using scoped_t = std::conditional_t<scoped, args_t, void>;
        using sndr_t   = std::invoke_result_t<maybe_const<Const, Fn>&, Ts&...>;

        continuing(stored_t&& stored, opstate* op)
          : stored(std::move(stored))
          , child(ex::connect(std::apply(op->fn(), args(op)), rcvr_t{op}))
        {}

        auto args(opstate* op) noexcept -> args_t&
        {
          if constexpr (overlapped)
          {
            return stored;
          }
          else if constexpr (scoped)
          {
            return op->phases_.template scoped<continuing>();
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
        requires(!bound)
        : phases_(static_cast<Rcvr&&>(rcvr),
                  carried{static_cast<Fn&&>(fn), {}},
                  static_cast<Sndr&&>(sndr),
                  this)
      {}

      explicit opstate(Rcvr rcvr)
        requires bound
        : phases_(static_cast<Rcvr&&>(rcvr), carried{})
      {}

      void start() & noexcept
      {
        phases_.begin(start_child, this);
      }

      auto fn() noexcept -> maybe_const<Const, Fn>&
      {
        if constexpr (bound)
        {
          return self().fn_;
        }
        else
        {
          return phases_.carried().fn;
        }
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
            else if constexpr (scoped)
            {
              return with_scoped{std::forward_as_tuple(static_cast<As&&>(as)...),
                                 std::tuple<nothing>()};
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

      // Which child sender a phase binds.
      auto bound_sender(running_tag) noexcept -> maybe_const<Const, Sndr>&
      {
        return self().sndr_;
      }

      // The sender this operation is bound to, found through the receiver
      // rather than stored.
      auto self() noexcept -> maybe_const<Const, let_value_sender>&
      {
        return phases_.receiver().bound_sender();
      }

      typename phased_with<Ending, Entry, Rcvr, carried, running, continuing, values_t>::type
        phases_;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) &&
    {
      if constexpr (bound)
      {
        return root<let_value_sender, Rcvr>(std::move(*this), static_cast<Rcvr&&>(rcvr));
      }
      else
      {
        return opstate<Rcvr>(static_cast<Sndr&&>(sndr_),
                             static_cast<Fn&&>(fn_),
                             static_cast<Rcvr&&>(rcvr));
      }
    }

    template <class Rcvr>
      requires bound && requires(Rcvr const & r) { r.bound_sender(); }
    auto bind(Rcvr rcvr) & -> opstate<Rcvr>
    {
      STDEXEC_ASSERT(&rcvr.bound_sender() == this);
      return opstate<Rcvr>(static_cast<Rcvr&&>(rcvr));
    }

    template <class Rcvr>
      requires bound && requires(Rcvr const & r) { r.bound_sender(); }
    auto bind(Rcvr rcvr) const & -> opstate<Rcvr, true>
    {
      STDEXEC_ASSERT(&rcvr.bound_sender() == this);
      return opstate<Rcvr, true>(static_cast<Rcvr&&>(rcvr));
    }

    Sndr sndr_;
    Fn   fn_;
  };

  template <class Ending = eager_teardown,
            class Layout = carried_layout,
            class Entry  = connect_entry,
            class Sndr,
            class Fn>
  auto let_value(Sndr&& sndr, Fn&& fn)
  {
    return let_value_sender<Ending, Layout, Entry, std::decay_t<Sndr>, std::decay_t<Fn>>{
      static_cast<Sndr&&>(sndr),
      static_cast<Fn&&>(fn)};
  }

  //////////////////////////////////////////////////////////////////////////////
  // sequence(first, second): runs first, discards its values, then runs
  // second. The second sender is carried until it's connected
  // (carried_layout), or waits in the first phase and crosses to the second
  // on the stack (overlapped_layout). It exists from connect, before any
  // phase that could own it, so scoped_layout carries it too. When bound,
  // both senders stay in the stored sender and each phase binds to its own;
  // the layout doesn't apply.

  template <class Ending, class Layout, class Entry, class Sndr1, class Sndr2>
  struct sequence_sender
  {
    using sender_concept = ex::sender_t;

    static constexpr bool bound      = std::is_same_v<Entry, start_entry>;
    static constexpr bool overlapped = !bound && std::is_same_v<Layout, overlapped_layout>;

    template <class Second, class Env>
    struct values_fn
    {
      template <class...>
      using f = ex::completion_signatures_of_t<Second, Env>;
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
                                    ex::completion_signatures_of_t<member_t<Self, Sndr1>, Env...>,
                                    values_fn<member_t<Self, Sndr2>, Env...>>();
      }
    }

    template <class Rcvr, bool Const = false>
    struct opstate
    {
      struct first_tag
      {};

      struct second_tag
      {};

      using first_rcvr_t  = phase_receiver<opstate, Rcvr, first_tag, maybe_const<Const, Sndr1>>;
      using second_rcvr_t = phase_receiver<opstate, Rcvr, second_tag, maybe_const<Const, Sndr2>>;
      using next_t        = std::conditional_t<overlapped, Sndr2, nothing>;

      template <class Sndr, class R>
      using child_t = std::conditional_t<bound,
                                         bind_result_t<maybe_const<Const, Sndr>, R>,
                                         ex::connect_result_t<Sndr, R>>;

      struct carried
      {
        STDEXEC_ATTRIBUTE(no_unique_address)
        std::conditional_t<overlapped || bound, nothing, Sndr2> next;
      };

      struct first
      {
        first(Sndr1&& sndr1, next_t&& next, opstate* op)
          : next(static_cast<next_t&&>(next))
          , child(ex::connect(static_cast<Sndr1&&>(sndr1), first_rcvr_t{op}))
        {}

        explicit first(opstate* op)
          : child(typestate::bind(op->self().sndr1_, first_rcvr_t{op}))
        {}

        STDEXEC_ATTRIBUTE(no_unique_address) next_t next;
        child_t<Sndr1, first_rcvr_t> child;
      };

      struct second
      {
        second(next_t&& next, opstate* op)
          : child(make_child(next, op))
        {}

        static auto make_child(next_t& next, opstate* op) -> child_t<Sndr2, second_rcvr_t>
        {
          if constexpr (bound)
          {
            return typestate::bind(op->self().sndr2_, second_rcvr_t{op});
          }
          else if constexpr (overlapped)
          {
            return ex::connect(static_cast<Sndr2&&>(next), second_rcvr_t{op});
          }
          else
          {
            return ex::connect(static_cast<Sndr2&&>(op->phases_.carried().next), second_rcvr_t{op});
          }
        }

        child_t<Sndr2, second_rcvr_t> child;
      };

      opstate(Sndr1&& sndr1, Sndr2&& sndr2, Rcvr rcvr)
        requires(!bound)
        : phases_(static_cast<Rcvr&&>(rcvr),
                  make_carried(sndr2),
                  static_cast<Sndr1&&>(sndr1),
                  make_next(sndr2),
                  this)
      {}

      explicit opstate(Rcvr rcvr)
        requires bound
        : phases_(static_cast<Rcvr&&>(rcvr), carried{})
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
        phases_.begin(start_child, this);
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

      // Which child sender each phase binds.
      auto bound_sender(first_tag) noexcept -> maybe_const<Const, Sndr1>&
      {
        return self().sndr1_;
      }

      auto bound_sender(second_tag) noexcept -> maybe_const<Const, Sndr2>&
      {
        return self().sndr2_;
      }

      // The sender this operation is bound to, found through the receiver
      // rather than stored.
      auto self() noexcept -> maybe_const<Const, sequence_sender>&
      {
        return phases_.receiver().bound_sender();
      }

      phased<Ending, Entry, Rcvr, carried, first, second> phases_;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) &&
    {
      if constexpr (bound)
      {
        return root<sequence_sender, Rcvr>(std::move(*this), static_cast<Rcvr&&>(rcvr));
      }
      else
      {
        return opstate<Rcvr>(static_cast<Sndr1&&>(sndr1_),
                             static_cast<Sndr2&&>(sndr2_),
                             static_cast<Rcvr&&>(rcvr));
      }
    }

    template <class Rcvr>
      requires bound && requires(Rcvr const & r) { r.bound_sender(); }
    auto bind(Rcvr rcvr) & -> opstate<Rcvr>
    {
      STDEXEC_ASSERT(&rcvr.bound_sender() == this);
      return opstate<Rcvr>(static_cast<Rcvr&&>(rcvr));
    }

    template <class Rcvr>
      requires bound && requires(Rcvr const & r) { r.bound_sender(); }
    auto bind(Rcvr rcvr) const & -> opstate<Rcvr, true>
    {
      STDEXEC_ASSERT(&rcvr.bound_sender() == this);
      return opstate<Rcvr, true>(static_cast<Rcvr&&>(rcvr));
    }

    Sndr1 sndr1_;
    Sndr2 sndr2_;
  };

  template <class Ending = eager_teardown,
            class Layout = carried_layout,
            class Entry  = connect_entry,
            class Sndr1,
            class Sndr2>
  auto sequence(Sndr1&& sndr1, Sndr2&& sndr2)
  {
    return sequence_sender<Ending, Layout, Entry, std::decay_t<Sndr1>, std::decay_t<Sndr2>>{
      static_cast<Sndr1&&>(sndr1),
      static_cast<Sndr2&&>(sndr2)};
  }

  //////////////////////////////////////////////////////////////////////////////
  // repeat_n(sndr, n): runs sndr n times in sequence, discarding its values,
  // then completes with set_value(). One phase, re-entered for each
  // iteration; each time, the phase binds the stored child sender without
  // consuming it, so a child that can bind is never copied. (A child that
  // can't, such as any stdexec sender, is connected as a const lvalue, which
  // copies it, as exec::repeat_n does.) The child is stored in the carried
  // region, or (when bound) in the stored sender.
  //
  // Prototype simplification: no trampoline, so a child that completes
  // inline makes each iteration a nested call.

  template <class Ending, class Entry, class Sndr>
  struct repeat_n_sender
  {
    using sender_concept = ex::sender_t;

    static constexpr bool bound = std::is_same_v<Entry, start_entry>;

    struct values_fn
    {
      template <class...>
      using f = ex::completion_signatures<ex::set_value_t()>;
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
        return detail::join_t<
          detail::sent_as<Ending>,
          detail::map_values_t<Ending,
                               ex::completion_signatures_of_t<Sndr const &, Env...>,
                               values_fn>,
          ex::completion_signatures<ex::set_value_t()>>();
      }
    }

    template <class Rcvr, bool Const = false>
    struct opstate
    {
      struct iteration_tag
      {};

      using rcvr_t  = phase_receiver<opstate, Rcvr, iteration_tag, Sndr const>;
      using child_t = bind_result_t<Sndr const, rcvr_t>;

      struct carried
      {
        STDEXEC_ATTRIBUTE(no_unique_address) std::conditional_t<bound, nothing, Sndr> sndr;
        std::size_t remaining;
      };

      struct iteration
      {
        explicit iteration(opstate* op)
          : child(typestate::bind(op->child_sender(), rcvr_t{op}))
        {}

        child_t child;
      };

      opstate(Sndr&& sndr, std::size_t count, Rcvr rcvr)
        requires(!bound)
        : phases_(static_cast<Rcvr&&>(rcvr), carried{static_cast<Sndr&&>(sndr), count}, this)
      {}

      explicit opstate(Rcvr rcvr)
        requires bound
        : opstate(rcvr.bound_sender().count_, static_cast<Rcvr&&>(rcvr))
      {}

      opstate(std::size_t count, Rcvr rcvr)
        requires bound
        : phases_(static_cast<Rcvr&&>(rcvr), carried{{}, count})
      {}

      auto child_sender() noexcept -> Sndr const &
      {
        if constexpr (bound)
        {
          return self().sndr_;
        }
        else
        {
          return phases_.carried().sndr;
        }
      }

      void start() & noexcept
      {
        phases_.begin(
          [this](iteration& it) noexcept
          {
            if (phases_.carried().remaining == 0)
            {
              phases_.template finish<iteration>(ex::set_value);
            }
            else
            {
              ex::start(it.child);
            }
          },
          this);
      }

      template <class... As>
      void complete(iteration_tag, ex::set_value_t, As&&...) noexcept
      {
        if (--phases_.carried().remaining == 0)
        {
          phases_.template finish<iteration>(ex::set_value);
        }
        else
        {
          phases_.template transition<iteration, iteration>([](iteration&) { return std::tuple(); },
                                                            this);
        }
      }

      template <class Tag, class... As>
      void complete(iteration_tag, Tag, As&&... as) noexcept
      {
        phases_.template finish<iteration>(Tag(), static_cast<As&&>(as)...);
      }

      // The child sender each iteration binds.
      auto bound_sender(iteration_tag) noexcept -> Sndr const &
      {
        return child_sender();
      }

      // The sender this operation is bound to, found through the receiver
      // rather than stored.
      auto self() noexcept -> maybe_const<Const, repeat_n_sender>&
      {
        return phases_.receiver().bound_sender();
      }

      phased<Ending, Entry, Rcvr, carried, iteration> phases_;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) &&
    {
      if constexpr (bound)
      {
        return root<repeat_n_sender, Rcvr>(std::move(*this), static_cast<Rcvr&&>(rcvr));
      }
      else
      {
        return opstate<Rcvr>(static_cast<Sndr&&>(sndr_), count_, static_cast<Rcvr&&>(rcvr));
      }
    }

    // repeat_n only ever reads its child sender, so binding it consumes
    // nothing either way.
    template <class Rcvr>
      requires bound && requires(Rcvr const & r) { r.bound_sender(); }
    auto bind(Rcvr rcvr) const & -> opstate<Rcvr, true>
    {
      STDEXEC_ASSERT(&rcvr.bound_sender() == this);
      return opstate<Rcvr, true>(static_cast<Rcvr&&>(rcvr));
    }

    Sndr        sndr_;
    std::size_t count_;
  };

  template <class Ending = eager_teardown, class Entry = connect_entry, class Sndr>
  auto repeat_n(Sndr&& sndr, std::size_t count)
  {
    return repeat_n_sender<Ending, Entry, std::decay_t<Sndr>>{static_cast<Sndr&&>(sndr), count};
  }
}  // namespace frames::typestate
