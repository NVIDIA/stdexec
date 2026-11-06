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
 * This header holds that machinery; the algorithms written against it are
 * in just.hpp, then.hpp, let_value.hpp, sequence.hpp, repeat_n.hpp,
 * when_all.hpp, finally.hpp and associate.hpp.
 *
 * Prototype simplifications: senders are connected as rvalues only, every
 * algorithm except just conservatively declares
 * set_error_t(std::exception_ptr), and let_value doesn't add a completion
 * scheduler to its second phase's environment.
 */
#pragma once

#include <stdexec/execution.hpp>

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <new>
#include <optional>
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
  // Either way, connect stores the sender and binds the operation to it.
  struct connect_entry
  {};

  struct start_entry
  {};

  // Where values one phase hands to the next are kept: let_value's values.
  struct carried_layout  // beside the phases, for the whole operation, as stdexec does
  {};

  struct overlapped_layout  // in the phases, moving across on the stack
  {};

  struct scoped_layout  // beside the phases, for as long as the phase that uses it
  {};

  // Where a sender's data lives once it's connected: in the sender, stored
  // once at the root, which every operation below binds to; or split among
  // the operations by connect, each moving in the parts it uses.
  struct stored_sender
  {};

  struct split_sender
  {};

  // The parts of a sender that its operation uses, as pointers to members.
  template <auto... Members>
  struct parts
  {};

  // The four policies, chosen together for a tree of operations.
  template <class Ending, class Layout, class Entry, class Data>
  struct policies
  {
    using ending = Ending;
    using layout = Layout;
    using entry  = Entry;
    using data   = Data;
  };

  using default_policies = policies<eager_teardown, carried_layout, connect_entry, stored_sender>;

  // What an operation asks its receiver's environment for its policies. It's
  // a forwarding query, so the operations in a tree all see the policies
  // asked for at its top, unless something in between asks for others.
  struct get_policies_t
  {
    template <class Env>
      requires requires(Env const & env, get_policies_t const & tag) { env.query(tag); }
    constexpr auto operator()(Env const & env) const noexcept
    {
      return env.query(*this);
    }

    static constexpr auto query(ex::forwarding_query_t) noexcept -> bool
    {
      return true;
    }
  };

  inline constexpr get_policies_t get_policies{};

  // The policies Env asks for, or default_policies.
  template <class Env>
  struct policies_of
  {
    using type = default_policies;
  };

  template <class Env>
    requires std::invocable<get_policies_t const &, Env const &>
  struct policies_of<Env>
  {
    using type = std::decay_t<std::invoke_result_t<get_policies_t const &, Env const &>>;
  };

  template <class Env>
  using policies_of_t = typename policies_of<Env>::type;

  template <class... Env>
  struct ending_of;

  template <class Env>
  struct ending_of<Env>
  {
    using type = typename policies_of_t<Env>::ending;
  };

  // The Ending policy Env asks for, for completion signatures.
  template <class... Env>
  using ending_of_t = typename ending_of<Env...>::type;

  // An environment asking for the policies P.
  template <class P>
  struct policies_env
  {
    constexpr auto query(get_policies_t) const noexcept -> P
    {
      return {};
    }
  };

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

    // Distinct empty types: two empty members of one type can't share an
    // address.
    struct no_carried_values
    {};

    struct no_scoped_region
    {};

    struct no_active_region
    {};

    struct no_slot_region
    {};

    struct no_slot_flag
    {};

    // What a phase receives: Phase::received_t if it names one, otherwise
    // void.
    template <class Phase>
    struct received_of
    {
      using type = void;
    };

    template <class Phase>
      requires requires { typename Phase::received_t; }
    struct received_of<Phase>
    {
      using type = typename Phase::received_t;
    };

    // What a phase holds only while it runs: Phase::active_t if it names
    // one, otherwise void.
    template <class Phase>
    struct active_of
    {
      using type = void;
    };

    template <class Phase>
      requires requires { typename Phase::active_t; }
    struct active_of<Phase>
    {
      using type = typename Phase::active_t;
    };

    // What a first phase enters just before itself and keeps until the
    // operation completes: Phase::slot_t if it names one, otherwise void.
    template <class Phase>
    struct slot_of
    {
      using type = void;
    };

    template <class Phase>
      requires requires { typename Phase::slot_t; }
    struct slot_of<Phase>
    {
      using type = typename Phase::slot_t;
    };

    template <class T>
    struct or_monostate
    {
      using type = T;
    };

    template <>
    struct or_monostate<void>
    {
      using type = std::monostate;
    };

    template <class Member>
    struct member_of;

    template <class T, class C>
    struct member_of<T C::*>
    {
      using type = T;
    };

    // No member: what phase_receiver's Part is for a child made at run time.
    template <>
    struct member_of<std::nullptr_t>
    {
      using type = void;
    };

    // The type of the member M points to.
    template <auto M>
    using member_type_t = typename member_of<decltype(M)>::type;

    // Whether an operation was bound through a const reference: every
    // phase-based operation is Sndr::opstate<Rcvr, Const> (see
    // phased_sender). Op may be incomplete. Any other Op is a mistake, so
    // there's no answer for it.
    template <class Op>
    struct bound_const;

    template <template <class, bool> class Op, class Rcvr, bool Const>
    struct bound_const<Op<Rcvr, Const>> : std::bool_constant<Const>
    {};

    template <auto A, auto B>
    consteval auto same_member() -> bool
    {
      if constexpr (std::is_same_v<decltype(A), decltype(B)>)
      {
        return A == B;
      }
      else
      {
        return false;
      }
    }

    // Whether `Connects`, what a phase names as `connects`, names the part
    // M points to: Connects is one pointer to a member, or parts<Ms...>
    // for a phase whose children connect several.
    template <class Connects>
    struct connects_of
    {
      template <auto M, auto C>
      static constexpr bool names = same_member<M, C>();
    };

    template <auto... Ms>
    struct connects_of<parts<Ms...>>
    {
      template <auto M, auto>
      static constexpr bool names = (same_member<M, Ms>() || ...);
    };

    template <auto M, auto Connects>
    inline constexpr bool connects_names =
      connects_of<std::remove_cv_t<decltype(Connects)>>::template names<M, Connects>;

    struct no_parts
    {};

    constexpr auto round_up(std::size_t size, std::size_t align) noexcept -> std::size_t
    {
      return (size + align - 1) / align * align;
    }

    template <class T>
    inline constexpr std::size_t size_of = sizeof(T);

    template <>
    inline constexpr std::size_t size_of<void> = 0;

    template <class T>
    inline constexpr std::size_t align_of = alignof(T);

    template <>
    inline constexpr std::size_t align_of<void> = 1;
  }  // namespace detail

  // What prepare returns to hand values to the phase it prepares (see
  // phased). The values may refer into the dying phase.
  template <class... As>
  struct handover
  {
    std::tuple<As&&...> values;
  };

  template <class... As>
  auto hand_over(As&&... as) noexcept -> handover<As...>
  {
    return {std::forward_as_tuple(static_cast<As&&>(as)...)};
  }

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

  //////////////////////////////////////////////////////////////////////////////
  // phased: an operation's receiver, a carried region that lives as long as
  // the operation, and storage for whichever phase is live. Its policies are
  // whatever its receiver's environment asks for (get_policies), or
  // default_policies.
  //
  // Algorithms never construct or destroy phases themselves. They call:
  //
  //   on_connect<Phase = First>(op, incoming...)
  //       from the operation's constructor: under connect_entry, enters
  //       Phase, constructed from op, and from the incoming sender too if
  //       the phase connects a part (below); under start_entry, does
  //       nothing. An operation whose first step depends on what it learns
  //       at connect enters another of its phases first (associate, refused
  //       by its scope, enters one that has no child).
  //
  //   part<M>()
  //       the part of the operation's sender that M points to.
  //
  //   live<Phase>()
  //       Phase, while it's live. A phase whose children see an environment
  //       of its own (see phase_receiver) builds it from what this returns.
  //
  //   connect_part<M>(rcvr, incoming...)
  //       from a phase's constructor: the phase's child, made from the part
  //       M points to. The phase names M as `connects`, or names parts<M,
  //       ...> when it has a child for each of several parts.
  //
  //   begin<Phase = First>(then, args...)
  //       from start: under start_entry, enters Phase, constructed from
  //       args...; then calls then(Phase&), e.g. start_child. Under
  //       connect_entry Phase was entered by on_connect<Phase> and args...
  //       are ignored.
  //
  //   transition<From, To>(prepare, extra...)
  //       from inside From's completion. `prepare(From&)` runs while From and
  //       the completion's arguments are still alive; it may store into the
  //       carried region, and returns a tuple of what To's constructor needs.
  //       Then From is destroyed, To is constructed from that tuple and
  //       `extra...`, and To's `child` is started. `extra...` must not refer
  //       into From. If To receives values (below), prepare returns
  //       hand_over(vs...) instead, and To is constructed from `extra...`.
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
  //       set_value(fn(as...)), or set_value() if fn returns void. Under
  //       eager_teardown, fn is called after Phase ends, with copies of
  //       as... in the value categories they arrived in, so the same
  //       overload of fn runs under either ending.
  //
  //   finish_from_slot<Phase>(tag, as...)
  //       like finish, from inside the completion of a child the slot holds
  //       (below), while Phase is live: as... may refer into the slot, which
  //       ends before the receiver is completed, so the operation completes
  //       with rvalue copies, made first.
  //
  //   slot()
  //       the slot, while it's live.
  //
  //   visit_live(fn)
  //       calls fn(Phase&) with the live phase, which must be one fn
  //       accepts: for a completion that doesn't say which phase is live,
  //       such as that of a child the slot holds, connected before anyone
  //       knew which phase would start it.
  //
  // A phase may hold something only while it runs, nested inside it: it
  // names its type as `active_t`, constructible without throwing from the
  // phase and the receiver's environment. phased enters it just before the
  // phase starts, by begin or by a transition, and ends it first thing in
  // whichever of transition, finish, finish_copies or finish_with the phase
  // ends with, under either Ending: a completed operation is never running,
  // whatever it still holds. when_all's stop forwarding is one: a callback
  // on the receiver's stop token, which must not outlive the operation's
  // run. An operation is never destroyed while it runs, so nothing records
  // whether one is live.
  //
  // A first phase may also name a `slot_t`: something set up as the
  // operation is entered and kept until it completes, whichever phase is
  // live then, such as the operation of a child connected up front but
  // started by a later phase. phased enters it just before the first phase,
  // constructed from what that phase is, and ends it as the operation
  // completes, under either Ending, after ending the phase if that goes too:
  // a completed operation never holds one, so a third party waiting on
  // something the slot holds isn't left waiting on the operation's
  // destruction. Something that must last until the operation is destroyed
  // belongs in the carried region instead. One byte records whether the
  // slot is live, for the destructor (under connect_entry, an operation
  // destroyed before it starts still holds it) and for failures, which can
  // come before or after it's entered. A child the slot holds completes
  // through a receiver chosen when the slot was entered, so it doesn't know
  // which phase is live then; the byte recording the live phase does, so
  // phased keeps it whenever there's a slot.
  //
  // An exception from any of these completes the operation with
  // set_error(std::current_exception()); under eager_teardown it ends the
  // live phase first. Each exception path knows statically which phase, if
  // any, is live.
  //
  // An operation is in one of three states: connected, from connect until
  // start; running, one phase at a time, from start until its receiver is
  // completed; and completed, until it's destroyed. It's destroyed only
  // when connected or completed, and two policies say what those states
  // hold. Under connect_entry, being connected holds the first phase,
  // entered but not yet started; under start_entry, nothing. Under
  // forward_results, being completed holds the phase the operation
  // completed from; under eager_teardown, nothing, so whatever the receiver
  // gets is copied out of that phase before it ends. When neither state
  // holds a phase (eager_teardown with start_entry) and there's no slot
  // (below), phased keeps no record of the live phase and its destructor
  // destroys only the receiver and the carried region. Otherwise one byte
  // records the live phase, for the destructor and for visit_live.
  //
  // A phase may receive values from the phase before it, built from that
  // phase's results while it's still alive, since they may refer into it:
  // let_value's second phase receives the first phase's values. The phase
  // names their type as `received_t` and reads them through received<Phase>.
  // Where they live is the Layout policy's choice, not the algorithm's:
  //
  //   carried_layout
  //       beside the phases, in a variant that lives as long as the
  //       operation state; they're constructed in place.
  //   overlapped_layout
  //       in the phases' bytes, just before the phase, reusing the dying
  //       phase's bytes; they're built on the stack first and move once more.
  //   scoped_layout
  //       beside the phases, in a region entered just before the phase and
  //       ended just after it; they're constructed in place. One region
  //       holds one phase's values at a time, so consecutive phases can't
  //       both receive any.
  //
  // Overlapped or scoped, the values end with the phase that receives them.
  //
  // Where the operation's sender is kept is the Data policy's choice:
  //
  //   stored_sender
  //       in the root, where connect stored it; part<M>() refers into it,
  //       and connect_part<M> binds the part.
  //   split_sender
  //       nowhere: connect hands the sender to phased, which moves into its
  //       own storage each of the parts the operation uses (Parts), for as
  //       long as the operation, except the parts a first phase connects
  //       as it's entered by connect, which go straight from the incoming
  //       sender into connect, as stdexec's algorithms do.
  //
  // Under AddressSanitizer, storage no phase occupies is poisoned, so
  // touching a phase after it has ended (a child that touches its own state
  // after completing, say) is reported. Bytes the next phase reuses are
  // live again, so this catches what eager teardown leaves dead. The same
  // goes for values received under overlapped_layout or scoped_layout.

  template <class Data, class Sndr, class Rcvr>
  struct part_child;

  template <class Sndr, class Rcvr>
  struct part_child<stored_sender, Sndr, Rcvr>
  {
    using type = bind_result_t<Sndr, Rcvr>;
  };

  template <class Sndr, class Rcvr>
  struct part_child<split_sender, Sndr, Rcvr>
  {
    using type = ex::connect_result_t<std::remove_const_t<Sndr>, Rcvr>;
  };

  // The type of a child that connect_part makes from a part of type Sndr
  // (const if the part is only read), completing to Rcvr.
  template <class Sndr, class Rcvr>
  using part_child_t =
    typename part_child<typename policies_of_t<ex::env_of_t<Rcvr>>::data, Sndr, Rcvr>::type;

  // Whether an operation connected with Rcvr binds to a stored sender:
  // Rcvr's environment asks for stored_sender, and Rcvr leads to the
  // sender.
  template <class Rcvr>
  concept binds_stored =
    std::is_same_v<typename policies_of_t<ex::env_of_t<Rcvr>>::data, stored_sender>
    && requires(Rcvr const & r) { r.bound_sender(); };

  template <class Rcvr, class Parts, class Carried, class First, class... Rest>
  class phased
  {
    using policies_t = policies_of_t<ex::env_of_t<Rcvr>>;
    using Ending     = typename policies_t::ending;
    using Layout     = typename policies_t::layout;
    using Entry      = typename policies_t::entry;
    using Data       = typename policies_t::data;

    static constexpr unsigned char none = 0xff;

    // What the connected and completed states hold (see above).
    static constexpr bool connected_holds_phase = std::is_same_v<Entry, connect_entry>;
    static constexpr bool completed_holds_phase = std::is_same_v<Ending, forward_results>;
    // A slot's child may complete whichever phase is live (see visit_live).
    static constexpr bool track_live = connected_holds_phase || completed_holds_phase
                                    || !std::is_void_v<typename detail::slot_of<First>::type>;
    static constexpr bool overlapped       = std::is_same_v<Layout, overlapped_layout>;
    static constexpr bool scoped           = std::is_same_v<Layout, scoped_layout>;
    static constexpr bool carried_received = !overlapped && !scoped;

    template <class Phase>
    static constexpr unsigned char index_of = detail::index_of<Phase, First, Rest...>();

    template <class Phase>
    using received_t = typename detail::received_of<Phase>::type;

    template <class Phase>
    static constexpr bool receives = !std::is_void_v<received_t<Phase>>;

    template <class Phase>
    using active_t = typename detail::active_of<Phase>::type;

    // Whether Phase may be entered first: it receives nothing, since no
    // phase came before it, and only First has a slot. Under split_sender,
    // the parts First connects on entry go straight from the incoming
    // sender into connect, so any other phase entered first must not
    // connect any of them.
    template <auto... Ms>
    static consteval auto any_connected_on_entry(parts<Ms...>) -> bool
    {
      return (connected_on_entry<Ms> || ...);
    }

    template <class Phase>
    static consteval auto connects_entry_part() -> bool
    {
      if constexpr (!requires { Phase::connects; })
      {
        return false;
      }
      else if constexpr (requires { any_connected_on_entry(Phase::connects); })
      {
        return any_connected_on_entry(Phase::connects);
      }
      else
      {
        return connected_on_entry<Phase::connects>;
      }
    }

    template <class Phase>
    static consteval auto check_entry() -> bool
    {
      static_assert(index_of<Phase> != none, "not one of this operation's phases");
      static_assert(!receives<Phase>,
                    "a phase entered first follows no phase, so it receives nothing");
      static_assert(std::is_same_v<Phase, First> || !has_slot,
                    "an operation whose first phase has a slot always enters that phase first");
      static_assert(std::is_same_v<Phase, First> || !connects_entry_part<Phase>(),
                    "a part the first phase connects on entry isn't kept for another phase");
      return true;
    }

    using live_t = std::conditional_t<track_live, unsigned char, detail::empty>;

    static constexpr bool split = std::is_same_v<Data, split_sender>;

    // What the first phase enters just before itself, if anything.
    using slot_t = typename detail::slot_of<First>::type;

    static constexpr bool has_slot = !std::is_void_v<slot_t>;

    // Whether T, the first phase or its slot, names M as a part it connects.
    template <class T, auto M>
    static constexpr bool connects_on_entry = []
    {
      if constexpr (requires { T::connects; })
      {
        return detail::connects_names<M, T::connects>;
      }
      else
      {
        return false;
      }
    }();

    // Whether M is a part the first phase or its slot connects as they're
    // entered by connect, straight from the incoming sender.
    template <auto M>
    static constexpr bool connected_on_entry = []
    {
      if constexpr (split && connected_holds_phase)
      {
        return connects_on_entry<First, M> || connects_on_entry<slot_t, M>;
      }
      else
      {
        return false;
      }
    }();

    // Where split_sender keeps the parts: one element per part, empty for
    // a part connected on entry.
    template <class P>
    struct split_parts;

    template <auto... Ms>
    struct split_parts<parts<Ms...>>
    {
      using type = std::tuple<
        std::conditional_t<connected_on_entry<Ms>, nothing, detail::member_type_t<Ms>>...>;

      template <auto M>
      static constexpr std::size_t index = []
      {
        std::size_t i     = 0;
        std::size_t found = sizeof...(Ms);
        ((found = detail::same_member<M, Ms>() ? i : found, ++i), ...);
        return found;
      }();

      template <auto M, class S>
      static auto take(S& sndr) noexcept -> decltype(auto)
      {
        if constexpr (connected_on_entry<M>)
        {
          return nothing();
        }
        else
        {
          return static_cast<detail::member_type_t<M>&&>(sndr.*M);
        }
      }

      template <class S>
      static auto make(S& sndr) -> type
      {
        return type(take<Ms>(sndr)...);
      }
    };

    using parts_t = std::conditional_t<split, split_parts<Parts>, void>;
    using parts_storage_t =
      std::conditional_t<split, typename split_parts<Parts>::type, detail::no_parts>;

    template <class... Incoming>
    static auto make_parts([[maybe_unused]] Incoming&... incoming) -> parts_storage_t
    {
      if constexpr (split)
      {
        static_assert(sizeof...(Incoming) == 1, "split_sender connects from the incoming sender");
        return parts_t::make(incoming...);
      }
      else
      {
        static_assert(sizeof...(Incoming) == 0, "stored_sender binds to the stored sender");
        return {};
      }
    }

   public:
    // Under split_sender, `incoming` is the sender being connected; the
    // parts the operation uses are moved out of it.
    template <class CarriedInit, class... Incoming>
    phased(Rcvr rcvr, CarriedInit&& carried, Incoming&... incoming)
      : rcvr_(static_cast<Rcvr&&>(rcvr))
      , carried_(static_cast<CarriedInit&&>(carried))
      , parts_(make_parts(incoming...))
    {}

    phased(phased&&) = delete;

    ~phased()
    {
      if constexpr (track_live)
      {
        destroy_live();
      }
      end_slot();
    }

    auto carried() noexcept -> Carried&
    {
      return carried_;
    }

    auto receiver() noexcept -> Rcvr&
    {
      return rcvr_;
    }

    // Phase, while it's live: for its children's environments and its
    // completions, which can only happen while it is.
    template <class Phase>
    auto live() noexcept -> Phase&
    {
      return get<Phase>();
    }

    template <class Slot = slot_t>
      requires has_slot
    auto slot() noexcept -> Slot&
    {
      return *std::launder(reinterpret_cast<Slot*>(slot_.bytes));
    }

    // Nothing may touch the operation once fn has returned, if fn finished
    // it, so live_ isn't read again after fn is called.
    template <class Fn>
      requires track_live
    void visit_live(Fn fn) noexcept
    {
      bool const visited = [&]<std::size_t... Is>(std::index_sequence<Is...>)
      {
        return (visit_if<std::tuple_element_t<Is, std::tuple<First, Rest...>>>(fn) || ...);
      }(std::index_sequence_for<First, Rest...>());
      STDEXEC_ASSERT(visited);
      (void) visited;
    }

    // The values Phase received, from just before Phase is entered until
    // just after it ends, or under carried_layout until the operation state
    // is destroyed.
    template <class Phase>
      requires receives<Phase>
    auto received() noexcept -> received_t<Phase>&
    {
      if constexpr (overlapped)
      {
        return *std::launder(reinterpret_cast<received_t<Phase>*>(storage_.bytes));
      }
      else if constexpr (scoped)
      {
        return *std::launder(reinterpret_cast<received_t<Phase>*>(scoped_.bytes));
      }
      else
      {
        return std::get<index_of<Phase> + 1>(received_);
      }
    }

    // Called from the operation's constructor body rather than from this
    // constructor, so that the first phase's children, which find their
    // senders through the operation, see a fully constructed phased.
    template <class Phase = First, class Op, class... Incoming>
    void on_connect([[maybe_unused]] Op* op, [[maybe_unused]] Incoming&... incoming)
    {
      static_assert(check_entry<Phase>());
      if constexpr (connected_holds_phase)
      {
        if constexpr (requires { slot_t::connects; })
        {
          enter_slot(op, incoming...);
        }
        else
        {
          enter_slot(op);
        }
        if constexpr (requires { Phase::connects; })
        {
          enter<Phase>(op, incoming...);
        }
        else
        {
          enter<Phase>(op);
        }
      }
    }

    template <auto M>
    auto part() noexcept -> decltype(auto)
    {
      if constexpr (split)
      {
        static_assert(!connected_on_entry<M>, "the first phase connected this part on entry");
        return (std::get<parts_t::template index<M>>(parts_));
      }
      else
      {
        return (rcvr_.bound_sender().*M);
      }
    }

    template <auto M, class R, class... Incoming>
    auto connect_part(R rcvr, [[maybe_unused]] Incoming&... incoming)
    {
      if constexpr (!split)
      {
        return typestate::bind(rcvr_.bound_sender().*M, static_cast<R&&>(rcvr));
      }
      else if constexpr (connected_on_entry<M>)
      {
        static_assert(sizeof...(Incoming) == 1, "a part connected on entry comes from the sender");
        return ex::connect(static_cast<detail::member_type_t<M>&&>(incoming.*M)...,
                           static_cast<R&&>(rcvr));
      }
      else
      {
        return ex::connect(static_cast<detail::member_type_t<M>&&>(part<M>()),
                           static_cast<R&&>(rcvr));
      }
    }

    // Leaves the connected state for the running one: Phase, entered now
    // unless being connected holds it, starts.
    template <class Phase = First, class Then, class... Args>
    void begin(Then then, [[maybe_unused]] Args&&... args) noexcept
    {
      static_assert(check_entry<Phase>());
      if constexpr (connected_holds_phase)
      {
        activate<Phase>();
        then(get<Phase>());
      }
      else
      {
        STDEXEC_TRY
        {
          enter_slot(args...);
          Phase& phase = enter<Phase>(static_cast<Args&&>(args)...);
          activate<Phase>();
          then(phase);
        }
        STDEXEC_CATCH_ALL
        {
          fail();  // no phase was entered; fail ends the slot if it was
        }
      }
    }

    template <class From, class To, class Prepare, class... Extra>
    void transition(Prepare prepare, Extra... extra) noexcept
    {
      static_assert(!(scoped && receives<From> && receives<To>),
                    "under scoped_layout, consecutive phases can't both receive values");
      deactivate<From>();
      STDEXEC_TRY
      {
        auto bundle = prepare(get<From>());
        if constexpr (receives<To>)
        {
          // Built while From and the completion's arguments are alive.
          auto staged = stage<To>(std::move(bundle.values));
          leave<From>();
          bool placed = false;
          STDEXEC_TRY
          {
            place<To>(staged);
            placed = true;
            To& to = enter<To>(extra...);
            activate<To>();
            ex::start(to.child);
          }
          STDEXEC_CATCH_ALL
          {
            if (placed || scoped)
            {
              leave_received<To>();
            }
            fail();  // no phase is live
          }
        }
        else
        {
          leave<From>();
          STDEXEC_TRY
          {
            To& to = std::apply([&](auto&... parts) -> To&
                                { return enter<To>(std::move(parts)..., extra...); },
                                bundle);
            activate<To>();
            ex::start(to.child);
          }
          STDEXEC_CATCH_ALL
          {
            fail();  // no phase is live
          }
        }
      }
      STDEXEC_CATCH_ALL
      {
        settle<From>();  // prepare or staging the values threw: From is live
        fail();
      }
    }

    template <class Phase, class Tag, class... As>
    void finish(Tag, As&&... as) noexcept
    {
      complete<Phase, false>(
        [this](auto&&... rs)
        { Tag()(static_cast<Rcvr&&>(rcvr_), static_cast<decltype(rs)&&>(rs)...); },
        static_cast<As&&>(as)...);
    }

    template <class Phase, class Tag, class... As>
    void finish_copies(Tag, As const &... as) noexcept
    {
      complete<Phase, true>(
        [this](auto&&... cs)
        { Tag()(static_cast<Rcvr&&>(rcvr_), static_cast<decltype(cs)&&>(cs)...); },
        as...);
    }

    template <class Phase, class Fn, class... As>
    void finish_with(Fn& fn, As&&... as) noexcept
    {
      complete<Phase, false>(
        [&](auto&&... args)
        { complete_with([&]() -> decltype(auto) { return fn(static_cast<As&&>(args)...); }); },
        static_cast<As&&>(as)...);
    }

    template <class Phase, class Tag, class... As>
      requires has_slot
    void finish_from_slot(Tag, As&&... as) noexcept
    {
      complete<Phase, true>(
        [this](auto&&... cs)
        { Tag()(static_cast<Rcvr&&>(rcvr_), static_cast<decltype(cs)&&>(cs)...); },
        static_cast<As&&>(as)...);
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

    // Where Phase starts in the phases' bytes: after the values it
    // received, under overlapped_layout.
    template <class Phase>
    static constexpr std::size_t offset_of = []
    {
      if constexpr (overlapped && receives<Phase>)
      {
        return detail::round_up(sizeof(received_t<Phase>), alignof(Phase));
      }
      else
      {
        return std::size_t(0);
      }
    }();

    template <class Phase>
    auto get() noexcept -> Phase&
    {
      return *std::launder(reinterpret_cast<Phase*>(storage_.bytes + offset_of<Phase>));
    }

    template <class Phase, class... Args>
    auto enter(Args&&... args) -> Phase&
    {
      static_assert(index_of<Phase> != none, "not one of this operation's phases");
      storage_.unpoison(offset_of<Phase> + sizeof(Phase));
      Phase& phase = *::new (static_cast<void*>(storage_.bytes + offset_of<Phase>))
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
      if constexpr (receives<Phase>)
      {
        leave_received<Phase>();
      }
      storage_.poison(offset_of<Phase> + sizeof(Phase));
    }

    template <class Phase, class Fn>
    auto visit_if(Fn& fn) noexcept -> bool
    {
      if constexpr (std::is_invocable_v<Fn&, Phase&>)
      {
        if (live_ == index_of<Phase>)
        {
          fn(get<Phase>());
          return true;
        }
      }
      return false;
    }

    // Enters the first phase's slot, if it names one.
    template <class... Args>
    void enter_slot([[maybe_unused]] Args&... args)
    {
      if constexpr (has_slot)
      {
        slot_.unpoison(sizeof(slot_t));
        STDEXEC_TRY
        {
          ::new (static_cast<void*>(slot_.bytes)) slot_t(args...);
        }
        STDEXEC_CATCH_ALL
        {
          slot_.poison(sizeof(slot_t));
          STDEXEC_THROW();
        }
        slot_live_ = true;
      }
    }

    // Ends the slot, if it's live.
    void end_slot() noexcept
    {
      if constexpr (has_slot)
      {
        if (slot_live_)
        {
          slot_live_ = false;
          std::destroy_at(&slot());
          slot_.poison(sizeof(slot_t));
        }
      }
    }

    // Enters what Phase holds while it runs, if anything.
    template <class Phase>
    void activate() noexcept
    {
      if constexpr (!std::is_void_v<active_t<Phase>>)
      {
        static_assert(std::is_nothrow_constructible_v<active_t<Phase>, Phase&, env_t const &>,
                      "what a phase holds while it runs is entered without throwing");
        active_.unpoison(sizeof(active_t<Phase>));
        ::new (static_cast<void*>(active_.bytes)) active_t<Phase>(get<Phase>(), ex::get_env(rcvr_));
      }
    }

    template <class Phase>
    void deactivate() noexcept
    {
      if constexpr (!std::is_void_v<active_t<Phase>>)
      {
        std::destroy_at(std::launder(reinterpret_cast<active_t<Phase>*>(active_.bytes)));
        active_.poison(sizeof(active_t<Phase>));
      }
    }

    // While the dying phase is alive: constructs the values Phase receives in
    // place, or under overlapped_layout returns them, built on the stack.
    template <class Phase, class Values>
    auto stage(Values&& values)
    {
      using values_t = received_t<Phase>;
      if constexpr (overlapped)
      {
        return std::make_from_tuple<values_t>(static_cast<Values&&>(values));
      }
      else
      {
        if constexpr (scoped)
        {
          scoped_.unpoison(sizeof(values_t));
          STDEXEC_TRY
          {
            std::apply(
              [this](auto&&... as)
              {
                ::new (static_cast<void*>(scoped_.bytes))
                  values_t(static_cast<decltype(as)&&>(as)...);
              },
              static_cast<Values&&>(values));
          }
          STDEXEC_CATCH_ALL
          {
            scoped_.poison(sizeof(values_t));
            STDEXEC_THROW();
          }
        }
        else
        {
          std::apply(
            [this](auto&&... as)
            {
              received_.template emplace<index_of<Phase> + 1>(static_cast<decltype(as)&&>(as)...);
            },
            static_cast<Values&&>(values));
        }
        return nothing_staged();
      }
    }

    // Once the dying phase is gone: under overlapped_layout, moves the
    // staged values into its bytes.
    template <class Phase, class Staged>
    void place([[maybe_unused]] Staged& staged)
    {
      if constexpr (overlapped)
      {
        using values_t = received_t<Phase>;
        storage_.unpoison(sizeof(values_t));
        STDEXEC_TRY
        {
          ::new (static_cast<void*>(storage_.bytes)) values_t(std::move(staged));
        }
        STDEXEC_CATCH_ALL
        {
          storage_.poison(sizeof(values_t));
          STDEXEC_THROW();
        }
      }
    }

    // Ends the values Phase received, unless they're carried.
    template <class Phase>
    void leave_received() noexcept
    {
      if constexpr (!carried_received)
      {
        std::destroy_at(&received<Phase>());
        if constexpr (overlapped)
        {
          storage_.poison(sizeof(received_t<Phase>));
        }
        else
        {
          scoped_.poison(sizeof(received_t<Phase>));
        }
      }
    }

    struct nothing_staged
    {};

    // Enters the completed state from Phase: ends Phase, unless being
    // completed holds it, in which case the destructor will.
    template <class Phase>
    void settle() noexcept
    {
      if constexpr (!completed_holds_phase)
      {
        leave<Phase>();
      }
    }

    // From inside Phase's completion, completes the operation: Phase stops
    // running, the operation settles, the slot ends, and deliver completes
    // the receiver.
    // If Phase ends as the operation settles, or Copy asks for copies,
    // deliver gets rvalue copies of as..., made while Phase is alive;
    // otherwise as... themselves. Whatever deliver throws completes the
    // receiver with set_error.
    template <class Phase, bool Copy, class Deliver, class... As>
    void complete(Deliver deliver, As&&... as) noexcept
    {
      deactivate<Phase>();
      if constexpr (Copy || !completed_holds_phase)
      {
        STDEXEC_TRY
        {
          std::tuple<std::decay_t<As>...> copies(static_cast<As&&>(as)...);
          settle<Phase>();
          end_slot();
          deliver_or_fail(deliver, std::move(copies));
        }
        STDEXEC_CATCH_ALL
        {
          settle<Phase>();  // only copying can throw here: Phase is live
          fail();
        }
      }
      else
      {
        end_slot();
        deliver_or_fail(deliver, std::forward_as_tuple(static_cast<As&&>(as)...));
      }
    }

    template <class Deliver, class Args>
    void deliver_or_fail(Deliver& deliver, Args&& args) noexcept
    {
      STDEXEC_TRY
      {
        std::apply(deliver, static_cast<Args&&>(args));
      }
      STDEXEC_CATCH_ALL
      {
        fail();
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

    // Completes the operation with the exception being handled. The live
    // phase, if any, has ended or settled already.
    void fail() noexcept
    {
      end_slot();
      ex::set_error(static_cast<Rcvr&&>(rcvr_), std::current_exception());
    }

    static constexpr bool any_receive = (receives<Rest> || ...);

    static constexpr std::size_t storage_size = std::max(
      {sizeof(First), (offset_of<Rest> + sizeof(Rest))...});
    static constexpr std::size_t storage_align = std::max(
      {alignof(First),
       alignof(Rest)...,
       (overlapped ? detail::align_of<received_t<Rest>> : std::size_t(1))...});

    static constexpr std::size_t scoped_size = std::max(
      {detail::size_of<received_t<Rest>>..., std::size_t(0)});
    static constexpr std::size_t scoped_align = std::max(
      {detail::align_of<received_t<Rest>>..., std::size_t(1)});

    using carried_values_t =
      std::conditional_t<carried_received && any_receive,
                         std::variant<std::monostate,
                                      typename detail::or_monostate<received_t<First>>::type,
                                      typename detail::or_monostate<received_t<Rest>>::type...>,
                         detail::no_carried_values>;

    using scoped_region_t = std::conditional_t<scoped && any_receive,
                                               detail::guarded_bytes<scoped_size, scoped_align>,
                                               detail::no_scoped_region>;

    using env_t = ex::env_of_t<Rcvr>;

    static constexpr std::size_t active_size = std::max(
      {detail::size_of<active_t<First>>, detail::size_of<active_t<Rest>>...});
    static constexpr std::size_t active_align = std::max(
      {detail::align_of<active_t<First>>, detail::align_of<active_t<Rest>>...});

    using active_region_t = std::conditional_t<active_size != 0,
                                               detail::guarded_bytes<active_size, active_align>,
                                               detail::no_active_region>;

    using slot_region_t =
      std::conditional_t<has_slot,
                         detail::guarded_bytes<detail::size_of<slot_t>, detail::align_of<slot_t>>,
                         detail::no_slot_region>;
    using slot_flag_t = std::conditional_t<has_slot, bool, detail::no_slot_flag>;

    Rcvr rcvr_;
    STDEXEC_ATTRIBUTE(no_unique_address) Carried carried_;
    STDEXEC_ATTRIBUTE(no_unique_address) parts_storage_t parts_;
    STDEXEC_ATTRIBUTE(no_unique_address) live_t live_ { none_or_empty() };
    STDEXEC_ATTRIBUTE(no_unique_address) slot_flag_t slot_live_ {};
    STDEXEC_ATTRIBUTE(no_unique_address) carried_values_t received_;
    STDEXEC_ATTRIBUTE(no_unique_address) scoped_region_t scoped_;
    STDEXEC_ATTRIBUTE(no_unique_address) active_region_t active_;
    STDEXEC_ATTRIBUTE(no_unique_address) slot_region_t slot_;
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

  // phased<Rcvr, Parts, Carried, First, Next<Ts...>...>, one Next per
  // type_list<Ts...>.
  template <class Rcvr,
            class Parts,
            class Carried,
            class First,
            template <class...> class Next,
            class Lists>
  struct phased_with;

  template <class Rcvr,
            class Parts,
            class Carried,
            class First,
            template <class...> class Next,
            class... Lists>
  struct phased_with<Rcvr, Parts, Carried, First, Next, detail::type_list<Lists...>>
  {
    template <class List>
    struct apply;

    template <class... Ts>
    struct apply<detail::type_list<Ts...>>
    {
      using type = Next<Ts...>;
    };

    using type = phased<Rcvr, Parts, Carried, First, typename apply<Lists>::type...>;
  };

  // Env, with Query answered by a value of its own instead: what a phase
  // gives its children to add or replace one query of the operation's
  // environment.
  template <class Query, class Value, class Env>
  struct env_with
  {
    [[nodiscard]]
    auto query(Query) const noexcept -> Value const &
    {
      return value;
    }

    template <class Q, class... As>
      requires(!std::is_same_v<Q, Query>)
           && requires(Env const & env, Q q, As&&... as) { env.query(q, static_cast<As&&>(as)...); }
    [[nodiscard]]
    auto query(Q q, As&&... as) const
      noexcept(noexcept(std::declval<Env const &>().query(q, static_cast<As&&>(as)...)))
        -> decltype(auto)
    {
      return env.query(q, static_cast<As&&>(as)...);
    }

    Value value;
    Env   env;
  };

  // Children of one phase that run at once and stop together: a stop source
  // of their own, and a count of the children still running. The phase
  // holds it before its children, whose environments carry token(). Each
  // child's completion arrives once: with arrive(), or with
  // stop_and_arrive() to stop the others first. The one that returns true
  // is the last; nothing else will touch the phase, so the operation may
  // end it.
  //
  // While the phase runs, its active_t holds a forwarding<Env> (see
  // phased), so that stop requests from the operation's environment reach
  // the scope too; phased ends it before the operation completes, waiting
  // for a request being forwarded on another thread. A stop request from
  // the environment can make the last child complete inside this scope's
  // request_stop, and so end the scope there; stdexec's inplace_stop_source
  // allows its own callbacks to destroy it.
  class join_scope
  {
    struct forward_stop
    {
      ex::inplace_stop_source* source;

      void operator()() const noexcept
      {
        source->request_stop();
      }
    };

   public:
    // Forwards Env's stop requests to a scope for as long as it lives.
    template <class Env>
    class forwarding
    {
      using callback_t = ex::stop_callback_for_t<ex::stop_token_of_t<Env const &>, forward_stop>;

     public:
      forwarding(join_scope& scope, Env const & env) noexcept
        : callback_(ex::get_stop_token(env), forward_stop{&scope.source_})
      {}

     private:
      callback_t callback_;
    };

    explicit join_scope(std::size_t count) noexcept
      : count_(count)
    {}

    join_scope(join_scope&&) = delete;

    [[nodiscard]]
    auto token() const noexcept -> ex::inplace_stop_token
    {
      return source_.get_token();
    }

    // A child that arrives this way never ends the scope during its own
    // request_stop: until it arrives, the count includes it.
    [[nodiscard]]
    auto stop_and_arrive() noexcept -> bool
    {
      source_.request_stop();
      return arrive();
    }

    [[nodiscard]]
    auto arrive() noexcept -> bool
    {
      return count_.fetch_sub(1, std::memory_order_acq_rel) == 1;
    }

   private:
    ex::inplace_stop_source  source_;
    std::atomic<std::size_t> count_;
  };

  // Calls fn(), returning what it threw, if anything.
  template <class Fn>
  auto attempt(Fn fn) noexcept -> std::exception_ptr
  {
    STDEXEC_TRY
    {
      fn();
      return nullptr;
    }
    STDEXEC_CATCH_ALL
    {
      return std::current_exception();
    }
  }

  // The receiver a phase's child completes through. `Tag` names the phase,
  // or the child when a phase runs several; `Part`, if not null, points to
  // the part of the operation's sender the child is made from, which the
  // child binds to under stored_sender.
  //
  // `Env`, if not void, is the environment the child sees instead of the
  // operation's receiver's: the operation's env(Tag()) returns it, built
  // from what the phase holds (its own stop source, say). Its type is named
  // up front because the operation is incomplete while its phases' children
  // are declared. It should carry over the receiver's environment, policies
  // included, adding or replacing what it must; env_with does that.
  template <class Op, class Rcvr, class Tag, auto Part = nullptr, class Env = void>
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

    using env_t = std::conditional_t<std::is_void_v<Env>, ex::env_of_t<Rcvr>, Env>;

    [[nodiscard]]
    auto get_env() const noexcept -> env_t
    {
      if constexpr (std::is_void_v<Env>)
      {
        return ex::get_env(op->phases_.receiver());
      }
      else
      {
        return op->env(Tag());
      }
    }

    // The child sender this phase binds, for a child that binds rather than
    // connects: const if the operation was bound without consuming, so the
    // child is bound the same way.
    [[nodiscard]]
    auto bound_sender() const noexcept -> std::add_lvalue_reference_t<
      maybe_const<detail::bound_const<Op>::value, detail::member_type_t<Part>>>
      requires(!std::is_null_pointer_v<decltype(Part)>)
    {
      return op->phases_.template part<Part>();
    }

    Op* op;
  };

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

  // What every typestate sender's connect returns: the sender, stored once,
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

  // What a phase-based sender's connect returns: under stored_sender, a root
  // storing the sender; under split_sender, the operation itself, which
  // moves the sender's parts out of it.
  template <class Sndr, class Rcvr>
  auto connect_sender(Sndr& sndr, Rcvr rcvr)
  {
    if constexpr (std::is_same_v<typename policies_of_t<ex::env_of_t<Rcvr>>::data, split_sender>)
    {
      return typename Sndr::template opstate<Rcvr>(static_cast<Rcvr&&>(rcvr), sndr);
    }
    else
    {
      return root<Sndr, Rcvr>(static_cast<Sndr&&>(sndr), static_cast<Rcvr&&>(rcvr));
    }
  }

  // How a phase-based sender's stored sender may be bound: may_consume binds
  // through a non-const reference when it can, so the operation may move
  // from the sender, as connecting an rvalue may; read_only always binds
  // through a const reference, so the stored sender can be bound again.
  struct may_consume
  {};

  struct read_only
  {};

  // The base of every phase-based sender, Derived, whose operations are
  // Derived::opstate<Rcvr, Const>: Const when bound through a const
  // reference. It connects, under either Data policy, and binds to a stored
  // sender. The base comes first, so Derived is initialized as {{}, ...}.
  template <class Derived, class Binding = may_consume>
  struct phased_sender
  {
    using sender_concept = ex::sender_t;

    template <class Rcvr>
    auto connect(Rcvr rcvr) &&
    {
      return connect_sender(static_cast<Derived&>(*this), static_cast<Rcvr&&>(rcvr));
    }

    template <class Rcvr>
      requires binds_stored<Rcvr> && std::is_same_v<Binding, may_consume>
    auto bind(Rcvr rcvr) &
    {
      STDEXEC_ASSERT(&rcvr.bound_sender() == static_cast<Derived*>(this));
      return typename Derived::template opstate<Rcvr>(static_cast<Rcvr&&>(rcvr));
    }

    template <class Rcvr>
      requires binds_stored<Rcvr>
    auto bind(Rcvr rcvr) const &
    {
      STDEXEC_ASSERT(&rcvr.bound_sender() == static_cast<Derived const *>(this));
      return typename Derived::template opstate<Rcvr, true>(static_cast<Rcvr&&>(rcvr));
    }
  };

  // From an operation's constructor: the part M points to of the sender
  // being connected, under split_sender, or else of the stored sender.
  template <auto M, class Rcvr, class... Incoming>
  auto peek(Rcvr const & rcvr, Incoming&... incoming) -> decltype(auto)
  {
    if constexpr (sizeof...(Incoming) == 1)
    {
      return ((incoming.*M), ...);
    }
    else
    {
      return (rcvr.bound_sender().*M);
    }
  }

  // sndr, with its operations asking for the policies P. Like any adaptor,
  // it adds a layer of connect, which moves sndr's data again; connecting
  // with a receiver whose environment asks for P costs nothing.
  template <class P, class Sndr>
  auto with_policies(Sndr&& sndr)
  {
    return ex::write_env(static_cast<Sndr&&>(sndr), policies_env<P>());
  }
}  // namespace frames::typestate
