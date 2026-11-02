/*
 * Exploration: let_value(sndr, fn) as a sequence of phases, written against the
 * machinery in typestate.hpp.
 */
#pragma once

#include "typestate.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace frames::typestate
{
  // let_value(sndr, fn): a first phase running the child, then one possible
  // second phase per value signature of the child, running the sender fn
  // returns. fn is used in place in the stored sender: the sender it returns
  // may refer to its captures. The first phase hands the values fn is
  // called with to the second, and phased places them as the Layout says.
  // Each second phase has its own receiver type, so its completion knows
  // which one is live. The sender fn returns is created at run time, so its
  // phase owns it.

  template <class Ending, class Layout, class Entry, class Sndr, class Fn>
  struct let_value_sender
  {
    using sender_concept = ex::sender_t;

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
      using child_t      = bind_result_t<maybe_const<Const, Sndr>, first_rcvr_t>;
      using values_t     = typename detail::value_lists<
            ex::completion_signatures_of_t<std::conditional_t<Const, Sndr const &, Sndr>,
                                           ex::env_of_t<Rcvr>>>::type;

      struct running
      {
        explicit running(opstate* op)
          : child(typestate::bind(op->self().sndr_, first_rcvr_t{op}))
        {}

        child_t child;
      };

      template <class... Ts>
      struct continuing
      {
        using rcvr_t     = phase_receiver<opstate, Rcvr, continuing_tag<Ts...>>;
        using received_t = std::tuple<Ts...>;
        using sndr_t     = std::invoke_result_t<maybe_const<Const, Fn>&, Ts&...>;

        explicit continuing(opstate* op)
          : child(ex::connect(std::apply(op->fn(), op->phases_.template received<continuing>()),
                              rcvr_t{op}))
        {}

        ex::connect_result_t<sndr_t, rcvr_t> child;
      };

      explicit opstate(Rcvr rcvr)
        : phases_(static_cast<Rcvr&&>(rcvr), nothing())
      {
        phases_.on_connect(this);
      }

      void start() & noexcept
      {
        phases_.begin(start_child, this);
      }

      auto fn() noexcept -> maybe_const<Const, Fn>&
      {
        return self().fn_;
      }

      template <class... As>
      void complete(running_tag, ex::set_value_t, As&&... as) noexcept
      {
        phases_.template transition<running, continuing<std::decay_t<As>...>>(
          [&](running&) { return hand_over(static_cast<As&&>(as)...); },
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

      typename phased_with<Ending, Layout, Entry, Rcvr, nothing, running, continuing, values_t>::type
        phases_;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) && -> root<let_value_sender, Rcvr>
    {
      return root<let_value_sender, Rcvr>(std::move(*this), static_cast<Rcvr&&>(rcvr));
    }

    template <class Rcvr>
      requires requires(Rcvr const & r) { r.bound_sender(); }
    auto bind(Rcvr rcvr) & -> opstate<Rcvr>
    {
      STDEXEC_ASSERT(&rcvr.bound_sender() == this);
      return opstate<Rcvr>(static_cast<Rcvr&&>(rcvr));
    }

    template <class Rcvr>
      requires requires(Rcvr const & r) { r.bound_sender(); }
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
}  // namespace frames::typestate
