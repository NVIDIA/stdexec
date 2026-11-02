/*
 * Exploration: then(sndr, fn) as a sequence of phases, written against the
 * machinery in typestate.hpp.
 */
#pragma once

#include "typestate.hpp"

#include <type_traits>
#include <utility>

namespace frames::typestate
{
  // then(sndr, fn): one phase, running the child. fn is used in place in the
  // stored sender.

  template <class Ending, class Entry, class Sndr, class Fn>
  struct then_sender
  {
    using sender_concept = ex::sender_t;

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
      using child_t = bind_result_t<maybe_const<Const, Sndr>, rcvr_t>;

      struct running
      {
        explicit running(opstate* op)
          : child(typestate::bind(op->self().sndr_, rcvr_t{op}))
        {}

        child_t child;
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

      phased<Ending, carried_layout, Entry, Rcvr, nothing, running> phases_;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) && -> root<then_sender, Rcvr>
    {
      return root<then_sender, Rcvr>(std::move(*this), static_cast<Rcvr&&>(rcvr));
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

  template <class Ending = eager_teardown, class Entry = connect_entry, class Sndr, class Fn>
  auto then(Sndr&& sndr, Fn&& fn)
  {
    return then_sender<Ending, Entry, std::decay_t<Sndr>, std::decay_t<Fn>>{static_cast<Sndr&&>(
                                                                              sndr),
                                                                            static_cast<Fn&&>(fn)};
  }
}  // namespace frames::typestate
