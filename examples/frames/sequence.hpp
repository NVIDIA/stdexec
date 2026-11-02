/*
 * Exploration: sequence(first, second) as a sequence of phases, written against the
 * machinery in typestate.hpp.
 */
#pragma once

#include "typestate.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace frames::typestate
{
  // sequence(first, second): runs first, discards its values, then runs
  // second. Both senders stay in the stored sender, and each phase binds to
  // its own, so nothing crosses the transition.

  template <class Ending, class Entry, class Sndr1, class Sndr2>
  struct sequence_sender
  {
    using sender_concept = ex::sender_t;

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

      template <class Sndr, class R>
      using child_t = bind_result_t<maybe_const<Const, Sndr>, R>;

      struct first
      {
        explicit first(opstate* op)
          : child(typestate::bind(op->self().sndr1_, first_rcvr_t{op}))
        {}

        child_t<Sndr1, first_rcvr_t> child;
      };

      struct second
      {
        explicit second(opstate* op)
          : child(typestate::bind(op->self().sndr2_, second_rcvr_t{op}))
        {}

        child_t<Sndr2, second_rcvr_t> child;
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

      template <class... As>
      void complete(first_tag, ex::set_value_t, As&&...) noexcept
      {
        phases_.template transition<first, second>([](first&) { return std::tuple(); }, this);
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

      phased<Ending, carried_layout, Entry, Rcvr, nothing, first, second> phases_;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) && -> root<sequence_sender, Rcvr>
    {
      return root<sequence_sender, Rcvr>(std::move(*this), static_cast<Rcvr&&>(rcvr));
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

    Sndr1 sndr1_;
    Sndr2 sndr2_;
  };

  template <class Ending = eager_teardown, class Entry = connect_entry, class Sndr1, class Sndr2>
  auto sequence(Sndr1&& sndr1, Sndr2&& sndr2)
  {
    return sequence_sender<Ending, Entry, std::decay_t<Sndr1>, std::decay_t<Sndr2>>{
      static_cast<Sndr1&&>(sndr1),
      static_cast<Sndr2&&>(sndr2)};
  }
}  // namespace frames::typestate
