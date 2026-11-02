/*
 * Exploration: repeat_n(sndr, n) as a sequence of phases, written against the
 * machinery in typestate.hpp.
 */
#pragma once

#include "typestate.hpp"

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace frames::typestate
{
  // repeat_n(sndr, n): runs sndr n times in sequence, discarding its values,
  // then completes with set_value(). One phase, re-entered for each
  // iteration; each time, the phase binds the stored child sender without
  // consuming it, so a child that can bind is never copied. (A child that
  // can't, such as any stdexec sender, is connected as a const lvalue, which
  // copies it, as exec::repeat_n does.) The child stays in the stored
  // sender.
  //
  // Prototype simplification: no trampoline, so a child that completes
  // inline makes each iteration a nested call.

  template <class Ending, class Entry, class Sndr>
  struct repeat_n_sender
  {
    using sender_concept = ex::sender_t;

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
        std::size_t remaining;
      };

      struct iteration
      {
        explicit iteration(opstate* op)
          : child(typestate::bind(op->child_sender(), rcvr_t{op}))
        {}

        child_t child;
      };

      explicit opstate(Rcvr rcvr)
        : opstate(rcvr.bound_sender().count_, static_cast<Rcvr&&>(rcvr))
      {}

      opstate(std::size_t count, Rcvr rcvr)
        : phases_(static_cast<Rcvr&&>(rcvr), carried{count})
      {
        phases_.on_connect(this);
      }

      auto child_sender() noexcept -> Sndr const &
      {
        return self().sndr_;
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

      phased<Ending, carried_layout, Entry, Rcvr, carried, iteration> phases_;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) && -> root<repeat_n_sender, Rcvr>
    {
      return root<repeat_n_sender, Rcvr>(std::move(*this), static_cast<Rcvr&&>(rcvr));
    }

    // repeat_n only ever reads its child sender, so binding it consumes
    // nothing either way.
    template <class Rcvr>
      requires requires(Rcvr const & r) { r.bound_sender(); }
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
