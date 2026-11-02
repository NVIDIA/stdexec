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
  // copies it, as exec::repeat_n does.)
  //
  // Prototype simplification: no trampoline, so a child that completes
  // inline makes each iteration a nested call.

  template <class Sndr>
  struct repeat_n_sender
  {
    using sender_concept = ex::sender_t;

    Sndr        sndr_;
    std::size_t count_;

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
          detail::sent_as<ending_of_t<Env...>>,
          detail::map_values_t<ending_of_t<Env...>,
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

      using parts_t = parts<&repeat_n_sender::sndr_>;
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

      template <class... Incoming>
      explicit opstate(Rcvr rcvr, Incoming&... incoming)
        : opstate(peek<&repeat_n_sender::count_>(rcvr, incoming...),
                  static_cast<Rcvr&&>(rcvr),
                  incoming...)
      {}

      template <class... Incoming>
      opstate(std::size_t count, Rcvr rcvr, Incoming&... incoming)
        : phases_(static_cast<Rcvr&&>(rcvr), carried{count}, incoming...)
      {
        phases_.on_connect(this, incoming...);
      }

      auto child_sender() noexcept -> Sndr const &
      {
        return phases_.template part<&repeat_n_sender::sndr_>();
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

      phased<Rcvr, parts_t, carried, iteration> phases_;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) &&
    {
      return connect_sender(*this, static_cast<Rcvr&&>(rcvr));
    }

    // repeat_n only ever reads its child sender, so binding it consumes
    // nothing either way.
    template <class Rcvr>
      requires binds_stored<Rcvr>
    auto bind(Rcvr rcvr) const & -> opstate<Rcvr, true>
    {
      STDEXEC_ASSERT(&rcvr.bound_sender() == this);
      return opstate<Rcvr, true>(static_cast<Rcvr&&>(rcvr));
    }
  };

  template <class Sndr>
  auto repeat_n(Sndr&& sndr, std::size_t count)
  {
    return repeat_n_sender<std::decay_t<Sndr>>{static_cast<Sndr&&>(sndr), count};
  }
}  // namespace frames::typestate
