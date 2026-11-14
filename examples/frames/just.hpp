/*
 * Exploration: just(vs...) as a sequence of phases, written against the
 * machinery in typestate.hpp.
 */
#pragma once

#include "typestate.hpp"

#include <exception>
#include <tuple>
#include <type_traits>
#include <utility>

namespace frames::typestate
{
  // just(vs...): one phase, sending the sender's values.

  template <class... Vs>
  struct just_sender : phased_sender<just_sender<Vs...>>
  {
    std::tuple<Vs...> values_;

    template <class Self, class... Env>
    static consteval auto get_completion_signatures()
    {
      if constexpr (sizeof...(Env) == 0)
      {
        return STDEXEC::__throw_dependent_sender_error<Self>();
      }
      else if constexpr (reads_only<Self>)
      {
        // Bound without consuming: the values are copied.
        return sent_as_copies(ex::completion_signatures<ex::set_value_t(Vs const &...)>());
      }
      else
      {
        return sent_as<ending_of_t<Env...>>(ex::completion_signatures<ex::set_value_t(Vs...)>());
      }
    }

    template <class Rcvr, bool Const = false>
    struct opstate
    {
      using parts_t = maybe_const<Const, parts<&just_sender::values_>>;

      struct ready
      {
        explicit ready(opstate *) noexcept {}
      };

      template <class... Incoming>
      explicit opstate(Rcvr rcvr, Incoming &...incoming)
        : phases_(static_cast<Rcvr &&>(rcvr), nothing(), incoming...)
      {
        phases_.on_connect(this, incoming...);
      }

      void start() & noexcept
      {
        phases_.begin(
          [this](ready &r) noexcept
          {
            if constexpr (Const)
            {
              std::apply([this](Vs const &...vs)
                         { phases_.template finish_copies<ready>(ex::set_value, vs...); },
                         values(r));
            }
            else
            {
              std::apply([this](Vs &...vs)
                         { phases_.template finish<ready>(ex::set_value, std::move(vs)...); },
                         values(r));
            }
          },
          this);
      }

      auto values(ready &) noexcept -> maybe_const<Const, std::tuple<Vs...>> &
      {
        return phases_.template part<&just_sender::values_>();
      }

      phased<Rcvr, parts_t, nothing, ready> phases_;
    };
  };

  template <class... Vs>
  auto just(Vs &&...vs)
  {
    return just_sender<std::decay_t<Vs>...>{{}, {static_cast<Vs &&>(vs)...}};
  }
}  // namespace frames::typestate
