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

  template <class Ending, class Entry, class Data, class... Vs>
  struct just_sender
  {
    using sender_concept = ex::sender_t;

    std::tuple<Vs...> values_;

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
      using parts_t = parts<&just_sender::values_>;

      struct ready
      {
        explicit ready(opstate*) noexcept {}
      };

      template <class... Incoming>
      explicit opstate(Rcvr rcvr, Incoming&... incoming)
        : phases_(static_cast<Rcvr&&>(rcvr), nothing(), incoming...)
      {
        phases_.on_connect(this, incoming...);
      }

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

      auto values(ready&) noexcept -> maybe_const<Const, std::tuple<Vs...>>&
      {
        return phases_.template part<&just_sender::values_>();
      }

      phased<Ending, carried_layout, Entry, Data, Rcvr, parts_t, nothing, ready> phases_;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) &&
    {
      return connect_sender<Data>(*this, static_cast<Rcvr&&>(rcvr));
    }

    template <class Rcvr>
      requires std::is_same_v<Data, stored_sender> && requires(Rcvr const & r) { r.bound_sender(); }
    auto bind(Rcvr rcvr) & -> opstate<Rcvr>
    {
      STDEXEC_ASSERT(&rcvr.bound_sender() == this);
      return opstate<Rcvr>(static_cast<Rcvr&&>(rcvr));
    }

    template <class Rcvr>
      requires std::is_same_v<Data, stored_sender> && requires(Rcvr const & r) { r.bound_sender(); }
    auto bind(Rcvr rcvr) const & -> opstate<Rcvr, true>
    {
      STDEXEC_ASSERT(&rcvr.bound_sender() == this);
      return opstate<Rcvr, true>(static_cast<Rcvr&&>(rcvr));
    }
  };

  template <class Ending = eager_teardown,
            class Entry  = connect_entry,
            class Data   = stored_sender,
            class... Vs>
  auto just(Vs&&... vs)
  {
    return just_sender<Ending, Entry, Data, std::decay_t<Vs>...>{{static_cast<Vs&&>(vs)...}};
  }
}  // namespace frames::typestate
