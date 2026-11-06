/*
 * Exploration: when_all(sndrs...) as one phase whose children all run at
 * once, written against the machinery in typestate.hpp.
 */
#pragma once

#include "typestate.hpp"

#include <atomic>
#include <cstddef>
#include <exception>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace frames::typestate
{
  // when_all(sndrs...): one phase, holding a child for each sender, all
  // running at once. The children see the operation's environment with a
  // stop token of the phase's own: the phase's join_scope stops them all
  // when one fails or stops, or when the operation's receiver asks. The last
  // child to complete finishes the operation, with every child's values if
  // all of them succeeded, otherwise with the first error, or else stopped.
  // Values and errors are kept in the phase until then, so they're sent as
  // decayed copies whatever the Ending policy.
  //
  // Like stdexec's when_all, each child may send at most one set of values.

  namespace detail
  {
    // A child that never sends values: when_all never sends any either.
    struct no_values
    {};

    template <class List>
    struct single_list;

    template <>
    struct single_list<type_list<>>
    {
      using tuple = no_values;
    };

    template <class... Ts>
    struct single_list<type_list<type_list<Ts...>>>
    {
      using tuple = std::tuple<Ts...>;
    };

    // A child's values, as a tuple, or no_values.
    template <class Sigs>
    using when_all_tuple_t = typename single_list<typename value_lists<Sigs>::type>::tuple;

    // set_value_t of every child's values in order, unless a child sends
    // none.
    template <class... Tuples>
    struct concat_values
    {
      using type = ex::completion_signatures<>;
    };

    template <class... Ts>
    struct concat_values<std::tuple<Ts...>>
    {
      using type = ex::completion_signatures<ex::set_value_t(Ts...)>;
    };

    template <class... As, class... Bs, class... Rest>
    struct concat_values<std::tuple<As...>, std::tuple<Bs...>, Rest...>
      : concat_values<std::tuple<As..., Bs...>, Rest...>
    {};

    // A child's completions other than its values.
    template <class Sig>
    struct non_value
    {
      using type = ex::completion_signatures<Sig>;
    };

    template <class... As>
    struct non_value<ex::set_value_t(As...)>
    {
      using type = ex::completion_signatures<>;
    };

    template <class Sigs>
    struct non_values;

    template <class... Sigs>
    struct non_values<ex::completion_signatures<Sigs...>>
    {
      using type = join_t<sent_as<eager_teardown>, typename non_value<Sigs>::type...>;
    };

    // std::variant<std::monostate, E...>, one E per decayed error of Sigs.
    template <class List, class... Sigs>
    struct errors_of
    {
      using type = List;
    };

    template <class List, class Sig, class... Rest>
    struct errors_of<List, Sig, Rest...> : errors_of<List, Rest...>
    {};

    template <class List, class E, class... Rest>
    struct errors_of<List, ex::set_error_t(E), Rest...>
      : errors_of<typename add_type<List, std::decay_t<E>>::type, Rest...>
    {};

    template <class List>
    struct variant_of;

    template <class... Es>
    struct variant_of<type_list<Es...>>
    {
      using type = std::variant<std::monostate, Es...>;
    };

    template <class Sigs>
    struct errors_variant;

    template <class... Sigs>
    struct errors_variant<ex::completion_signatures<Sigs...>>
    {
      using type =
        typename variant_of<typename errors_of<type_list<std::exception_ptr>, Sigs...>::type>::type;
    };
  }  // namespace detail

  // One of when_all's senders, as a member an operation can point to.
  template <std::size_t I, class Sndr>
  struct when_all_slot
  {
    Sndr sndr;
  };

  // What a when_all child sees: the operation's environment, with the
  // phase's stop token.
  template <class Env>
  using when_all_env_t = env_with<ex::get_stop_token_t, ex::inplace_stop_token, Env>;

  template <class Indices, class... Sndrs>
  struct when_all_sender;

  template <std::size_t... Is, class... Sndrs>
  struct when_all_sender<std::index_sequence<Is...>, Sndrs...>
    : phased_sender<when_all_sender<std::index_sequence<Is...>, Sndrs...>>
    , when_all_slot<Is, Sndrs>...
  {
    template <std::size_t I>
    using sndr_t = std::tuple_element_t<I, std::tuple<Sndrs...>>;

    template <std::size_t I>
    static constexpr auto member = &when_all_slot<I, sndr_t<I>>::sndr;

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
          detail::sent_as<eager_teardown>,
          typename detail::concat_values<detail::when_all_tuple_t<
            ex::completion_signatures_of_t<member_t<Self, Sndrs>, when_all_env_t<Env>...>>...>::type,
          typename detail::non_values<
            ex::completion_signatures_of_t<member_t<Self, Sndrs>, when_all_env_t<Env>...>>::type...,
          ex::completion_signatures<ex::set_error_t(std::exception_ptr)>>();
      }
    }

    template <class Rcvr, bool Const = false>
    struct opstate
    {
      template <std::size_t I>
      struct child_tag
      {};

      using env_t = when_all_env_t<ex::env_of_t<Rcvr>>;

      template <std::size_t I>
      using rcvr_t = phase_receiver<opstate, Rcvr, child_tag<I>, member<I>, env_t>;

      template <std::size_t I>
      using child_t = part_child_t<maybe_const<Const, sndr_t<I>>, rcvr_t<I>>;

      template <std::size_t I>
      using sigs_t =
        ex::completion_signatures_of_t<std::conditional_t<Const, sndr_t<I> const &, sndr_t<I>>,
                                       env_t>;

      using values_t = std::tuple<std::optional<detail::when_all_tuple_t<sigs_t<Is>>>...>;

      // Otherwise the last child can't arrive with every child's values.
      static constexpr bool all_send_values =
        (!std::is_same_v<detail::when_all_tuple_t<sigs_t<Is>>, detail::no_values> && ...);
      using errors_t = typename detail::errors_variant<
        detail::join_t<detail::sent_as<forward_results>, sigs_t<Is>...>>::type;

      enum class outcome : unsigned char
      {
        values,
        error,
        stopped
      };

      template <std::size_t I>
      struct child_slot
      {
        template <class... Incoming>
        explicit child_slot(opstate* op, Incoming&... incoming)
          : child(op->phases_.template connect_part<member<I>>(rcvr_t<I>{op}, incoming...))
        {}

        child_t<I> child;
      };

      struct children_t : child_slot<Is>...
      {
        template <class... Incoming>
        explicit children_t(opstate* op, Incoming&... incoming)
          : child_slot<Is>(op, incoming...)...
        {}
      };

      struct running;

      // While the operation runs, its receiver's stop requests reach the
      // phase's scope.
      struct listening
      {
        listening(running& phase, ex::env_of_t<Rcvr> const & env) noexcept
          : forward(phase.scope, env)
        {}

        join_scope::forwarding<ex::env_of_t<Rcvr>> forward;
      };

      struct running
      {
        static constexpr parts<member<Is>...> connects{};
        using active_t = listening;

        template <class... Incoming>
        explicit running(opstate* op, Incoming&... incoming)
          : children(op, incoming...)
        {}

        // Before the children, whose environments carry its token.
        join_scope           scope{sizeof...(Sndrs)};
        std::atomic<outcome> state{outcome::values};
        values_t             values;
        errors_t             error;
        children_t           children;
      };

      template <class... Incoming>
      explicit opstate(Rcvr rcvr, Incoming&... incoming)
        : phases_(static_cast<Rcvr&&>(rcvr), nothing(), incoming...)
      {
        phases_.on_connect(this, incoming...);
      }

      void start() & noexcept
      {
        phases_.begin([](running& phase) noexcept
                      { (ex::start(static_cast<child_slot<Is>&>(phase.children).child), ...); },
                      this);
      }

      template <std::size_t I>
      auto env(child_tag<I>) noexcept -> env_t
      {
        return {phases_.template live<running>().scope.token(), ex::get_env(phases_.receiver())};
      }

      template <std::size_t I, class... As>
      void complete(child_tag<I>, ex::set_value_t, As&&... as) noexcept
      {
        running& phase = phases_.template live<running>();
        if (phase.state.load(std::memory_order_relaxed) == outcome::values)
        {
          if (std::exception_ptr e = attempt(
                [&] { std::get<I>(phase.values).emplace(static_cast<As&&>(as)...); }))
          {
            fail(phase, std::move(e));
            return;
          }
        }
        done(phase.scope.arrive());
      }

      template <std::size_t I, class E>
      void complete(child_tag<I>, ex::set_error_t, E&& e) noexcept
      {
        fail(phases_.template live<running>(), static_cast<E&&>(e));
      }

      template <std::size_t I>
      void complete(child_tag<I>, ex::set_stopped_t) noexcept
      {
        running& phase    = phases_.template live<running>();
        outcome  expected = outcome::values;
        phase.state.compare_exchange_strong(expected, outcome::stopped);
        done(phase.scope.stop_and_arrive());
      }

      // The first error wins, and outranks stopped.
      template <class E>
      void fail(running& phase, E&& e) noexcept
      {
        if (phase.state.exchange(outcome::error) != outcome::error)
        {
          if (std::exception_ptr thrown = attempt(
                [&] { phase.error.template emplace<std::decay_t<E>>(static_cast<E&&>(e)); }))
          {
            phase.error.template emplace<std::exception_ptr>(std::move(thrown));
          }
        }
        done(phase.scope.stop_and_arrive());
      }

      void done(bool last) noexcept
      {
        if (!last)
        {
          return;
        }
        running& phase = phases_.template live<running>();
        switch (phase.state.load(std::memory_order_relaxed))
        {
        case outcome::values:
          if constexpr (all_send_values)
          {
            std::apply([this](auto&... vs)
                       { phases_.template finish<running>(ex::set_value, std::move(vs)...); },
                       std::tuple_cat(std::apply([](auto&... v)
                                                 { return std::forward_as_tuple(v...); },
                                                 *std::get<Is>(phase.values))...));
          }
          break;
        case outcome::error:
          std::visit(
            [this]<class E>(E& e)
            {
              if constexpr (!std::is_same_v<E, std::monostate>)
              {
                phases_.template finish<running>(ex::set_error, std::move(e));
              }
            },
            phase.error);
          break;
        case outcome::stopped:
          phases_.template finish<running>(ex::set_stopped);
          break;
        }
      }

      phased<Rcvr, parts<member<Is>...>, nothing, running> phases_;
    };
  };

  template <class... Sndrs>
  auto when_all(Sndrs&&... sndrs)
  {
    return when_all_sender<std::index_sequence_for<Sndrs...>, std::decay_t<Sndrs>...>{
      {},
      {static_cast<Sndrs&&>(sndrs)}...};
  }
}  // namespace frames::typestate
