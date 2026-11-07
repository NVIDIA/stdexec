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
  // running at once. If any child can fail, the children see the
  // operation's environment with a stop token of the phase's own: the
  // phase's join_scope stops them all when one fails or stops, or when the
  // operation's receiver asks. Otherwise they see the operation's
  // environment, as stdexec's when_all's do. The last
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
    using when_all_tuple_t = typename single_list<value_lists_t<Sigs>>::tuple;

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

    // No room for errors, for children that can't fail.
    struct no_errors
    {};

    template <class List>
    struct errors_variant_of
    {
      using type = no_errors;
    };

    template <class E, class... Es>
    struct errors_variant_of<type_list<E, Es...>>
    {
      using type = std::variant<std::monostate, E, Es...>;
    };

    // std::variant<std::monostate, E...>, one E per decayed error of Sigs,
    // or no_errors if there are none.
    template <class Sigs>
    using errors_variant_t = typename errors_variant_of<error_list_t<Sigs>>::type;
  }  // namespace detail

  // One of when_all's senders, as a member an operation can point to.
  template <std::size_t I, class Sndr>
  struct when_all_slot
  {
    Sndr sndr;
  };

  namespace detail
  {
    template <class Sig>
    inline constexpr bool is_value_sig = false;

    template <class... As>
    inline constexpr bool is_value_sig<ex::set_value_t(As...)> = true;

    // Whether a child that completes with Sigs can fail: send an error or
    // stopped, or throw as its values are kept.
    template <class... Sigs>
    consteval auto can_fail(ex::completion_signatures<Sigs...> sigs) -> bool
    {
      return (!is_value_sig<Sigs> || ...) || any_copy_may_throw(sigs);
    }

    // Whether any of when_all's children, used as CvSndrs, can fail under
    // Env.
    template <class Env, class... CvSndrs>
    inline constexpr bool any_can_fail = (can_fail(ex::get_completion_signatures<CvSndrs, Env>())
                                          || ...);
  }  // namespace detail

  // What when_all's children see, used as CvSndrs: if any can fail, the
  // operation's environment with a stop token of the phase's own, which
  // stops them all; otherwise, as with stdexec's when_all, the operation's
  // environment itself, since nothing but the receiver would ever stop them.
  template <class Env, class... CvSndrs>
  using when_all_env_t =
    std::conditional_t<detail::any_can_fail<Env, CvSndrs...>,
                       env_with<ex::get_stop_token_t, ex::inplace_stop_token, Env>,
                       Env>;

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
        using env_t       = std::tuple_element_t<0, std::tuple<Env...>>;
        using child_env_t = when_all_env_t<env_t, member_t<Self, Sndrs>...>;
        // The children's values and errors are kept as decayed copies,
        // which may throw.
        return exec::concat_completion_signatures(
          sent_as<ending_of_t<Env...>>(exec::concat_completion_signatures(
            typename detail::concat_values<detail::when_all_tuple_t<
              ex::completion_signatures_of_t<member_t<Self, Sndrs>, child_env_t>>...>::type(),
            exec::transform_completion_signatures(
              ex::get_completion_signatures<member_t<Self, Sndrs>, child_env_t>(),
              exec::ignore_completion(),
              detail::decayed<ex::set_error_t>())...)),
          eptr_if<(any_copy_may_throw(
                     ex::get_completion_signatures<member_t<Self, Sndrs>, child_env_t>())
                   || ...)
                  || start_enters<Env...>>());
      }
    }

    template <class Rcvr, bool Const = false>
    struct opstate
    {
      template <std::size_t I>
      struct child_tag
      {};

      template <std::size_t I>
      using cv_sndr_t = std::conditional_t<Const, sndr_t<I> const &, sndr_t<I>>;

      // Whether the children can stop one another.
      static constexpr bool stops = detail::any_can_fail<ex::env_of_t<Rcvr>, cv_sndr_t<Is>...>;

      using env_t = when_all_env_t<ex::env_of_t<Rcvr>, cv_sndr_t<Is>...>;

      template <std::size_t I>
      using rcvr_t = phase_receiver<opstate, Rcvr, child_tag<I>, member<I>, env_t>;

      template <std::size_t I>
      using child_t = part_child_t<maybe_const<Const, sndr_t<I>>, rcvr_t<I>>;

      template <std::size_t I>
      using sigs_t = ex::completion_signatures_of_t<cv_sndr_t<I>, env_t>;

      using values_t = std::tuple<std::optional<detail::when_all_tuple_t<sigs_t<Is>>>...>;

      // Otherwise the last child can't arrive with every child's values.
      static constexpr bool all_send_values =
        (!std::is_same_v<detail::when_all_tuple_t<sigs_t<Is>>, detail::no_values> && ...);

      // Whether keeping a child's values or error may throw.
      static constexpr bool keeping_may_throw = (any_copy_may_throw(sigs_t<Is>()) || ...);

      // Room for the first error: a child's, or what keeping one threw.
      using errors_sigs_t =
        decltype(exec::concat_completion_signatures(sigs_t<Is>()..., eptr_if<keeping_may_throw>()));
      using errors_t = detail::errors_variant_t<errors_sigs_t>;

      // Whether errors_t has room for E.
      template <class E>
      static constexpr bool keeps = detail::in_list<detail::error_list_t<errors_sigs_t>, E>;

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

      using join_t = std::conditional_t<stops, join_scope, join_count>;

      // The phase is its children's join scope, set up before them, since
      // their environments carry its token. While it runs, its receiver's
      // stop requests reach it. Children that can't fail only need counting.
      struct running : join_t
      {
        static constexpr parts<member<Is>...> connects{};
        using active_t =
          std::conditional_t<stops, join_scope::forwarding<ex::env_of_t<Rcvr>>, void>;

        template <class... Incoming>
        explicit running(opstate* op, Incoming&... incoming)
          : join_t(sizeof...(Sndrs))
          , children(op, incoming...)
        {}

        std::atomic<outcome> state{outcome::values};
        values_t             values;
        STDEXEC_ATTRIBUTE(no_unique_address) errors_t error;
        children_t children;
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
        if constexpr (stops)
        {
          return {phases_.template live<running>().token(), ex::get_env(phases_.receiver())};
        }
        else
        {
          return ex::get_env(phases_.receiver());
        }
      }

      template <std::size_t I, class... As>
      void complete(child_tag<I>, ex::set_value_t, As&&... as) noexcept
      {
        running& phase = phases_.template live<running>();
        if (phase.state.load(std::memory_order_relaxed) == outcome::values)
        {
          auto keep = [&]
          {
            std::get<I>(phase.values).emplace(static_cast<As&&>(as)...);
          };
          if constexpr (typestate::copy_may_throw<As...>)
          {
            if (std::exception_ptr e = attempt(keep))
            {
              fail(phase, std::move(e));
              return;
            }
          }
          else
          {
            keep();
          }
        }
        done(phase.arrive());
      }

      template <std::size_t I, class E>
      void complete(child_tag<I>, ex::set_error_t, E&& e) noexcept
      {
        if constexpr (keeps<std::decay_t<E>>)
        {
          fail(phases_.template live<running>(), static_cast<E&&>(e));
        }
        else
        {
          // The children's completion signatures rule this out. It's named
          // anyway by failure paths that can't be taken: every phase-based
          // operation's failure paths name set_error(std::exception_ptr).
          STDEXEC_UNREACHABLE();
        }
      }

      template <std::size_t I>
      void complete(child_tag<I>, ex::set_stopped_t) noexcept
      {
        if constexpr (stops)
        {
          running& phase    = phases_.template live<running>();
          outcome  expected = outcome::values;
          phase.state.compare_exchange_strong(expected, outcome::stopped);
          done(phase.stop_and_arrive());
        }
        else
        {
          // The children's completion signatures rule this out.
          STDEXEC_UNREACHABLE();
        }
      }

      // The first error wins, and outranks stopped.
      template <class E>
      void fail(running& phase, E&& e) noexcept
      {
        if (phase.state.exchange(outcome::error) != outcome::error)
        {
          auto keep = [&]
          {
            phase.error.template emplace<std::decay_t<E>>(static_cast<E&&>(e));
          };
          if constexpr (typestate::copy_may_throw<E>)
          {
            if (std::exception_ptr thrown = attempt(keep))
            {
              phase.error.template emplace<std::exception_ptr>(std::move(thrown));
            }
          }
          else
          {
            keep();
          }
        }
        done(phase.stop_and_arrive());
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
          if constexpr (!std::is_same_v<errors_t, detail::no_errors>)
          {
            std::visit(
              [this]<class E>(E& e)
              {
                if constexpr (!std::is_same_v<E, std::monostate>)
                {
                  phases_.template finish<running>(ex::set_error, std::move(e));
                }
              },
              phase.error);
          }
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
