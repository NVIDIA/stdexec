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
  // then(sndr, fn): one phase, running the child. fn is used in place.

  template <class Sndr, class Fn>
  struct then_sender : phased_sender<then_sender<Sndr, Fn>>
  {
    Sndr sndr_;
    Fn   fn_;

    template <class Self, class... Env>
    static consteval auto get_completion_signatures()
    {
      if constexpr (sizeof...(Env) == 0)
      {
        return STDEXEC::__throw_dependent_sender_error<Self>();
      }
      else
      {
        // Bound without consuming, fn is called as const.
        using fn_ref =
          std::conditional_t<std::is_const_v<std::remove_reference_t<Self>>, Fn const &, Fn&>;
        using ending_t = ending_of_t<Env...>;
        // fn may throw, as may copying its arguments under eager_teardown,
        // or its result if it returns a reference.
        auto values = []<class... As>()
        {
          using result_t             = std::invoke_result_t<fn_ref, As...>;
          constexpr bool copies_args = std::is_same_v<ending_t, eager_teardown>
                                    && !(std::is_nothrow_constructible_v<std::decay_t<As>, As>
                                         && ...);
          constexpr bool copies_result =
            !std::is_void_v<result_t>
            && !std::is_nothrow_constructible_v<std::decay_t<result_t>, result_t>;
          return exec::concat_completion_signatures(
            detail::value_of<result_t>(),
            eptr_if<!std::is_nothrow_invocable_v<fn_ref, As...> || copies_args || copies_result>());
        };
        return exec::transform_completion_signatures(
          ex::get_completion_signatures<member_t<Self, Sndr>, Env...>(),
          values,
          sent_as_fn<ending_t, ex::set_error_t>(),
          exec::keep_completion<ex::set_stopped_t>(),
          eptr_if<start_enters<Env...>>());
      }
    }

    template <class Rcvr, bool Const = false>
    struct opstate
    {
      struct running_tag
      {};

      using parts_t = parts<&then_sender::sndr_, &then_sender::fn_>;
      using rcvr_t  = phase_receiver<opstate, Rcvr, running_tag, &then_sender::sndr_>;
      using child_t = part_child_t<maybe_const<Const, Sndr>, rcvr_t>;

      struct running
      {
        static constexpr auto connects = &then_sender::sndr_;

        template <class... Incoming>
        explicit running(opstate* op, Incoming&... incoming)
          : child(op->phases_.template connect_part<connects>(rcvr_t{op}, incoming...))
        {}

        child_t child;
      };

      template <class... Incoming>
      explicit opstate(Rcvr rcvr, Incoming&... incoming)
        : phases_(static_cast<Rcvr&&>(rcvr), nothing(), incoming...)
      {
        phases_.on_connect(this, incoming...);
      }

      void start() & noexcept
      {
        phases_.begin(start_child, this);
      }

      auto fn() noexcept -> maybe_const<Const, Fn>&
      {
        return phases_.template part<&then_sender::fn_>();
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

      phased<Rcvr, parts_t, nothing, running> phases_;
    };
  };

  template <class Sndr, class Fn>
  auto then(Sndr&& sndr, Fn&& fn)
  {
    return then_sender<std::decay_t<Sndr>, std::decay_t<Fn>>{{},
                                                             static_cast<Sndr&&>(sndr),
                                                             static_cast<Fn&&>(fn)};
  }
}  // namespace frames::typestate
