/*
 * Exploration: finally(initial, final) as a sequence of phases, written
 * against the machinery in typestate.hpp.
 */
#pragma once

#include "typestate.hpp"

#include <exception>
#include <tuple>
#include <type_traits>
#include <utility>

namespace frames::typestate
{
  // finally(initial, final): runs initial, then final, whatever initial
  // completed with; if final succeeds, completes as initial did, otherwise
  // as final did. final must complete with set_value() and no other values.
  //
  // final is like a destructor: it's connected with initial, just before
  // it, under every policy, so a failure to connect either means neither
  // runs, and its operation lasts until finally completes. It's the slot of
  // the first phase, which runs initial. Then, as with let_value, there's
  // one possible second phase per completion signature of initial,
  // receiving that completion's arguments and starting final; the
  // completion is kept where the Layout puts the values a phase receives,
  // and after eager teardown it's replayed from copies. final's operation
  // was connected before anyone knew which second phase would start it, so
  // its completions go to whichever is live (visit_live).
  //
  // final sees the receiver's environment, plus initial's completion
  // scheduler when initial has exactly one kind of completion, and its
  // domain, as exec::finally gives it.

  namespace detail
  {
    // What final's environment adds to the receiver's, given initial,
    // used as CvInitial: as exec::finally does.
    template <class CvInitial, class Env>
    using finally_env2_t = STDEXEC::
      __secondary_env_t<CvInitial, Env, ex::set_value_t, ex::set_error_t, ex::set_stopped_t>;

    template <class CvInitial, class Env>
    using final_env_t =
      STDEXEC::__join_env_t<finally_env2_t<CvInitial, Env> const &, STDEXEC::__fwd_env_t<Env>>;
  }  // namespace detail

  template <class Initial, class Final>
  struct finally_sender : phased_sender<finally_sender<Initial, Final>>
  {
    Initial initial_;
    Final   final_;

    template <class Self, class... Env>
    static consteval auto get_completion_signatures()
    {
      if constexpr (sizeof...(Env) == 0)
      {
        return STDEXEC::__throw_dependent_sender_error<Self>();
      }
      else
      {
        // final's completions other than set_value().
        auto final_failures = exec::transform_completion_signatures(
          ex::get_completion_signatures<member_t<Self, Final>,
                                        detail::final_env_t<member_t<Self, Initial>, Env>...>(),
          []<class... As>()
          {
            static_assert(sizeof...(As) == 0,
                          "finally's final sender must complete with set_value() and no other "
                          "values");
            return ex::completion_signatures<>();
          });
        // initial's completions are replayed from the decayed copies a
        // second phase receives, and final's failures are sent as copies
        // made before its operation, the slot, ends.
        auto initial_sigs = ex::get_completion_signatures<member_t<Self, Initial>, Env...>();
        return exec::concat_completion_signatures(
          sent_as<ending_of_t<Env...>>(
            exec::transform_completion_signatures(initial_sigs,
                                                  detail::decayed<ex::set_value_t>(),
                                                  detail::decayed<ex::set_error_t>())),
          sent_as_copies(final_failures),
          eptr_if<any_hand_over_may_throw(initial_sigs) || start_enters<Env...>>());
      }
    }

    template <class Rcvr, bool Const = false>
    struct opstate
    {
      struct running_tag
      {};

      struct final_tag
      {};

      using cv_initial_t = std::conditional_t<Const, Initial const &, Initial>;
      using env2_t       = detail::finally_env2_t<cv_initial_t, ex::env_of_t<Rcvr>>;
      using final_env_t  = detail::final_env_t<cv_initial_t, ex::env_of_t<Rcvr>>;

      using first_rcvr_t = phase_receiver<opstate, Rcvr, running_tag, &finally_sender::initial_>;
      using final_rcvr_t =
        phase_receiver<opstate, Rcvr, final_tag, &finally_sender::final_, final_env_t>;
      using child_t       = part_child_t<maybe_const<Const, Initial>, first_rcvr_t>;
      using final_child_t = part_child_t<maybe_const<Const, Final>, final_rcvr_t>;

      // What initial may complete with, decayed, once each, as
      // type_list<Tag, Ts...>.
      using results_t = detail::completion_lists_t<
        ex::completion_signatures_of_t<cv_initial_t, ex::env_of_t<Rcvr>>>;

      // final's operation.
      struct final_op
      {
        static constexpr auto connects = &finally_sender::final_;

        template <class... Incoming>
        explicit final_op(opstate* op, Incoming&... incoming)
          : child(op->phases_.template connect_part<connects>(final_rcvr_t{op}, incoming...))
        {}

        final_child_t child;
      };

      struct running
      {
        static constexpr auto connects = &finally_sender::initial_;
        using slot_t                   = final_op;

        template <class... Incoming>
        explicit running(opstate* op, Incoming&... incoming)
          : child(op->phases_.template connect_part<connects>(first_rcvr_t{op}, incoming...))
        {}

        child_t child;
      };

      // Starts final, having received the arguments of initial's completion
      // with Tag.
      template <class Tag, class... Ts>
      struct finalizing
      {
        using received_t = std::tuple<Ts...>;

        explicit finalizing(opstate* op) noexcept
          : child(op->phases_.slot().child)
        {}

        final_child_t& child;
      };

      // Works out final's environment from initial before either is
      // connected.
      template <class... Incoming>
      explicit opstate(Rcvr rcvr, Incoming&... incoming)
        : opstate(
            STDEXEC::__mk_secondary_env_t<ex::set_value_t, ex::set_error_t, ex::set_stopped_t>()(
              peek<&finally_sender::initial_>(rcvr, incoming...),
              ex::get_env(rcvr)),
            static_cast<Rcvr&&>(rcvr),
            incoming...)
      {}

      template <class... Incoming>
      opstate(env2_t env2, Rcvr&& rcvr, Incoming&... incoming)
        : phases_(static_cast<Rcvr&&>(rcvr), static_cast<env2_t&&>(env2), incoming...)
      {
        phases_.on_connect(this, incoming...);
      }

      void start() & noexcept
      {
        phases_.begin(start_child, this);
      }

      auto env(final_tag) noexcept -> final_env_t
      {
        return STDEXEC::__env::__join(std::as_const(phases_.carried()),
                                      STDEXEC::__fwd_env(ex::get_env(phases_.receiver())));
      }

      template <class Tag, class... As>
      void complete(running_tag, Tag, As&&... as) noexcept
      {
        if constexpr (detail::in_list<results_t, detail::type_list<Tag, std::decay_t<As>...>>)
        {
          phases_.template transition<running, finalizing<Tag, std::decay_t<As>...>>(
            [&](running&) { return hand_over(static_cast<As&&>(as)...); },
            this);
        }
        else
        {
          // initial's completion signatures rule this out. It's named
          // anyway by failure paths that can't be taken: every phase-based
          // operation's failure paths name set_error(std::exception_ptr).
          STDEXEC_UNREACHABLE();
        }
      }

      // final succeeded: complete as initial did.
      void complete(final_tag, ex::set_value_t) noexcept
      {
        phases_.visit_live(
          [this]<class Tag, class... Ts>(finalizing<Tag, Ts...>&)
          {
            using phase_t = finalizing<Tag, Ts...>;
            std::apply([this](Ts&... ts)
                       { phases_.template finish<phase_t>(Tag(), std::move(ts)...); },
                       phases_.template received<phase_t>());
          });
      }

      template <class FinalTag, class... As>
      void complete(final_tag, FinalTag, As&&... as) noexcept
      {
        phases_.visit_live(
          [&]<class Tag, class... Ts>(finalizing<Tag, Ts...>&)
          {
            phases_.template finish_from_slot<finalizing<Tag, Ts...>>(FinalTag(),
                                                                      static_cast<As&&>(as)...);
          });
      }

      typename phased_with<Rcvr,
                           parts<&finally_sender::initial_, &finally_sender::final_>,
                           env2_t,
                           running,
                           finalizing,
                           results_t>::type phases_;
    };
  };
  template <class Initial, class Final>
  auto finally(Initial&& initial, Final&& final)
  {
    return finally_sender<std::decay_t<Initial>, std::decay_t<Final>>{{},
                                                                      static_cast<Initial&&>(
                                                                        initial),
                                                                      static_cast<Final&&>(final)};
  }
}  // namespace frames::typestate
