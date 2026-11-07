/*
 * Exploration: associate(sndr, token) as a sequence of phases, written
 * against the machinery in typestate.hpp.
 */
#pragma once

#include "typestate.hpp"

#include <concepts>
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>

namespace frames::typestate
{
  // associate(sndr, token): asks token's scope for an association when it's
  // called, as [exec.associate] does. If the scope gives one, the operation
  // runs token.wrap(sndr) in a phase; otherwise it enters a phase with no
  // child instead, which completes with set_stopped(), and never connects
  // the wrapped sender.
  //
  // The wrapped sender and the association travel together, as
  // [exec.associate]'s associate-data: from the sender into the operation's
  // carried region, so under either ending they live until the operation
  // state is destroyed, after the operation has completed. That's what
  // [exec.associate] specifies: its op_state holds the association, and
  // nothing releases it on completion. So a scope's join can't finish until
  // whoever owns the operation destroys it. The wrapped sender is only ever
  // destroyed while the association is held, or as soon as the scope
  // refuses one, since what it refers to may only stay valid while it's
  // associated.
  //
  // The operation owns its wrapped sender, so it connects it, under either
  // Data policy, as let_value connects the sender its function returns.

  namespace detail
  {
    // The wrapped sender and its association, or neither if the scope
    // refused: [exec.associate]'s associate-data. A copy associates again,
    // and copies the wrapped sender only if it got an association; a move
    // takes both, destroying what's left of the source's wrapped sender
    // while the association is still held.
    template <class Sndr, class Assoc>
    class associate_data
    {
      // Destroys the wrapped sender it points to, unless released.
      struct destroyer
      {
        void operator()(Sndr* sndr) const noexcept
        {
          std::destroy_at(sndr);
        }
      };

      using guard_t = std::unique_ptr<Sndr, destroyer>;

     public:
      template <class Token, class S>
      associate_data(Token const & token, S&& sndr)
        : sndr_(token.wrap(static_cast<S&&>(sndr)))
        , assoc_(
            [&]
            {
              guard_t guard(std::addressof(sndr_));
              Assoc   assoc = token.try_associate();
              if (assoc)
              {
                (void) guard.release();
              }
              return assoc;
            }())
      {}

      associate_data(associate_data const & other)
        requires std::copy_constructible<Sndr>
        : assoc_(other.assoc_.try_associate())
      {
        if (assoc_)
        {
          std::construct_at(std::addressof(sndr_), other.sndr_);
        }
      }

      // A moved-from association is disengaged, so other's destructor
      // leaves its wrapped sender alone.
      associate_data(associate_data&& other) noexcept(std::is_nothrow_move_constructible_v<Sndr>)
        : assoc_(static_cast<Assoc&&>(other.assoc_))
      {
        if (assoc_)
        {
          guard_t source(std::addressof(other.sndr_));
          std::construct_at(std::addressof(sndr_), static_cast<Sndr&&>(*source));
        }
      }

      auto operator=(associate_data const &) -> associate_data& = delete;

      ~associate_data()
      {
        if (assoc_)
        {
          std::destroy_at(std::addressof(sndr_));
        }
      }

      explicit operator bool() const noexcept
      {
        return static_cast<bool>(assoc_);
      }

      // The wrapped sender, if there's an association.
      auto sender() noexcept -> Sndr&
      {
        return sndr_;
      }

     private:
      union
      {
        Sndr sndr_;
      };

      Assoc assoc_;
    };
  }  // namespace detail

  template <class Sndr, class Assoc>
  struct associate_sender : phased_sender<associate_sender<Sndr, Assoc>>
  {
    detail::associate_data<Sndr, Assoc> data_;

    template <class Self, class... Env>
    static consteval auto get_completion_signatures()
    {
      if constexpr (sizeof...(Env) == 0)
      {
        return STDEXEC::__throw_dependent_sender_error<Self>();
      }
      else
      {
        return exec::concat_completion_signatures(sent_as<ending_of_t<Env...>>(
                                                    ex::get_completion_signatures<Sndr, Env...>()),
                                                  ex::completion_signatures<ex::set_stopped_t()>(),
                                                  detail::eptr_completion());
      }
    }

    template <class Rcvr, bool Const = false>
    struct opstate
    {
      struct running_tag
      {};

      using rcvr_t    = phase_receiver<opstate, Rcvr, running_tag>;
      using child_t   = ex::connect_result_t<Sndr, rcvr_t>;
      using carried_t = detail::associate_data<Sndr, Assoc>;

      struct running
      {
        explicit running(opstate* op)
          : child(ex::connect(static_cast<Sndr&&>(op->phases_.carried().sender()), rcvr_t{op}))
        {}

        child_t child;
      };

      // What an operation whose scope refused runs: nothing, completing
      // with set_stopped(), as just completes with its values.
      struct refused
      {
        explicit refused(opstate*) noexcept {}
      };

      // Takes the sender's associate-data: moves it out, or under a bind
      // that doesn't consume, copies it, associating again.
      template <class... Incoming>
      explicit opstate(Rcvr rcvr, Incoming&... incoming)
        : opstate(std::move(peek<&associate_sender::data_>(rcvr, incoming...)),
                  static_cast<Rcvr&&>(rcvr),
                  incoming...)
      {}

      template <class... Incoming>
      opstate(carried_t data, Rcvr&& rcvr, Incoming&... incoming)
        : phases_(static_cast<Rcvr&&>(rcvr), static_cast<carried_t&&>(data), incoming...)
      {
        if (associated())
        {
          phases_.template on_connect<running>(this, incoming...);
        }
        else
        {
          phases_.template on_connect<refused>(this);
        }
      }

      [[nodiscard]]
      auto associated() noexcept -> bool
      {
        return static_cast<bool>(phases_.carried());
      }

      void start() & noexcept
      {
        if (associated())
        {
          phases_.template begin<running>(start_child, this);
        }
        else
        {
          phases_.template begin<refused>([this](refused&) noexcept
                                          { phases_.template finish<refused>(ex::set_stopped); },
                                          this);
        }
      }

      template <class Tag, class... As>
      void complete(running_tag, Tag, As&&... as) noexcept
      {
        phases_.template finish<running>(Tag(), static_cast<As&&>(as)...);
      }

      phased<Rcvr, parts<>, carried_t, running, refused> phases_;
    };
  };

  template <ex::sender Sndr, ex::scope_token Token>
  auto associate(Sndr&& sndr, Token const token)
  {
    using wrapped_t = std::remove_cvref_t<decltype(token.wrap(static_cast<Sndr&&>(sndr)))>;
    using assoc_t   = decltype(token.try_associate());
    return associate_sender<wrapped_t, assoc_t>{
      {},
      detail::associate_data<wrapped_t, assoc_t>(token, static_cast<Sndr&&>(sndr))};
  }
}  // namespace frames::typestate
