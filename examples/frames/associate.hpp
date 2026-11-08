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
  // [exec.associate]'s associate-data, because the wrapped sender may refer
  // to what the scope keeps alive: it's only ever destroyed while the
  // association is held, or as soon as the scope refuses one. The operation
  // takes the association into its carried region, so under either ending
  // it lasts until the operation state is destroyed, after the operation
  // has completed. That's what [exec.associate] specifies: its op_state
  // holds the association, and nothing releases it on completion. So a
  // scope's join can't finish until whoever owns the operation destroys it.
  //
  // Where the wrapped sender goes depends on how the operation got it:
  //
  //   consuming it (connected, or bound through a non-const reference)
  //       both move into the carried region, and the operation connects
  //       the wrapped sender it now owns, as let_value connects the sender
  //       its function returns;
  //   bound without consuming
  //       the stored sender keeps its own association, which covers its
  //       wrapped sender until the stored sender is destroyed, after this
  //       operation. So the operation asks the scope for an association of
  //       its own, as copying the sender would, and binds the stored
  //       wrapped sender in place rather than copying it.

  namespace detail
  {
    // The wrapped sender and its association, or neither if the scope
    // refused: [exec.associate]'s associate-data. A copy associates again,
    // and copies the wrapped sender only if it got an association; a move
    // takes both, destroying what's left of the source's wrapped sender
    // while the association is still held, and so does release().
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

      auto sender() const noexcept -> Sndr const &
      {
        return sndr_;
      }

      // Another association with the same scope, as a copy would take.
      [[nodiscard]]
      auto reassociate() const -> Assoc
      {
        return assoc_.try_associate();
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
        return exec::concat_completion_signatures(
          sent_as<ending_of_t<Env...>>(
            ex::get_completion_signatures<member_t<Self, Sndr>, Env...>()),
          ex::completion_signatures<ex::set_stopped_t()>(),
          eptr_if<start_enters<Env...>>());
      }
    }

    template <class Rcvr, bool Const = false>
    struct opstate
    {
      struct running_tag
      {};

      using data_t = detail::associate_data<Sndr, Assoc>;

      // Bound without consuming, the wrapped sender's operation finds it in
      // the stored sender through its receiver, as any bound operation does.
      struct bound_rcvr : phase_receiver<opstate, Rcvr, running_tag>
      {
        [[nodiscard]]
        auto bound_sender() const noexcept -> Sndr const &
        {
          return this->op->data().sender();
        }
      };

      using rcvr_t =
        std::conditional_t<Const, bound_rcvr, phase_receiver<opstate, Rcvr, running_tag>>;
      using child_t = std::conditional_t<Const,
                                         bind_result_t<Sndr const, rcvr_t>,
                                         ex::connect_result_t<Sndr, rcvr_t>>;

      // Bound without consuming, the operation carries only its own
      // association; otherwise it carries the sender's associate-data.
      using carried_t = std::conditional_t<Const, Assoc, data_t>;

      struct running
      {
        template <class... Incoming>
          requires Const
        explicit running(opstate* op, Incoming&...)
          : child(typestate::bind(op->data().sender(), rcvr_t{{op}}))
        {}

        explicit running(opstate* op)
          requires(!Const)
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

      // Takes the sender's associate-data, or, bound without consuming,
      // an association of its own.
      template <class... Incoming>
      explicit opstate(Rcvr rcvr, Incoming&... incoming)
        : opstate(take(rcvr, incoming...), static_cast<Rcvr&&>(rcvr), incoming...)
      {}

      template <class... Incoming>
      opstate(carried_t carried, Rcvr&& rcvr, Incoming&... incoming)
        : phases_(static_cast<Rcvr&&>(rcvr), static_cast<carried_t&&>(carried), incoming...)
      {
        if (runs())
        {
          phases_.template on_connect<running>(this, incoming...);
        }
        else
        {
          phases_.template on_connect<refused>(this);
        }
      }

      void start() & noexcept
      {
        if (runs())
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

      template <class... Incoming>
      static auto take(Rcvr const & rcvr, Incoming&... incoming) -> carried_t
      {
        if constexpr (Const)
        {
          return peek<&associate_sender::data_>(rcvr).reassociate();
        }
        else
        {
          return std::move(peek<&associate_sender::data_>(rcvr, incoming...));
        }
      }

      // The stored sender's associate-data, bound without consuming.
      auto data() noexcept -> data_t const &
      {
        return phases_.template part<&associate_sender::data_>();
      }

      // Whether the operation runs the wrapped sender rather than refusing.
      [[nodiscard]]
      auto runs() noexcept -> bool
      {
        return static_cast<bool>(phases_.carried());
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
