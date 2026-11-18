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
  //       as the phase that runs it is entered, the association moves into
  //       the carried region and the wrapped sender straight into connect,
  //       and what's left of it is destroyed while the association is held,
  //       as [exec.associate]'s get-state does with release();
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

     public:
      // Owns what's left of a wrapped sender once it's been moved from.
      using guard_t = std::unique_ptr<Sndr, destroyer>;

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

      // Hands over the association, and the wrapped sender if there is one,
      // leaving this disengaged: the caller destroys the wrapped sender, by
      // destroying the guard, before it releases the association.
      auto release() && noexcept -> std::pair<Assoc, guard_t>
      {
        guard_t sndr(assoc_ ? std::addressof(sndr_) : nullptr);
        return {static_cast<Assoc&&>(assoc_), static_cast<guard_t&&>(sndr)};
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

      // Reading its sender only, the operation binds the wrapped sender in
      // place: under stored_sender, the wrapped sender's operation finds it
      // in the stored sender through its receiver, as any bound operation
      // does; under split_sender, it's connected from the operation's copy.
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

      struct running
      {
        static constexpr auto connects = &associate_sender::data_;

        template <class... Incoming>
          requires Const
        explicit running(opstate* op, Incoming&... incoming)
          : child(typestate::bind(op->data(incoming...).sender(), rcvr_t{{op}}))
        {}

        template <class... Incoming>
          requires(!Const)
        explicit running(opstate* op, Incoming&... incoming)
          : running(op, op->take(incoming...))
        {}

        // sndr destroys what's left of the wrapped sender once it's
        // connected, or if connect throws, before the operation can release
        // the association.
        running(opstate* op, typename data_t::guard_t sndr)
          : child(ex::connect(static_cast<Sndr&&>(*sndr), rcvr_t{op}))
        {}

        child_t child;
      };

      // What an operation whose scope refused runs: nothing, completing
      // with set_stopped(), as just completes with its values.
      struct refused
      {
        explicit refused(opstate*) noexcept {}
      };

      // Reading its sender only (bound without consuming, or connected as
      // an lvalue), the operation takes its own association now; otherwise
      // it has none until running is entered.
      template <class... Incoming>
      explicit opstate(Rcvr rcvr, Incoming&... incoming)
        : opstate(own_association(rcvr, incoming...), static_cast<Rcvr&&>(rcvr), incoming...)
      {}

      template <class... Incoming>
      opstate(Assoc assoc, Rcvr&& rcvr, Incoming&... incoming)
        : phases_(static_cast<Rcvr&&>(rcvr), static_cast<Assoc&&>(assoc), incoming...)
      {
        if (runs_from_connect(incoming...))
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
        if (runs_from_start())
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
      static auto own_association(Rcvr const & rcvr, Incoming&... incoming) -> Assoc
      {
        if constexpr (Const)
        {
          return peek<&associate_sender::data_>(rcvr, incoming...).reassociate();
        }
        else
        {
          return Assoc();
        }
      }

      // The sender's associate-data: the incoming sender's while connect
      // runs under split_sender, or wherever phased keeps it.
      template <class... Incoming>
      auto data(Incoming&... incoming) noexcept -> decltype(auto)
      {
        if constexpr (sizeof...(Incoming) == 1)
        {
          return ((incoming.data_), ...);
        }
        else
        {
          return phases_.template part<&associate_sender::data_>();
        }
      }

      static constexpr bool enters_on_connect =
        std::is_same_v<typename policies_of_t<ex::env_of_t<Rcvr>>::entry, connect_entry>;

      // Whether the operation runs the wrapped sender rather than refusing:
      // whether it has its own association, or, until it takes one, whether
      // the sender's associate-data has one.
      template <class... Incoming>
      auto runs_from_connect(Incoming&... incoming) noexcept -> bool
      {
        if constexpr (Const)
        {
          return static_cast<bool>(phases_.carried());
        }
        else
        {
          return static_cast<bool>(data(incoming...));
        }
      }

      auto runs_from_start() noexcept -> bool
      {
        if constexpr (Const || enters_on_connect)
        {
          return static_cast<bool>(phases_.carried());
        }
        else
        {
          return static_cast<bool>(data());
        }
      }

      // Moves the association out of the sender's associate-data into the
      // carried region, and returns what owns the wrapped sender.
      template <class... Incoming>
      auto take(Incoming&... incoming) noexcept -> typename data_t::guard_t
      {
        auto [assoc, sndr] = static_cast<data_t&&>(data(incoming...)).release();
        phases_.carried()  = static_cast<Assoc&&>(assoc);
        return static_cast<typename data_t::guard_t&&>(sndr);
      }

      phased<Rcvr, maybe_const<Const, parts<&associate_sender::data_>>, Assoc, running, refused>
        phases_;
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
