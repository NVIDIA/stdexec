/*
 * Copyright (c) 2021-2024 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "__concepts.hpp"
#include "__debug.hpp"  // IWYU pragma: keep
#include "__diagnostics.hpp"
#include "__domain.hpp"
#include "__env.hpp"
#include "__into_variant.hpp"
#include "__meta.hpp"
#include "__receivers.hpp"
#include "__run_loop.hpp"
#include "__senders.hpp"
#include "__transform_sender.hpp"
#include "__type_traits.hpp"

#include <exception>
#include <optional>
#include <system_error>
#include <tuple>
#include <variant>

#include "__prologue.hpp"

STDEXEC_PRAGMA_IGNORE_MSVC(4714)  // marked as __forceinline not inlined

namespace STDEXEC::__sync_wait
{
  /////////////////////////////////////////////////////////////////////////////
  // [exec.sync.wait]
  // [exec.sync.wait.var]
  struct __env
  {
    template <__one_of<get_scheduler_t, get_start_scheduler_t, get_delegation_scheduler_t> _Query>
    [[nodiscard]]
    constexpr auto query(_Query) const noexcept -> run_loop::scheduler
    {
      return __loop_->get_scheduler();
    }

    [[nodiscard]]
    constexpr auto query(__root_t) const noexcept -> bool
    {
      return true;
    }

    run_loop* __loop_ = nullptr;
  };

  // What should sync_wait(just_stopped()) return?
  template <class _CvSender, class _Continuation>
  using __result_t = __value_types_of_t<_CvSender,
                                        __env,
                                        __mtransform<__q<__decay_t>, _Continuation>,
                                        __q<__msingle>>;

  template <class _CvSender>
  using __sync_wait_result_t = __result_t<_CvSender, __qq<std::tuple>>;

  template <class _CvSender>
  using __sync_wait_with_variant_result_t =
    __result_t<__result_of<into_variant, _CvSender>, __q<__midentity>>;

  struct __state
  {
    std::exception_ptr __eptr_;
    run_loop           __loop_;
  };

  template <class... _Values>
  struct __receiver
  {
    using receiver_concept = receiver_tag;

    template <class... _As>
    constexpr void set_value(_As&&... __as) noexcept
    {
      static_assert(__std::constructible_from<std::tuple<_Values...>, _As...>);
      STDEXEC_TRY
      {
        __values_->emplace(static_cast<_As&&>(__as)...);
      }
      STDEXEC_CATCH_ALL
      {
        __state_->__eptr_ = std::current_exception();
      }
      __state_->__loop_.finish();
    }

    template <class _Error>
    constexpr void set_error(_Error __err) noexcept
    {
      if constexpr (__same_as<_Error, std::exception_ptr>)
      {
        STDEXEC_ASSERT(__err != nullptr);  // std::exception_ptr must not be null.
        __state_->__eptr_ = static_cast<_Error&&>(__err);
      }
      else if constexpr (__same_as<_Error, std::error_code>)
      {
        __state_->__eptr_ = std::make_exception_ptr(std::system_error(__err));
      }
      else
      {
        __state_->__eptr_ = std::make_exception_ptr(static_cast<_Error&&>(__err));
      }
      __state_->__loop_.finish();
    }

    constexpr void set_stopped() noexcept
    {
      __state_->__loop_.finish();
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> __env
    {
      return __env{&__state_->__loop_};
    }
    __state*                               __state_;
    std::optional<std::tuple<_Values...>>* __values_;
  };

  template <class _CvSender>
  using __receiver_t = __result_t<_CvSender, __q<__receiver>>;

  // These are for hiding the metaprogramming in diagnostics
  template <class _CvSender>
  struct __sync_receiver_for
  {
    using __t = __receiver_t<_CvSender>;
  };
  template <class _CvSender>
  using __sync_receiver_for_t = __t<__sync_receiver_for<_CvSender>>;

  template <class _CvSender>
  struct __value_tuple_for
  {
    using __t = __sync_wait_result_t<_CvSender>;
  };
  template <class _CvSender>
  using __value_tuple_for_t = __t<__value_tuple_for<_CvSender>>;

  template <class _CvSender>
  struct __variant
  {
    using __t = __sync_wait_with_variant_result_t<_CvSender>;
  };
  template <class _CvSender>
  using __variant_for_t = __t<__variant<_CvSender>>;

  struct _SENDER_HAS_TOO_MANY_SUCCESSFUL_COMPLETIONS_
  {};
  struct _USE_SYNC_WAIT_WITH_VARIANT_INSTEAD_
  {};

  template <class _Reason, class _CvSender, class _Env = __env>
  using __sync_wait_error_t =
    __mexception<_WHAT_(_INVALID_ARGUMENT_),
                 _WHERE_(_IN_ALGORITHM_, sync_wait_t),
                 _WHY_(_Reason),
                 _WITH_PRETTY_SENDER_<_CvSender>,
                 _WITH_ENVIRONMENT_(_Env),
                 _TO_FIX_THIS_ERROR_(_USE_SYNC_WAIT_WITH_VARIANT_INSTEAD_)>;

  template <class _CvSender, class>
  using __too_many_successful_completions_error_t =
    __sync_wait_error_t<_SENDER_HAS_TOO_MANY_SUCCESSFUL_COMPLETIONS_, _CvSender>;

  template <class _CvSender>
  concept __valid_sync_wait_argument = __ok<__minvoke<
    __mtry_catch_q<__single_value_variant_sender_t, __q<__too_many_successful_completions_error_t>>,
    _CvSender,
    __env>>;
}  // namespace STDEXEC::__sync_wait

STDEXEC_P2300_NAMESPACE_BEGIN(this_thread)
  ////////////////////////////////////////////////////////////////////////////
  // [exec.sync.wait]

  //! @brief A sender consumer that synchronously blocks the calling thread
  //!        until a sender completes and returns its result.
  //!
  //! @c sync_wait is the bridge from the asynchronous sender world back into
  //! synchronous code. You give it a sender; it connects the sender to a
  //! built-in receiver, starts the resulting operation, then drives an
  //! internal @c run_loop on the calling thread until the operation
  //! completes. The result is returned as a <tt>std::optional</tt> of a
  //! tuple of the value-completion datums.
  //!
  //! This is the most common way to "run" a sender in a top-level program or
  //! a test — it's what you reach for in a @c main() or when synchronously
  //! waiting on a single sub-pipeline. For fire-and-forget execution, prefer
  //! @c exec::start_detached or @c stdexec::spawn.
  //!
  //! @code{.cpp}
  //! auto [v] = stdexec::sync_wait(stdexec::just(42)).value();
  //! // v == 42
  //! @endcode
  //!
  //! See [exec.sync.wait] in the C++26 working draft for the normative
  //! specification.
  //!
  //! **Completion behavior.**
  //!
  //! Given an input sender @c sndr that, in some environment, completes with
  //! exactly one of:
  //!
  //! | Sender completion             | What @c sync_wait does                                          |
  //! | ----------------------------- | --------------------------------------------------------------- |
  //! | @c set_value_t(Vs...)         | Returns @c std::optional<std::tuple<Vs...>> engaged.            |
  //! | @c set_error_t(std::exception_ptr) | Rethrows the exception via @c std::rethrow_exception.     |
  //! | @c set_error_t(std::error_code)    | Throws @c std::system_error(error_code).                  |
  //! | @c set_error_t(E)             | Throws @c E directly.                                           |
  //! | @c set_stopped_t()            | Returns an empty (disengaged) @c std::optional.                 |
  //!
  //! **Single-value-completion requirement.**
  //!
  //! @c sync_wait *mandates* that its argument sender have exactly one
  //! @c set_value_t completion signature. A sender that can succeed in more
  //! than one way (e.g. <tt>just(1) | when_all(just(std::string{"x"}))</tt>
  //! yielding two distinct tuples) requires @c sync_wait_with_variant
  //! instead. The static assertion in @c sync_wait will point this out at
  //! compile time, with a hint to use the variant form.
  //!
  //! **Delegation scheduler.**
  //!
  //! The internal @c run_loop is exposed via @c get_delegation_scheduler on
  //! the receiver's environment, so senders that need to enqueue work back
  //! onto the waiting thread (e.g. continuations after an I/O wait) can do
  //! so safely. This is what enables algorithms like @c continues_on to
  //! return execution to the calling thread of @c sync_wait.
  //!
  //! **When *not* to use** @c sync_wait **:**
  //! - On any thread that participates in an event loop or executor — you
  //!   will block it. @c sync_wait is for top-level synchronization
  //!   (main, tests, leaf utilities), not pipeline composition.
  //! - When you don't need the result. Use @c exec::start_detached or
  //!   @c stdexec::spawn for fire-and-forget.
  //!
  //! @see stdexec::sync_wait_with_variant  — sync_wait for multi-completion senders
  //! @see exec::start_detached             — fire-and-forget consumer (no result)
  //! @see stdexec::spawn                   — fire-and-forget into a scope
  //! @see stdexec::spawn_future            — spawn into a scope and observe via a sender
  struct sync_wait_t
  {
    //! @brief Connect @c __sndr to an internal receiver, start the operation,
    //!        and drive a @c run_loop until completion.
    //!
    //! @tparam _CvSender A type satisfying @c stdexec::sender_in for the
    //!                   built-in @c sync_wait environment.
    //! @param __sndr     The sender to drive to completion. Must have
    //!                   exactly one @c set_value_t completion signature.
    //!
    //! @returns @c std::optional<std::tuple<Vs...>> where @c Vs... are the
    //!          value-completion datum types of @c __sndr. The optional is
    //!          engaged on @c set_value, disengaged on @c set_stopped.
    //!
    //! @throws The error datum, if @c __sndr completes with @c set_error
    //!         (rethrown via @c std::rethrow_exception for
    //!         @c std::exception_ptr, via @c std::system_error for
    //!         @c std::error_code, or directly otherwise).
    //!
    //! @pre @c __sndr must have exactly one @c set_value_t completion
    //!      signature, otherwise the program is ill-formed with a
    //!      diagnostic pointing at @c sync_wait_with_variant.
    template <STDEXEC::sender_in<STDEXEC::__sync_wait::__env> _CvSender>
    auto operator()(_CvSender&& __sndr) const
    {
      using __domain_t = STDEXEC::__completion_domain_of_t<STDEXEC::set_value_t,
                                                           _CvSender,
                                                           STDEXEC::__sync_wait::__env>;
      constexpr auto __success_completion_count =
        STDEXEC::__count_of<STDEXEC::set_value_t, _CvSender, STDEXEC::__sync_wait::__env>::value;

      static_assert(__success_completion_count != 0,
                    "The argument to " STDEXEC_PP_STRINGIZE(STDEXEC)  //
                    "::sync_wait() is a sender that cannot complete successfully. "
                    "STDEXEC::sync_wait() requires a sender that can complete successfully in "
                    "exactly one way. In other words, the sender's completion signatures must "
                    "include exactly one signature of the form `set_value_t(value-types...)`.");

      static_assert(__success_completion_count <= 1,
                    "The sender passed to " STDEXEC_PP_STRINGIZE(STDEXEC)  //
                    "::sync_wait() can complete successfully in more than one way. "
                    "Use " STDEXEC_PP_STRINGIZE(STDEXEC) "::sync_wait_"
                                                         "with_variant() instead.");

      if constexpr (1 == __success_completion_count)
      {
        if constexpr (STDEXEC::__same_as<__domain_t, STDEXEC::default_domain>)
        {
          if constexpr (STDEXEC::sender_to<_CvSender,
                                           STDEXEC::__sync_wait::__receiver_t<_CvSender>>)
          {
            using __opstate_t =
              STDEXEC::connect_result_t<_CvSender, STDEXEC::__sync_wait::__receiver_t<_CvSender>>;
            if constexpr (STDEXEC::operation_state<__opstate_t>)
            {
              // success path, dispatch to the default domain's sync_wait
              return STDEXEC::default_domain().apply_sender(*this,
                                                            static_cast<_CvSender&&>(__sndr));
            }
            else
            {
              static_assert(STDEXEC::operation_state<__opstate_t>,
                            "The `connect` member function of the sender passed to "
                            "STDEXEC::sync_wait() does not return an operation state. An operation "
                            "state is required to have a no-throw .start() member function.");
            }
          }
          else
          {
            // This shoud generate a useful error message about why the sender cannot
            // be connected to the receiver:
            STDEXEC::connect(static_cast<_CvSender&&>(__sndr),
                             STDEXEC::__sync_wait::__receiver_t<_CvSender>{});
            static_assert(
              STDEXEC::sender_to<_CvSender, STDEXEC::__sync_wait::__receiver_t<_CvSender>>,
              STDEXEC_ERROR_SYNC_WAIT_CANNOT_CONNECT_SENDER_TO_RECEIVER);
          }
        }
        else if constexpr (!STDEXEC::__has_implementation_for<sync_wait_t, __domain_t, _CvSender>)
        {
          static_assert(STDEXEC::__has_implementation_for<sync_wait_t, __domain_t, _CvSender>,
                        "The sender passed to " STDEXEC_PP_STRINGIZE(STDEXEC)  //
                        "::sync_wait() has a domain that does not provide a usable implementation "
                        "for sync_wait().");
        }
        else
        {
          // success path, dispatch to the custom domain's sync_wait
          return STDEXEC::apply_sender(__domain_t(), *this, static_cast<_CvSender&&>(__sndr));
        }
      }
    }

    template <class _CvSender>
    constexpr auto operator()(_CvSender&&) const
    {
      STDEXEC::__diagnose_sender_concept_failure<_CvSender, STDEXEC::__sync_wait::__env>();
      // dummy return type to silence follow-on errors
      return std::optional<std::tuple<int>>{};
    }

    //! @internal
    //! @brief Default-domain implementation of @c sync_wait. Connects
    //! @c __sndr, starts the operation, drives an internal @c run_loop, and
    //! returns/throws per @ref sync_wait_t. Not normally called by users.
    template <STDEXEC::sender_in<STDEXEC::__sync_wait::__env> _CvSender>
    STDEXEC_CONSTEXPR_CXX23 auto apply_sender(_CvSender&& __sndr) const  //
      -> std::optional<STDEXEC::__sync_wait::__value_tuple_for_t<_CvSender>>
    {
      STDEXEC::__sync_wait::__state                                       __local_state{};
      std::optional<STDEXEC::__sync_wait::__value_tuple_for_t<_CvSender>> __result{};

      // Launch the sender with a continuation that will fill in the __result optional or set the
      // exception_ptr in __local_state.
      [[maybe_unused]]
      auto __op = STDEXEC::connect(static_cast<_CvSender&&>(__sndr),
                                   STDEXEC::__sync_wait::__receiver_t<_CvSender>{&__local_state,
                                                                                 &__result});
      STDEXEC::start(__op);

      // Wait for the variant to be filled in.
      __local_state.__loop_.run();

      if (__local_state.__eptr_)
      {
        std::rethrow_exception(static_cast<std::exception_ptr&&>(__local_state.__eptr_));
      }

      return __result;
    }
  };

  ////////////////////////////////////////////////////////////////////////////
  // [exec.sync.wait.var]

  //! @brief A sender consumer that synchronously blocks the calling thread
  //!        until a multi-value-completion sender completes, returning the
  //!        result as a variant of tuples.
  //!
  //! @c sync_wait_with_variant is the multi-completion sibling of
  //! @ref sync_wait_t. A sender that can succeed in more than one way — for
  //! example, an algorithm that may complete with either an @c int or a
  //! @c std::string — cannot be passed to @c sync_wait, because the latter
  //! returns a single fixed tuple type. @c sync_wait_with_variant accepts
  //! such senders and returns the result as a @c std::variant of all the
  //! possible value-tuple shapes.
  //!
  //! @code{.cpp}
  //! // sndr completes with either set_value_t(int) or set_value_t(std::string).
  //! auto opt = stdexec::sync_wait_with_variant(std::move(sndr));
  //! if (opt) {
  //!   std::visit([](auto&& tup) {
  //!     // tup is either std::tuple<int> or std::tuple<std::string>.
  //!   }, *opt);
  //! }
  //! @endcode
  //!
  //! See [exec.sync.wait.var] in the C++26 working draft for the normative
  //! specification.
  //!
  //! **Completion behavior.**
  //!
  //! Given an input sender @c sndr with value-completion signatures
  //! <tt>set_value_t(Vs1...), set_value_t(Vs2...), ...</tt>, the return type is
  //!
  //! @code{.cpp}
  //! std::optional<std::variant<std::tuple<Vs1...>, std::tuple<Vs2...>, ...>>
  //! @endcode
  //!
  //! The handling of @c set_error_t and @c set_stopped_t matches
  //! @ref sync_wait_t : errors are thrown, @c set_stopped yields a disengaged
  //! optional.
  //!
  //! **When to use** @c sync_wait_with_variant **vs.** @c sync_wait **:**
  //! Use @c sync_wait when the sender has *exactly one* value-completion
  //! shape; use @c sync_wait_with_variant otherwise. @c sync_wait's static
  //! assertion will steer you here if needed.
  //!
  //! @see stdexec::sync_wait          — for single-value-completion senders
  //! @see stdexec::into_variant       — adaptor that collapses multi-completion senders into a variant
  struct sync_wait_with_variant_t
  {
    //! @brief Connect @c __sndr, start the operation, drive a @c run_loop
    //!        until completion, and return the result as a variant of tuples.
    //!
    //! @tparam _CvSender A type satisfying @c stdexec::sender_in for the
    //!                   @c sync_wait environment.
    //! @param __sndr     The sender to drive to completion. May have any
    //!                   number of @c set_value_t completion signatures.
    //!
    //! @returns @c std::optional<std::variant<std::tuple<Vs1...>, ...>>
    //!          engaged on @c set_value, disengaged on @c set_stopped.
    //!
    //! @throws The error datum, if @c __sndr completes with @c set_error,
    //!         using the same rules as @ref sync_wait_t.
    template <STDEXEC::sender_in<STDEXEC::__sync_wait::__env> _CvSender>
      requires STDEXEC::__callable<STDEXEC::apply_sender_t,
                                   STDEXEC::__completion_domain_of_t<STDEXEC::set_value_t,
                                                                     _CvSender,
                                                                     STDEXEC::__sync_wait::__env>,
                                   sync_wait_with_variant_t,
                                   _CvSender>
    auto operator()(_CvSender&& __sndr) const -> decltype(auto)
    {
      using __result_t =
        STDEXEC::__call_result_t<STDEXEC::apply_sender_t,
                                 STDEXEC::__completion_domain_of_t<STDEXEC::set_value_t,
                                                                   _CvSender,
                                                                   STDEXEC::__sync_wait::__env>,
                                 sync_wait_with_variant_t,
                                 _CvSender>;
      static_assert(STDEXEC::__is_instance_of<__result_t, std::optional>);
      using __variant_t = __result_t::value_type;
      static_assert(STDEXEC::__is_instance_of<__variant_t, std::variant>);

      using _Domain = STDEXEC::__completion_domain_of_t<STDEXEC::set_value_t,
                                                        _CvSender,
                                                        STDEXEC::__sync_wait::__env>;
      return STDEXEC::apply_sender(_Domain(), *this, static_cast<_CvSender&&>(__sndr));
    }

    template <class _CvSender>
      requires STDEXEC::__callable<sync_wait_t,
                                   STDEXEC::__result_of<STDEXEC::into_variant, _CvSender>>
    auto apply_sender(_CvSender&& __sndr) const
      -> std::optional<STDEXEC::__sync_wait::__variant_for_t<_CvSender>>
    {
      if (auto __opt_values = sync_wait_t()(
            STDEXEC::into_variant(static_cast<_CvSender&&>(__sndr))))
      {
        return std::move(std::get<0>(*__opt_values));
      }
      return std::nullopt;
    }
  };

  //! @brief The customization point object for the @c sync_wait sender consumer.
  //!
  //! @c sync_wait is an instance of @ref sync_wait_t. See @ref sync_wait_t
  //! for the full description, completion-behavior table, and a usage example.
  //!
  //! @hideinitializer
  inline constexpr sync_wait_t sync_wait{};

  //! @brief The customization point object for the @c sync_wait_with_variant
  //!        sender consumer.
  //!
  //! @c sync_wait_with_variant is an instance of @ref sync_wait_with_variant_t.
  //! See @ref sync_wait_with_variant_t for the full description and a usage
  //! example.
  //!
  //! @hideinitializer
  inline constexpr sync_wait_with_variant_t sync_wait_with_variant{};

STDEXEC_P2300_NAMESPACE_END(this_thread)

#include "__epilogue.hpp"
