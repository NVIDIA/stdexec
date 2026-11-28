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

#include "__config.hpp"

#if STDEXEC_USE_MODULES() && !defined(STDEXEC_IN_MODULE_PURVIEW)

import stdexec;

#else

#  include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#  include "__concepts.hpp"
#  include "__continues_on.hpp"
#  include "__diagnostics.hpp"
#  include "__domain.hpp"
#  include "__env.hpp"
#  include "__into_variant.hpp"
#  include "__just.hpp"
#  include "__meta.hpp"
#  include "__optional.hpp"
#  include "__schedulers.hpp"
#  include "__sender_facade.hpp"
#  include "__sender_introspection.hpp"
#  include "__senders.hpp"
#  include "__transform_completion_signatures.hpp"
#  include "__tuple.hpp"
#  include "__type_traits.hpp"
#  include "__utility.hpp"
#  include "__variant.hpp"

#  include "../stop_token.hpp"

#  include "__atomic.hpp"
#  if !STDEXEC_USE_MODULES()
#    include <exception>
#  endif

#  include "__prologue.hpp"

STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  //////////////////////////////////////////////////////////////////////////////
  // [exec.when.all]
  //! @brief A variadic sender factory that runs multiple senders concurrently
  //!        and completes when all of them have completed, concatenating
  //!        their value datums.
  //!
  //! @c when_all is the canonical *parallel composition* primitive in the
  //! sender model. You give it zero or more senders; it returns a single
  //! sender that, when connected and started, starts *all* of the input
  //! senders concurrently. When every input has completed, @c when_all's
  //! sender completes with a value tuple that is the concatenation of every
  //! input's value datums.
  //!
  //! If any one input can fail or be stopped, @c when_all uses an internal
  //! @c inplace_stop_source. An unhappy completion requests stop on the
  //! other inputs before the combined operation completes.
  //!
  //! @code{.cpp}
  //! auto s = stdexec::when_all(
  //!   stdexec::just(1),
  //!   stdexec::just(2.5),
  //!   stdexec::just(std::string{"x"}));
  //! auto [i, d, str] = stdexec::sync_wait(std::move(s)).value();
  //! // i == 1, d == 2.5, str == "x"
  //! @endcode
  //!
  //! See [exec.when.all] in the C++26 working draft for the normative
  //! specification.
  //!
  //! **Single value-completion requirement.**
  //!
  //! @c when_all requires that each input sender have exactly one
  //! @c set_value_t completion signature — otherwise the *output* value
  //! signature would be a combinatorial explosion of all possible
  //! concatenations. The constraint is enforced at connect time with a
  //! diagnostic pointing at @ref when_all_with_variant_t, which lifts the
  //! restriction (at the cost of producing a variant per input).
  //!
  //! **Completion signatures.**
  //!
  //! Given inputs @c sndr_i with completion signatures
  //!
  //! @code{.cpp}
  //! // For each i in 1..n:
  //! set_value_t(Vi...)        // exactly one such signature per input
  //! set_error_t(Eij)...       // zero or more per input
  //! set_stopped_t()           // optional per input
  //! @endcode
  //!
  //! the resulting sender has completion signatures:
  //!
  //! @code{.cpp}
  //! set_value_t(V1..., V2..., ..., Vn...) // concatenation of every input
  //! set_error_t(Eij)...                   // union across all inputs
  //! set_error_t(std::exception_ptr)       // added if any decay-copy may throw
  //! set_stopped_t()                       // added if any input has it
  //! @endcode
  //!
  //! The value datums of each input are decay-copied into the resulting
  //! sender's state while it waits for the slowest input to finish; the
  //! final tuple is built from those decay-copies. If any decay-copy
  //! throws, the operation transitions to the error path.
  //!
  //! **Concurrency.**
  //!
  //! "Concurrently" here means *the inputs are not sequenced relative to
  //! each other* — @c when_all starts every child operation in a fold
  //! expression before returning from @c start. Whether they actually
  //! execute in parallel depends on the schedulers they're attached to:
  //! independent @c starts_on / @c continues_on branches across different
  //! schedulers truly run in parallel; multiple @c just-rooted branches
  //! all complete synchronously inside @c when_all's @c start.
  //!
  //! **Error and stop semantics.**
  //!
  //! At most *one* completion is delivered to the downstream receiver. If
  //! several children produce errors or stopped completions, the first
  //! one observed wins; subsequent failures are dropped. Concretely:
  //!
  //! - First child to call @c set_error wins; its error becomes the result.
  //! - First child to call @c set_stopped wins (if no error has been seen).
  //! - On either, the internal stop source is signalled so the remaining
  //!   children can wind down promptly.
  //!
  //! **Cancellation.**
  //!
  //! @c when_all chains the receiver's stop token to its internal
  //! stop-source, so an outer stop request propagates to every child.
  //!
  //! **Example.**
  //!
  //! @code{.cpp}
  //! #include <stdexec/execution.hpp>
  //!
  //! int main() {
  //!   using namespace stdexec;
  //!   auto sched = get_parallel_scheduler();
  //!
  //!   auto pipeline = when_all(
  //!     starts_on(sched, just(10) | then([](int x){ return x * 2; })),
  //!     starts_on(sched, just(5)  | then([](int x){ return x + 1; })));
  //!
  //!   auto [a, b] = sync_wait(std::move(pipeline)).value();
  //!   // a == 20, b == 6, computed in parallel on `sched`
  //! }
  //! @endcode
  //!
  //! @see stdexec::when_all_with_variant — for inputs with multiple
  //!                                       value-completion shapes
  //! @see stdexec::transfer_when_all     — when_all + scheduler transfer
  //!                                       (stdexec extension)
  //! @see stdexec::spawn_future          — start a sender eagerly and observe
  //!                                       via a sender
  struct when_all_t
  {
    //! @brief Compose @c __sndrs... into a sender that completes when every
    //!        input has completed.
    //!
    //! @tparam _Senders A pack of types each satisfying @c stdexec::sender.
    //!                  Each must have exactly one @c set_value_t completion
    //!                  signature in the ambient environment.
    //!
    //! @param __sndrs   The senders to compose. Forwarded into the result.
    //!
    //! @returns @c just() for no inputs, the input sender for one input, or a
    //!          sender that concurrently starts every input and concatenates
    //!          their value datums for two or more inputs.
    constexpr auto operator()() const noexcept
    {
      return just();
    }

    template <sender _Sender>
    constexpr auto operator()(_Sender&& __sndr) const noexcept(__nothrow_decay_copyable<_Sender>)
    {
      return static_cast<_Sender&&>(__sndr);
    }

    template <sender _Sender0, sender _Sender1, sender... _Senders>
    constexpr auto operator()(_Sender0&& __sndr0, _Sender1&& __sndr1, _Senders&&... __sndrs) const
      noexcept(__nothrow_decay_copyable<_Sender0, _Sender1, _Senders...>) -> auto
    {
      return STDEXEC::__check_well_formed(__facade::__sender{when_all_t(),
                                                             __(),
                                                             static_cast<_Sender0&&>(__sndr0),
                                                             static_cast<_Sender1&&>(__sndr1),
                                                             static_cast<_Senders&&>(__sndrs)...});
    }
  };

  //! @brief A variadic sender factory like @c when_all that lifts the
  //!        "exactly one value completion per input" restriction by
  //!        wrapping each input in @c into_variant.
  //!
  //! @c when_all_with_variant is for the case where one or more of the
  //! inputs can complete with more than one value-completion shape. Where
  //! @c when_all would refuse to compile, @c when_all_with_variant
  //! transforms each input via @c stdexec::into_variant first (collapsing
  //! that input's multiple shapes into a single
  //! @c std::variant<std::tuple<...>, ...> value), and then composes the
  //! results with the ordinary @c when_all rules.
  //!
  //! @code{.cpp}
  //! // sndr_a value-completes with either int or std::string;
  //! // sndr_b value-completes with float.
  //! auto s = stdexec::when_all_with_variant(sndr_a, sndr_b);
  //! auto [variant_a, variant_b] = stdexec::sync_wait(std::move(s)).value();
  //! //   variant_a: std::variant<std::tuple<int>, std::tuple<std::string>>
  //! //   variant_b: std::variant<std::tuple<float>>
  //! @endcode
  //!
  //! See [exec.when.all] in the C++26 working draft for the normative
  //! specification (where @c when_all_with_variant is specified alongside
  //! @c when_all).
  //!
  //! **Equivalence.**
  //!
  //! `when_all_with_variant(sndrs...)` is specified as
  //! expression-equivalent to `when_all(into_variant(sndrs)...)`
  //! (after @c transform_sender), so all of @c when_all's concurrency,
  //! error, and cancellation semantics carry over unchanged.
  //!
  //! @see stdexec::when_all      — single-value-completion variant
  //! @see stdexec::into_variant  — the adaptor that lifts each input
  struct when_all_with_variant_t
  {
   private:
    static constexpr auto __mk_transform_fn() noexcept
    {
      return []<class... _Child>(__ignore, __ignore, _Child&&... __child)
      {
        return when_all(into_variant(static_cast<_Child&&>(__child))...);
      };
    }

   public:
    //! @brief Compose @c __sndrs... into a sender that completes when every
    //!        input has completed, with each input wrapped in
    //!        @c into_variant.
    //!
    //! @tparam _Senders A non-empty pack of types each satisfying
    //!                  @c stdexec::sender. Inputs may have multiple
    //!                  value-completion shapes.
    //!
    //! @returns A sender equivalent to
    //!          `when_all(into_variant(__sndrs)...)`.
    template <sender... _Senders>
    constexpr auto operator()(_Senders&&... __sndrs) const -> auto
    {
      return STDEXEC::__check_well_formed(
        __facade::__sender{when_all_with_variant_t(), __(), static_cast<_Senders&&>(__sndrs)...});
    }

    template <class _Sender>
    static constexpr auto transform_sender(set_value_t, _Sender&& __sndr, __ignore)
    {
      // transform when_all_with_variant(sndrs...) into when_all(into_variant(sndrs)...).
      return __apply(__mk_transform_fn(), static_cast<_Sender&&>(__sndr));
    }
  };

  //! @brief Like @c when_all, but transfers execution to a scheduler before
  //!        delivering the combined completion.
  //!
  //! @deprecated @c transfer_when_all is deprecated. It is not part of the
  //!             C++26 working draft and is retained only for backwards
  //!             compatibility. Write
  //!             `when_all(sndrs...) | continues_on(sch)` instead;
  //!             the behavior is identical.
  //!
  //! Composition of @c when_all with @c continues_on:
  //! `transfer_when_all(sch, sndrs...)` is expression-equivalent
  //! to `continues_on(when_all(sndrs...), sch)`. The inputs run
  //! concurrently (wherever their respective schedulers run them), and
  //! once all have completed, the combined result is delivered on
  //! @c sch's execution resource.
  //!
  //! @see stdexec::when_all      — without the scheduler transfer
  //! @see stdexec::continues_on  — the underlying transfer primitive
  struct __transfer_when_all_t
  {
    //! @brief Compose @c __sndrs... and deliver the combined completion on
    //!        @c __sched's execution resource.
    //!
    //! @tparam _Scheduler A type satisfying @c stdexec::scheduler.
    //! @tparam _Senders   A non-empty pack of @c stdexec::sender types.
    //
    // The sender is continues_on(when_all(sndrs...), sch) with this tag, so that a domain
    // can customize it: it has the parts of that sender (the scheduler, and when_all's
    // sender as continues_on keeps it), and continues_on's algorithm.
    template <scheduler _Scheduler, sender... _Senders>
    constexpr auto operator()(_Scheduler __sched, _Senders&&... __sndrs) const -> auto
    {
      return STDEXEC::__check_well_formed(__facade::__retagged<__transfer_when_all_t>(
        continues_on(when_all(static_cast<_Senders&&>(__sndrs)...),
                     static_cast<_Scheduler&&>(__sched))));
    }

    // transform __transfer_when_all(sch, sndrs...) into continues_on(when_all(sndrs...), sch):
    // the same parts with continues_on's tag
    template <class _Sender>
    static constexpr auto transform_sender(set_value_t, _Sender&& __sndr, __ignore)
    {
      return __facade::__retagged<continues_on_t>(static_cast<_Sender&&>(__sndr));
    }
  };

  //! @brief Like @c when_all_with_variant, but transfers execution to a
  //!        scheduler before delivering the combined completion.
  //!
  //! @deprecated @c transfer_when_all_with_variant is deprecated. It is not
  //!             part of the C++26 working draft and is retained only for
  //!             backwards compatibility. Write
  //!             `when_all_with_variant(sndrs...) | continues_on(sch)`
  //!             instead; the behavior is identical.
  //!
  //! Composition of @c when_all_with_variant and @c continues_on:
  //! `transfer_when_all_with_variant(sch, sndrs...)` is
  //! expression-equivalent to
  //! `continues_on(when_all_with_variant(sndrs...), sch)`.
  //!
  //! @see stdexec::when_all_with_variant
  //! @see stdexec::transfer_when_all
  struct __transfer_when_all_with_variant_t
  {
    //! @brief Compose @c __sndrs... (each wrapped in @c into_variant) and
    //!        deliver the combined completion on @c __sched's execution
    //!        resource.
    //
    // The sender is __transfer_when_all(sch, into_variant(sndrs)...) with this tag: it has
    // the parts of that sender, and continues_on's algorithm.
    template <scheduler _Scheduler, sender... _Senders>
    constexpr auto operator()(_Scheduler&& __sched, _Senders&&... __sndrs) const -> auto
    {
      return STDEXEC::__check_well_formed(__facade::__retagged<__transfer_when_all_with_variant_t>(
        continues_on(when_all(into_variant(static_cast<_Senders&&>(__sndrs))...),
                     static_cast<_Scheduler&&>(__sched))));
    }

    // transform __transfer_when_all_with_variant(sch, sndrs...) into
    // __transfer_when_all(sch, into_variant(sndrs)...): the same parts with that tag
    template <class _Sender>
    static constexpr auto transform_sender(set_value_t, _Sender&& __sndr, __ignore)
    {
      return __facade::__retagged<__transfer_when_all_t>(static_cast<_Sender&&>(__sndr));
    }
  };

  //! @brief The customization point object for the @c when_all sender factory.
  //!
  //! @c when_all is an instance of @c when_all_t. See @ref when_all_t for
  //! the full description, completion signatures, error/stop semantics, and
  //! a usage example.
  //!
  //! @hideinitializer
  inline constexpr when_all_t when_all{};

  //! @brief The customization point object for the @c when_all_with_variant
  //!        sender factory.
  //!
  //! @c when_all_with_variant is an instance of @c when_all_with_variant_t.
  //! See @ref when_all_with_variant_t for the full description.
  //!
  //! @hideinitializer
  inline constexpr when_all_with_variant_t when_all_with_variant{};

  //! @brief The customization point object for the @c __transfer_when_all
  //!        sender factory.
  //!
  //! @deprecated See @ref __transfer_when_all_t. Use
  //!             `when_all(...) | continues_on(sch)` instead.
  //!
  //! @hideinitializer
  inline constexpr __transfer_when_all_t __transfer_when_all{};

  //! @brief The customization point object for the
  //!        @c __transfer_when_all_with_variant sender factory.
  //!
  //! @deprecated See @ref __transfer_when_all_with_variant_t. Use
  //!             `when_all_with_variant(...) | continues_on(sch)` instead.
  //!
  //! @hideinitializer
  inline constexpr __transfer_when_all_with_variant_t __transfer_when_all_with_variant{};

  namespace __when_all
  {
    enum __state_t
    {
      __started,
      __error,
      __stopped
    };

    template <class _Env>
    constexpr auto __mk_env(_Env&& __env, inplace_stop_source const & __stop_source) noexcept
    {
      return __env::__join(prop{get_stop_token, __stop_source.get_token()},
                           static_cast<_Env&&>(__env));
    }

    template <class _Env>
    using __stoppable_env_t = decltype(__when_all::__mk_env(__declval<_Env>(),
                                                            __declval<inplace_stop_source&>()));

    struct _THE_GIVEN_SENDER_CAN_COMPLETE_SUCCESSFULLY_IN_MORE_THAN_ONE_WAY_;
    struct _USE_WHEN_ALL_WITH_VARIANT_INSTEAD_;

    template <class _CmplSigs>
    concept __can_fail = _CmplSigs::__count(set_error) != 0    //
                      || _CmplSigs::__count(set_stopped) != 0  //
                      || !__nothrow_decay_copyable_results_t<_CmplSigs>();

    template <class _Env, class... _CvSenders>
    using __uses_stop_source_t =
      __mbool<(__can_fail<__completion_signatures_of_t<_CvSenders, _Env>> || ...)>;

    template <class _Env, class... _CvSenders>
    using __env_t = __if<__uses_stop_source_t<_Env, _CvSenders...>, __stoppable_env_t<_Env>, _Env>;

    struct __algo;

    // Is the sender a when_all_with_variant, which stores the values of its children as
    // variants? (Its operation is when_all's, except for that. See __with_variant_algo.)
    template <class _Sender>
    inline constexpr bool __with_variant_v = __same_as<tag_of_t<_Sender>, when_all_with_variant_t>;

    // Can the child fail: complete with an error or stopped, or fail to store its results?
    // (A child that is not a sender in the environment is counted as one that can fail. The
    // walk reports its error.)
    template <class _Child, class... _Env>
    consteval auto __can_fail_in() noexcept -> bool
    {
      if constexpr (sender_in<_Child, _Env...>)
        return __can_fail<__completion_signatures_of_t<_Child, _Env...>>;
      else
        return true;
    }

    // Does when_all need a stop source, whose token its children see? Only if a child can
    // fail. This is decided in the environment of the receiver, before the children's
    // environment is known.
    template <class _Sexpr, class _Indices, class... _Env>
    inline constexpr bool __uses_stop_source_v = false;

    template <class _Sexpr, std::size_t... _Is, class... _Env>
    inline constexpr bool __uses_stop_source_v<_Sexpr, __indices<_Is...>, _Env...> =
      (__when_all::__can_fail_in<__facade::__nth_child_of_t<_Sexpr, _Is>, _Env...>() || ...);

    // (when_all_with_variant(sndrs...) is when_all(into_variant(sndrs)...), whose children
    // can always fail, because into_variant declares an exception_ptr error. Unless there is
    // only one: when_all(sndr) is sndr.)
    template <class _Sender, class... _Env>
    inline constexpr bool __uses_stop_source =
      __with_variant_v<_Sender> ? (__nbr_children_of<_Sender> > 1)
                                : __uses_stop_source_v<__decay_t<_Sender>,
                                                       __make_indices<__nbr_children_of<_Sender>>,
                                                       __decay_t<_Env>...>;

    template <class _Sender, class _Receiver>
    inline constexpr bool __uses_stop_source_for = __uses_stop_source<_Sender, env_of_t<_Receiver>>;

    template <class _Sender>
    inline constexpr bool __uses_stop_source_for<_Sender, __facade::__no_env_receiver> =
      __uses_stop_source<_Sender>;

    template <class _Sig>
    struct __value_tuple_of_sig
    {
      using __t = __mlist<>;
    };

    template <class... _Vs>
    struct __value_tuple_of_sig<set_value_t(_Vs...)>
    {
      using __t = __mlist<__optional<__decayed_tuple<_Vs...>>>;
    };

    template <class _Sig>
    struct __error_of_sig
    {
      using __t = __mlist<>;
    };

    template <class _Error>
    struct __error_of_sig<set_error_t(_Error)>
    {
      using __t = __mlist<_Error>;
    };

    template <class _Sigs, bool _WithVariant>
    struct __child_results;

    template <class... _Sigs>
    struct __child_results<completion_signatures<_Sigs...>, false>
    {
      // The value tuple of a child that completes with a value in exactly one way
      using __values_t = __mcall<__mconcat<>, typename __value_tuple_of_sig<_Sigs>::__t...>;
      using __errors_t = __mcall<__mconcat<>, typename __error_of_sig<_Sigs>::__t...>;
    };

    // when_all_with_variant stores a child's values as a variant of the value tuples of
    // each way it completes with a value, as into_variant does.
    template <class... _Sigs>
    struct __child_results<completion_signatures<_Sigs...>, true>
    {
      using __values_t = __mlist<__optional<__value_types_t<completion_signatures<_Sigs...>,
                                                            __qq<__decayed_std_tuple>,
                                                            __qq<__std_variant>>>>;
      using __errors_t = __mcall<__mconcat<>, typename __error_of_sig<_Sigs>::__t...>;
    };

    // The types of when_all's operation state, computed from the children's completions
    template <class _Sender, class _Receiver, class _Indices>
    struct __traits;

    template <class _Sender, class _Receiver, std::size_t... _Is>
    struct __traits<_Sender, _Receiver, __indices<_Is...>>
    {
      template <std::size_t _Idx>
      using __cmplsigs_t =
        __facade::__completions_of_t<__algo, __facade::__child<_Idx>, _Sender, _Receiver>;

      // A when_all_with_variant's children may complete with a value in more than one way.
      static constexpr bool __with_variant = __with_variant_v<_Sender>;

      template <std::size_t _Idx>
      static constexpr bool __sends_values = __with_variant
                                             ? __cmplsigs_t<_Idx>::__count(set_value) != 0
                                             : __cmplsigs_t<_Idx>::__count(set_value) == 1;

      static constexpr bool __too_many_values = !__with_variant
                                             && ((__cmplsigs_t<_Is>::__count(set_value) > 1)
                                                 || ...);
      static constexpr bool __all_send_values = (__sends_values<_Is> && ...);
      static constexpr bool __sends_stopped   = ((__cmplsigs_t<_Is>::__count(set_stopped) != 0)
                                               || ...);
      static constexpr bool __nothrow_decay =
        (__nothrow_decay_copyable_results_t<__cmplsigs_t<_Is>>::value && ...);
      static constexpr bool __uses_stop_source = __uses_stop_source_for<_Sender, _Receiver>;

      // A tuple of optional value tuples (or variants of them), one per child, or __ignore if
      // some child cannot complete with a value (so when_all never does).
      using __values_t =
        __if_c<__all_send_values,
               __mcall<__mconcat<__qq<__tuple>>,
                       typename __child_results<__cmplsigs_t<_Is>, __with_variant>::__values_t...>,
               __ignore>;

      // (when_all_with_variant can complete with an exception_ptr even when saving the values
      // cannot throw, because into_variant declares that error.)
      using __errors_t =
        __mcall<__mconcat<__qq<__uniqued_variant>>,
                typename __child_results<__cmplsigs_t<_Is>, __with_variant>::__errors_t...,
                __if_c<__nothrow_decay && !__with_variant, __mlist<>, __mlist<std::exception_ptr>>>;
    };

    template <class _Sender, class _Receiver>
    using __traits_t = __traits<_Sender, _Receiver, __make_indices<__nbr_children_of<_Sender>>>;

    template <bool _UsesStopSource, class _Receiver, class _Self>
    using __stop_base_t =
      __if_c<_UsesStopSource,
             __stoppable_base_for_t<stop_token_of_t<env_of_t<_Receiver>>, _Self>,
             __empty>;

    // when_all's operation state. (Its type does not name when_all's sender. The receivers
    // of the children name the state, and the children's operations name their receivers,
    // so the types of the operations, and the debug info that names them, would grow with
    // the square of the sender's size.)
    template <class _Receiver,
              class _Values,
              class _Errors,
              std::size_t _Np,
              bool        _WithVariant,
              bool        _UsesStopSource,
              bool        _SendsStopped>
    struct __state
      : __stop_base_t<
          _UsesStopSource,
          _Receiver,
          __state<_Receiver, _Values, _Errors, _Np, _WithVariant, _UsesStopSource, _SendsStopped>>
    {
      static constexpr std::size_t __nbr_children     = _Np;
      static constexpr bool        __with_variant     = _WithVariant;
      static constexpr bool        __uses_stop_source = _UsesStopSource;
      static constexpr bool        __sends_stopped    = _SendsStopped;
      using __stop_source_t = __if_c<__uses_stop_source, inplace_stop_source, __empty>;

      constexpr explicit __state(_Receiver&& __rcvr) noexcept
        : __rcvr_(static_cast<_Receiver&&>(__rcvr))
      {}

      STDEXEC_IMMOVABLE(__state);

      // Called when the receiver's stop token is stopped
      void on_stop_requested() noexcept
      {
        __stop_source_.request_stop();
      }

      constexpr void __register_stop_callback() noexcept
      {
        if constexpr (__uses_stop_source)
          this->register_stoppable(get_stop_token(STDEXEC::get_env(__rcvr_)));
      }

      constexpr void __deregister_stop_callback() noexcept
      {
        if constexpr (__uses_stop_source)
          this->deregister_stoppable();
      }

      constexpr void __request_stop() noexcept
      {
        if constexpr (__uses_stop_source)
          __stop_source_.request_stop();
      }

      _Receiver                  __rcvr_;
      __std::atomic<std::size_t> __count_{__nbr_children};
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      __stop_source_t            __stop_source_{};
      __std::atomic<__state_t>   __disp_{__started};
      _Errors                    __errors_{__no_init};
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Values                    __values_{};
    };

    template <class _Sender, class _Receiver, class _Traits = __traits_t<_Sender, _Receiver>>
    using __state_for_t = __state<_Receiver,
                                  typename _Traits::__values_t,
                                  typename _Traits::__errors_t,
                                  __nbr_children_of<_Sender>,
                                  _Traits::__with_variant,
                                  _Traits::__uses_stop_source,
                                  _Traits::__sends_stopped>;

    // when_all(sndrs...): the children run concurrently, and the last one to complete
    // completes the operation: with all their values if each child succeeded, and otherwise
    // with an error or stopped. An error or stopped requests stop on a stop source whose
    // token the children see, if any child can fail.
    //
    // when_all_with_variant(sndrs...) is the same, except that it saves the values of each
    // child as a variant of tuples, as into_variant does, and completes with the variants.
    struct __algo : __facade::__defaults
    {
      // The children see when_all's stop token, if it has a stop source.
      template <std::size_t _Idx, class _Sender, class _Env>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __env_for(__facade::__child<_Idx>,
                                      _Sender const &,
                                      _Env&& __env) noexcept -> decltype(auto)
      {
        if constexpr (__uses_stop_source<_Sender, _Env>)
          return __env::__join(prop{get_stop_token, inplace_stop_token{}},
                               static_cast<_Env&&>(__env));
        else
          return static_cast<_Env>(static_cast<_Env&&>(__env));
      }

      template <std::size_t _Idx, class _State>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __get_env(__facade::__child<_Idx>, _State const & __state) noexcept
        -> decltype(auto)
      {
        if constexpr (_State::__uses_stop_source)
          return __env::__join(prop{get_stop_token, __state.__stop_source_.get_token()},
                               STDEXEC::get_env(__state.__rcvr_));
        else
          return STDEXEC::get_env(__state.__rcvr_);
      }

      // A child that completes successfully in more than one way is an error: use
      // when_all_with_variant instead.
      template <class _Sender, class _Receiver>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __get_state(_Sender&&, _Receiver __rcvr) noexcept  //
        -> __state_for_t<_Sender, _Receiver>
      {
        return __state_for_t<_Sender, _Receiver>(static_cast<_Receiver&&>(__rcvr));
      }

      template <class _Sender, class _Receiver>
        requires __traits_t<_Sender, _Receiver>::__too_many_values
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __get_state(_Sender&&, _Receiver) noexcept
      {
        return __mexception<_WHAT_(_INVALID_ARGUMENT_),
                            _WHERE_(_IN_ALGORITHM_, when_all_t),
                            _WHY_(
                              _THE_GIVEN_SENDER_CAN_COMPLETE_SUCCESSFULLY_IN_MORE_THAN_ONE_WAY_),
                            _TO_FIX_THIS_ERROR_(_USE_WHEN_ALL_WITH_VARIANT_INSTEAD_)>();
      }

      template <class _State, class _Children>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __start(_State& __state, _Children& __children) noexcept
      {
        if constexpr (_State::__nbr_children == 0)
        {
          // (Only when_all_with_variant() has no children. when_all() is just().)
          return __facade::__send(__state.__rcvr_, set_value);
        }
        else
        {
          __state.__register_stop_callback();
          return __children.__start_all();
        }
      }

      // A child completed with values. Save them, unless the operation has failed.
      template <std::size_t _Idx, class _State, class... _Args>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __complete(__facade::__child<_Idx>,
                                       _State& __state,
                                       set_value_t,
                                       _Args&&... __args) noexcept
      {
        if constexpr (!__decay_copyable<_Args...>)
        {
          return __facade::__decay_copy_error<when_all_t, _Args...>();
        }
        else
        {
          if constexpr (!__same_as<decltype(__state.__values_), __ignore>)
          {
            if (__state.__disp_.load(__std::memory_order_relaxed) == __started)
            {
              auto& __opt_values = STDEXEC::__get<_Idx>(__state.__values_);
              if constexpr (__nothrow_decay_copyable<_Args...>)
              {
                __algo::__save<_State::__with_variant>(__opt_values,
                                                       static_cast<_Args&&>(__args)...);
              }
              else
              {
                STDEXEC_TRY
                {
                  __algo::__save<_State::__with_variant>(__opt_values,
                                                         static_cast<_Args&&>(__args)...);
                }
                STDEXEC_CATCH_ALL
                {
                  __algo::__set_error(__state, std::current_exception());
                }
              }
            }
          }
          return __algo::__arrive(__state);
        }
      }

      // A child failed. The operation cannot complete with values now.
      template <std::size_t _Idx, class _State, class _Error>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __complete(__facade::__child<_Idx>,
                                       _State& __state,
                                       set_error_t,
                                       _Error&& __err) noexcept
      {
        if constexpr (!__decay_copyable<_Error>)
        {
          return __facade::__decay_copy_error<when_all_t, _Error>();
        }
        else
        {
          __algo::__set_error(__state, static_cast<_Error&&>(__err));
          // Once a child has failed, the operation completes with an error.
          return __facade::__when(__algo::__last_to_arrive(__state),
                                  [&]() noexcept { return __algo::__send_error(__state); });
        }
      }

      template <std::size_t _Idx, class _State>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __complete(__facade::__child<_Idx>,
                                       _State& __state,
                                       set_stopped_t) noexcept
      {
        __state_t __expected = __started;
        if (__state.__disp_.compare_exchange_strong(__expected, __stopped))
        {
          __state.__request_stop();
        }
        // The operation completes with stopped, or with the error of a child that failed.
        return __facade::__when(__algo::__last_to_arrive(__state),
                                [&]() noexcept
                                {
                                  return __facade::__choose(
                                    __state.__disp_.load(__std::memory_order_relaxed) == __error,
                                    [&]() noexcept { return __algo::__send_error(__state); },
                                    [&]() noexcept { return __algo::__send_stopped(__state); });
                                });
      }

     private:
      // Saves a child's values in its optional value tuple, or in its optional variant of
      // value tuples
      template <bool _WithVariant, class _Values, class... _Args>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr void __save(__optional<_Values>& __opt_values, _Args&&... __args)
        noexcept(__nothrow_decay_copyable<_Args...>)
      {
        if constexpr (_WithVariant)
        {
          __opt_values.emplace(std::in_place_type<__decayed_std_tuple<_Args...>>,
                               static_cast<_Args&&>(__args)...);
        }
        else
        {
          using __tuple_t = __decayed_tuple<_Args...>;
          static_assert(__same_as<_Values, __tuple_t>,
                        "One of the senders in this when_all() is fibbing about what types it "
                        "sends");
          __opt_values.emplace(__tuple_t{static_cast<_Args&&>(__args)...});
        }
      }

      template <class _State, class _Error>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr void __set_error(_State& __state, _Error&& __err) noexcept
      {
        switch (__state.__disp_.exchange(__error))
        {
        case __started:
          __state.__request_stop();
          [[fallthrough]];
        case __stopped:
          if constexpr (__nothrow_decay_copyable<_Error>)
          {
            __state.__errors_.template emplace<__decay_t<_Error>>(static_cast<_Error&&>(__err));
          }
          else
          {
            STDEXEC_TRY
            {
              __state.__errors_.template emplace<__decay_t<_Error>>(static_cast<_Error&&>(__err));
            }
            STDEXEC_CATCH_ALL
            {
              __state.__errors_.template emplace<std::exception_ptr>(std::current_exception());
            }
          }
          break;
        case __error:;  // We're already in the "error" state. Ignore the error.
        }
      }

      // Is this the last child to complete? If so, it completes the operation.
      template <class _State>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __last_to_arrive(_State& __state) noexcept -> bool
      {
        if (1 != __state.__count_.fetch_sub(1, __std::memory_order_acq_rel))
          return false;
        __state.__deregister_stop_callback();
        return true;
      }

      template <class _State>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __arrive(_State& __state) noexcept
      {
        return __facade::__when(__algo::__last_to_arrive(__state),
                                [&]() noexcept
                                {
                                  auto const __disp = __state.__disp_.load(
                                    __std::memory_order_relaxed);
                                  return __facade::__choose(
                                    __disp == __started,
                                    [&]() noexcept { return __algo::__send_values(__state); },
                                    [&]() noexcept
                                    {
                                      return __facade::__choose(
                                        __disp == __error,
                                        [&]() noexcept { return __algo::__send_error(__state); },
                                        [&]() noexcept { return __algo::__send_stopped(__state); });
                                    });
                                });
      }

      template <class _State>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __send_values(_State& __state) noexcept
      {
        if constexpr (__same_as<decltype(__state.__values_), __ignore>)
        {
          // Unreachable: some child cannot complete with a value, so the operation cannot.
          return __facade::__effects<>();
        }
        else if constexpr (_State::__with_variant)
        {
          // when_all_with_variant sends each child's variant.
          return STDEXEC::__apply(
            [&]<class... _Variants>(__optional<_Variants>&... __opts) noexcept
            {
              return __facade::__send(__state.__rcvr_,
                                      set_value,
                                      static_cast<_Variants&&>(*__opts)...);
            },
            __state.__values_);
        }
        else
        {
          return STDEXEC::__apply(
            [&]<class... _OptTuples>(_OptTuples&... __opts) noexcept
            {
              return STDEXEC::__cat_apply(
                [&]<class... _Vs>(_Vs&&... __vs) noexcept
                {
                  return __facade::__send(__state.__rcvr_, set_value, static_cast<_Vs&&>(__vs)...);
                },
                static_cast<__decay_t<decltype(*__opts)>&&>(*__opts)...);
            },
            __state.__values_);
        }
      }

      template <class _State>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __send_error(_State& __state) noexcept
      {
        return __facade::__visit(
          __state.__errors_,
          [&]<class _Error>(_Error& __err) noexcept
          { return __facade::__send(__state.__rcvr_, set_error, static_cast<_Error&&>(__err)); });
      }

      template <class _State>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __send_stopped(_State& __state) noexcept
      {
        if constexpr (_State::__sends_stopped)
          return __facade::__send(__state.__rcvr_, set_stopped);
        else
          return __facade::__effects<>();  // Unreachable: no child can be stopped.
      }
    };

    // when_all_with_variant(sndrs...) is lowered to when_all(into_variant(sndrs)...) (see
    // when_all_with_variant_t::transform_sender), so it has that sender's completions. (The
    // facade derives the same ones from the algorithm, in another order, except one: for a
    // child that never completes with a value, into_variant still declares a value
    // completion.) Its attributes are derived from what that sender does, which is what
    // when_all's algorithm does for a when_all_with_variant.
    struct __with_variant_algo : __algo
    {
      template <class _Sender, class... _Env>
      static consteval auto __get_completion_signatures()
      {
        using __sndr_t = __detail::__transform_sender_result_t<when_all_with_variant_t,
                                                               set_value_t,
                                                               _Sender,
                                                               env<>>;
        return STDEXEC::get_completion_signatures<__sndr_t, _Env...>();
      }
    };
  }  // namespace __when_all

  template <>
  struct __facade::__algorithm<when_all_t> : __when_all::__algo
  {};

  template <>
  struct __facade::__algorithm<when_all_with_variant_t> : __when_all::__with_variant_algo
  {};

  // __transfer_when_all(sch, sndrs...) and __transfer_when_all_with_variant(sch, sndrs...)
  // are continues_on senders with tags of their own (see their transform_sender members).
  template <>
  struct __facade::__algorithm<__transfer_when_all_t> : __trnsfr::__algo
  {};

  template <>
  struct __facade::__algorithm<__transfer_when_all_with_variant_t> : __trnsfr::__algo
  {};
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
