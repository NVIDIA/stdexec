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
#include "__basic_sender.hpp"
#include "__concepts.hpp"
#include "__continues_on.hpp"
#include "__diagnostics.hpp"
#include "__domain.hpp"
#include "__env.hpp"
#include "__into_variant.hpp"
#include "__meta.hpp"
#include "__optional.hpp"
#include "__schedulers.hpp"
#include "__senders.hpp"
#include "__transform_completion_signatures.hpp"
#include "__tuple.hpp"
#include "__type_traits.hpp"
#include "__utility.hpp"
#include "__variant.hpp"

#include "../stop_token.hpp"

#include "__atomic.hpp"
#include <exception>

#include "__prologue.hpp"

namespace STDEXEC
{
  /////////////////////////////////////////////////////////////////////////////
  // [exec.when.all]
  //! @brief A variadic sender factory that runs multiple senders concurrently
  //!        and completes when all of them have completed, concatenating
  //!        their value datums.
  //!
  //! @c when_all is the canonical *parallel composition* primitive in the
  //! sender model. You give it one or more senders; it returns a single
  //! sender that, when connected and started, starts *all* of the input
  //! senders concurrently. When every input has completed, @c when_all's
  //! sender completes with a value tuple that is the concatenation of every
  //! input's value datums.
  //!
  //! If any one input fails or is stopped, @c when_all requests stop on the
  //! others (via an internal @c inplace_stop_source) and completes with
  //! that error (or with @c set_stopped). This makes @c when_all naturally
  //! fail-fast: as soon as one branch has gone bad, the rest are asked to
  //! wind down.
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
  //! set_value_t(V1..., V2..., ..., Vn...)   // concatenation of every input
  //! set_error_t(Eij)...                     // union across all inputs
  //! set_error_t(std::exception_ptr)         // added if any decay-copy may throw
  //! set_stopped_t()                         // added if any input has it,
  //!                                         //   or if cancellation may happen
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
  //! @see stdexec::when_all_with_variant       — for inputs with multiple value-completion shapes
  //! @see stdexec::transfer_when_all           — when_all + scheduler transfer (stdexec extension)
  //! @see stdexec::spawn_future                — start a sender eagerly and observe via a sender
  struct when_all_t
  {
    //! @brief Compose @c __sndrs... into a sender that completes when every
    //!        input has completed.
    //!
    //! @tparam _Senders A pack of types each satisfying @c stdexec::sender.
    //!                  Must be non-empty. Each must have exactly one
    //!                  @c set_value_t completion signature in the
    //!                  ambient environment.
    //!
    //! @param __sndrs   The senders to compose. Forwarded into the result.
    //!
    //! @returns A sender that, when connected and started, concurrently
    //!          starts every input and value-completes with the
    //!          concatenation of the input's value datums.
    template <sender... _Senders>
    constexpr auto operator()(_Senders&&... __sndrs) const -> __well_formed_sender auto
    {
      return __make_sexpr<when_all_t>(__(), static_cast<_Senders&&>(__sndrs)...);
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
  //! <tt>when_all_with_variant(sndrs...)</tt> is specified as
  //! expression-equivalent to <tt>when_all(into_variant(sndrs)...)</tt>
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
    //!          <tt>when_all(into_variant(__sndrs)...)</tt>.
    template <sender... _Senders>
    constexpr auto operator()(_Senders&&... __sndrs) const -> __well_formed_sender auto
    {
      return __make_sexpr<when_all_with_variant_t>(__(), static_cast<_Senders&&>(__sndrs)...);
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
  //!             <tt>when_all(sndrs...) | continues_on(sch)</tt> instead;
  //!             the behavior is identical.
  //!
  //! Composition of @c when_all with @c continues_on:
  //! <tt>transfer_when_all(sch, sndrs...)</tt> is expression-equivalent
  //! to <tt>continues_on(when_all(sndrs...), sch)</tt>. The inputs run
  //! concurrently (wherever their respective schedulers run them), and
  //! once all have completed, the combined result is delivered on
  //! @c sch's execution resource.
  //!
  //! @see stdexec::when_all      — without the scheduler transfer
  //! @see stdexec::continues_on  — the underlying transfer primitive
  struct transfer_when_all_t
  {
    //! @brief Compose @c __sndrs... and deliver the combined completion on
    //!        @c __sched's execution resource.
    //!
    //! @tparam _Scheduler A type satisfying @c stdexec::scheduler.
    //! @tparam _Senders   A non-empty pack of @c stdexec::sender types.
    template <scheduler _Scheduler, sender... _Senders>
    constexpr auto
    operator()(_Scheduler __sched, _Senders&&... __sndrs) const -> __well_formed_sender auto
    {
      return __make_sexpr<transfer_when_all_t>(static_cast<_Scheduler&&>(__sched),
                                               static_cast<_Senders&&>(__sndrs)...);
    }

    template <class _Sender>
    static constexpr auto transform_sender(set_value_t, _Sender&& __sndr, __ignore)
    {
      // transform transfer_when_all(sch, sndrs...) into
      // continues_on(when_all(sndrs...), sch).
      return __apply(
        [&]<class _Data, class... _Child>(__ignore, _Data&& __data, _Child&&... __child)
        {
          return continues_on(when_all(static_cast<_Child&&>(__child)...),
                              static_cast<_Data&&>(__data));
        },
        static_cast<_Sender&&>(__sndr));
    }
  };

  //! @brief Like @c when_all_with_variant, but transfers execution to a
  //!        scheduler before delivering the combined completion.
  //!
  //! @deprecated @c transfer_when_all_with_variant is deprecated. It is not
  //!             part of the C++26 working draft and is retained only for
  //!             backwards compatibility. Write
  //!             <tt>when_all_with_variant(sndrs...) | continues_on(sch)</tt>
  //!             instead; the behavior is identical.
  //!
  //! Composition of @c when_all_with_variant and @c continues_on:
  //! <tt>transfer_when_all_with_variant(sch, sndrs...)</tt> is
  //! expression-equivalent to
  //! <tt>continues_on(when_all_with_variant(sndrs...), sch)</tt>.
  //!
  //! @see stdexec::when_all_with_variant
  //! @see stdexec::transfer_when_all
  struct transfer_when_all_with_variant_t
  {
    //! @brief Compose @c __sndrs... (each wrapped in @c into_variant) and
    //!        deliver the combined completion on @c __sched's execution
    //!        resource.
    template <scheduler _Scheduler, sender... _Senders>
    constexpr auto
    operator()(_Scheduler&& __sched, _Senders&&... __sndrs) const -> __well_formed_sender auto
    {
      return __make_sexpr<transfer_when_all_with_variant_t>(static_cast<_Scheduler&&>(__sched),
                                                            static_cast<_Senders&&>(__sndrs)...);
    }

    template <class _Sender>
    static constexpr auto transform_sender(set_value_t, _Sender&& __sndr, __ignore)
    {
      // transform the transfer_when_all_with_variant(sch, sndrs...) into
      // transfer_when_all(sch, into_variant(sndrs...))
      return __apply(
        [&]<class _Data, class... _Child>(__ignore, _Data&& __data, _Child&&... __child)
        {
          return transfer_when_all_t()(static_cast<_Data&&>(__data),
                                       into_variant(static_cast<_Child&&>(__child))...);
        },
        static_cast<_Sender&&>(__sndr));
    }
  };

  //! @brief The customization point object for the @c when_all sender factory.
  //!
  //! @c when_all is an instance of @ref when_all_t. See @ref when_all_t for
  //! the full description, completion signatures, error/stop semantics, and
  //! a usage example.
  //!
  //! @hideinitializer
  inline constexpr when_all_t when_all{};

  //! @brief The customization point object for the @c when_all_with_variant
  //!        sender factory.
  //!
  //! @c when_all_with_variant is an instance of @ref when_all_with_variant_t.
  //! See @ref when_all_with_variant_t for the full description.
  //!
  //! @hideinitializer
  inline constexpr when_all_with_variant_t when_all_with_variant{};

  //! @brief The customization point object for the @c transfer_when_all
  //!        sender factory.
  //!
  //! @deprecated See @ref transfer_when_all_t. Use
  //!             <tt>when_all(...) | continues_on(sch)</tt> instead.
  //!
  //! @hideinitializer
  inline constexpr transfer_when_all_t transfer_when_all{};

  //! @brief The customization point object for the
  //!        @c transfer_when_all_with_variant sender factory.
  //!
  //! @deprecated See @ref transfer_when_all_with_variant_t. Use
  //!             <tt>when_all_with_variant(...) | continues_on(sch)</tt> instead.
  //!
  //! @hideinitializer
  inline constexpr transfer_when_all_with_variant_t transfer_when_all_with_variant{};

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
    using __env_t = decltype(__when_all::__mk_env(__declval<_Env>(),
                                                  __declval<inplace_stop_source&>()));

    template <class _Sender, class _Env>
    concept __max1_sender =
      sender_in<_Sender, _Env>
      && __minvocable_q<__value_types_of_t, _Sender, _Env, __mconst<int>, __msingle_or<void>>;

    struct _THE_GIVEN_SENDER_CAN_COMPLETE_SUCCESSFULLY_IN_MORE_THAN_ONE_WAY_
    {};
    struct _USE_WHEN_ALL_WITH_VARIANT_INSTEAD_
    {};

    template <class _Sender, class... _Env>
    using __too_many_value_completions_error_t =
      __mexception<_WHAT_(_INVALID_ARGUMENT_),
                   _WHERE_(_IN_ALGORITHM_, when_all_t),
                   _WHY_(_THE_GIVEN_SENDER_CAN_COMPLETE_SUCCESSFULLY_IN_MORE_THAN_ONE_WAY_),
                   _TO_FIX_THIS_ERROR_(_USE_WHEN_ALL_WITH_VARIANT_INSTEAD_),
                   _WITH_PRETTY_SENDER_<_Sender>,
                   __fn_t<_WITH_ENVIRONMENT_, _Env>...>;

    template <class _Error>
    using __set_error_t = completion_signatures<set_error_t(__decay_t<_Error>)>;

    template <class _Sender, class... _Env>
    using __nothrow_decay_copyable_results_t =
      STDEXEC::__nothrow_decay_copyable_results_t<__completion_signatures_of_t<_Sender, _Env...>>;

    template <class... _Env>
    struct __completions
    {
      // TODO(ericniebler): check that all senders have a common completion domain
      template <class... _Senders>
      using __all_nothrow_decay_copyable_results_t =
        __mand<__nothrow_decay_copyable_results_t<_Senders, _Env...>...>;

      template <class _Sender, class _ValueTuple, class... _Rest>
      using __value_tuple_t = __minvoke<__if_c<(0 == sizeof...(_Rest)),
                                               __mconst<_ValueTuple>,
                                               __q<__too_many_value_completions_error_t>>,
                                        _Sender,
                                        _Env...>;

      template <class _Sender>
      using __single_values_of_t = __value_types_t<__completion_signatures_of_t<_Sender, _Env...>,
                                                   __mtransform<__q<__decay_t>, __q<__mlist>>,
                                                   __mbind_front_q<__value_tuple_t, _Sender>>;

      template <class... _Senders>
      using __set_values_sig_t =
        __minvoke_q<completion_signatures,
                    __minvoke<__mconcat<__qf<set_value_t>>, __single_values_of_t<_Senders>...>>;

      template <class... _Senders>
      using __f = __minvoke_q<
        __concat_completion_signatures_t,
        __minvoke_q<__eptr_completion_unless_t, __all_nothrow_decay_copyable_results_t<_Senders...>>,
        __minvoke<__mwith_default<__qq<__set_values_sig_t>, completion_signatures<>>, _Senders...>,
        __transform_reduce_completion_signatures_t<__completion_signatures_of_t<_Senders, _Env...>,
                                                   __mconst<completion_signatures<>>::__f,
                                                   __set_error_t,
                                                   completion_signatures<set_stopped_t()>,
                                                   __concat_completion_signatures_t>...>;
    };

    template <class _Receiver, class _ValuesTuple>
    constexpr void __set_values(_Receiver& __rcvr, _ValuesTuple& __values) noexcept
    {
      STDEXEC::__apply(
        [&]<class... OptTuples>(OptTuples&&... __opt_vals) noexcept -> void
        {
          STDEXEC::__cat_apply(__mk_completion_fn(set_value, __rcvr),
                               *static_cast<OptTuples&&>(__opt_vals)...);
        },
        static_cast<_ValuesTuple&&>(__values));
    }

    template <class _Env, class _Sender>
    using __values_opt_tuple_t =
      value_types_of_t<_Sender, __env_t<_Env>, __decayed_tuple, __optional>;

    template <class _Env, __max1_sender<__env_t<_Env>>... _Senders>
    struct __traits
    {
      // tuple<optional<tuple<Vs1...>>, optional<tuple<Vs2...>>, ...>
      using __values_tuple = __minvoke<
        __mwith_default<__mtransform<__mbind_front_q<__values_opt_tuple_t, _Env>, __q<__tuple>>,
                        __ignore>,
        _Senders...>;

      using __collect_errors = __mbind_front_q<__mset_insert, __mset<>>;

      using __errors_list =
        __minvoke<__mconcat<>,
                  __if<__mand<__nothrow_decay_copyable_results_t<_Senders, _Env>...>,
                       __mlist<>,
                       __mlist<std::exception_ptr>>,
                  __error_types_of_t<_Senders, __env_t<_Env>, __q<__mlist>>...>;

      using __errors_variant = __mapply<__q<__uniqued_variant>, __errors_list>;
    };

    struct _INVALID_ARGUMENTS_TO_WHEN_ALL_
    {};

    template <class _State>
    struct __forward_stop_request
    {
      constexpr void operator()() const noexcept
      {
        // Temporarily increment the count to avoid concurrent/recursive arrivals to
        // pull the rug under our feet. Relaxed memory order is fine here.
        __state_->__count_.fetch_add(1, __std::memory_order_relaxed);

        __state_t __expected = __started;
        // Transition to the "stopped" state if and only if we're in the
        // "started" state. (If this fails, it's because we're in an
        // error state, which trumps cancellation.)
        if (__state_->__state_.compare_exchange_strong(__expected, __stopped))
        {
          __state_->__stop_source_.request_stop();
        }

        // Arrive in order to decrement the count again and complete if needed.
        __state_->__arrive();
      }

      _State* __state_;
    };

    template <class _ErrorsVariant, class _ValuesTuple, class _Receiver, bool _SendsStopped>
    struct __state
    {
      using __receiver_t = _Receiver;
      using __stop_callback_t =
        stop_callback_for_t<stop_token_of_t<env_of_t<_Receiver>>, __forward_stop_request<__state>>;

      constexpr void __arrive() noexcept
      {
        if (1 == __count_.fetch_sub(1, __std::memory_order_acq_rel))
        {
          __complete();
        }
      }

      constexpr void __complete() noexcept
      {
        // Stop callback is no longer needed. Destroy it.
        __on_stop_.reset();
        // All child operations have completed and arrived at the barrier.
        switch (__state_.load(__std::memory_order_relaxed))
        {
        case __stopped:
          if constexpr (_SendsStopped)
          {
            STDEXEC::set_stopped(static_cast<_Receiver&&>(__rcvr_));
            break;
          }
          // This is reachable because the stop callback sets __stopped whether
          // or not any child can send set_stopped, therefore we handle this the
          // same as __started
          [[fallthrough]];
        case __started:
          if constexpr (!__std::same_as<_ValuesTuple, __ignore>)
          {
            // All child operations completed successfully:
            __when_all::__set_values(__rcvr_, __values_);
          }
          break;
        case __error:
          if constexpr (!__same_as<_ErrorsVariant, __variant<>>)
          {
            // One or more child operations completed with an error:
            STDEXEC::__visit(__mk_completion_fn(set_error, __rcvr_),
                             static_cast<_ErrorsVariant&&>(__errors_));
          }
          break;
        default:;
        }
      }

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Receiver                  __rcvr_;
      __std::atomic<std::size_t> __count_;
      inplace_stop_source        __stop_source_{};
      // Could be non-atomic here and atomic_ref everywhere except __completion_fn
      __std::atomic<__state_t>      __state_{__started};
      _ErrorsVariant                __errors_{__no_init};
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _ValuesTuple                  __values_{};
      __optional<__stop_callback_t> __on_stop_{};
    };

    template <class... _Senders>
    struct __attrs
    {
      template <class _Tag, class... _Env>
      using __when_all_domain_t =
        __common_domain_t<__completion_domain_of_t<set_value_t, _Senders, _Env...>...>;

      template <class... _Env>
      [[nodiscard]]
      constexpr auto query(get_completion_domain_t<set_value_t>, _Env const &...) const noexcept
        -> __when_all_domain_t<set_value_t, _Env...>;

      template <class... _Env>
      [[nodiscard]]
      constexpr auto query(get_completion_domain_t<set_error_t>, _Env const &...) const noexcept
        -> __common_domain_t<__when_all_domain_t<set_value_t, _Env...>,
                             __when_all_domain_t<set_error_t, _Env...>,
                             __when_all_domain_t<set_stopped_t, _Env...>>;

      template <class... _Env>
      [[nodiscard]]
      constexpr auto query(get_completion_domain_t<set_stopped_t>, _Env const &...) const noexcept
        -> __common_domain_t<__when_all_domain_t<set_value_t, _Env...>,
                             __when_all_domain_t<set_stopped_t, _Env...>>;

      template <class _Tag, class... _Env>
      [[nodiscard]]
      constexpr auto query(__get_completion_behavior_t<_Tag>, _Env const &...) const noexcept
      {
        return __completion_behavior::__common(
          STDEXEC::__get_completion_behavior<_Tag, _Senders, _Env...>()...);
      }
    };

    // A when_all with no senders completes inline with no values.
    template <>
    struct __attrs<>
    {
      [[nodiscard]]
      constexpr auto query(__get_completion_behavior_t<set_value_t>) const noexcept
      {
        return __completion_behavior::__inline_completion;
      }

      [[nodiscard]]
      constexpr auto query(__get_completion_behavior_t<set_stopped_t>) const noexcept
      {
        return __completion_behavior::__inline_completion;
      }
    };

    template <class _Receiver>
    static constexpr auto __mk_state_fn(_Receiver&& __rcvr) noexcept
    {
      return [&]<__max1_sender<__env_t<env_of_t<_Receiver>>>... _Child>(__ignore,
                                                                        __ignore,
                                                                        _Child&&...) noexcept
      {
        using _Traits        = __traits<env_of_t<_Receiver>, _Child...>;
        using _ErrorsVariant = _Traits::__errors_variant;
        using _ValuesTuple   = _Traits::__values_tuple;
        using _State         = __state<_ErrorsVariant,
                                       _ValuesTuple,
                                       _Receiver,
                                       (sends_stopped<_Child, env_of_t<_Receiver>> || ...)>;
        return _State{static_cast<_Receiver&&>(__rcvr), sizeof...(_Child)};
      };
    }

    template <class _Receiver>
    using __mk_state_fn_t = decltype(__when_all::__mk_state_fn(__declval<_Receiver>()));

    struct __when_all_impl : __sexpr_defaults
    {
      template <class _Self, class... _Env>
      using __completions_t = __children_of<_Self, __when_all::__completions<__env_t<_Env>...>>;

      static constexpr auto __get_attrs =
        []<class... _Child>(__ignore, __ignore, _Child const &...) noexcept
      {
        return __when_all::__attrs<_Child...>{};
      };

      template <class _Self, class... _Env>
      static consteval auto __get_completion_signatures()
      {
        static_assert(__sender_for<_Self, when_all_t>);
        if constexpr (__minvocable_q<__completions_t, _Self, _Env...>)
        {
          // TODO: update this to use constant evaluation:
          return __completions_t<_Self, _Env...>{};
        }
        else if constexpr (sizeof...(_Env) == 0)
        {
          return STDEXEC::__throw_dependent_sender_error<_Self>();
        }
        else
        {
          return STDEXEC::__throw_compile_time_error<
            _INVALID_ARGUMENTS_TO_WHEN_ALL_,
            __children_of<_Self, __qq<_WITH_PRETTY_SENDERS_>>,
            __fn_t<_WITH_ENVIRONMENT_, _Env>...>();
        }
      }

      static constexpr auto __get_env = []<class _State>(__ignore, _State const & __state) noexcept
        -> __env_t<env_of_t<typename _State::__receiver_t const &>>
      {
        return __when_all::__mk_env(STDEXEC::get_env(__state.__rcvr_), __state.__stop_source_);
      };

      static constexpr auto __get_state =
        []<class _Self, class _Receiver>(_Self&& __self, _Receiver&& __rcvr) noexcept
        -> __apply_result_t<__mk_state_fn_t<_Receiver>, _Self>
      {
        return __apply(__when_all::__mk_state_fn(static_cast<_Receiver&&>(__rcvr)),
                       static_cast<_Self&&>(__self));
      };

      static constexpr auto __start =
        []<class _State, class... _Operations>(_State& __state,
                                               _Operations&... __child_ops) noexcept -> void
      {
        // register stop callback:
        __state.__on_stop_.emplace(get_stop_token(STDEXEC::get_env(__state.__rcvr_)),
                                   __forward_stop_request<_State>{&__state});
        (STDEXEC::start(__child_ops), ...);
        if constexpr (sizeof...(__child_ops) == 0)
        {
          __state.__complete();
        }
      };

      template <class _State, class _Error>
      static constexpr void __set_error(_State& __state, _Error&& __err) noexcept
      {
        // Transition to the "error" state and switch on the prior state.
        // TODO: What memory orderings are actually needed here?
        switch (__state.__state_.exchange(__error))
        {
        case __started:
          // We must request stop. When the previous state is __error or __stopped, then stop has
          // already been requested.
          __state.__stop_source_.request_stop();
          [[fallthrough]];
        case __stopped:
          // We are the first child to complete with an error, so we must save the error. (Any
          // subsequent errors are ignored.)
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

      static constexpr auto __complete = []<class _Index, class _State, class _Set, class... _Args>(
                                           _Index,
                                           _State& __state,
                                           _Set,
                                           _Args&&... __args) noexcept -> void
      {
        using _ValuesTuple = decltype(_State::__values_);
        if constexpr (__same_as<_Set, set_error_t>)
        {
          __set_error(__state, static_cast<_Args&&>(__args)...);
        }
        else if constexpr (__same_as<_Set, set_stopped_t>)
        {
          __state_t __expected = __started;
          // Transition to the "stopped" state if and only if we're in the
          // "started" state. (If this fails, it's because we're in an
          // error state, which trumps cancellation.)
          if (__state.__state_.compare_exchange_strong(__expected, __stopped))
          {
            __state.__stop_source_.request_stop();
          }
        }
        else if constexpr (!__same_as<_ValuesTuple, __ignore>)
        {
          auto& __opt_values = STDEXEC::__get<_Index::value>(__state.__values_);
          using _Tuple       = __decayed_tuple<_Args...>;
          static_assert(__same_as<decltype(*__opt_values), _Tuple&>,
                        "One of the senders in this when_all() is fibbing about what types it "
                        "sends");
          if constexpr ((__nothrow_decay_copyable<_Args> && ...))
          {
            __opt_values.emplace(_Tuple{static_cast<_Args&&>(__args)...});
          }
          else
          {
            STDEXEC_TRY
            {
              __opt_values.emplace(_Tuple{static_cast<_Args&&>(__args)...});
            }
            STDEXEC_CATCH_ALL
            {
              __set_error(__state, std::current_exception());
            }
          }
        }

        __state.__arrive();
      };
    };

    struct __when_all_with_variant_impl : __sexpr_defaults
    {
      static constexpr auto __get_attrs =
        []<class... _Child>(__ignore, __ignore, _Child const &...) noexcept
      {
        return __when_all::__attrs<_Child...>{};
      };

      template <class _Sender, class... _Env>
      static consteval auto __get_completion_signatures()
      {
        using __sndr_t = __detail::__transform_sender_result_t<when_all_with_variant_t,
                                                               set_value_t,
                                                               _Sender,
                                                               env<>>;
        return STDEXEC::get_completion_signatures<__sndr_t, _Env...>();
      };
    };

    struct __transfer_when_all_impl : __sexpr_defaults
    {
      static constexpr auto __get_attrs =
        []<class _Scheduler, class... _Child>(__ignore,
                                              _Scheduler const & __sched,
                                              _Child const &...) noexcept
      {
        // TODO(ericniebler): check this use of __sched_attrs
        return __sched_attrs{__sched};
      };

      template <class _Sender, class... _Env>
      static consteval auto __get_completion_signatures()
      {
        using __sndr_t =
          __detail::__transform_sender_result_t<transfer_when_all_t, set_value_t, _Sender, env<>>;
        return STDEXEC::get_completion_signatures<__sndr_t, _Env...>();
      };
    };

    struct __transfer_when_all_with_variant_impl : __sexpr_defaults
    {
      static constexpr auto __get_attrs =
        []<class _Scheduler, class... _Child>(__ignore,
                                              _Scheduler const & __sched,
                                              _Child const &...) noexcept
      {
        return __sched_attrs{__sched};
      };

      template <class _Sender, class... _Env>
      static consteval auto __get_completion_signatures()
      {
        using __sndr_t = __detail::__transform_sender_result_t<transfer_when_all_with_variant_t,
                                                               set_value_t,
                                                               _Sender,
                                                               env<>>;
        return STDEXEC::get_completion_signatures<__sndr_t, _Env...>();
      };
    };
  }  // namespace __when_all

  template <>
  struct __sexpr_impl<when_all_t> : __when_all::__when_all_impl
  {};

  template <>
  struct __sexpr_impl<when_all_with_variant_t> : __when_all::__when_all_with_variant_impl
  {};

  template <>
  struct __sexpr_impl<transfer_when_all_t> : __when_all::__transfer_when_all_impl
  {};

  template <>
  struct __sexpr_impl<transfer_when_all_with_variant_t>
    : __when_all::__transfer_when_all_with_variant_impl
  {};
}  // namespace STDEXEC

#include "__epilogue.hpp"
