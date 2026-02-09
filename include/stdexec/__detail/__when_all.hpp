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

namespace STDEXEC {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.when_all]
  // [execution.senders.adaptors.when_all_with_variant]
  namespace __when_all {
    struct when_all_t;

    enum __state_t {
      __started,
      __error,
      __stopped
    };

    template <class _Env>
    constexpr auto __mk_env(_Env&& __env, const inplace_stop_source& __stop_source) noexcept {
      return __env::__join(
        prop{get_stop_token, __stop_source.get_token()}, static_cast<_Env&&>(__env));
    }

    template <class _Env>
    using __env_t =
      decltype(__when_all::__mk_env(__declval<_Env>(), __declval<inplace_stop_source&>()));

    template <class _Sender, class _Env>
    concept __max1_sender =
      sender_in<_Sender, _Env>
      && __minvocable_q<__value_types_of_t, _Sender, _Env, __mconst<int>, __msingle_or<void>>;

    struct _THE_GIVEN_SENDER_CAN_COMPLETE_SUCCESSFULLY_IN_MORE_THAN_ONE_WAY_ { };
    struct _USE_WHEN_ALL_WITH_VARIANT_INSTEAD_ { };

    template <class _Sender, class... _Env>
    using __too_many_value_completions_error_t = __mexception<
      _WHAT_(_INVALID_ARGUMENT_),
      _WHERE_(_IN_ALGORITHM_, when_all_t),
      _WHY_(_THE_GIVEN_SENDER_CAN_COMPLETE_SUCCESSFULLY_IN_MORE_THAN_ONE_WAY_),
      _TO_FIX_THIS_ERROR_(_USE_WHEN_ALL_WITH_VARIANT_INSTEAD_),
      _WITH_PRETTY_SENDER_<_Sender>,
      __fn_t<_WITH_ENVIRONMENT_, _Env>...
    >;

    template <class _Error>
    using __set_error_t = completion_signatures<set_error_t(__decay_t<_Error>)>;

    template <class _Sender, class... _Env>
    using __nothrow_decay_copyable_results_t = __cmplsigs::__partitions_of_t<
      __completion_signatures_of_t<_Sender, _Env...>
    >::__nothrow_decay_copyable::__all;

    template <class... _Env>
    struct __completions {
      // TODO(ericniebler): check that all senders have a common completion domain
      template <class... _Senders>
      using __all_nothrow_decay_copyable_results_t =
        __mand<__nothrow_decay_copyable_results_t<_Senders, _Env...>...>;

      template <class _Sender, class _ValueTuple, class... _Rest>
      using __value_tuple_t = __minvoke<
        __if_c<
          (0 == sizeof...(_Rest)),
          __mconst<_ValueTuple>,
          __q<__too_many_value_completions_error_t>
        >,
        _Sender,
        _Env...
      >;

      template <class _Sender>
      using __single_values_of_t = __value_types_t<
        __completion_signatures_of_t<_Sender, _Env...>,
        __mtransform<__q<__decay_t>, __q<__mlist>>,
        __mbind_front_q<__value_tuple_t, _Sender>
      >;

      template <class... _Senders>
      using __set_values_sig_t = __minvoke_q<
        completion_signatures,
        __minvoke<__mconcat<__qf<set_value_t>>, __single_values_of_t<_Senders>...>
      >;

      template <class... _Senders>
      using __f = __minvoke_q<
        __concat_completion_signatures_t,
        __minvoke_q<__eptr_completion_unless_t, __all_nothrow_decay_copyable_results_t<_Senders...>>,
        __minvoke<__mwith_default<__qq<__set_values_sig_t>, completion_signatures<>>, _Senders...>,
        __transform_completion_signatures_t<
          __completion_signatures_of_t<_Senders, _Env...>,
          __mconst<completion_signatures<>>::__f,
          __set_error_t,
          completion_signatures<set_stopped_t()>,
          __concat_completion_signatures_t
        >...
      >;
    };

    template <class _Receiver, class _ValuesTuple>
    constexpr void __set_values(_Receiver& __rcvr, _ValuesTuple& __values) noexcept {
      STDEXEC::__apply(
        [&]<class... OptTuples>(OptTuples&&... __opt_vals) noexcept -> void {
          STDEXEC::__cat_apply(
            __mk_completion_fn(set_value, __rcvr), *static_cast<OptTuples&&>(__opt_vals)...);
        },
        static_cast<_ValuesTuple&&>(__values));
    }

    template <class _Env, class _Sender>
    using __values_opt_tuple_t =
      value_types_of_t<_Sender, __env_t<_Env>, __decayed_tuple, __optional>;

    template <class _Env, __max1_sender<__env_t<_Env>>... _Senders>
    struct __traits {
      // tuple<optional<tuple<Vs1...>>, optional<tuple<Vs2...>>, ...>
      using __values_tuple = __minvoke<
        __mwith_default<
          __mtransform<__mbind_front_q<__values_opt_tuple_t, _Env>, __q<__tuple>>,
          __ignore
        >,
        _Senders...
      >;

      using __collect_errors = __mbind_front_q<__mset_insert, __mset<>>;

      using __errors_list = __minvoke<
        __mconcat<>,
        __if<
          __mand<__nothrow_decay_copyable_results_t<_Senders, _Env>...>,
          __mlist<>,
          __mlist<std::exception_ptr>
        >,
        __error_types_of_t<_Senders, __env_t<_Env>, __q<__mlist>>...
      >;

      using __errors_variant = __mapply<__q<__uniqued_variant>, __errors_list>;
    };

    struct _INVALID_ARGUMENTS_TO_WHEN_ALL_ { };

    template <class _State>
    struct __forward_stop_request {
      constexpr void operator()() const noexcept {
        // Temporarily increment the count to avoid concurrent/recursive arrivals to
        // pull the rug under our feet. Relaxed memory order is fine here.
        __state_->__count_.fetch_add(1, __std::memory_order_relaxed);

        __state_t __expected = __started;
        // Transition to the "stopped" state if and only if we're in the
        // "started" state. (If this fails, it's because we're in an
        // error state, which trumps cancellation.)
        if (__state_->__state_.compare_exchange_strong(__expected, __stopped)) {
          __state_->__stop_source_.request_stop();
        }

        // Arrive in order to decrement the count again and complete if needed.
        __state_->__arrive();
      }

      _State* __state_;
    };

    template <class _ErrorsVariant, class _ValuesTuple, class _Receiver, bool _SendsStopped>
    struct __state {
      using __receiver_t = _Receiver;
      using __stop_callback_t =
        stop_callback_for_t<stop_token_of_t<env_of_t<_Receiver>>, __forward_stop_request<__state>>;

      constexpr void __arrive() noexcept {
        if (1 == __count_.fetch_sub(1, __std::memory_order_acq_rel)) {
          __complete();
        }
      }

      constexpr void __complete() noexcept {
        // Stop callback is no longer needed. Destroy it.
        __on_stop_.reset();
        // All child operations have completed and arrived at the barrier.
        switch (__state_.load(__std::memory_order_relaxed)) {
        case __started:
          if constexpr (!__std::same_as<_ValuesTuple, __ignore>) {
            // All child operations completed successfully:
            __when_all::__set_values(__rcvr_, __values_);
          }
          break;
        case __error:
          if constexpr (!__same_as<_ErrorsVariant, __variant<>>) {
            // One or more child operations completed with an error:
            STDEXEC::__visit(
              __mk_completion_fn(set_error, __rcvr_), static_cast<_ErrorsVariant&&>(__errors_));
          }
          break;
        case __stopped:
          if constexpr (_SendsStopped) {
            STDEXEC::set_stopped(static_cast<_Receiver&&>(__rcvr_));
          } else {
            STDEXEC_UNREACHABLE();
          }
          break;
        default:;
        }
      }

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Receiver __rcvr_;
      __std::atomic<std::size_t> __count_;
      inplace_stop_source __stop_source_{};
      // Could be non-atomic here and atomic_ref everywhere except __completion_fn
      __std::atomic<__state_t> __state_{__started};
      _ErrorsVariant __errors_{__no_init};
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _ValuesTuple __values_{};
      __optional<__stop_callback_t> __on_stop_{};
    };

    template <class... _Senders>
    struct __attrs {
      template <class _Tag, class... _Env>
      using __when_all_domain_t =
        __common_domain_t<__completion_domain_of_t<set_value_t, _Senders, _Env...>...>;

      template <class... _Env>
      [[nodiscard]]
      constexpr auto query(get_completion_domain_t<set_value_t>, const _Env&...) const noexcept
        -> __when_all_domain_t<set_value_t, _Env...>;

      template <class... _Env>
      [[nodiscard]]
      constexpr auto query(get_completion_domain_t<set_error_t>, const _Env&...) const noexcept
        -> __common_domain_t<
          __when_all_domain_t<set_value_t, _Env...>,
          __when_all_domain_t<set_error_t, _Env...>,
          __when_all_domain_t<set_stopped_t, _Env...>
        >;

      template <class... _Env>
      [[nodiscard]]
      constexpr auto query(get_completion_domain_t<set_stopped_t>, const _Env&...) const noexcept
        -> __common_domain_t<
          __when_all_domain_t<set_value_t, _Env...>,
          __when_all_domain_t<set_stopped_t, _Env...>
        >;

      template <class _Tag, class... _Env>
      [[nodiscard]]
      constexpr auto query(get_completion_behavior_t<_Tag>, const _Env&...) const noexcept {
        return completion_behavior::weakest(
          STDEXEC::get_completion_behavior<_Tag, _Senders, _Env...>()...);
      }
    };

    // A when_all with no senders completes inline with no values.
    template <>
    struct __attrs<> {
      [[nodiscard]]
      constexpr auto query(get_completion_behavior_t<set_value_t>) const noexcept {
        return completion_behavior::inline_completion;
      }

      [[nodiscard]]
      constexpr auto query(get_completion_behavior_t<set_stopped_t>) const noexcept {
        return completion_behavior::inline_completion;
      }
    };

    template <class _Receiver>
    static constexpr auto __mk_state_fn(_Receiver&& __rcvr) noexcept {
      using __env_of_t = env_of_t<_Receiver>;
      return [&]<__max1_sender<__env_t<__env_of_t>>... _Child>(
               __ignore, __ignore, _Child&&...) noexcept {
        using _Traits = __traits<__env_of_t, _Child...>;
        using _ErrorsVariant = _Traits::__errors_variant;
        using _ValuesTuple = _Traits::__values_tuple;
        using _State = __state<
          _ErrorsVariant,
          _ValuesTuple,
          _Receiver,
          (sends_stopped<_Child, __env_of_t> || ...)
        >;
        return _State{static_cast<_Receiver&&>(__rcvr), sizeof...(_Child)};
      };
    }

    template <class _Receiver>
    using __mk_state_fn_t = decltype(__when_all::__mk_state_fn(__declval<_Receiver>()));

    struct when_all_t {
      template <sender... _Senders>
      constexpr auto operator()(_Senders&&... __sndrs) const -> __well_formed_sender auto {
        return __make_sexpr<when_all_t>(__(), static_cast<_Senders&&>(__sndrs)...);
      }
    };

    struct __when_all_impl : __sexpr_defaults {
      template <class _Self, class... _Env>
      using __completions_t = __children_of<_Self, __when_all::__completions<__env_t<_Env>...>>;

      static constexpr auto get_attrs =
        []<class... _Child>(__ignore, __ignore, const _Child&...) noexcept {
          return __when_all::__attrs<_Child...>{};
        };

      template <class _Self, class... _Env>
      static consteval auto get_completion_signatures() {
        static_assert(sender_expr_for<_Self, when_all_t>);
        if constexpr (__minvocable_q<__completions_t, _Self, _Env...>) {
          // TODO: update this to use constant evaluation:
          return __completions_t<_Self, _Env...>{};
        } else if constexpr (sizeof...(_Env) == 0) {
          return STDEXEC::__dependent_sender<_Self>();
        } else {
          return STDEXEC::__throw_compile_time_error<
            _INVALID_ARGUMENTS_TO_WHEN_ALL_,
            __children_of<_Self, __qq<_WITH_PRETTY_SENDERS_>>,
            __fn_t<_WITH_ENVIRONMENT_, _Env>...
          >();
        }
      }

      static constexpr auto get_env = []<class _State>(__ignore, const _State& __state) noexcept
        -> __env_t<env_of_t<const typename _State::__receiver_t&>> {
        return __when_all::__mk_env(STDEXEC::get_env(__state.__rcvr_), __state.__stop_source_);
      };

      static constexpr auto get_state =
        []<class _Self, class _Receiver>(_Self&& __self, _Receiver&& __rcvr) noexcept
        -> __apply_result_t<__mk_state_fn_t<_Receiver>, _Self> {
        return __apply(
          __when_all::__mk_state_fn(static_cast<_Receiver&&>(__rcvr)),
          static_cast<_Self&&>(__self));
      };

      static constexpr auto start = []<class _State, class... _Operations>(
                                      _State& __state,
                                      _Operations&... __child_ops) noexcept -> void {
        // register stop callback:
        __state.__on_stop_.emplace(
          get_stop_token(STDEXEC::get_env(__state.__rcvr_)),
          __forward_stop_request<_State>{&__state});
        (STDEXEC::start(__child_ops), ...);
        if constexpr (sizeof...(__child_ops) == 0) {
          __state.__complete();
        }
      };

      template <class _State, class _Error>
      static constexpr void __set_error(_State& __state, _Error&& __err) noexcept {
        // Transition to the "error" state and switch on the prior state.
        // TODO: What memory orderings are actually needed here?
        switch (__state.__state_.exchange(__error)) {
        case __started:
          // We must request stop. When the previous state is __error or __stopped, then stop has
          // already been requested.
          __state.__stop_source_.request_stop();
          [[fallthrough]];
        case __stopped:
          // We are the first child to complete with an error, so we must save the error. (Any
          // subsequent errors are ignored.)
          if constexpr (__nothrow_decay_copyable<_Error>) {
            __state.__errors_.template emplace<__decay_t<_Error>>(static_cast<_Error&&>(__err));
          } else {
            STDEXEC_TRY {
              __state.__errors_.template emplace<__decay_t<_Error>>(static_cast<_Error&&>(__err));
            }
            STDEXEC_CATCH_ALL {
              __state.__errors_.template emplace<std::exception_ptr>(std::current_exception());
            }
          }
          break;
        case __error:; // We're already in the "error" state. Ignore the error.
        }
      }

      static constexpr auto complete = []<class _Index, class _State, class _Set, class... _Args>(
                                         _Index,
                                         _State& __state,
                                         _Set,
                                         _Args&&... __args) noexcept -> void {
        using _ValuesTuple = decltype(_State::__values_);
        if constexpr (__same_as<_Set, set_error_t>) {
          __set_error(__state, static_cast<_Args&&>(__args)...);
        } else if constexpr (__same_as<_Set, set_stopped_t>) {
          __state_t __expected = __started;
          // Transition to the "stopped" state if and only if we're in the
          // "started" state. (If this fails, it's because we're in an
          // error state, which trumps cancellation.)
          if (__state.__state_.compare_exchange_strong(__expected, __stopped)) {
            __state.__stop_source_.request_stop();
          }
        } else if constexpr (!__same_as<_ValuesTuple, __ignore>) {
          // We only need to bother recording the completion values
          // if we're not already in the "error" or "stopped" state.
          if (__state.__state_.load() == __started) {
            auto& __opt_values = STDEXEC::__get<_Index::value>(__state.__values_);
            using _Tuple = __decayed_tuple<_Args...>;
            static_assert(
              __same_as<decltype(*__opt_values), _Tuple&>,
              "One of the senders in this when_all() is fibbing about what types it sends");
            if constexpr ((__nothrow_decay_copyable<_Args> && ...)) {
              __opt_values.emplace(_Tuple{static_cast<_Args&&>(__args)...});
            } else {
              STDEXEC_TRY {
                __opt_values.emplace(_Tuple{static_cast<_Args&&>(__args)...});
              }
              STDEXEC_CATCH_ALL {
                __set_error(__state, std::current_exception());
              }
            }
          }
        }

        __state.__arrive();
      };
    };

    struct when_all_with_variant_t {
      template <sender... _Senders>
      constexpr auto operator()(_Senders&&... __sndrs) const -> __well_formed_sender auto {
        return __make_sexpr<when_all_with_variant_t>(__(), static_cast<_Senders&&>(__sndrs)...);
      }

      template <class _Sender>
      static constexpr auto transform_sender(set_value_t, _Sender&& __sndr, __ignore) {
        // transform when_all_with_variant(sndrs...) into when_all(into_variant(sndrs)...).
        return __apply(
          [&]<class... _Child>(__ignore, __ignore, _Child&&... __child) {
            return when_all_t()(into_variant(static_cast<_Child&&>(__child))...);
          },
          static_cast<_Sender&&>(__sndr));
      }
    };

    struct __when_all_with_variant_impl : __sexpr_defaults {
      static constexpr auto get_attrs =
        []<class... _Child>(__ignore, __ignore, const _Child&...) noexcept {
          return __when_all::__attrs<_Child...>{};
        };

      template <class _Sender, class... _Env>
      static consteval auto get_completion_signatures() {
        using __sndr_t = __detail::__transform_sender_result_t<
          when_all_with_variant_t,
          set_value_t,
          _Sender,
          env<>
        >;
        return STDEXEC::get_completion_signatures<__sndr_t, _Env...>();
      };
    };

    struct transfer_when_all_t {
      template <scheduler _Scheduler, sender... _Senders>
      constexpr auto
        operator()(_Scheduler __sched, _Senders&&... __sndrs) const -> __well_formed_sender auto {
        return __make_sexpr<transfer_when_all_t>(
          static_cast<_Scheduler&&>(__sched), static_cast<_Senders&&>(__sndrs)...);
      }

      template <class _Sender>
      static constexpr auto transform_sender(set_value_t, _Sender&& __sndr, __ignore) {
        // transform transfer_when_all(sch, sndrs...) into
        // continues_on(when_all(sndrs...), sch).
        return __apply(
          [&]<class _Data, class... _Child>(__ignore, _Data&& __data, _Child&&... __child) {
            return continues_on(
              when_all_t()(static_cast<_Child&&>(__child)...), static_cast<_Data&&>(__data));
          },
          static_cast<_Sender&&>(__sndr));
      }
    };

    struct __transfer_when_all_impl : __sexpr_defaults {
      static constexpr auto get_attrs = []<class _Scheduler, class... _Child>(
                                          __ignore,
                                          const _Scheduler& __sched,
                                          const _Child&...) noexcept {
        // TODO(ericniebler): check this use of __sched_attrs
        return __sched_attrs{std::cref(__sched)};
      };

      template <class _Sender, class... _Env>
      static consteval auto get_completion_signatures() {
        using __sndr_t =
          __detail::__transform_sender_result_t<transfer_when_all_t, set_value_t, _Sender, env<>>;
        return STDEXEC::get_completion_signatures<__sndr_t, _Env...>();
      };
    };

    struct transfer_when_all_with_variant_t {
      template <scheduler _Scheduler, sender... _Senders>
      constexpr auto
        operator()(_Scheduler&& __sched, _Senders&&... __sndrs) const -> __well_formed_sender auto {
        return __make_sexpr<transfer_when_all_with_variant_t>(
          static_cast<_Scheduler&&>(__sched), static_cast<_Senders&&>(__sndrs)...);
      }

      template <class _Sender>
      static constexpr auto transform_sender(set_value_t, _Sender&& __sndr, __ignore) {
        // transform the transfer_when_all_with_variant(sch, sndrs...) into
        // transfer_when_all(sch, into_variant(sndrs...))
        return __apply(
          [&]<class _Data, class... _Child>(__ignore, _Data&& __data, _Child&&... __child) {
            return transfer_when_all_t()(
              static_cast<_Data&&>(__data), into_variant(static_cast<_Child&&>(__child))...);
          },
          static_cast<_Sender&&>(__sndr));
      }
    };

    struct __transfer_when_all_with_variant_impl : __sexpr_defaults {
      static constexpr auto get_attrs = []<class _Scheduler, class... _Child>(
                                          __ignore,
                                          const _Scheduler& __sched,
                                          const _Child&...) noexcept {
        return __sched_attrs{std::cref(__sched)};
      };

      template <class _Sender, class... _Env>
      static consteval auto get_completion_signatures() {
        using __sndr_t = __detail::__transform_sender_result_t<
          transfer_when_all_with_variant_t,
          set_value_t,
          _Sender,
          env<>
        >;
        return STDEXEC::get_completion_signatures<__sndr_t, _Env...>();
      };
    };
  } // namespace __when_all

  using __when_all::when_all_t;
  inline constexpr when_all_t when_all{};

  using __when_all::when_all_with_variant_t;
  inline constexpr when_all_with_variant_t when_all_with_variant{};

  using __when_all::transfer_when_all_t;
  inline constexpr transfer_when_all_t transfer_when_all{};

  using __when_all::transfer_when_all_with_variant_t;
  inline constexpr transfer_when_all_with_variant_t transfer_when_all_with_variant{};

  template <>
  struct __sexpr_impl<when_all_t> : __when_all::__when_all_impl { };

  template <>
  struct __sexpr_impl<when_all_with_variant_t> : __when_all::__when_all_with_variant_impl { };

  template <>
  struct __sexpr_impl<transfer_when_all_t> : __when_all::__transfer_when_all_impl { };

  template <>
  struct __sexpr_impl<transfer_when_all_with_variant_t>
    : __when_all::__transfer_when_all_with_variant_impl { };
} // namespace STDEXEC
