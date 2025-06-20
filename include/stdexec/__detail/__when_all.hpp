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
#include "__transform_sender.hpp"
#include "__tuple.hpp"
#include "__type_traits.hpp"
#include "__utility.hpp"
#include "__variant.hpp"

#include "../stop_token.hpp"

#include <atomic>
#include <exception>

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.when_all]
  // [execution.senders.adaptors.when_all_with_variant]
  namespace __when_all {
    enum __state_t {
      __started,
      __error,
      __stopped
    };

    struct __on_stop_request {
      inplace_stop_source& __stop_source_;

      void operator()() noexcept {
        __stop_source_.request_stop();
      }
    };

    template <class _Env>
    auto __mkenv(_Env&& __env, const inplace_stop_source& __stop_source) noexcept {
      return __env::__join(
        prop{get_stop_token, __stop_source.get_token()}, static_cast<_Env&&>(__env));
    }

    template <class _Env>
    using __env_t =
      decltype(__when_all::__mkenv(__declval<_Env>(), __declval<inplace_stop_source&>()));

    template <class _Sender, class _Env>
    concept __max1_sender =
      sender_in<_Sender, _Env>
      && __mvalid<__value_types_of_t, _Sender, _Env, __mconst<int>, __msingle_or<void>>;

    template <
      __mstring _Context = "In stdexec::when_all()..."_mstr,
      __mstring _Diagnostic =
        "The given sender can complete successfully in more that one way. "
        "Use stdexec::when_all_with_variant() instead."_mstr
    >
    struct _INVALID_WHEN_ALL_ARGUMENT_;

    template <class _Sender, class... _Env>
    using __too_many_value_completions_error = __mexception<
      _INVALID_WHEN_ALL_ARGUMENT_<>,
      _WITH_SENDER_<_Sender>,
      _WITH_ENVIRONMENT_<_Env>...
    >;

    template <class... _Args>
    using __all_nothrow_decay_copyable = __mbool<(__nothrow_decay_copyable<_Args> && ...)>;

    template <class _Error>
    using __set_error_t = completion_signatures<set_error_t(__decay_t<_Error>)>;

    template <class _Sender, class... _Env>
    using __nothrow_decay_copyable_results = __for_each_completion_signature<
      __completion_signatures_of_t<_Sender, _Env...>,
      __all_nothrow_decay_copyable,
      __mand_t
    >;

    template <class... _Env>
    struct __completions_t {
      template <class... _Senders>
      using __all_nothrow_decay_copyable_results =
        __mand<__nothrow_decay_copyable_results<_Senders, _Env...>...>;

      template <class _Sender, class _ValueTuple, class... _Rest>
      using __value_tuple_t = __minvoke<
        __if_c<
          (0 == sizeof...(_Rest)),
          __mconst<_ValueTuple>,
          __q<__too_many_value_completions_error>
        >,
        _Sender,
        _Env...
      >;

      template <class _Sender>
      using __single_values_of_t = __value_types_t<
        __completion_signatures_of_t<_Sender, _Env...>,
        __mtransform<__q<__decay_t>, __q<__types>>,
        __mbind_front_q<__value_tuple_t, _Sender>
      >;

      template <class... _Senders>
      using __set_values_sig_t = __meval<
        completion_signatures,
        __minvoke<__mconcat<__qf<set_value_t>>, __single_values_of_t<_Senders>...>
      >;

      template <class... _Senders>
      using __f = __meval<
        __concat_completion_signatures,
        __meval<__eptr_completion_if_t, __all_nothrow_decay_copyable_results<_Senders...>>,
        completion_signatures<set_stopped_t()>,
        __minvoke<__with_default<__qq<__set_values_sig_t>, completion_signatures<>>, _Senders...>,
        __transform_completion_signatures<
          __completion_signatures_of_t<_Senders, _Env...>,
          __mconst<completion_signatures<>>::__f,
          __set_error_t,
          completion_signatures<>,
          __concat_completion_signatures
        >...
      >;
    };

    template <class _Receiver, class _ValuesTuple>
    void __set_values(_Receiver& __rcvr, _ValuesTuple& __values) noexcept {
      __values.apply(
        [&]<class... OptTuples>(OptTuples&&... __opt_vals) noexcept -> void {
          __tup::__cat_apply(
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
        __with_default<
          __mtransform<__mbind_front_q<__values_opt_tuple_t, _Env>, __q<__tuple_for>>,
          __ignore
        >,
        _Senders...
      >;

      using __collect_errors = __mbind_front_q<__mset_insert, __mset<>>;

      using __errors_list = __minvoke<
        __mconcat<>,
        __if<
          __mand<__nothrow_decay_copyable_results<_Senders, _Env>...>,
          __types<>,
          __types<std::exception_ptr>
        >,
        __error_types_of_t<_Senders, __env_t<_Env>, __q<__types>>...
      >;

      using __errors_variant = __mapply<__q<__uniqued_variant_for>, __errors_list>;
    };

    struct _INVALID_ARGUMENTS_TO_WHEN_ALL_ { };

    template <class _ErrorsVariant, class _ValuesTuple, class _StopToken>
    struct __when_all_state {
      using __stop_callback_t = stop_callback_for_t<_StopToken, __on_stop_request>;

      template <class _Receiver>
      void __arrive(_Receiver& __rcvr) noexcept {
        if (1 == __count_.fetch_sub(1)) {
          __complete(__rcvr);
        }
      }

      template <class _Receiver>
      void __complete(_Receiver& __rcvr) noexcept {
        // Stop callback is no longer needed. Destroy it.
        __on_stop_.reset();
        // All child operations have completed and arrived at the barrier.
        switch (__state_.load(std::memory_order_relaxed)) {
        case __started:
          if constexpr (!same_as<_ValuesTuple, __ignore>) {
            // All child operations completed successfully:
            __when_all::__set_values(__rcvr, __values_);
          }
          break;
        case __error:
          if constexpr (!__same_as<_ErrorsVariant, __variant_for<>>) {
            // One or more child operations completed with an error:
            __errors_.visit(
              __mk_completion_fn(set_error, __rcvr), static_cast<_ErrorsVariant&&>(__errors_));
          }
          break;
        case __stopped:
          stdexec::set_stopped(static_cast<_Receiver&&>(__rcvr));
          break;
        default:;
        }
      }

      std::atomic<std::size_t> __count_;
      inplace_stop_source __stop_source_{};
      // Could be non-atomic here and atomic_ref everywhere except __completion_fn
      std::atomic<__state_t> __state_{__started};
      _ErrorsVariant __errors_{};
      STDEXEC_ATTRIBUTE(no_unique_address) _ValuesTuple __values_ { };
      __optional<__stop_callback_t> __on_stop_{};
    };

    template <class _Env>
    static auto __mk_state_fn(const _Env&) noexcept {
      return []<__max1_sender<__env_t<_Env>>... _Child>(__ignore, __ignore, _Child&&...) {
        using _Traits = __traits<_Env, _Child...>;
        using _ErrorsVariant = typename _Traits::__errors_variant;
        using _ValuesTuple = typename _Traits::__values_tuple;
        using _State = __when_all_state<_ErrorsVariant, _ValuesTuple, stop_token_of_t<_Env>>;
        return _State{sizeof...(_Child)};
      };
    }

    template <class _Env>
    using __mk_state_fn_t = decltype(__when_all::__mk_state_fn(__declval<_Env>()));

    struct when_all_t {
      template <sender... _Senders>
        requires __has_common_domain<_Senders...>
      auto operator()(_Senders&&... __sndrs) const -> __well_formed_sender auto {
        auto __domain = __common_domain_t<_Senders...>();
        return stdexec::transform_sender(
          __domain, __make_sexpr<when_all_t>(__(), static_cast<_Senders&&>(__sndrs)...));
      }
    };

    struct __when_all_impl : __sexpr_defaults {
      template <class _Self, class _Env>
      using __error_t = __mexception<
        _INVALID_ARGUMENTS_TO_WHEN_ALL_,
        __children_of<_Self, __q<_WITH_SENDERS_>>,
        _WITH_ENVIRONMENT_<_Env>
      >;

      template <class _Self, class... _Env>
      using __completions = __children_of<_Self, __completions_t<__env_t<_Env>...>>;

      static constexpr auto get_attrs = []<class... _Child>(__ignore, const _Child&...) noexcept {
        using _Domain = __common_domain_t<_Child...>;
        if constexpr (__same_as<_Domain, default_domain>) {
          return env();
        } else {
          return prop{get_domain, _Domain()};
        }
      };

      static constexpr auto get_completion_signatures =
        []<class _Self, class... _Env>(_Self&&, _Env&&...) noexcept {
          static_assert(sender_expr_for<_Self, when_all_t>);
          return __minvoke<__mtry_catch<__q<__completions>, __q<__error_t>>, _Self, _Env...>();
        };

      static constexpr auto get_env =
        []<class _State, class _Receiver>(
          __ignore,
          _State& __state,
          const _Receiver& __rcvr) noexcept -> __env_t<env_of_t<const _Receiver&>> {
        return __mkenv(stdexec::get_env(__rcvr), __state.__stop_source_);
      };

      static constexpr auto get_state =
        []<class _Self, class _Receiver>(_Self&& __self, _Receiver& __rcvr)
        -> __sexpr_apply_result_t<_Self, __mk_state_fn_t<env_of_t<_Receiver>>> {
        return __sexpr_apply(
          static_cast<_Self&&>(__self), __when_all::__mk_state_fn(stdexec::get_env(__rcvr)));
      };

      static constexpr auto start = []<class _State, class _Receiver, class... _Operations>(
                                      _State& __state,
                                      _Receiver& __rcvr,
                                      _Operations&... __child_ops) noexcept -> void {
        // register stop callback:
        __state.__on_stop_.emplace(
          get_stop_token(stdexec::get_env(__rcvr)), __on_stop_request{__state.__stop_source_});
        if (__state.__stop_source_.stop_requested()) {
          // Stop has already been requested. Don't bother starting
          // the child operations.
          stdexec::set_stopped(static_cast<_Receiver&&>(__rcvr));
        } else {
          (stdexec::start(__child_ops), ...);
          if constexpr (sizeof...(__child_ops) == 0) {
            __state.__complete(__rcvr);
          }
        }
      };

      template <class _State, class _Receiver, class _Error>
      static void __set_error(_State& __state, _Receiver&, _Error&& __err) noexcept {
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

      static constexpr auto complete =
        []<class _Index, class _State, class _Receiver, class _Set, class... _Args>(
          _Index,
          _State& __state,
          _Receiver& __rcvr,
          _Set,
          _Args&&... __args) noexcept -> void {
        using _ValuesTuple = decltype(_State::__values_);
        if constexpr (__same_as<_Set, set_error_t>) {
          __set_error(__state, __rcvr, static_cast<_Args&&>(__args)...);
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
            auto& __opt_values = _ValuesTuple::template __get<__v<_Index>>(__state.__values_);
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
                __set_error(__state, __rcvr, std::current_exception());
              }
            }
          }
        }

        __state.__arrive(__rcvr);
      };
    };

    struct when_all_with_variant_t {
      template <sender... _Senders>
        requires __has_common_domain<_Senders...>
      auto operator()(_Senders&&... __sndrs) const -> __well_formed_sender auto {
        auto __domain = __common_domain_t<_Senders...>();
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<when_all_with_variant_t>(__(), static_cast<_Senders&&>(__sndrs)...));
      }

      template <class _Sender, class _Env>
      static auto transform_sender(_Sender&& __sndr, const _Env&) {
        // transform the when_all_with_variant into a regular when_all (looking for
        // early when_all customizations), then transform it again to look for
        // late customizations.
        return __sexpr_apply(
          static_cast<_Sender&&>(__sndr),
          [&]<class... _Child>(__ignore, __ignore, _Child&&... __child) {
            return when_all_t()(into_variant(static_cast<_Child&&>(__child))...);
          });
      }
    };

    struct __when_all_with_variant_impl : __sexpr_defaults {
      static constexpr auto get_attrs = []<class... _Child>(__ignore, const _Child&...) noexcept {
        using _Domain = __common_domain_t<_Child...>;
        if constexpr (same_as<_Domain, default_domain>) {
          return env();
        } else {
          return prop{get_domain, _Domain()};
        }
      };

      static constexpr auto get_completion_signatures = []<class _Sender>(_Sender&&) noexcept
        -> __completion_signatures_of_t<transform_sender_result_t<default_domain, _Sender, env<>>> {
        return {};
      };
    };

    struct transfer_when_all_t {
      template <scheduler _Scheduler, sender... _Senders>
        requires __has_common_domain<_Senders...>
      auto
        operator()(_Scheduler __sched, _Senders&&... __sndrs) const -> __well_formed_sender auto {
        auto __domain = query_or(get_domain, __sched, default_domain());
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<transfer_when_all_t>(
            static_cast<_Scheduler&&>(__sched), static_cast<_Senders&&>(__sndrs)...));
      }

      template <class _Sender, class _Env>
      static auto transform_sender(_Sender&& __sndr, const _Env&) {
        // transform the transfer_when_all into a regular transform | when_all
        // (looking for early customizations), then transform it again to look for
        // late customizations.
        return __sexpr_apply(
          static_cast<_Sender&&>(__sndr),
          [&]<class _Data, class... _Child>(__ignore, _Data&& __data, _Child&&... __child) {
            return continues_on(
              when_all_t()(static_cast<_Child&&>(__child)...), static_cast<_Data&&>(__data));
          });
      }
    };

    struct __transfer_when_all_impl : __sexpr_defaults {
      static constexpr auto get_attrs = []<class _Scheduler, class... _Child>(
                                          const _Scheduler& __sched,
                                          const _Child&...) noexcept {
        using __sndr_t = __call_result_t<when_all_t, _Child...>;
        using __domain_t = __detail::__early_domain_of_t<__sndr_t, __none_such>;
        return __sched_attrs{std::cref(__sched), __domain_t{}};
      };

      static constexpr auto get_completion_signatures = []<class _Sender>(_Sender&&) noexcept
        -> __completion_signatures_of_t<transform_sender_result_t<default_domain, _Sender, env<>>> {
        return {};
      };
    };

    struct transfer_when_all_with_variant_t {
      template <scheduler _Scheduler, sender... _Senders>
        requires __has_common_domain<_Senders...>
      auto
        operator()(_Scheduler&& __sched, _Senders&&... __sndrs) const -> __well_formed_sender auto {
        auto __domain = query_or(get_domain, __sched, default_domain());
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<transfer_when_all_with_variant_t>(
            static_cast<_Scheduler&&>(__sched), static_cast<_Senders&&>(__sndrs)...));
      }

      template <class _Sender, class _Env>
      static auto transform_sender(_Sender&& __sndr, const _Env&) {
        // transform the transfer_when_all_with_variant into regular transform_when_all
        // and into_variant calls/ (looking for early customizations), then transform it
        // again to look for late customizations.
        return __sexpr_apply(
          static_cast<_Sender&&>(__sndr),
          [&]<class _Data, class... _Child>(__ignore, _Data&& __data, _Child&&... __child) {
            return transfer_when_all_t()(
              static_cast<_Data&&>(__data), into_variant(static_cast<_Child&&>(__child))...);
          });
      }
    };

    struct __transfer_when_all_with_variant_impl : __sexpr_defaults {
      static constexpr auto get_attrs = []<class _Scheduler, class... _Child>(
                                          const _Scheduler& __sched,
                                          const _Child&...) noexcept {
        using __sndr_t = __call_result_t<when_all_with_variant_t, _Child...>;
        using __domain_t = __detail::__early_domain_of_t<__sndr_t, __none_such>;
        return __sched_attrs{std::cref(__sched), __domain_t{}};
      };

      static constexpr auto get_completion_signatures = []<class _Sender>(_Sender&&) noexcept
        -> __completion_signatures_of_t<transform_sender_result_t<default_domain, _Sender, env<>>> {
        return {};
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
} // namespace stdexec
