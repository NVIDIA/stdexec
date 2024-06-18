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
#include "__diagnostics.hpp"
#include "__domain.hpp"
#include "__env.hpp"
#include "__inline_scheduler.hpp"
#include "__meta.hpp"
#include "__receiver_ref.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__senders.hpp"
#include "__tag_invoke.hpp"
#include "__transform_sender.hpp"
#include "__transform_completion_signatures.hpp"
#include "__variant.hpp"

#include <exception>

namespace stdexec {
  //////////////////////////////////////////////////////////////////////////////
  // [exec.let]
  namespace __let {
    // A dummy scheduler that is used by the metaprogramming below when the input sender doesn't
    // have a completion scheduler.
    struct __unknown_scheduler {
      struct __env {
        static constexpr bool query(__is_scheduler_affine_t) noexcept {
          return true;
        }

        constexpr auto query(get_completion_scheduler_t<set_value_t>) const noexcept {
          return __unknown_scheduler{};
        }
      };

      struct __sender {
        using sender_concept = sender_t;

        constexpr auto get_env() const noexcept -> __env {
          return __env();
        }
      };

      auto schedule() const noexcept {
        return __sender();
      }

      bool operator==(const __unknown_scheduler&) const noexcept = default;
    };

    inline constexpr auto __get_rcvr = [](auto& __op_state) noexcept -> decltype(auto) {
      return (__op_state.__rcvr_);
    };

    inline constexpr auto __get_env = [](auto& __op_state) noexcept -> decltype(auto) {
      return __op_state.__state_.__get_env(__op_state.__rcvr_);
    };

    template <class _Set, class _Domain = dependent_domain>
    struct __let_t;

    template <class _Set>
    inline constexpr __mstring __in_which_let_msg{"In stdexec::let_value(Sender, Function)..."};

    template <>
    inline constexpr __mstring __in_which_let_msg<set_error_t>{
      "In stdexec::let_error(Sender, Function)..."};

    template <>
    inline constexpr __mstring __in_which_let_msg<set_stopped_t>{
      "In stdexec::let_stopped(Sender, Function)..."};

    template <class _Set>
    using __on_not_callable = __callable_error<__in_which_let_msg<_Set>>;

    template <class _Receiver, class _Scheduler>
    struct __receiver_with_sched {
      using receiver_concept = receiver_t;
      _Receiver __rcvr_;
      _Scheduler __sched_;

      template <class... _As>
      void set_value(_As&&... __as) noexcept {
        stdexec::set_value(static_cast<_Receiver&&>(__rcvr_), static_cast<_As&&>(__as)...);
      }

      template <class _Error>
      void set_error(_Error&& __err) noexcept {
        stdexec::set_error(static_cast<_Receiver&&>(__rcvr_), static_cast<_Error&&>(__err));
      }

      void set_stopped() noexcept {
        stdexec::set_stopped(static_cast<_Receiver&&>(__rcvr_));
      }

      auto get_env() const noexcept {
        return __env::__join(
          __env::__with(__sched_, get_scheduler),
          __env::__without(stdexec::get_env(__rcvr_), get_domain));
      }
    };

    template <class _Receiver, class _Scheduler>
    __receiver_with_sched(_Receiver, _Scheduler) -> __receiver_with_sched<_Receiver, _Scheduler>;

    // If the input sender knows its completion scheduler, make it the current scheduler
    // in the environment seen by the result sender.
    template <class _Scheduler, class _Env>
    using __result_env_t = __if_c<
      __is_scheduler_affine<schedule_result_t<_Scheduler>>,
      _Env,
      __env::__join_t< //
        __env::__with<_Scheduler, get_scheduler_t>,
        __env::__without_t<_Env, get_domain_t>>>;

    template <__mstring _Where, __mstring _What>
    struct _FUNCTION_MUST_RETURN_A_VALID_SENDER_IN_THE_CURRENT_ENVIRONMENT_ { };

#if STDEXEC_NVHPC()
    template <class _Sender, class _Set, class... _Env>
    struct __bad_result_sender_ {
      using __t = __mexception<
        _FUNCTION_MUST_RETURN_A_VALID_SENDER_IN_THE_CURRENT_ENVIRONMENT_<
          __in_which_let_msg<_Set>,
          "The function must return a valid sender for the current environment"_mstr>,
        _WITH_SENDER_<_Sender>,
        _WITH_ENVIRONMENT_<_Env>...>;
    };
    template <class _Sender, class _Set, class... _Env>
    using __bad_result_sender = __t<__bad_result_sender_<_Sender, _Set, _Env...>>;
#else
    template <class _Sender, class _Set, class... _Env>
    using __bad_result_sender = __mexception<
      _FUNCTION_MUST_RETURN_A_VALID_SENDER_IN_THE_CURRENT_ENVIRONMENT_<
        __in_which_let_msg<_Set>,
        "The function must return a valid sender for the current environment"_mstr>,
      _WITH_SENDER_<_Sender>,
      _WITH_ENVIRONMENT_<_Env>...>;
#endif

    template <class _Sender, class... _Env>
    concept __potentially_valid_sender_in =
      sender_in<_Sender, _Env...> || (sender<_Sender> && (sizeof...(_Env) == 0));

    template <class _Set, class _Sender, class... _Env>
    using __ensure_sender = //
      __minvoke_if_c<
        __potentially_valid_sender_in<_Sender, _Env...>,
        __q<__midentity>,
        __mbind_back_q<__bad_result_sender, _Set, _Env...>,
        _Sender>;

    // A metafunction that computes the result sender type for a given set of argument types
    template <class _Set, class _Fun, class _Sched, class... _Env>
    struct __result_sender_fn {
      template <class... _Args>
      using __f = //
        __meval<
          __ensure_sender,
          _Set,
          __mcall<__mtry_catch_q<__call_result_t, __on_not_callable<_Set>>, _Fun, __decay_t<_Args>&...>,
          __result_env_t<_Sched, _Env>...>;
    };

    // The receiver that gets connected to the result sender is the input receiver,
    // possibly augmented with the input sender's completion scheduler (which is
    // where the result sender will be started).
    template <class _Receiver, class _Scheduler>
    using __result_receiver_t = __if_c<
      __is_scheduler_affine<schedule_result_t<_Scheduler>>,
      _Receiver,
      __receiver_with_sched<_Receiver, _Scheduler>>;

    template <class _ResultSender, class _Scheduler, class... _Env>
    using __receiver_ref_t = //
      __meval<
        __any_::__receiver_ref,
        __completion_signatures_of_t<_ResultSender, __result_env_t<_Scheduler, _Env>...>,
        __result_env_t<_Scheduler, _Env>...>;

    template <class _ResultSender, class _Scheduler, class _Receiver>
    concept __needs_receiver_ref =
      __nothrow_connectable<
        _ResultSender,
        __receiver_ref_t<_ResultSender, _Scheduler, env_of_t<_Receiver>>>
      && !__nothrow_connectable<_ResultSender, __result_receiver_t<_Receiver, _Scheduler>>;

    template <class _Sender, class _Receiver>
    using __nothrow_connectable_t = __mbool<__nothrow_connectable<_Sender, _Receiver>>;

    template <class _ResultSender, class _Scheduler, class... _Env>
    using __nothrow_connectable_receiver_ref_t = __meval<
      __nothrow_connectable_t,
      _ResultSender,
      __receiver_ref_t<_ResultSender, _Scheduler, _Env...>>;

    template <class _ResultSender, class _Scheduler, class _Receiver>
    using __checked_result_receiver_t = //
      __if_c<
        __needs_receiver_ref<_ResultSender, _Scheduler, _Receiver>,
        __receiver_ref_t<_ResultSender, _Scheduler, env_of_t<_Receiver>>,
        __result_receiver_t<_Receiver, _Scheduler>>;

    template <class _ResultSender, class _Scheduler, class _Receiver>
    using __op_state_t = connect_result_t<
      _ResultSender,
      __checked_result_receiver_t<_ResultSender, _Scheduler, _Receiver>>;

    template <class _SetTag, class _Fun, class _Sched, class... _Env>
    struct __transform_signal_fn {
      template <class... _Args>
      using __nothrow_connect = __mand<
        __mbool<(__nothrow_decay_copyable<_Args> && ...) && __nothrow_callable<_Fun, _Args...>>,
        __nothrow_connectable_receiver_ref_t<
          __mcall<__result_sender_fn<_SetTag, _Fun, _Sched, _Env...>, _Args...>,
          _Sched,
          _Env...>>;

      template <class... _Args>
      using __f = //
        __mcall<
          __mtry_q<__concat_completion_signatures>,
          __completion_signatures_of_t<
            __mcall<__result_sender_fn<_SetTag, _Fun, _Sched, _Env...>, _Args...>,
            __result_env_t<_Sched, _Env>...>,
          __eptr_completion_if_t<__nothrow_connect<_Args...>>>;
    };

    template <class _Sender, class _Set>
    using __completion_sched =
      __query_result_or_t<get_completion_scheduler_t<_Set>, env_of_t<_Sender>, __unknown_scheduler>;

    template <class _LetTag, class _Fun, class _CvrefSender, class... _Env>
    using __completions = //
      __gather_completion_signatures<
        __completion_signatures_of_t<_CvrefSender, _Env...>,
        __t<_LetTag>,
        __transform_signal_fn<
          __t<_LetTag>,
          _Fun,
          __completion_sched<_CvrefSender, __t<_LetTag>>,
          _Env...>::template __f,
        __sigs::__default_completion,
        __mtry_q<__concat_completion_signatures>::__f>;

    template <__mstring _Where, __mstring _What>
    struct _NO_COMMON_DOMAIN_ { };

    template <class _Set>
    using __no_common_domain_t = //
      _NO_COMMON_DOMAIN_<
        __in_which_let_msg<_Set>,
        "The senders returned by Function do not all share a common domain"_mstr>;

    template <class _Set>
    struct __try_common_domain_fn {
      struct __error_fn {
        template <class... _Senders>
        using __f = __mexception<__no_common_domain_t<_Set>, _WITH_SENDERS_<_Senders...>>;
      };

      template <class... _Senders>
      using __f = __mcall<__mtry_catch_q<__domain::__common_domain_t, __error_fn>, _Senders...>;
    };

    // Compute all the domains of all the result senders and make sure they're all the same
    template <class _Set, class _Child, class _Fun, class _Env, class _Sched>
    using __result_domain_t = //
      __gather_completions<
        _Set,
        __completion_signatures_of_t<_Child, _Env>,
        __result_sender_fn<_Set, _Fun, _Sched, _Env>,
        __try_common_domain_fn<_Set>>;

    template <class _LetTag, class _Env>
    auto __mk_transform_env_fn(_Env&& __env) noexcept {
      using _Set = __t<_LetTag>;
      return [&]<class _Fun, class _Child>(__ignore, _Fun&&, _Child&& __child) -> decltype(auto) {
        using __completions_t = __completion_signatures_of_t<_Child, _Env>;
        if constexpr (__merror<__completions_t>) {
          return __completions_t();
        } else {
          using _Scheduler = __completion_sched<_Child, _Set>;
          if constexpr (__is_scheduler_affine<schedule_result_t<_Scheduler>>) {
            return (__env);
          } else {
            return __env::__join(
              __env::__with(
                get_completion_scheduler<_Set>(stdexec::get_env(__child)), get_scheduler),
              __env::__without(static_cast<_Env&&>(__env), get_domain));
          }
        }
      };
    }

    template <class _LetTag, class _Env>
    auto __mk_transform_sender_fn(_Env&&) noexcept {
      using _Set = __t<_LetTag>;

      return []<class _Fun, class _Child>(__ignore, _Fun&& __fun, _Child&& __child) {
        using __completions_t = __completion_signatures_of_t<_Child, _Env>;

        if constexpr (__merror<__completions_t>) {
          return __completions_t();
        } else {
          using _Sched = __completion_sched<_Child, _Set>;
          using _Domain = __result_domain_t<_Set, _Child, _Fun, _Env, _Sched>;

          if constexpr (__merror<_Domain>) {
            return _Domain();
          } else if constexpr (same_as<_Domain, dependent_domain>) {
            using _Domain2 = __late_domain_of_t<_Child, _Env>;
            return __make_sexpr<__let_t<_Set, _Domain2>>(
              static_cast<_Fun&&>(__fun), static_cast<_Child&&>(__child));
          } else {
            static_assert(!same_as<_Domain, __unknown_scheduler>);
            return __make_sexpr<__let_t<_Set, _Domain>>(
              static_cast<_Fun&&>(__fun), static_cast<_Child&&>(__child));
          }
        }
      };
    }

    template <class _Receiver, class _Fun, class _Set, class _Sched>
    struct __op_state_for {
      template <class... _Args>
      using __f = __op_state_t<
        __mcall<__result_sender_fn<_Set, _Fun, _Sched, env_of_t<_Receiver>>, _Args...>,
        _Sched,
        _Receiver>;
    };

    template <class _Receiver, class _Fun, class _Set, class _Sched, class... _Tuples>
    struct __let_state {
      using __fun_t = _Fun;
      using __sched_t = _Sched;
      using __env_t = __result_env_t<_Sched, env_of_t<_Receiver>>;
      using __result_variant = __variant_for<__monostate, _Tuples...>;
      using __op_state_variant = //
        __variant_for<
          __monostate,
          __mapply<__op_state_for<_Receiver, _Fun, _Set, _Sched>, _Tuples>...>;

      template <class _ResultSender, class _OpState>
      auto __get_result_receiver(const _ResultSender&, _OpState& __op_state) -> decltype(auto) {
        if constexpr (__needs_receiver_ref<_ResultSender, _Sched, _Receiver>) {
          using __receiver_ref = __receiver_ref_t<_ResultSender, _Sched, env_of_t<_Receiver>>;
          return __receiver_ref{__op_state, __let::__get_env, __let::__get_rcvr};
        } else {
          _Receiver& __rcvr = __op_state.__rcvr_;
          if constexpr (__is_scheduler_affine<schedule_result_t<_Sched>>) {
            return static_cast<_Receiver&&>(__rcvr);
          } else {
            return __receiver_with_sched{static_cast<_Receiver&&>(__rcvr), this->__sched_};
          }
        }
      }

      auto __get_env(const _Receiver& __rcvr) const noexcept -> __env_t {
        if constexpr (__is_scheduler_affine<schedule_result_t<_Sched>>) {
          return stdexec::get_env(__rcvr);
        } else {
          return __env::__join(
            __env::__with(__sched_, get_scheduler),
            __env::__without(stdexec::get_env(__rcvr), get_domain));
        }
      }

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS _Fun __fun_;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS _Sched __sched_;
      __result_variant __args_{};
      __op_state_variant __op_state3_{};
    };

    template <class _Set, class _Domain>
    struct __let_t {
      using __domain_t = _Domain;
      using __t = _Set;

      template <sender _Sender, __movable_value _Fun>
      auto operator()(_Sender&& __sndr, _Fun __fun) const -> __well_formed_sender auto {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<__let_t<_Set>>(static_cast<_Fun&&>(__fun), static_cast<_Sender&&>(__sndr)));
      }

      template <class _Fun>
      STDEXEC_ATTRIBUTE((always_inline))
      auto
        operator()(_Fun __fun) const -> __binder_back<__let_t, _Fun> {
        return {{static_cast<_Fun&&>(__fun)}, {}, {}};
      }

      using _Sender = __1;
      using _Function = __0;
      using __legacy_customizations_t = __types<
        tag_invoke_t(
          __let_t,
          get_completion_scheduler_t<set_value_t>(get_env_t(const _Sender&)),
          _Sender,
          _Function),
        tag_invoke_t(__let_t, _Sender, _Function)>;

      template <sender_expr_for<__let_t<_Set>> _Sender, class _Env>
      static auto transform_env(_Sender&& __sndr, const _Env& __env) -> decltype(auto) {
        return __sexpr_apply(
          static_cast<_Sender&&>(__sndr), __mk_transform_env_fn<__let_t<_Set>>(__env));
      }

      template <sender_expr_for<__let_t<_Set>> _Sender, class _Env>
        requires same_as<__early_domain_of_t<_Sender>, dependent_domain>
      static auto transform_sender(_Sender&& __sndr, const _Env& __env) -> decltype(auto) {
        return __sexpr_apply(
          static_cast<_Sender&&>(__sndr), __mk_transform_sender_fn<__let_t<_Set>>(__env));
      }
    };

    template <class _Set, class _Domain>
    struct __let_impl : __sexpr_defaults {
      static constexpr auto get_attrs = //
        []<class _Child>(__ignore, const _Child& __child) noexcept {
          return __env::__join(__env::__with(_Domain(), get_domain), stdexec::get_env(__child));
        };

      static constexpr auto get_completion_signatures = //
        []<class _Self, class... _Env>(_Self&&, _Env&&...) noexcept
        -> __completions<__let_t<_Set, _Domain>, __data_of<_Self>, __child_of<_Self>, _Env...> {
        static_assert(sender_expr_for<_Self, __let_t<_Set, _Domain>>);
        return {};
      };

      static constexpr auto get_state = //
        []<class _Sender, class _Receiver>(_Sender&& __sndr, _Receiver&) {
          static_assert(sender_expr_for<_Sender, __let_t<_Set, _Domain>>);
          using _Fun = __data_of<_Sender>;
          using _Child = __child_of<_Sender>;
          using _Sched = __completion_sched<_Child, _Set>;
          using __mk_let_state = __mbind_front_q<__let_state, _Receiver, _Fun, _Set, _Sched>;

          using __let_state_t = __gather_completions_of<
            _Set,
            _Child,
            env_of_t<_Receiver>,
            __q<__decayed_tuple>,
            __mk_let_state>;

          return __sndr.apply(
            static_cast<_Sender&&>(__sndr),
            [&]<class _Fn, class _Child>(__ignore, _Fn&& __fn, _Child&& __child) {
              _Sched __sched = query_or(
                get_completion_scheduler<_Set>, stdexec::get_env(__child), __unknown_scheduler());
              return __let_state_t{static_cast<_Fn&&>(__fn), __sched};
            });
        };

      template <class _State, class _OpState, class... _As>
      static void __bind_(_State& __state, _OpState& __op_state, _As&&... __as) {
        auto& __args = __state.__args_.emplace_from(__tup::__mktuple, static_cast<_As&&>(__as)...);
        auto __sndr2 = __args.apply(std::move(__state.__fun_), __args);
        auto __rcvr2 = __state.__get_result_receiver(__sndr2, __op_state);
        auto& __op2 = __state.__op_state3_.emplace_from(
          stdexec::connect, std::move(__sndr2), std::move(__rcvr2));
        stdexec::start(__op2);
      }

      template <class _OpState, class... _As>
      static void __bind(_OpState& __op_state, _As&&... __as) noexcept {
        using _State = decltype(__op_state.__state_);
        using _Receiver = decltype(__op_state.__rcvr_);
        using _Fun = typename _State::__fun_t;
        using _Sched = typename _State::__sched_t;
        using _ResultSender =
          __mcall<__result_sender_fn<_Set, _Fun, _Sched, env_of_t<_Receiver>>, _As...>;

        _State& __state = __op_state.__state_;
        _Receiver& __rcvr = __op_state.__rcvr_;

        if constexpr (
          (__nothrow_decay_copyable<_As> && ...) && __nothrow_callable<_Fun, _As...>
          && __v<__nothrow_connectable_receiver_ref_t<_ResultSender, _Sched, env_of_t<_Receiver>>>) {
          __bind_(__state, __op_state, static_cast<_As&&>(__as)...);
        } else {
          try {
            __bind_(__state, __op_state, static_cast<_As&&>(__as)...);
          } catch (...) {
            using _Receiver = decltype(__op_state.__rcvr_);
            stdexec::set_error(static_cast<_Receiver&&>(__rcvr), std::current_exception());
          }
        }
      }

      static constexpr auto complete = //
        []<class _OpState, class _Tag, class... _As>(
          __ignore,
          _OpState& __op_state,
          _Tag,
          _As&&... __as) noexcept -> void {
        if constexpr (__same_as<_Tag, _Set>) {
          __bind(__op_state, static_cast<_As&&>(__as)...);
        } else {
          using _Receiver = decltype(__op_state.__rcvr_);
          _Tag()(static_cast<_Receiver&&>(__op_state.__rcvr_), static_cast<_As&&>(__as)...);
        }
      };
    };
  } // namespace __let

  using let_value_t = __let::__let_t<set_value_t>;
  inline constexpr let_value_t let_value{};

  using let_error_t = __let::__let_t<set_error_t>;
  inline constexpr let_error_t let_error{};

  using let_stopped_t = __let::__let_t<set_stopped_t>;
  inline constexpr let_stopped_t let_stopped{};

  template <class _Set, class _Domain>
  struct __sexpr_impl<__let::__let_t<_Set, _Domain>> : __let::__let_impl<_Set, _Domain> { };
} // namespace stdexec
