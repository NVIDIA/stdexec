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
#include "__meta.hpp"
#include "__any_receiver_ref.hpp" // IWYU pragma: keep for __any::__receiver_ref
#include "__schedulers.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__senders.hpp"
#include "__submit.hpp"
#include "__transform_completion_signatures.hpp"
#include "__variant.hpp"
#include "__utility.hpp"

#include <exception>

namespace stdexec {
  //////////////////////////////////////////////////////////////////////////////
  // [exec.let]
  namespace __let {
    template <class _SetTag>
    struct __let_t;

    template <class _SetTag>
    inline constexpr __mstring __in_which_let_msg{"In stdexec::let_value(Sender, Function)..."};

    template <>
    inline constexpr __mstring __in_which_let_msg<set_error_t>{
      "In stdexec::let_error(Sender, Function)..."};

    template <>
    inline constexpr __mstring __in_which_let_msg<set_stopped_t>{
      "In stdexec::let_stopped(Sender, Function)..."};

    template <class _SetTag>
    using __on_not_callable = __callable_error<__in_which_let_msg<_SetTag>>;

    // This environment is part of the receiver used to connect the secondary sender.
    template <class _SetTag, class _Attrs, class... _Env>
    constexpr auto __mk_env2(const _Attrs& __attrs, const _Env&... __env) noexcept {
      if constexpr (__callable<
                      get_completion_scheduler_t<_SetTag>,
                      const _Attrs&,
                      __fwd_env_t<const _Env&>...
                    >) {
        return __mk_sch_env(
          get_completion_scheduler<_SetTag>(__attrs, __fwd_env(__env)...), __fwd_env(__env)...);
      } else if constexpr (
        __callable<get_completion_domain_t<_SetTag>, const _Attrs&, __fwd_env_t<const _Env&>...>) {
        using __domain_t = __call_result_t<
          get_completion_domain_t<_SetTag>,
          const _Attrs&,
          __fwd_env_t<const _Env&>...
        >;
        return prop{get_domain, __domain_t{}};
      } else {
        return env{};
      }
    }

    template <class _SetTag, class _Attrs, class... _Env>
    using __env2_t = decltype(__let::__mk_env2<_SetTag>(__declval<_Attrs>(), __declval<_Env>()...));

    template <class _SetTag, class _Attrs, class _Env>
    using __result_env_t = __join_env_t<__env2_t<_SetTag, _Attrs, _Env>, _Env>;

    template <class _ReceiverId, class _Env2Id>
    struct __rcvr_env {
      using _Receiver = stdexec::__t<_ReceiverId>;
      using _Env2 = stdexec::__t<_Env2Id>;

      struct __t {
        using receiver_concept = receiver_t;
        using __id = __rcvr_env;

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
          return __env::__join(__env_, stdexec::get_env(__rcvr_));
        }

        _Receiver __rcvr_;
        const _Env2& __env_;
      };
    };

    template <class _Receiver, class _Env2>
    using __receiver_with_env_t = __t<__rcvr_env<__id<_Receiver>, __id<_Env2>>>;

    template <__mstring _Where, __mstring _What>
    struct _FUNCTION_MUST_RETURN_A_VALID_SENDER_IN_THE_CURRENT_ENVIRONMENT_ { };

#if STDEXEC_EDG()
    template <class _Sender, class _SetTag, class... _JoinEnv2>
    struct __bad_result_sender_ {
      using __t = __mexception<
        _FUNCTION_MUST_RETURN_A_VALID_SENDER_IN_THE_CURRENT_ENVIRONMENT_<
          __in_which_let_msg<_SetTag>,
          "The function must return a valid sender for the current environment"_mstr
        >,
        _WITH_SENDER_<_Sender>,
        _WITH_ENVIRONMENT_<_JoinEnv2>...
      >;
    };
    template <class _Sender, class _SetTag, class... _JoinEnv2>
    using __bad_result_sender = __t<__bad_result_sender_<_Sender, _SetTag, _JoinEnv2...>>;
#else
    template <class _Sender, class _SetTag, class... _JoinEnv2>
    using __bad_result_sender = __mexception<
      _FUNCTION_MUST_RETURN_A_VALID_SENDER_IN_THE_CURRENT_ENVIRONMENT_<
        __in_which_let_msg<_SetTag>,
        "The function must return a valid sender for the current environment"_mstr
      >,
      _WITH_SENDER_<_Sender>,
      _WITH_ENVIRONMENT_<_JoinEnv2>...
    >;
#endif

    template <class _Sender, class... _JoinEnv2>
    concept __potentially_valid_sender_in = sender_in<_Sender, _JoinEnv2...>
                                         || (sender<_Sender> && (sizeof...(_JoinEnv2) == 0));

    template <class _SetTag, class _Sender, class... _JoinEnv2>
    using __ensure_sender = __minvoke_if_c<
      __potentially_valid_sender_in<_Sender, _JoinEnv2...>,
      __q1<__midentity>,
      __mbind_back_q<__bad_result_sender, _SetTag, _JoinEnv2...>,
      _Sender
    >;

    // A metafunction that computes the result sender type for a given set of argument types
    template <class _SetTag, class _Fun, class... _JoinEnv2>
    struct __result_sender_fn {
      template <class... _Args>
      using __f = __meval<
        __ensure_sender,
        _SetTag,
        __mcall<
          __mtry_catch_q<__call_result_t, __on_not_callable<_SetTag>>,
          _Fun,
          __decay_t<_Args>&...
        >,
        _JoinEnv2...
      >;
    };

    // The receiver that gets connected to the result sender is the input receiver,
    // possibly augmented with the input sender's completion scheduler (which is
    // where the result sender will be started).
    template <class _Receiver, class _Env2>
    using __result_receiver_t = __receiver_with_env_t<_Receiver, _Env2>;

    template <class _ResultSender, class _Env2, class _Receiver>
    using __checked_result_receiver_t = __result_receiver_t<_Receiver, _Env2>;

    template <class _ResultSender, class _Env2, class _Receiver>
    using __submit_result =
      submit_result<_ResultSender, __checked_result_receiver_t<_ResultSender, _Env2, _Receiver>>;

    template <class _SetTag, class _Fun, class _JoinEnv2>
    struct __transform_signal_fn {
      template <class... _Args>
      using __nothrow_connect = __mbool<
        __nothrow_decay_copyable<_Args...> //
        && __nothrow_callable<_Fun, __decay_t<_Args>&...>
        && __nothrow_connectable<
          __mcall<__result_sender_fn<_SetTag, _Fun, _JoinEnv2>, _Args...>,
          __receiver_archetype<_JoinEnv2>
        >
      >;

      template <class... _Args>
      using __f = __mcall<
        __mtry_q<__concat_completion_signatures>,
        __completion_signatures_of_t<
          __mcall<__result_sender_fn<_SetTag, _Fun, _JoinEnv2>, _Args...>,
          _JoinEnv2
        >,
        __eptr_completion_unless_t<__nothrow_connect<_Args...>>
      >;
    };

    template <class _LetTag, class _Fun, class _CvrefSender, class _Env>
    using __completions_t = __gather_completion_signatures<
      __completion_signatures_of_t<_CvrefSender, _Env>,
      __t<_LetTag>,
      __transform_signal_fn<
        __t<_LetTag>,
        _Fun,
        __result_env_t<__t<_LetTag>, env_of_t<_CvrefSender>, _Env>
      >::template __f,
      __sigs::__default_completion,
      __mtry_q<__concat_completion_signatures>::__f
    >;

    template <__mstring _Where, __mstring _What>
    struct _NO_COMMON_DOMAIN_ { };

    template <class _SetTag>
    using __no_common_domain_t = _NO_COMMON_DOMAIN_<
      __in_which_let_msg<_SetTag>,
      "The senders returned by Function do not all share a common domain"_mstr
    >;

    template <class _SetTag, class... _Env>
    struct __try_common_domain_fn {
      struct __error_fn {
        template <class... _Senders>
        using __f = __mexception<__no_common_domain_t<_SetTag>, _WITH_SENDERS_<_Senders...>>;
      };

      // TODO(ericniebler): this needs to be updated:
      template <class... _Senders>
      using __f = __mcall<
        __mtry_catch_q<__common_domain_t, __error_fn>,
        __compl_domain_t<_SetTag, _Senders, _Env...>...
      >;
    };

    // Compute all the domains of all the result senders and make sure they're all the same
    template <class _SetTag, class _Child, class _Fun, class _Env>
    using __result_domain_t = __gather_completions<
      _SetTag,
      __completion_signatures_of_t<_Child, _Env>,
      __result_sender_fn<_SetTag, _Fun, __result_env_t<_SetTag, env_of_t<_Child>, _Env>>,
      __try_common_domain_fn<_SetTag, _Env>
    >;

    //! Metafunction creating the operation state needed to connect the result of calling
    //! the sender factory function, `_Fun`, and passing its result to a receiver.
    template <class _Receiver, class _Fun, class _SetTag, class _Env2>
    struct __submit_datum_for {
      // compute the result of calling submit with the result of executing _Fun
      // with _Args. if the result is void, substitute with __ignore.
      template <class... _Args>
      using __f = __submit_result<
        __mcall<
          __result_sender_fn<_SetTag, _Fun, __join_env_t<_Env2, env_of_t<_Receiver>>>,
          _Args...
        >,
        _Env2,
        _Receiver
      >;
    };

    //! The core of the operation state for `let_*`.
    //! This gets bundled up into a larger operation state (`__detail::__op_state<...>`).
    template <class _Receiver, class _Fun, class _SetTag, class _Env2, class... _Tuples>
    struct __let_state {
      using __fun_t = _Fun;
      using __env2_t = _Env2;
      using __env_t = __join_env_t<_Env2, env_of_t<_Receiver>>;
      using __rcvr_t = __receiver_with_env_t<_Receiver, _Env2>;
      using __result_variant = __variant_for<__monostate, _Tuples...>;
      using __submit_variant = __variant_for<
        __monostate,
        __mapply<__submit_datum_for<_Receiver, _Fun, _SetTag, _Env2>, _Tuples>...
      >;

      template <class _ResultSender, class _OpState>
      auto __get_result_receiver(const _ResultSender&, _OpState& __op_state) -> decltype(auto) {
        return __rcvr_t{static_cast<_Receiver&&>(__op_state.__rcvr_), __env2_};
      }

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Fun __fun_;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Env2 __env2_;
      //! Variant to hold the results passed from upstream before passing them to the function:
      __result_variant __args_{};
      //! Variant type for holding the operation state from connecting
      //! the function result to the downstream receiver:
      __submit_variant __storage_{};
    };

    //! Implementation of the `let_*_t` types, where `_SetTag` is, e.g., `set_value_t` for `let_value`.
    template <class _SetTag>
    struct __let_t {
      using __t = _SetTag;

      template <sender _Sender, __movable_value _Fun>
      auto operator()(_Sender&& __sndr, _Fun __fun) const -> __well_formed_sender auto {
        return __make_sexpr<__let_t<_SetTag>>(
          static_cast<_Fun&&>(__fun), static_cast<_Sender&&>(__sndr));
      }

      template <class _Fun>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(_Fun __fun) const -> __binder_back<__let_t, _Fun> {
        return {{static_cast<_Fun&&>(__fun)}, {}, {}};
      }
    };

    template <class _SetTag>
    struct __let_impl : __sexpr_defaults {
      static constexpr auto get_attrs =
        []<class _Fun, class _Child>(const _Fun&, const _Child& __child) noexcept {
          //return __env::__join(prop{get_domain, _Domain()}, stdexec::get_env(__child));
          // TODO(ericniebler): this needs to be updated:
          return stdexec::get_env(__child);
        };

      static constexpr auto get_completion_signatures =
        []<class _Self, class _Env>(_Self&&, _Env&&...) noexcept {
          static_assert(sender_expr_for<_Self, __let_t<_SetTag>>);
          if constexpr (__decay_copyable<_Self>) {
            using __result_t =
              __completions_t<__let_t<_SetTag>, __data_of<_Self>, __child_of<_Self>, _Env>;
            return __result_t{};
          } else {
            return __mexception<_SENDER_TYPE_IS_NOT_COPYABLE_, _WITH_SENDER_<_Self>>{};
          }
        };

      static constexpr auto get_state =
        []<class _Receiver, __decay_copyable _Sender>(_Sender&& __sndr, const _Receiver& __rcvr)
        requires sender_in<__child_of<_Sender>, env_of_t<_Receiver>>
      {
        static_assert(sender_expr_for<_Sender, __let_t<_SetTag>>);
        using _Fun = __decay_t<__data_of<_Sender>>;
        using _Child = __child_of<_Sender>;
        using _Env2 = __env2_t<_SetTag, env_of_t<const _Child&>, env_of_t<const _Receiver&>>;
        using __mk_let_state = __mbind_front_q<__let_state, _Receiver, _Fun, _SetTag, _Env2>;

        using __let_state_t = __gather_completions_of<
          _SetTag,
          _Child,
          env_of_t<_Receiver>,
          __q<__decayed_tuple>,
          __mk_let_state
        >;

        return __sndr
          .apply(
            static_cast<_Sender&&>(__sndr),
            [&]<class _Fn, class _Child>(__ignore, _Fn&& __fn, _Child&& __child) {
              // TODO(ericniebler): this needs a fallback
              _Env2 __env2 =
                __let::__mk_env2<_SetTag>(stdexec::get_env(__child), stdexec::get_env(__rcvr));
              return __let_state_t{static_cast<_Fn&&>(__fn), static_cast<_Env2&&>(__env2)};
            });
      };

      //! Helper function to actually invoke the function to produce `let_*`'s sender,
      //! connect it to the downstream receiver, and start it. This is the heart of
      //! `let_*`.
      template <class _State, class _OpState, class... _As>
      static void __bind_(_State& __state, _OpState& __op_state, _As&&... __as) {
        // Store the passed-in (received) args:
        auto& __args = __state.__args_.emplace_from(__tup::__mktuple, static_cast<_As&&>(__as)...);
        // Apply the function to the args to get the sender:
        auto __sndr2 = __args.apply(std::move(__state.__fun_), __args);
        // Create a receiver based on the state, the computed sender, and the operation state:
        auto __rcvr2 = __state.__get_result_receiver(__sndr2, __op_state);
        // Connect the sender to the receiver and start it:
        using __result_t = decltype(submit_result{std::move(__sndr2), std::move(__rcvr2)});
        auto& __op = __state.__storage_
                       .template emplace<__result_t>(std::move(__sndr2), std::move(__rcvr2));
        __op.submit(std::move(__sndr2), std::move(__rcvr2));
      }

      template <class _OpState, class... _As>
      static void __bind(_OpState& __op_state, _As&&... __as) noexcept {
        using _State = decltype(__op_state.__state_);
        using _Receiver = decltype(__op_state.__rcvr_);
        using _Fun = _State::__fun_t;
        using _Env2 = _State::__env2_t;
        using _JoinEnv2 = __join_env_t<_Env2, env_of_t<_Receiver>>;
        using _ResultSender = __mcall<__result_sender_fn<_SetTag, _Fun, _JoinEnv2>, _As...>;

        _State& __state = __op_state.__state_;
        _Receiver& __rcvr = __op_state.__rcvr_;

        if constexpr (
          (__nothrow_decay_copyable<_As> && ...) && __nothrow_callable<_Fun, _As...>
          && __nothrow_connectable<_ResultSender, __result_receiver_t<_Receiver, _Env2>>) {
          __bind_(__state, __op_state, static_cast<_As&&>(__as)...);
        } else {
          STDEXEC_TRY {
            __bind_(__state, __op_state, static_cast<_As&&>(__as)...);
          }
          STDEXEC_CATCH_ALL {
            using _Receiver = decltype(__op_state.__rcvr_);
            stdexec::set_error(static_cast<_Receiver&&>(__rcvr), std::current_exception());
          }
        }
      }

      static constexpr auto complete = []<class _OpState, class _Tag, class... _As>(
                                         __ignore,
                                         _OpState& __op_state,
                                         _Tag,
                                         _As&&... __as) noexcept -> void {
        if constexpr (__same_as<_Tag, _SetTag>) {
          // Intercept the channel of interest to compute the sender and connect it:
          __bind(__op_state, static_cast<_As&&>(__as)...);
        } else {
          // Forward the other channels downstream:
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

  template <class _SetTag>
  struct __sexpr_impl<__let::__let_t<_SetTag>> : __let::__let_impl<_SetTag> { };
} // namespace stdexec
