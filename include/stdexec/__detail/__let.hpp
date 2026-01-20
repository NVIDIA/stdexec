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
#include "__any_receiver_ref.hpp" // IWYU pragma: keep for __any::__receiver_ref
#include "__basic_sender.hpp"
#include "__diagnostics.hpp"
#include "__domain.hpp"
#include "__env.hpp"
#include "__meta.hpp"
#include "__schedulers.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__senders.hpp"
#include "__submit.hpp"
#include "__transform_completion_signatures.hpp"
#include "__utility.hpp"
#include "__variant.hpp"

#include <exception>
#include <type_traits>

namespace STDEXEC {
  //////////////////////////////////////////////////////////////////////////////
  // [exec.let]
  namespace __let {
    template <class _SetTag>
    struct __let_t;

    template <class _SetTag>
    struct __let_tag {
      using __t = _SetTag;
    };

    template <class _SetTag>
    inline constexpr __mstring __in_which_let_msg{"In STDEXEC::let_value(Sender, Function)..."};

    template <>
    inline constexpr __mstring __in_which_let_msg<set_error_t>{
      "In STDEXEC::let_error(Sender, Function)..."};

    template <>
    inline constexpr __mstring __in_which_let_msg<set_stopped_t>{
      "In STDEXEC::let_stopped(Sender, Function)..."};

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
      using _Receiver = STDEXEC::__t<_ReceiverId>;
      using _Env2 = STDEXEC::__t<_Env2Id>;

      struct __t {
        using receiver_concept = receiver_t;
        using __id = __rcvr_env;

        template <class... _As>
        void set_value(_As&&... __as) noexcept {
          STDEXEC::set_value(static_cast<_Receiver&&>(__rcvr_), static_cast<_As&&>(__as)...);
        }

        template <class _Error>
        void set_error(_Error&& __err) noexcept {
          STDEXEC::set_error(static_cast<_Receiver&&>(__rcvr_), static_cast<_Error&&>(__err));
        }

        void set_stopped() noexcept {
          STDEXEC::set_stopped(static_cast<_Receiver&&>(__rcvr_));
        }

        auto get_env() const noexcept {
          return __env::__join(__env_, STDEXEC::get_env(__rcvr_));
        }

        _Receiver& __rcvr_;
        const _Env2& __env_;
      };
    };

    template <class _Receiver, class _Env2>
    using __receiver_with_env_t = __t<__rcvr_env<__id<_Receiver>, __id<_Env2>>>;

    template <__mstring _Where, __mstring _What>
    struct _FUNCTION_MUST_RETURN_A_VALID_SENDER_IN_THE_CURRENT_ENVIRONMENT_ { };

    template <class...>
    struct _NESTED_ERROR_;

    template <class _Sender, class... _Env>
    using __try_completion_signatures_of_t = __meval_or<
      __completion_signatures_of_t,
      __unrecognized_sender_error_t<_Sender, _Env...>,
      _Sender,
      _Env...
    >;

    template <class _Sender, class _SetTag, class... _JoinEnv2>
    using __bad_result_sender = __not_a_sender<
      _FUNCTION_MUST_RETURN_A_VALID_SENDER_IN_THE_CURRENT_ENVIRONMENT_<
        __in_which_let_msg<_SetTag>,
        "The function must return a valid sender for the current environment"_mstr
      >,
      _WITH_PRETTY_SENDER_<_Sender>,
      __fn_t<_WITH_ENVIRONMENT_, _JoinEnv2>...,
      __mapply_q<_NESTED_ERROR_, __try_completion_signatures_of_t<_Sender, _JoinEnv2...>>
    >;

    template <class _Sender, class... _JoinEnv2>
    concept __potentially_valid_sender_in = sender_in<_Sender, _JoinEnv2...>
                                         || (sender<_Sender> && (sizeof...(_JoinEnv2) == 0));

    template <class _SetTag, class _Sender, class... _JoinEnv2>
    using __ensure_sender_t = __minvoke_if_c<
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
        __ensure_sender_t,
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
    using __submit_result_t =
      submit_result<_ResultSender, __checked_result_receiver_t<_ResultSender, _Env2, _Receiver>>;

    template <class _SetTag, class _Fun, class... _JoinEnv2>
    struct __transform_signal_fn {
      template <class... _Args>
      static constexpr bool
        __nothrow_connect_v = __nothrow_decay_copyable<_Args...>
                           && __nothrow_callable<_Fun, __decay_t<_Args>&...>
                           && (__nothrow_connectable<
                                 __mcall<__result_sender_fn<_SetTag, _Fun, _JoinEnv2>, _Args...>,
                                 __receiver_archetype<_JoinEnv2>
                               >
                               && ...);

      template <class... _Args>
      using __f = __mcall<
        __mtry_q<__concat_completion_signatures_t>,
        __completion_signatures_of_t<
          __mcall<__result_sender_fn<_SetTag, _Fun, _JoinEnv2...>, _Args...>,
          _JoinEnv2...
        >,
        __eptr_completion_unless<__nothrow_connect_v<_Args...>>
      >;
    };

    template <class _LetTag, class _Fun, class _CvrefSender, class... _Env>
    using __completions_t = __gather_completion_signatures_t<
      __completion_signatures_of_t<_CvrefSender, _Env...>,
      __t<_LetTag>,
      __transform_signal_fn<
        __t<_LetTag>,
        _Fun,
        __result_env_t<__t<_LetTag>, env_of_t<_CvrefSender>, _Env>...
      >::template __f,
      __cmplsigs::__default_completion,
      __mtry_q<__concat_completion_signatures_t>::__f
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
        using __f = __mexception<__no_common_domain_t<_SetTag>, _WITH_PRETTY_SENDERS_<_Senders...>>;
      };

      // TODO(ericniebler): this needs to be updated:
      template <class... _Senders>
      using __f = __mcall<
        __mtry_catch_q<__common_domain_t, __error_fn>,
        __completion_domain_of_t<_SetTag, _Senders, _Env...>...
      >;
    };

    // Compute all the domains of all the result senders and make sure they're all the same
    template <class _SetTag, class _Child, class _Fun, class _Env>
    using __result_domain_t = __gather_completions_t<
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
      using __f = __submit_result_t<
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
    template <class _SetTag, class _Sender, class _Fun, class _Receiver, class... _Tuples>
    struct __let_state {
      using __env2_t =
        __let::__env2_t<_SetTag, env_of_t<const _Sender&>, env_of_t<const _Receiver&>>;
      using __second_rcvr_t = __receiver_with_env_t<_Receiver, __env2_t>;
      template <class _Tag, class... _Args>
      constexpr void __impl(_Receiver& __rcvr, _Tag __tag, _Args&&... __args) noexcept {
        if constexpr (std::is_same_v<_SetTag, _Tag>) {
          using __sender_t = __call_result_t<_Fun, __decay_t<_Args>&...>;
          using __submit_t = __submit_result_t<__sender_t, __env2_t, _Receiver>;
          constexpr bool __nothrow_store = (__nothrow_decay_copyable<_Args> && ...);
          constexpr bool __nothrow_invoke = __nothrow_callable<_Fun, __decay_t<_Args>&...>;
          constexpr bool __nothrow_submit = noexcept(
            __storage_
              .template emplace<__submit_t>(__declval<__sender_t>(), __declval<__second_rcvr_t>()));
          STDEXEC_TRY {
            auto& __tuple = __args_.emplace_from(__mktuple, static_cast<_Args&&>(__args)...);
            auto&& __sender = ::STDEXEC::__apply(static_cast<_Fun&&>(__fn_), __tuple);
            __storage_.template emplace<__monostate>();
            __second_rcvr_t __r{__rcvr, static_cast<__env2_t&&>(__env2_)};
            auto& __op = __storage_.template emplace<__submit_t>(
              static_cast<__sender_t&&>(__sender), static_cast<__second_rcvr_t&&>(__r));
            __op.submit(static_cast<__sender_t&&>(__sender), static_cast<__second_rcvr_t&&>(__r));
          }
          STDEXEC_CATCH_ALL {
            if constexpr (!(__nothrow_store && __nothrow_invoke && __nothrow_submit)) {
              ::STDEXEC::set_error(static_cast<_Receiver&&>(__rcvr), std::current_exception());
            }
          }
        } else {
          __tag(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
        }
      }
      struct __first_rcvr_t {
        using receiver_concept = ::STDEXEC::receiver_t;
        __let_state& __state;
        _Receiver& __rcvr;
        template <class... _Args>
        constexpr void set_value(_Args&&... __args) noexcept {
          __state.__impl(__rcvr, ::STDEXEC::set_value, static_cast<_Args&&>(__args)...);
        }
        template <class... _Args>
        constexpr void set_error(_Args&&... __args) noexcept {
          __state.__impl(__rcvr, ::STDEXEC::set_error, static_cast<_Args&&>(__args)...);
        }
        template <class... _Args>
        constexpr void set_stopped(_Args&&... __args) noexcept {
          __state.__impl(__rcvr, ::STDEXEC::set_stopped, static_cast<_Args&&>(__args)...);
        }
        constexpr decltype(auto) get_env() const noexcept {
          return ::STDEXEC::get_env(__rcvr);
        }
      };

      using __result_variant = __variant_for<__monostate, _Tuples...>;
      using __op_state_variant = __variant_for<
        __monostate,
        ::STDEXEC::connect_result_t<_Sender, __first_rcvr_t>,
        __mapply<__submit_datum_for<_Receiver, _Fun, _SetTag, __env2_t>, _Tuples>...
      >;

      constexpr explicit __let_state(_Sender&& __sender, _Fun __fn, _Receiver& __r) noexcept(
        __nothrow_connectable<_Sender, __first_rcvr_t>
        && std::is_nothrow_move_constructible_v<_Fun>)
        : __fn_(static_cast<_Fun&&>(__fn))
        , __env2_(
            // TODO(ericniebler): this needs a fallback
            __let::__mk_env2<_SetTag>(::STDEXEC::get_env(__sender), ::STDEXEC::get_env(__r))) {
        __storage_.emplace_from(
          ::STDEXEC::connect, static_cast<_Sender&&>(__sender), __first_rcvr_t{*this, __r});
      }

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Fun __fn_;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      __env2_t __env2_;
      //! Variant to hold the results passed from upstream before passing them to the function:
      __result_variant __args_{};
      //! Variant type for holding the operation state of the currently in flight operation
      __op_state_variant __storage_{};
    };

    // The set_value completions of:
    //
    //   * a let_value sender are:
    //       * the value completions of the secondary senders
    //
    //   * a let_error sender are:
    //       * the value completions of the predecessor sender
    //       * the value completions of the secondary senders
    //
    //   * a let_stopped sender are:
    //       * the value completions of the predecessor sender
    //       * the value completions of the secondary sender
    //
    // The set_stopped completions of:
    //
    //   * a let_value sender are:
    //       * the stopped completions of the predecessor sender
    //       * the stopped completions of the secondary senders
    //
    //   * a let_error sender are:
    //       * the stopped completions of the predecessor sender
    //       * the stopped completions of the secondary senders
    //
    //   * a let_stopped sender are:
    //       * the stopped completions of the secondary senders
    //
    // The set_error completions of:
    //
    //   * a let_value sender are:
    //       * the error completions of the predecessor sender
    //       * the error completions of the secondary senders
    //       * the value completions of the predecessor sender if decay copying the arguments can throw
    //
    //   * a let_error sender are:
    //       * the error completions of the secondary senders
    //       * the error completions of the predecessor sender if decay copying the errors can throw
    //
    //   * a let_stopped sender are:
    //       * the error completions of the predecessor sender
    //       * the error completions of the secondary senders

    // A metafunction to check whether the predecessor's completion results are nothrow
    // decay-copyable and whether connecting the secondary sender is nothrow.
    template <class _SetTag, class _Sndr, class _Fn, class _Env>
    struct __has_nothrow_completions_fn {
      using __env2_t = __let::__result_env_t<_SetTag, env_of_t<_Sndr>, _Env>;
      using __rcvr2_t = __receiver_archetype<__env2_t>;

      template <class... _Ts>
      using __f = __mbool<
        __nothrow_decay_copyable<_Ts...>
        && __nothrow_connectable<__call_result_t<_Fn, __decay_t<_Ts>&...>, __rcvr2_t>
      >;
    };

    template <class _SetTag, class _Sndr, class _Fn, class _Env>
    using __has_nothrow_completions = __gather_completions_t<
      completion_signatures_of_t<_Sndr, _Env>,
      _SetTag,
      __has_nothrow_completions_fn<_SetTag, _Sndr, _Fn, _Env>,
      __qq<__mand_t>
    >;

    template <class _SetTag, class _Fn, class... _JoinEnv2>
    struct __result_completion_behavior_fn {
      template <class... _Ts>
      [[nodiscard]]
      static constexpr auto __impl() noexcept {
        using __result_sender_fn = __let::__result_sender_fn<_SetTag, _Fn, _JoinEnv2...>;
        if constexpr (__minvocable<__result_sender_fn, _Ts...>) {
          using __sndr2_t = __mcall<__result_sender_fn, _Ts...>;
          return STDEXEC::get_completion_behavior<_SetTag, __sndr2_t, _JoinEnv2...>();
        } else {
          return completion_behavior::unknown;
        }
      }

      template <class... _Ts>
      using __f = decltype(__impl<_Ts...>());
    };

    template <class _SetTag, class _Fn, class _Attrs, class... _Env>
    struct __domain_transform_fn {
      using __result_sender_fn =
        __let::__result_sender_fn<_SetTag, _Fn, __result_env_t<_SetTag, _Attrs, _Env>...>;

      template <class... _As>
      using __f = __completion_domain_of_t<
        _SetTag,
        __mcall<__result_sender_fn, _As...>,
        __result_env_t<_SetTag, _Attrs, _Env>...
      >;
    };

    //! @tparam _LetTag The tag type for the let_ operation.
    //! @tparam _SetTag The completion signal of the let_ sender itself that is being
    //! queried. For example, you may be querying a let_value sender for its set_error
    //! completion domain.
    template <class _LetTag, class _SetTag, class _Sndr, class _Fn, class... _Env>
    [[nodiscard]]
    consteval auto __get_completion_domain() noexcept {
      if constexpr (sender_in<_Sndr, _Env...>) {
        using __domain_transform_fn =
          __let::__domain_transform_fn<_SetTag, _Fn, env_of_t<_Sndr>, _Env...>;
        return __minvoke_or_q<
          __gather_completions_t,
          indeterminate_domain<>,
          __t<_LetTag>,
          __completion_signatures_of_t<_Sndr, _Env...>,
          __domain_transform_fn,
          __qq<__common_domain_t>
        >();
      } else {
        return indeterminate_domain<>{};
      }
    }

    template <class _SetTag, class _SetTag2, class _Sndr, class _Fn, class... _Env>
    using __let_completion_domain_t = __unless_one_of_t<
      decltype(__let::__get_completion_domain<_SetTag, _SetTag2, _Sndr, _Fn, _Env...>()),
      indeterminate_domain<>
    >;

    template <class _LetTag, class _Sndr, class _Fn>
    struct __attrs {
      using __t = __attrs;
      using __id = __attrs;
      using __set_tag_t = STDEXEC::__t<_LetTag>;

      template <class _Tag>
      constexpr auto query(get_completion_scheduler_t<_Tag>) const = delete;

      template <class... _Env>
      [[nodiscard]]
      constexpr auto query(get_completion_domain_t<__set_tag_t>, const _Env&...) const noexcept
        -> __ensure_valid_domain_t<
          __let_completion_domain_t<_LetTag, __set_tag_t, _Sndr, _Fn, _Env...>
        > {
        return {};
      }

      template <__one_of<set_error_t, set_stopped_t> _Tag, class... _Env>
        requires(__has_nothrow_completions<__set_tag_t, _Sndr, _Fn, _Env>::value && ...)
      [[nodiscard]]
      constexpr auto query(get_completion_domain_t<_Tag>, const _Env&...) const noexcept
        -> __ensure_valid_domain_t<__common_domain_t<
          __completion_domain_of_t<_Tag, _Sndr, __fwd_env_t<_Env>...>,
          __let_completion_domain_t<_LetTag, _Tag, _Sndr, _Fn, _Env...>
        >> {
        return {};
      }

      template <class _Env>
        requires(!__has_nothrow_completions<__set_tag_t, _Sndr, _Fn, _Env>::value)
      [[nodiscard]]
      constexpr auto query(get_completion_domain_t<set_error_t>, const _Env&) const noexcept
        -> __ensure_valid_domain_t<__common_domain_t<
          __completion_domain_of_t<__set_tag_t, _Sndr, __fwd_env_t<_Env>>,
          __completion_domain_of_t<set_error_t, _Sndr, __fwd_env_t<_Env>>,
          __let_completion_domain_t<_LetTag, set_error_t, _Sndr, _Fn, _Env>
        >> {
        return {};
      }

      template <class... _Env>
      [[nodiscard]]
      constexpr auto query(get_completion_behavior_t<__set_tag_t>, const _Env&...) const noexcept {
        if constexpr (sender_in<_Sndr, __fwd_env_t<_Env>...>) {
          // The completion behavior of let_value(sndr, fn) is the weakest completion
          // behavior of sndr and all the senders that fn can potentially produce. (MSVC
          // needs the constexpr computation broken up, hence the local variables.)
          using __transform_fn = __result_completion_behavior_fn<
            __set_tag_t,
            _Fn,
            __result_env_t<__set_tag_t, env_of_t<_Sndr>, _Env>...
          >;
          using __completions_t = __completion_signatures_of_t<_Sndr, __fwd_env_t<_Env>...>;

          constexpr auto __pred_behavior =
            STDEXEC::get_completion_behavior<__set_tag_t, _Sndr, __fwd_env_t<_Env>...>();
          constexpr auto __result_behavior = __gather_completions_t<
            __set_tag_t,
            __completions_t,
            __transform_fn,
            __qq<__common_completion_behavior_t>
          >();

          return completion_behavior::weakest(__pred_behavior, __result_behavior);
        } else {
          return completion_behavior::unknown;
        }
      }
    };

    template <class _Sender, class _Fun>
    struct __data {
      _Sender __sndr;
      _Fun __fn;
    };

    template <class _Sender, class _Fun>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __data(_Sender, _Fun) -> __data<_Sender, _Fun>;

    template <class _Sender>
    using __child_of_t = decltype((__declval<__data_of<_Sender>>().__sndr));
    template <class _Sender>
    using __fn_of_t = decltype((__declval<__data_of<_Sender>>().__fn));

    //! Implementation of the `let_*_t` types, where `_SetTag` is, e.g., `set_value_t` for `let_value`.
    template <class _SetTag>
    struct __let_t {
      template <sender _Sender, __movable_value _Fun>
      auto operator()(_Sender&& __sndr, _Fun __fn) const -> __well_formed_sender auto {
        return __make_sexpr<__let_t<_SetTag>>(
          static_cast<_Fun&&>(__fn), static_cast<_Sender&&>(__sndr));
      }

      template <class _Fun>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(_Fun __fn) const {
        return __closure(*this, static_cast<_Fun&&>(__fn));
      }

      template <class _Sender>
      static auto transform_sender(set_value_t, _Sender&& __sndr, __ignore) {
        if constexpr (__decay_copyable<_Sender>) {
          auto& [__tag, __fn, __child] = __sndr;
          return __make_sexpr<__let_tag<_SetTag>>(__data{
            STDEXEC::__forward_like<_Sender>(__child), STDEXEC::__forward_like<_Sender>(__fn)});
        } else {
          return __not_a_sender<_SENDER_TYPE_IS_NOT_COPYABLE_, _WITH_PRETTY_SENDER_<_Sender>>();
        }
      }
    };

    template <class _SetTag>
    struct __let_impl : __sexpr_defaults {
      static constexpr auto get_attrs =
        []<class _Child, class _Fun>(
          const __data<_Child, _Fun>& __data) noexcept -> decltype(auto) {
        // BUGBUG: TODO(ericniebler): this needs a proper implementation
        return __fwd_env(STDEXEC::get_env(__data.__sndr));
      };

      template <class _Sender, class... _Env>
      static consteval auto get_completion_signatures() {
        static_assert(sender_expr_for<_Sender, __let_tag<_SetTag>>);
        if constexpr (__decay_copyable<_Sender>) {
          using __fn_t = __decay_t<__fn_of_t<_Sender>>;
          // TODO: update this to use constant evaluation
          using __result_t =
            __completions_t<__let_tag<_SetTag>, __fn_t, __child_of_t<_Sender>, _Env...>;
          if constexpr (__ok<__result_t>) {
            return __result_t();
          } else {
            return STDEXEC::__throw_compile_time_error(__result_t());
          }
        } else {
          return STDEXEC::__throw_compile_time_error<
            _SENDER_TYPE_IS_NOT_COPYABLE_,
            _WITH_PRETTY_SENDER_<_Sender>
          >();
        }
      }

      static constexpr auto get_state =
        []<class _Receiver, __decay_copyable _Sender>(_Sender&& __sndr, _Receiver& __rcvr)
        requires sender_in<__child_of_t<_Sender>, env_of_t<_Receiver>>
      {
        static_assert(sender_expr_for<_Sender, __let_tag<_SetTag>>);
        using _Child = __child_of_t<_Sender>;
        using _Fun = __decay_t<__fn_of_t<_Sender>>;
        using __mk_let_state = __mbind_front_q<__let_state, _SetTag, _Child, _Fun, _Receiver>;
        using __let_state_t = __gather_completions_of_t<
          _SetTag,
          _Child,
          env_of_t<_Receiver>,
          __q<__decayed_tuple>,
          __mk_let_state
        >;
        auto&& [__tag, __data] = static_cast<_Sender&&>(__sndr);
        return __let_state_t(
          __forward_like<_Sender>(__data).__sndr, __forward_like<_Sender>(__data).__fn, __rcvr);
      };

      static constexpr auto start =
        []<class _State, class _Receiver>(_State& __state, _Receiver&) noexcept {
          ::STDEXEC::start(__state.__storage_.template get<1>());
        };
    };
  } // namespace __let

  inline constexpr let_value_t let_value{};
  inline constexpr let_error_t let_error{};
  inline constexpr let_stopped_t let_stopped{};

  template <class _SetTag>
  struct __sexpr_impl<__let::__let_tag<_SetTag>> : __let::__let_impl<_SetTag> { };

  template <class _SetTag>
  struct __sexpr_impl<__let::__let_t<_SetTag>> : __sexpr_defaults {
    template <class _Sender, class... _Env>
    static consteval auto get_completion_signatures() {
      static_assert(sender_expr_for<_Sender, __let::__let_t<_SetTag>>);
      using __sndr_t =
        __detail::__transform_sender_result_t<__let::__let_t<_SetTag>, set_value_t, _Sender, env<>>;
      return STDEXEC::get_completion_signatures<__sndr_t, _Env...>();
    }
  };
} // namespace STDEXEC
