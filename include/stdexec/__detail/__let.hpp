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
#include "__schedulers.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__senders.hpp"
#include "__submit.hpp"
#include "__transform_completion_signatures.hpp"
#include "__utility.hpp"
#include "__variant.hpp"

#include <exception>

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

    template <class _SetTag, class...>
    extern __undefined<_SetTag> __let_from_set;

    template <class... _Ign>
    extern let_value_t __let_from_set<set_value_t, _Ign...>;

    template <class... _Ign>
    extern let_error_t __let_from_set<set_error_t, _Ign...>;

    template <class... _Ign>
    extern let_stopped_t __let_from_set<set_stopped_t, _Ign...>;

    template <class _SetTag>
    using __on_not_callable =
      __mbind_front_q<__callable_error_t, decltype(__let_from_set<_SetTag>)>;

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

    template <class _Receiver, class _Env2>
    struct __rcvr_env {
      using receiver_concept = receiver_t;
      template <class... _As>
      constexpr void set_value(_As&&... __as) noexcept {
        STDEXEC::set_value(static_cast<_Receiver&&>(__rcvr_), static_cast<_As&&>(__as)...);
      }

      template <class _Error>
      constexpr void set_error(_Error&& __err) noexcept {
        STDEXEC::set_error(static_cast<_Receiver&&>(__rcvr_), static_cast<_Error&&>(__err));
      }

      constexpr void set_stopped() noexcept {
        STDEXEC::set_stopped(static_cast<_Receiver&&>(__rcvr_));
      }

      [[nodiscard]]
      constexpr decltype(__env::__join(
        __declval<const _Env2&>(),
        __declval<STDEXEC::env_of_t<_Receiver&>>())) get_env() const noexcept {
        return __env::__join(__env_, STDEXEC::get_env(__rcvr_));
      }

      _Receiver& __rcvr_;
      const _Env2& __env_;
    };

    struct _FUNCTION_MUST_RETURN_A_VALID_SENDER_IN_THE_CURRENT_ENVIRONMENT_ { };

    template <class...>
    struct _NESTED_ERROR_;

    template <class _Sender, class... _Env>
    using __try_completion_signatures_of_t = __minvoke_or_q<
      __completion_signatures_of_t,
      __unrecognized_sender_error_t<_Sender, _Env...>,
      _Sender,
      _Env...
    >;

    template <class _Sender, class _SetTag, class... _JoinEnv2>
    using __bad_result_sender = __not_a_sender<
      _WHAT_(_FUNCTION_MUST_RETURN_A_VALID_SENDER_IN_THE_CURRENT_ENVIRONMENT_),
      _WHERE_(_IN_ALGORITHM_, decltype(__let_from_set<_SetTag>)),
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
      using __f = __minvoke_q<
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
    using __result_receiver_t = __rcvr_env<_Receiver, _Env2>;

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

    template <class _LetTag, class _Fun, class _CvSender, class... _Env>
    using __completions_t = __gather_completion_signatures_t<
      __completion_signatures_of_t<_CvSender, _Env...>,
      __t<_LetTag>,
      __transform_signal_fn<
        __t<_LetTag>,
        _Fun,
        __result_env_t<__t<_LetTag>, env_of_t<_CvSender>, _Env>...
      >::template __f,
      __cmplsigs::__default_completion,
      __mtry_q<__concat_completion_signatures_t>::__f
    >;

    struct _THE_SENDERS_RETURNED_BY_THE_GIVEN_FUNCTION_DO_NOT_SHARE_A_COMMON_DOMAIN_ { };

    template <class _SetTag, class... _Env>
    struct __try_common_domain_fn {
      struct __error_fn {
        template <class... _Senders>
        using __f = __mexception<
          _WHAT_(_DOMAIN_ERROR_),
          _WHY_(_THE_SENDERS_RETURNED_BY_THE_GIVEN_FUNCTION_DO_NOT_SHARE_A_COMMON_DOMAIN_),
          _WHERE_(_IN_ALGORITHM_, decltype(__let_from_set<_SetTag>)),
          _WITH_PRETTY_SENDERS_<_Senders...>
        >;
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

    // A metafunction to check whether the predecessor's completion results are nothrow
    // decay-copyable and whether connecting the secondary sender is nothrow.
    template <class _Fn, class _Env2>
    struct __has_nothrow_completions_fn {
      using __rcvr2_t = __receiver_archetype<_Env2>;

      template <class... _Ts>
      using __f = __mbool<
        __nothrow_decay_copyable<_Ts...>
        && __nothrow_connectable<__call_result_t<_Fn, __decay_t<_Ts>&...>, __rcvr2_t>
      >;
    };

    template <class _SetTag, class _Child, class _Fn, class _Env>
    using __has_nothrow_completions_t = __gather_completions_t<
      completion_signatures_of_t<_Child, _Env>,
      _SetTag,
      __has_nothrow_completions_fn<_Fn, __result_env_t<_SetTag, env_of_t<_Child>, _Env>>,
      __qq<__mand_t>
    >;

    //! The core of the operation state for `let_*`.
    //! This gets bundled up into a larger operation state (`__detail::__op_state<...>`).
    template <class _SetTag, class _Fun, class _Receiver, class _Env2, class... _Tuples>
    struct __opstate_base : __immovable {
      using __env2_t = _Env2;
      using __second_rcvr_t = __rcvr_env<_Receiver, _Env2>;

      template <class _Attrs>
      constexpr explicit __opstate_base(
        _SetTag,
        const _Attrs& __attrs,
        _Fun __fn,
        _Receiver&& __rcvr) noexcept
        : __rcvr_(static_cast<_Receiver&&>(__rcvr))
        , __fn_(static_cast<_Fun&&>(__fn))
        // TODO(ericniebler): this needs a fallback:
        , __env2_(__let::__mk_env2<_SetTag>(__attrs, STDEXEC::get_env(__rcvr_))) {
      }

      constexpr virtual void __start_next() = 0;

      template <class _Tag, class... _Args>
      constexpr void __impl(_Tag, _Args&&... __args) noexcept {
        if constexpr (__same_as<_SetTag, _Tag>) {
          using __sender_t = __call_result_t<_Fun, __decay_t<_Args>&...>;
          using __submit_t = __submit_result_t<__sender_t, _Env2, _Receiver>;

          constexpr bool __nothrow_store = (__nothrow_decay_copyable<_Args> && ...);
          constexpr bool __nothrow_invoke = __nothrow_callable<_Fun, __decay_t<_Args>&...>;
          constexpr bool __nothrow_submit =
            __nothrow_constructible_from<__submit_t, __sender_t, __second_rcvr_t>;

          STDEXEC_TRY {
            __args_.__emplace_from(__mktuple, static_cast<_Args&&>(__args)...);
            __start_next();
          }
          STDEXEC_CATCH_ALL {
            if constexpr (!(__nothrow_store && __nothrow_invoke && __nothrow_submit)) {
              STDEXEC::set_error(static_cast<_Receiver&&>(this->__rcvr_), std::current_exception());
            }
          }
        } else {
          _Tag()(static_cast<_Receiver&&>(this->__rcvr_), static_cast<_Args&&>(__args)...);
        }
      }

      _Receiver __rcvr_;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Fun __fn_;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Env2 __env2_;
      //! Variant to hold the child sender's results before passing them to the function:
      __variant<_Tuples...> __args_{__no_init};
    };

    template <class _SetTag, class _Fun, class _Receiver, class _Env2, class... _Tuples>
    struct __first_rcvr {
      using receiver_concept = STDEXEC::receiver_t;
      template <class... _Args>
      constexpr void set_value(_Args&&... __args) noexcept {
        __state_->__impl(STDEXEC::set_value, static_cast<_Args&&>(__args)...);
      }

      template <class... _Args>
      constexpr void set_error(_Args&&... __args) noexcept {
        __state_->__impl(STDEXEC::set_error, static_cast<_Args&&>(__args)...);
      }

      template <class... _Args>
      constexpr void set_stopped(_Args&&... __args) noexcept {
        __state_->__impl(STDEXEC::set_stopped, static_cast<_Args&&>(__args)...);
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> env_of_t<_Receiver> {
        return STDEXEC::get_env(__state_->__rcvr_);
      }

      __opstate_base<_SetTag, _Fun, _Receiver, _Env2, _Tuples...>* __state_;
    };

    constexpr auto __start_next_fn =
      []<class _Fun, class _Receiver, class _Env2, class _Storage, class _Tuple>(
        _Fun& __fn,
        _Receiver& __rcvr,
        _Env2& __env2,
        _Storage& __storage,
        _Tuple& __tupl) {
        using __sender_t = __apply_result_t<_Fun, decltype(__tupl)>;
        auto&& __sndr = STDEXEC::__apply(static_cast<_Fun&&>(__fn), __tupl);
        using __submit_t = __submit_result_t<__sender_t, _Env2, _Receiver>;
        using __second_rcvr_t = __rcvr_env<_Receiver, _Env2>;
        __second_rcvr_t __rcvr2{__rcvr, static_cast<_Env2&&>(__env2)};

        auto& __op = __storage.template emplace<__submit_t>(
          static_cast<__sender_t&&>(__sndr), static_cast<__second_rcvr_t&&>(__rcvr2));
        __op.submit(static_cast<__sender_t&&>(__sndr), static_cast<__second_rcvr_t&&>(__rcvr2));
      };

    //! The core of the operation state for `let_*`.
    //! This gets bundled up into a larger operation state (`__detail::__op_state<...>`).
    template <class _SetTag, class _Child, class _Fun, class _Receiver, class... _Tuples>
    struct __opstate final
      : __opstate_base<
          _SetTag,
          _Fun,
          _Receiver,
          __let::__env2_t<_SetTag, env_of_t<_Child>, env_of_t<_Receiver>>,
          _Tuples...
        > {
      using __env2_t = __opstate::__opstate_base::__env2_t;
      using __first_rcvr_t = __first_rcvr<_SetTag, _Fun, _Receiver, __env2_t, _Tuples...>;
      using __second_rcvr_t = __opstate::__opstate_base::__second_rcvr_t;

      using __op_state_variant_t = __variant<
        connect_result_t<_Child, __first_rcvr_t>,
        __mapply<__submit_datum_for<_Receiver, _Fun, _SetTag, __env2_t>, _Tuples>...
      >;

      constexpr explicit __opstate(_Child&& __child, _Fun __fn, _Receiver&& __rcvr) noexcept(
        __nothrow_connectable<_Child, __first_rcvr_t> && __nothrow_move_constructible<_Fun>)
        : __opstate::__opstate_base(
            _SetTag(),
            STDEXEC::get_env(__child),
            static_cast<_Fun&&>(__fn),
            static_cast<_Receiver&&>(__rcvr)) {
        __storage_
          .__emplace_from(STDEXEC::connect, static_cast<_Child&&>(__child), __first_rcvr_t{this});
      }

      constexpr void start() noexcept {
        STDEXEC_ASSERT(__storage_.index() == 0);
        STDEXEC::start(__var::__get<0>(__storage_));
      };

      constexpr void __start_next() final {
        STDEXEC_ASSERT(__storage_.index() == 0);
        if constexpr (sizeof...(_Tuples) != 0) {
          STDEXEC::__visit(
            __start_next_fn, this->__args_, this->__fn_, this->__rcvr_, this->__env2_, __storage_);
        }
      }

      //! Variant type for holding the operation state of the currently in flight operation
      __op_state_variant_t __storage_{__no_init};
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
      __result_of<__let::__get_completion_domain<_SetTag, _SetTag2, _Sndr, _Fn, _Env...>>,
      indeterminate_domain<>
    >;

    template <class _LetTag, class _Sndr, class _Fn>
    struct __attrs {
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
        requires(__has_nothrow_completions_t<__set_tag_t, _Sndr, _Fn, _Env>::value && ...)
      [[nodiscard]]
      constexpr auto query(get_completion_domain_t<_Tag>, const _Env&...) const noexcept
        -> __ensure_valid_domain_t<__common_domain_t<
          __completion_domain_of_t<_Tag, _Sndr, __fwd_env_t<_Env>...>,
          __let_completion_domain_t<_LetTag, _Tag, _Sndr, _Fn, _Env...>
        >> {
        return {};
      }

      template <class _Env>
        requires(!__has_nothrow_completions_t<__set_tag_t, _Sndr, _Fn, _Env>::value)
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

    template <class _Tag, class = void>
    extern const __undefined<_Tag> __set_tag_from_let_v;

    template <class _Void>
    extern const __declfn_t<set_value_t> __set_tag_from_let_v<let_value_t, _Void>;

    template <class _Void>
    extern const __declfn_t<set_error_t> __set_tag_from_let_v<let_error_t, _Void>;

    template <class _Void>
    extern const __declfn_t<set_stopped_t> __set_tag_from_let_v<let_stopped_t, _Void>;

    //! Implementation of the `let_*_t` types, where `_SetTag` is, e.g., `set_value_t` for `let_value`.
    template <class _LetTag>
    struct __let_t { // NOLINT(bugprone-crtp-constructor-accessibility)
      using __t = decltype(__set_tag_from_let_v<_LetTag>());

      template <sender _Sender, __movable_value _Fun>
      constexpr auto operator()(_Sender&& __sndr, _Fun __fn) const -> __well_formed_sender auto {
        return __make_sexpr<_LetTag>(static_cast<_Fun&&>(__fn), static_cast<_Sender&&>(__sndr));
      }

      template <class _Fun>
      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()(_Fun __fn) const {
        return __closure(*this, static_cast<_Fun&&>(__fn));
      }
    };

    template <class _LetTag>
    struct __impls : __sexpr_defaults {
     private:
      template <class _Sender>
      using __fn_t = __decay_t<__data_of<_Sender>>;

      template <class _Sender, class _Receiver>
      using __opstate_t = __gather_completions_of_t<
        __t<_LetTag>,
        __child_of<_Sender>,
        __fwd_env_t<env_of_t<_Receiver>>,
        __q<__decayed_tuple>,
        __mbind_front_q<__opstate, __t<_LetTag>, __child_of<_Sender>, __fn_t<_Sender>, _Receiver>
      >;
     public:
      static constexpr auto get_attrs =
        []<class _Child>(__ignore, __ignore, const _Child& __child) noexcept -> decltype(auto) {
        // TODO(ericniebler): this needs a proper implementation
        return __fwd_env(STDEXEC::get_env(__child));
      };

      template <class _Sender, class... _Env>
      static consteval auto get_completion_signatures() {
        static_assert(sender_expr_for<_Sender, _LetTag>);
        if constexpr (__decay_copyable<_Sender>) {
          // TODO: update this to use constant evaluation
          using __result_t =
            __completions_t<_LetTag, __fn_t<_Sender>, __child_of<_Sender>, _Env...>;
          if constexpr (__ok<__result_t>) {
            return __result_t();
          } else {
            return STDEXEC::__throw_compile_time_error(__result_t());
          }
        } else {
          return STDEXEC::__throw_compile_time_error<
            _SENDER_TYPE_IS_NOT_DECAY_COPYABLE_,
            _WITH_PRETTY_SENDER_<_Sender>
          >();
        }
      }

      static constexpr auto connect =
        []<class _CvSender, class _Receiver>(_CvSender&& __sndr, _Receiver __rcvr) noexcept(
          __nothrow_constructible_from<
            __opstate_t<_CvSender, _Receiver>,
            __child_of<_CvSender>,
            __data_of<_CvSender>,
            _Receiver
          >) -> __opstate_t<_CvSender, _Receiver> {
        static_assert(sender_expr_for<_CvSender, _LetTag>);
        auto& [__tag, __fn, __child] = __sndr;
        return __opstate_t<_CvSender, _Receiver>(
          STDEXEC::__forward_like<_CvSender>(__child),
          STDEXEC::__forward_like<_CvSender>(__fn),
          static_cast<_Receiver&&>(__rcvr));
      };
    };
  } // namespace __let

  struct let_value_t : __let::__let_t<let_value_t> { };
  struct let_error_t : __let::__let_t<let_error_t> { };
  struct let_stopped_t : __let::__let_t<let_stopped_t> { };

  inline constexpr let_value_t let_value{};
  inline constexpr let_error_t let_error{};
  inline constexpr let_stopped_t let_stopped{};

  template <>
  struct __sexpr_impl<let_value_t> : __let::__impls<let_value_t> { };

  template <>
  struct __sexpr_impl<let_error_t> : __let::__impls<let_error_t> { };

  template <>
  struct __sexpr_impl<let_stopped_t> : __let::__impls<let_stopped_t> { };
} // namespace STDEXEC
