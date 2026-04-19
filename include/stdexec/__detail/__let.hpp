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
#include "__completion_info.hpp"
#include "__diagnostics.hpp"
#include "__domain.hpp"
#include "__env.hpp"
#include "__meta.hpp"
#include "__schedulers.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__senders.hpp"
#include "__submit.hpp"
#include "__utility.hpp"
#include "__variant.hpp"

#include "../functional.hpp"

#include <exception>

namespace STDEXEC
{
  //////////////////////////////////////////////////////////////////////////////
  // [exec.let]
  namespace __let
  {
    template <class _SetTag>
    struct __let_t;

    template <class _SetTag, class _Sender, class _Env>
    using __env2_t = __secondary_env_t<_Sender, _Env, _SetTag>;

    template <class _SetTag, class _Sender, class _Env>
    using __result_env_t = __join_env_t<__env2_t<_SetTag, _Sender, _Env>, _Env>;

    template <class _Receiver, class _Env2>
    struct __rcvr_env
    {
      using receiver_concept = receiver_tag;

      _Receiver&    __rcvr_;
      _Env2 const & __env_;

      template <class... _As>
      constexpr void set_value(_As&&... __as) noexcept
      {
        STDEXEC::set_value(static_cast<_Receiver&&>(__rcvr_), static_cast<_As&&>(__as)...);
      }

      template <class _Error>
      constexpr void set_error(_Error&& __err) noexcept
      {
        STDEXEC::set_error(static_cast<_Receiver&&>(__rcvr_), static_cast<_Error&&>(__err));
      }

      constexpr void set_stopped() noexcept
      {
        STDEXEC::set_stopped(static_cast<_Receiver&&>(__rcvr_));
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept  //
        -> decltype(__env::__join(__env_, STDEXEC::get_env(__rcvr_)))
      {
        return __env::__join(__env_, STDEXEC::get_env(__rcvr_));
      }
    };

    struct _FUNCTION_MUST_RETURN_A_VALID_SENDER_IN_THE_CURRENT_ENVIRONMENT_;

    template <class...>
    struct _NESTED_ERROR_;

    // BUGBUG: when get_completion_signatures is using constexpr exceptions, this
    // __try_completion_signatures_of_t machinery hides the nested error from trying to
    // compute the result senders.
    template <class _Sender, class... _Env>
    using __try_completion_signatures_of_t =
      __minvoke_or_q<__completion_signatures_of_t,
                     __unrecognized_sender_error_t<_Sender, _Env...>,
                     _Sender,
                     _Env...>;

    template <class _Sender, class _LetTag, class... _JoinEnv2>
    using __bad_result_sender = __mexception<
      _WHAT_(_FUNCTION_MUST_RETURN_A_VALID_SENDER_IN_THE_CURRENT_ENVIRONMENT_),
      _WHERE_(_IN_ALGORITHM_, _LetTag),
      _WITH_PRETTY_SENDER_<_Sender>,
      __fn_t<_WITH_ENVIRONMENT_, _JoinEnv2>...,
      __mapply_q<_NESTED_ERROR_, __try_completion_signatures_of_t<_Sender, _JoinEnv2...>>>;

    template <class _LetTag, class... _Args>
    using __not_decay_copyable_error_t =
      __mexception<_WHAT_(_SENDER_RESULTS_ARE_NOT_DECAY_COPYABLE_),
                   _WHERE_(_IN_ALGORITHM_, _LetTag),
                   _WITH_ARGUMENTS_(_Args...)>;

    template <class _Sender, class... _JoinEnv2>
    concept __potentially_valid_sender_in = sender_in<_Sender, _JoinEnv2...>
                                         || (sender<_Sender> && (sizeof...(_JoinEnv2) == 0));

    //! Metafunction creating the operation state needed to connect the result of calling
    //! the sender factory function, `_Fun`, and passing its result to a receiver.
    template <class _Receiver, class _Fun, class _SetTag, class _Env2>
    struct __submit_datum_for
    {
      // compute the result of calling submit with the result of executing _Fun
      // with _Args. if the result is void, substitute with __ignore.
      template <class... _Args>
      using __f =
        submit_result<__invoke_result_t<_Fun, __decay_t<_Args>&...>, __rcvr_env<_Receiver, _Env2>>;
    };

    // A metafunction to check whether the predecessor's completion results are nothrow
    // decay-copyable and whether connecting the secondary sender is nothrow.
    template <class _Fn, class _Env2>
    struct __has_nothrow_completions_fn
    {
      template <class... _Ts>
      using __sndr2_t = __invoke_result_t<_Fn, __decay_t<_Ts>&...>;
      using __rcvr2_t = __receiver_archetype<_Env2>;

      template <class... _Ts>
      using __f = __mbool<__nothrow_decay_copyable<_Ts...>                 //
                          && __nothrow_invocable<_Fn, __decay_t<_Ts>&...>  //
                          && __nothrow_connectable<__sndr2_t<_Ts...>, __rcvr2_t>>;
    };

    template <class _SetTag, class _Child, class _Fn, class _Env>
    using __has_nothrow_completions_t = __gather_completions_t<
      completion_signatures_of_t<_Child, _Env>,
      _SetTag,
      __has_nothrow_completions_fn<_Fn, __result_env_t<_SetTag, _Child, _Env>>,
      __qq<__mand_t>>;

    //! The core of the operation state for `let_*`.
    //! This gets bundled up into a larger operation state (`__detail::__op_state<...>`).
    template <class _SetTag, class _Fun, class _Receiver, class _Env2, class... _Tuples>
    struct __opstate_base
    {
      using __env2_t        = _Env2;
      using __second_rcvr_t = __rcvr_env<_Receiver, _Env2>;

      template <class _CvFn, class _Sender>
      constexpr explicit __opstate_base(_SetTag,
                                        _CvFn           __cv,
                                        _Sender const & __sndr,
                                        _Fun            __fn,
                                        _Receiver&&     __rcvr) noexcept
        : __rcvr_(static_cast<_Receiver&&>(__rcvr))
        , __fn_(STDEXEC::__allocator_aware_forward(static_cast<_Fun&&>(__fn), __rcvr_))
        // TODO(ericniebler): this needs a fallback:
        , __env2_(__mk_secondary_env_t<_SetTag>()(__cv, __sndr, STDEXEC::get_env(__rcvr_)))
      {}

      STDEXEC_IMMOVABLE(__opstate_base);

      constexpr virtual void __start_next() = 0;

      template <class _Tag, class... _Args>
      constexpr void __impl(_Tag, _Args&&... __args) noexcept
      {
        if constexpr (__same_as<_SetTag, _Tag>)
        {
          using __sender_t = __invoke_result_t<_Fun, __decay_t<_Args>&...>;
          using __submit_t = submit_result<__sender_t, __rcvr_env<_Receiver, _Env2>>;

          constexpr bool __nothrow_store  = (__nothrow_decay_copyable<_Args> && ...);
          constexpr bool __nothrow_invoke = __nothrow_invocable<_Fun, __decay_t<_Args>&...>;
          constexpr bool __nothrow_submit =
            __nothrow_constructible_from<__submit_t, __sender_t, __second_rcvr_t>;

          STDEXEC_TRY
          {
            __args_.__emplace_from(__mktuple, static_cast<_Args&&>(__args)...);
            __start_next();
          }
          STDEXEC_CATCH_ALL
          {
            if constexpr (!(__nothrow_store && __nothrow_invoke && __nothrow_submit))
            {
              STDEXEC::set_error(static_cast<_Receiver&&>(this->__rcvr_), std::current_exception());
            }
          }
        }
        else
        {
          _Tag()(static_cast<_Receiver&&>(this->__rcvr_), static_cast<_Args&&>(__args)...);
        }
      }

      _Receiver __rcvr_;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Fun      __fn_;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Env2     __env2_;
      //! Variant to hold the child sender's results before passing them to the function:
      __variant<_Tuples...> __args_{__no_init};
    };

    template <class _SetTag, class _Fun, class _Receiver, class _Env2, class... _Tuples>
    struct __first_rcvr
    {
      using receiver_concept = STDEXEC::receiver_tag;
      template <class... _Args>
      constexpr void set_value(_Args&&... __args) noexcept
      {
        __state_->__impl(STDEXEC::set_value, static_cast<_Args&&>(__args)...);
      }

      template <class... _Args>
      constexpr void set_error(_Args&&... __args) noexcept
      {
        __state_->__impl(STDEXEC::set_error, static_cast<_Args&&>(__args)...);
      }

      template <class... _Args>
      constexpr void set_stopped(_Args&&... __args) noexcept
      {
        __state_->__impl(STDEXEC::set_stopped, static_cast<_Args&&>(__args)...);
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> env_of_t<_Receiver>
      {
        return STDEXEC::get_env(__state_->__rcvr_);
      }

      __opstate_base<_SetTag, _Fun, _Receiver, _Env2, _Tuples...>* __state_;
    };

    constexpr auto __mk_result_sndr =
      []<class _Fun, class... _Args>(_Fun& __fn, _Args&... __args) noexcept(
        __nothrow_invocable<_Fun, _Args&...>) -> decltype(auto)
    {
      return STDEXEC::__invoke(static_cast<_Fun&&>(__fn), __args...);
    };

    inline constexpr auto __start_next_fn =
      []<class _Fun, class _Receiver, class _Env2, class _Storage, class _Tuple>(
        _Fun&      __fn,
        _Receiver& __rcvr,
        _Env2&     __env2,
        _Storage&  __storage,
        _Tuple&    __tupl)
    {
      decltype(auto) __sndr = STDEXEC::__apply(__mk_result_sndr, __tupl, __fn);
      using __sender_t      = decltype(__sndr);
      using __submit_t      = submit_result<__sender_t, __rcvr_env<_Receiver, _Env2>>;
      using __second_rcvr_t = __rcvr_env<_Receiver, _Env2>;
      __second_rcvr_t __rcvr2{__rcvr, static_cast<_Env2&&>(__env2)};

      auto& __op = __storage.template emplace<__submit_t>(static_cast<__sender_t&&>(__sndr),
                                                          static_cast<__second_rcvr_t&&>(__rcvr2));
      __op.submit(static_cast<__sender_t&&>(__sndr), static_cast<__second_rcvr_t&&>(__rcvr2));
    };

    //! The core of the operation state for `let_*`.
    //! This gets bundled up into a larger operation state (`__detail::__op_state<...>`).
    template <class _SetTag, class _CvChild, class _Fun, class _Receiver, class... _Tuples>
    struct __opstate final
      : __opstate_base<_SetTag,
                       _Fun,
                       _Receiver,
                       __let::__env2_t<_SetTag, _CvChild, env_of_t<_Receiver>>,
                       _Tuples...>
    {
      using __env2_t        = __opstate::__opstate_base::__env2_t;
      using __first_rcvr_t  = __first_rcvr<_SetTag, _Fun, _Receiver, __env2_t, _Tuples...>;
      using __second_rcvr_t = __opstate::__opstate_base::__second_rcvr_t;

      using __op_state_variant_t =
        __variant<connect_result_t<_CvChild, __first_rcvr_t>,
                  __mapply<__submit_datum_for<_Receiver, _Fun, _SetTag, __env2_t>, _Tuples>...>;

      constexpr explicit __opstate(_CvChild&& __child, _Fun __fn, _Receiver&& __rcvr)
        noexcept(__nothrow_connectable<_CvChild, __first_rcvr_t>
                 && __nothrow_move_constructible<_Fun>)
        : __opstate::__opstate_base(_SetTag(),
                                    __copy_cvref_fn<_CvChild>{},
                                    __child,
                                    static_cast<_Fun&&>(__fn),
                                    static_cast<_Receiver&&>(__rcvr))
      {
        __storage_.__emplace_from(STDEXEC::connect,
                                  static_cast<_CvChild&&>(__child),
                                  __first_rcvr_t{this});
      }

      constexpr void start() noexcept
      {
        STDEXEC_ASSERT(__storage_.index() == 0);
        STDEXEC::start(__var::__get<0>(__storage_));
      };

      constexpr void __start_next() final
      {
        STDEXEC_ASSERT(__storage_.index() == 0);
        if constexpr (sizeof...(_Tuples) != 0)
        {
          STDEXEC::__visit(__start_next_fn,
                           this->__args_,
                           this->__fn_,
                           this->__rcvr_,
                           this->__env2_,
                           __storage_);
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
    //       * the stopped completions of the secondary sender
    //
    template <class _SetTag, class _Fun, class... _JoinEnv2>
    struct __result_completion_behavior_fn
    {
      template <class... _Ts>
      [[nodiscard]]
      static constexpr auto __impl() noexcept
      {
        using __sndr_t =
          __minvoke_or_q<__invoke_result_t, __not_a_sender<>, _Fun, __decay_t<_Ts>&...>;
        return STDEXEC::__get_completion_behavior<_SetTag, __sndr_t, _JoinEnv2...>();
      }

      template <class... _Ts>
      using __f = decltype(__impl<_Ts...>());
    };

    template <class _SetTag, class _Fun, class _Sender, class... _Env>
    struct __domain_transform_fn
    {
      template <class... _As>
      using __f = __completion_domain_of_t<_SetTag,
                                           __invoke_result_t<_Fun, __decay_t<_As>&...>,
                                           __result_env_t<_SetTag, _Sender, _Env>...>;
    };

    //! @tparam _LetTag The tag type for the let_ operation.
    //! @tparam _SetTag The completion signal of the let_ sender itself that is being
    //! queried. For example, you may be querying a let_value sender for its set_error
    //! completion domain.
    template <class _LetTag, class _SetTag, class _Sndr, class _Fun, class... _Env>
    [[nodiscard]]
    consteval auto __get_completion_domain() noexcept
    {
      if constexpr (sender_in<_Sndr, _Env...>)
      {
        using __domain_transform_fn = __let::__domain_transform_fn<_SetTag, _Fun, _Sndr, _Env...>;
        return __minvoke_or_q<__gather_completions_t,
                              indeterminate_domain<>,
                              __t<_LetTag>,
                              __completion_signatures_of_t<_Sndr, _Env...>,
                              __domain_transform_fn,
                              __qq<__common_domain_t>>();
      }
      else
      {
        return indeterminate_domain<>{};
      }
    }

    template <class _LetTag, class _SetTag, class _Sndr, class _Fun, class... _Env>
    using __let_completion_domain_t = __unless_one_of_t<
      __result_of<__let::__get_completion_domain<_LetTag, _SetTag, _Sndr, _Fun, _Env...>>,
      indeterminate_domain<>>;

    template <class _LetTag, class _Sndr, class _Fun>
    struct __attrs
    {
      using __set_tag_t = STDEXEC::__t<_LetTag>;

      template <class _Tag>
      constexpr auto query(get_completion_scheduler_t<_Tag>) const = delete;

      template <class... _Env>
      [[nodiscard]]
      constexpr auto query(get_completion_domain_t<__set_tag_t>, _Env const &...) const noexcept
        -> __ensure_valid_domain_t<
          __let_completion_domain_t<_LetTag, __set_tag_t, _Sndr, _Fun, _Env...>>
      {
        return {};
      }

      template <__one_of<set_error_t, set_stopped_t> _Tag, class... _Env>
        requires(__has_nothrow_completions_t<__set_tag_t, _Sndr, _Fun, _Env>::value && ...)
      [[nodiscard]]
      constexpr auto query(get_completion_domain_t<_Tag>, _Env const &...) const noexcept
        -> __ensure_valid_domain_t<
          __common_domain_t<__completion_domain_of_t<_Tag, _Sndr, __fwd_env_t<_Env>...>,
                            __let_completion_domain_t<_LetTag, _Tag, _Sndr, _Fun, _Env...>>>
      {
        return {};
      }

      template <class _Env>
        requires(!__has_nothrow_completions_t<__set_tag_t, _Sndr, _Fun, _Env>::value)
      [[nodiscard]]
      constexpr auto query(get_completion_domain_t<set_error_t>, _Env const &) const noexcept
        -> __ensure_valid_domain_t<
          __common_domain_t<__completion_domain_of_t<__set_tag_t, _Sndr, __fwd_env_t<_Env>>,
                            __completion_domain_of_t<set_error_t, _Sndr, __fwd_env_t<_Env>>,
                            __let_completion_domain_t<_LetTag, set_error_t, _Sndr, _Fun, _Env>>>
      {
        return {};
      }

      template <class... _Env>
      [[nodiscard]]
      constexpr auto query(__get_completion_behavior_t<__set_tag_t>, _Env const &...) const noexcept
      {
        if constexpr (sender_in<_Sndr, __fwd_env_t<_Env>...>)
        {
          // The completion behavior of let_value(sndr, fn) is the union of the completion
          // behavior of sndr and all the senders that fn can potentially produce. (MSVC
          // needs the constexpr computation broken up, hence the local variables.)
          using __transform_fn =
            __result_completion_behavior_fn<__set_tag_t,
                                            _Fun,
                                            __result_env_t<__set_tag_t, _Sndr, _Env>...>;
          using __completions_t = __completion_signatures_of_t<_Sndr, __fwd_env_t<_Env>...>;

          constexpr auto __pred_behavior =
            STDEXEC::__get_completion_behavior<__set_tag_t, _Sndr, __fwd_env_t<_Env>...>();
          constexpr auto __result_behaviors = __gather_completions_t<
            __set_tag_t,
            __completions_t,
            __transform_fn,
            __mbind_front_q<__call_result_t, __completion_behavior::__common_t>>();

          return __pred_behavior | __result_behaviors;
        }
        else
        {
          return __completion_behavior::__unknown;
        }
      }
    };

    //! Implementation of the `let_*_t` types, where `_SetTag` is, e.g., `set_value_t` for `let_value`.
    template <class _LetTag>
    struct __let_t
    {
      template <sender _Sender, __movable_value _Fun>
      constexpr auto operator()(_Sender&& __sndr, _Fun __fn) const -> __well_formed_sender auto
      {
        return __make_sexpr<_LetTag>(static_cast<_Fun&&>(__fn), static_cast<_Sender&&>(__sndr));
      }

      template <class _Fun>
      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()(_Fun __fn) const
      {
        return __closure(*this, static_cast<_Fun&&>(__fn));
      }

     private:
      friend _LetTag;
      __let_t() = default;
    };

    template <class _LetTag>
    struct __impls : __sexpr_defaults
    {
     private:
      using __set_t      = __t<_LetTag>;
      using __eptr_sig_t = set_error_t(std::exception_ptr);

      template <class _CvSender>
      using __fn_t = __decay_t<__data_of<_CvSender>>;

      template <class _CvSender, class _Env>
      using __env2_t = __let::__result_env_t<__set_t, _CvSender, _Env>;

      template <class _CvSender, class _Env>
      using __rcvr2_t = __receiver_archetype<__env2_t<_CvSender, _Env>>;

      template <class _CvSender, class _Receiver>
      using __opstate_t = __gather_completions_of_t<
        __set_t,
        __child_of<_CvSender>,
        __fwd_env_t<env_of_t<_Receiver>>,
        __q<__decayed_tuple>,
        __mbind_front_q<__opstate, __set_t, __child_of<_CvSender>, __fn_t<_CvSender>, _Receiver>>;

      template <class _Fun, class _Child, class... _Env>
      static constexpr auto __transform_cmplsig =                        //
        []<class... _As>(__set_t (*)(_As...), __completion_info __info)  //
        -> decltype(auto)
      {
        if constexpr (!__decay_copyable<_As...>)
        {
          using __what_t = __not_decay_copyable_error_t<_LetTag, _As...>;
          return STDEXEC::__throw_compile_time_error(__what_t());
        }
        else if constexpr (!__invocable<_Fun, __decay_t<_As>&...>)
        {
          using __what_t = __callable_error_t<_LetTag, _Fun, __decay_t<_As>&...>;
          return STDEXEC::__throw_compile_time_error(__what_t());
        }
        else if constexpr (!__potentially_valid_sender_in<
                             __invoke_result_t<_Fun, __decay_t<_As>&...>,
                             __env2_t<_Child, _Env>...>)
        {
          using __sndr_t = __invoke_result_t<_Fun, __decay_t<_As>&...>;
          using __what_t = __bad_result_sender<__sndr_t, _LetTag, __env2_t<_Child, _Env>...>;
          return STDEXEC::__throw_compile_time_error(__what_t());
        }
        else
        {
          using __sndr2_t = __invoke_result_t<_Fun, __decay_t<_As>&...>;
          auto __cmpls    = STDEXEC::__get_completion_info<__sndr2_t, __env2_t<_Child, _Env>...>();
          STDEXEC_IF_OK(__cmpls)
          {
            if constexpr (!__nothrow_decay_copyable<_As...>
                          || !__nothrow_invocable<_Fun, __decay_t<_As>&...>
                          || (!__nothrow_connectable<__sndr2_t, __rcvr2_t<_Child, _Env>> || ...))
            {
              __completion_info const __eptr_info(__signature<__eptr_sig_t>,
                                                  __info.__domain,
                                                  __info.__behavior);
              return __cmpls + STDEXEC::__make_static_vector(__eptr_info);
            }
            else
            {
              return __cmpls;
            }
          }
        }
      };

      template <__completion_info _Info>
      static constexpr auto __maybe_transform_cmplsig = [](auto __transform) -> decltype(auto)
      {
        if constexpr (_Info.__disposition != __set_t::__disposition)
          return STDEXEC::__make_static_vector(_Info);
        else
          return __transform(__signature<__msplice<_Info.__signature>>, _Info);
      };

      //! @tparam _Info A `__static_vector` of `__completion_info` objects representing
      //! the completions of the predecessor sender.
      template <auto _Info, auto _Transform>
      static constexpr auto __get_cmpl_info_i = []<std::size_t... _Is>(__indices<_Is...>)
      {
        return []
        {
          __static_vector<__completion_info, 0> __result;
          // NB: this fold uses an overloaded addition operator that propagates
          // __mexception objects when constexpr exceptions are not available.
          return (__maybe_transform_cmplsig<_Info[_Is]>(_Transform) + ... + __result);
        };
      };

      template <class _Fun, class _Child, class... _Env>
      struct __get_cmpl_info
      {
        constexpr auto operator()() const
        {
          constexpr auto __transform   = __transform_cmplsig<_Fun, _Child, _Env...>;
          constexpr auto __get_sig     = &__completion_info::__signature;
          constexpr auto __eptr_sig_id = __mtypeid<__eptr_sig_t>;
          constexpr auto __cmpls       = STDEXEC::__get_completion_info<_Child, _Env...>();

          STDEXEC_IF_OK(__cmpls)
          {
            constexpr auto __idx        = __make_indices<__cmpls.size()>();
            constexpr auto __get_cmpls2 = __get_cmpl_info_i<__cmpls, __transform>(__idx);
            constexpr auto __cmpls2     = __cmplsigs::__completion_info_from(__get_cmpls2);

            STDEXEC_IF_OK(__cmpls2)
            {
              if constexpr (std::ranges::find(__cmpls2, __eptr_sig_id, __get_sig) == __cmpls2.end()
                            && sizeof...(_Env) == 0)
                return STDEXEC::__dependent_sender<_Child>();
              else
                return __cmpls2;
            }
          }
        }
      };

     public:
      static constexpr auto __get_attrs =
        []<class _Child>(__ignore, __ignore, _Child const & __child) noexcept -> decltype(auto)
      {
        // TODO(ericniebler): this needs a proper implementation
        return __fwd_env(STDEXEC::get_env(__child));
      };

      template <class _CvSender, class... _Env>
      static consteval auto __get_completion_signatures()
      {
        static_assert(__sender_for<_CvSender, _LetTag>);
        constexpr auto __get_cmpl_info =
          __impls::__get_cmpl_info<__fn_t<_CvSender>, __child_of<_CvSender>, _Env...>();

        if constexpr (!__decay_copyable<_CvSender>)
          return STDEXEC::__throw_compile_time_error<_SENDER_TYPE_IS_NOT_DECAY_COPYABLE_,
                                                     _WITH_PRETTY_SENDER_<_CvSender>>();
        else
          return __cmplsigs::__completion_sigs_from(__get_cmpl_info);
      }

      static constexpr auto __connect =
        []<class _CvSender, class _Receiver>(_CvSender&& __sndr, _Receiver __rcvr) noexcept(
          __nothrow_constructible_from<__opstate_t<_CvSender, _Receiver>,
                                       __child_of<_CvSender>,
                                       __data_of<_CvSender>,
                                       _Receiver>) -> __opstate_t<_CvSender, _Receiver>
      {
        static_assert(__sender_for<_CvSender, _LetTag>);
        auto& [__tag, __fn, __child] = __sndr;
        return __opstate_t<_CvSender, _Receiver>(STDEXEC::__forward_like<_CvSender>(__child),
                                                 STDEXEC::__forward_like<_CvSender>(__fn),
                                                 static_cast<_Receiver&&>(__rcvr));
      };
    };
  }  // namespace __let

  struct let_value_t : __let::__let_t<let_value_t>
  {
    using __t     = set_value_t;
    let_value_t() = default;
  };
  struct let_error_t : __let::__let_t<let_error_t>
  {
    using __t     = set_error_t;
    let_error_t() = default;
  };
  struct let_stopped_t : __let::__let_t<let_stopped_t>
  {
    using __t       = set_stopped_t;
    let_stopped_t() = default;
  };

  inline constexpr let_value_t   let_value{};
  inline constexpr let_error_t   let_error{};
  inline constexpr let_stopped_t let_stopped{};

  template <>
  struct __sexpr_impl<let_value_t> : __let::__impls<let_value_t>
  {};

  template <>
  struct __sexpr_impl<let_error_t> : __let::__impls<let_error_t>
  {};

  template <>
  struct __sexpr_impl<let_stopped_t> : __let::__impls<let_stopped_t>
  {};
}  // namespace STDEXEC
