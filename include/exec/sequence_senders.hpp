/*
 * Copyright (c) 2023 Maikel Nadolski
 * Copyright (c) 2023 NVIDIA Corporation
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

#include "../stdexec/execution.hpp"

namespace exec {
  struct sequence_tag { };

  namespace __sequence_sndr {
    using namespace stdexec;

    template <class _Haystack>
    struct __mall_contained_in_impl {
      template <class... _Needles>
      using __f = __mand<__mapply<__contains<_Needles>, _Haystack>...>;
    };

    template <class _Needles, class _Haystack>
    using __mall_contained_in = __mapply<__mall_contained_in_impl<_Haystack>, _Needles>;

    template <class _Needles, class _Haystack>
    concept __all_contained_in = __mall_contained_in<_Needles, _Haystack>::value;

    // This concept checks if a given sender satisfies the requirements to be returned from `set_next`.
    template <class _Sender, class _Env = empty_env>
    concept next_sender =      //
      sender_in<_Sender, _Env> //
      && __all_contained_in<
        completion_signatures_of_t<_Sender, _Env>,
        completion_signatures<set_value_t(), set_stopped_t()>>;

    // This is a sequence-receiver CPO that is used to apply algorithms on an input sender and it
    // returns a next-sender. `set_next` is usually called in a context where a sender will be
    // connected to a receiver. Since calling `set_next` usually involves constructing senders it
    // is allowed to throw an excpetion, which needs to be handled by a calling sequence-operation.
    // The returned object is a sender that can complete with `set_value_t()` or `set_stopped_t()`.
    struct set_next_t {
      template <receiver _Receiver, sender _Item>
        requires tag_invocable<set_next_t, _Receiver&, _Item>
      auto operator()(_Receiver& __rcvr, _Item&& __item) const
        noexcept(nothrow_tag_invocable<set_next_t, _Receiver&, _Item>)
          -> tag_invoke_result_t<set_next_t, _Receiver&, _Item> {
        static_assert(
          next_sender<tag_invoke_result_t<set_next_t, _Receiver&, _Item>>,
          "The sender returned from set_next is required to complete with set_value_t() or "
          "set_stopped_t()");
        return tag_invoke(*this, __rcvr, (_Item&&) __item);
      }
    };
  } // namespace __sequence_sndr

  using __sequence_sndr::set_next_t;
  inline constexpr set_next_t set_next;

  template <class _Receiver, class _Sender>
  using __next_sender_of_t = decltype(exec::set_next(
    stdexec::__declval<std::__decay_t<_Receiver>&>(),
    stdexec::__declval<_Sender>()));

  namespace __sequence_sndr {
    struct __nop_operation {
      friend void tag_invoke(start_t, __nop_operation&) noexcept {
      }
    };

    template <__is_completion_signatures _Sigs>
    struct __unspecified_sender_of {
      using is_sender = void;
      using completion_signatures = _Sigs;
      using __id = __unspecified_sender_of;
      using __t = __unspecified_sender_of;

      template <class R>
      friend __nop_operation tag_invoke(connect_t, __unspecified_sender_of, R&&) {
        return {};
      }
    };
  } // namespace __sequence_sndr

  template <class _Sender>
  concept __enable_sequence_sender =             //
    requires { typename _Sender::is_sender; } && //
    stdexec::same_as<typename _Sender::is_sender, sequence_tag>;

  template <class _Sender>
  inline constexpr bool enable_sequence_sender = __enable_sequence_sender<_Sender>;

  /////////////////////////////////////////////////////////////////////////////
  // get_sequence_signatures
  //
  namespace __get_sequence_signatures {
    using namespace stdexec;

    struct get_sequence_signatures_t;

    template <class _Sender, class _Env>
    using __sequence_signatures_of_t =
      stdexec::__call_result_t<get_sequence_signatures_t, _Sender, _Env>;

    template <class _Sender, class _Env>
    concept __r7_style_sender = same_as<_Env, no_env> && enable_sequence_sender<__decay_t<_Sender>>;

    template <class _Sender, class _Env>
    concept __with_tag_invoke =
      __valid<tag_invoke_result_t, get_sequence_signatures_t, _Sender, _Env>;

    template <class _Sender, class...>
    using __member_alias_t = typename __decay_t<_Sender>::sequence_signatures;

    template <class _Sender>
    concept __with_member_alias = __valid<__member_alias_t, _Sender>;

    struct get_sequence_signatures_t {
      template <class _Sender, class _Env>
      static auto __impl() {
        static_assert(STDEXEC_LEGACY_R5_CONCEPTS() || !same_as<_Env, no_env>);
        static_assert(sizeof(_Sender), "Incomplete type used with get_sequence_signatures");
        static_assert(sizeof(_Env), "Incomplete type used with get_sequence_signatures");
        if constexpr (__with_tag_invoke<_Sender, _Env>) {
          using _Result = tag_invoke_result_t<get_sequence_signatures_t, _Sender, _Env>;
          if constexpr (same_as<_Env, no_env> && __merror<_Result>) {
            return (dependent_completion_signatures<no_env>(*)()) nullptr;
          } else {
            return (_Result(*)()) nullptr;
          }
        } else if constexpr (__with_member_alias<_Sender>) {
          return (__member_alias_t<_Sender, _Env>(*)()) nullptr;
        } else if constexpr (__callable<get_completion_signatures_t, _Sender, _Env>) {
          using _Result = __call_result_t<get_completion_signatures_t, _Sender, _Env>;
          if constexpr (same_as<_Result, dependent_completion_signatures<no_env>>) {
            return (dependent_completion_signatures<no_env>(*)()) nullptr;
          } else {
            return (_Result(*)()) nullptr;
          }
        } else
#if STDEXEC_LEGACY_R5_CONCEPTS()
          if constexpr (__r7_style_sender<_Sender, _Env>) {
          return (dependent_completion_signatures<no_env>(*)()) nullptr;
        } else
#endif
          if constexpr (__is_debug_env<_Env>) {
          using __tag_invoke::tag_invoke;
          // This ought to cause a hard error that indicates where the problem is.
          using _Completions
            [[maybe_unused]] = tag_invoke_result_t<get_sequence_signatures_t, _Sender, _Env>;
          return (__debug::__completion_signatures(*)()) nullptr;
        } else {
          return (void (*)()) nullptr;
        }
      }

      template <class _Sender, class _Env = __default_env>
        requires(
          __with_tag_invoke<_Sender, _Env> ||                       //
          __with_member_alias<_Sender> ||                           //
          __callable<get_completion_signatures_t, _Sender, _Env> || //
#if STDEXEC_LEGACY_R5_CONCEPTS()                                    //
          __r7_style_sender<_Sender, _Env> ||                       //
#endif                                                              //
          __is_debug_env<_Env>)                                     //
      constexpr auto operator()(_Sender&& __sndr, const _Env& __env) const noexcept
        -> decltype(__impl<_Sender, _Env>()()) {
        return {};
      }
    };

    template <class _Tag, class _Sigs, class _Tuple, class _Variant>
    using __gather_sigs_for = //
      __meval<                //
        __gather_signal,
        _Tag,
        _Sigs,
        _Tuple,
        _Variant>;

    template <                             //
      class _Sigs,                         //
      class _Tuple = __q<__decayed_tuple>, //
      class _Variant = __q<__variant>>
    using __try_value_sigs_of_t =          //
      __gather_sigs_for<set_value_t, _Sigs, _Tuple, _Variant>;

    template <                             //
      class _Sigs,                         //
      class _Tuple = __q<__decayed_tuple>, //
      class _Variant = __q<__variant>>
      requires __is_completion_signatures<_Sigs>
    using __value_sigs_of_t = //
      __msuccess_or_t<__try_value_types_of_t<_Sigs, _Tuple, _Variant>>;

    template <class _Sigs, class _Variant = __q<__variant>>
    using __try_error_sigs_of_t = __gather_sigs_for<set_error_t, _Sigs, __q<__midentity>, _Variant>;

    template <class _Sigs, class _Variant = __q<__variant>>
    using __error_sigs_of_t = __msuccess_or_t<__try_error_sigs_of_t<_Sigs, _Variant>>;

    template <class _Tag, class _Sigs>
    using __try_count_sigs_of = //
      __compl_sigs::__maybe_for_all_sigs< _Sigs, __q<__mfront>, __mcount<_Tag>>;

    template <class _Tag, class _Sigs>
      requires __is_completion_signatures<_Sigs>
    using __count_sigs_of = __msuccess_or_t<__try_count_sigs_of<_Tag, _Sigs>>;

    template <class _Sender, class _Env, class _Sigs, class _SetVal, class _SetErr, class _SetStp>
    using __seq_sigs_impl = //
      __concat_completion_signatures_t<
        _Sigs,
        __mtry_eval<
          __try_value_sigs_of_t,
          __sequence_signatures_of_t<_Sender, _Env>,
          _SetVal,
          __q<__compl_sigs::__ensure_concat>>,
        __mtry_eval<
          __try_error_sigs_of_t,
          __sequence_signatures_of_t<_Sender, _Env>,
          __transform<_SetErr, __q<__compl_sigs::__ensure_concat>>>,
        __if<
          __try_count_sigs_of<set_stopped_t, __sequence_signatures_of_t<_Sender, _Env>>,
          _SetStp,
          completion_signatures<>>>;

    template <class _Sender, class _Env, class _Sigs, class _SetVal, class _SetErr, class _SetStp>
      requires __valid<__seq_sigs_impl, _Sender, _Env, _Sigs, _SetVal, _SetErr, _SetStp>
    extern __seq_sigs_impl<_Sender, _Env, _Sigs, _SetVal, _SetErr, _SetStp> __seq_sigs_v;

    template <class _Sender, class _Env, class _Sigs, class _SetVal, class _SetErr, class _SetStp>
    using __seq_sigs_t = decltype(__seq_sigs_v<_Sender, _Env, _Sigs, _SetVal, _SetErr, _SetStp>);

    template <bool>
    struct __make_seq_sigs {
      template <class _Sender, class _Env, class _Sigs, class _SetVal, class _SetErr, class _SetStp>
      using __f = __seq_sigs_t<_Sender, _Env, _Sigs, _SetVal, _SetErr, _SetStp>;
    };

    template <>
    struct __make_seq_sigs<true> {
      template <class _Sender, class _Env, class _Sigs, class _SetVal, class _SetErr, class _SetStp>
      using __f = //
        __msuccess_or_t<
          __seq_sigs_t<_Sender, _Env, _Sigs, _SetVal, _SetErr, _SetStp>,
          dependent_completion_signatures<_Env>>;
    };


    template <                                                    //
      class _Sender,                                              //
      class _Env = __default_env,                                 //
      class _Sigs = completion_signatures<>,                      //
      class _SetValue = __q<__compl_sigs::__default_set_value>,   //
      class _SetError = __q<__compl_sigs::__default_set_error>,   //
      class _SetStopped = completion_signatures<set_stopped_t()>> //
    using __try_make_sequence_signatures =                        //
      __minvoke<
        __make_seq_sigs<same_as<_Env, no_env>>,
        _Sender,
        _Env,
        _Sigs,
        _SetValue,
        _SetError,
        _SetStopped>;
  } // namespace __get_sequence_signatures

  using __get_sequence_signatures::__value_sigs_of_t;
  using __get_sequence_signatures::__error_sigs_of_t;
  using __get_sequence_signatures::__try_make_sequence_signatures;
  using __get_sequence_signatures::__sequence_signatures_of_t;

  using __get_sequence_signatures::get_sequence_signatures_t;
  inline constexpr get_sequence_signatures_t get_sequence_signatures{};

  template <class _Sender>
  concept sequence_sender =     //
    stdexec::sender<_Sender> && //
    enable_sequence_sender<stdexec::__decay_t<_Sender>>;

  template <class _Sender, class _Env>
  concept sequence_sender_in =           //
    stdexec::sender_in<_Sender, _Env> && //
    sequence_sender<_Sender> &&          //
    requires(_Sender&& __sndr, _Env&& __env) {
      get_sequence_signatures((_Sender&&) __sndr, (_Env&&) __env);
    } && //
    stdexec::__valid_completion_signatures<__sequence_signatures_of_t<_Sender, _Env>, _Env>;


  template <class _Receiver, class _SequenceSigs>
  concept sequence_receiver_of =    //
    stdexec::receiver<_Receiver> && //
    stdexec::__callable<
      set_next_t,
      stdexec::__decay_t<_Receiver>&,
      __sequence_sndr::__unspecified_sender_of<_SequenceSigs>>;

  template <class _Receiver, class _Sender>
  concept sequence_receiver_from =                               //
    stdexec::receiver<_Receiver> &&                              //
    stdexec::sender_in<_Sender, stdexec::env_of_t<_Receiver>> && //
    sequence_receiver_of<
      _Receiver,
      __sequence_signatures_of_t<_Sender, stdexec::env_of_t<_Receiver>>>
    && //
    ((sequence_sender_in<_Sender, stdexec::env_of_t<_Receiver>>
      && stdexec::__receiver_from<_Receiver, _Sender>)
     || //
     (!sequence_sender_in<_Sender, stdexec::env_of_t<_Receiver>>
      && stdexec::__receiver_from<_Receiver, __next_sender_of_t<_Receiver, _Sender>>) );

  namespace __sequence_sndr {
    struct sequence_connect_t;

    template <class _ReceiverId>
    struct __stopped_means_break {
      struct __t {
        using is_receiver = void;
        using __id = __stopped_means_break;
        using _Receiver = stdexec::__t<_ReceiverId>;
        using _Token = stop_token_of_t<env_of_t<_Receiver>>;
        STDEXEC_NO_UNIQUE_ADDRESS _Receiver __rcvr_;

        template <same_as<get_env_t> _GetEnv, same_as<__t> _Self>
        friend env_of_t<_Receiver> tag_invoke(_GetEnv, const _Self& __self) noexcept {
          return stdexec::get_env(__self.__rcvr_);
        }

        template <same_as<set_value_t> _SetValue, same_as<__t> _Self>
          requires __callable<set_value_t, _Receiver&&>
        friend void tag_invoke(_SetValue, _Self&& __self) noexcept {
          return stdexec::set_value(static_cast<_Receiver&&>(__self.__rcvr_));
        }

        template <same_as<set_stopped_t> _SetStopped, same_as<__t> _Self>
          requires __callable<set_value_t, _Receiver&&>
                && (unstoppable_token<_Token> || __callable<set_stopped_t, _Receiver&&>)
        friend void tag_invoke(_SetStopped, _Self&& __self) noexcept {
          if constexpr (unstoppable_token<_Token>) {
            stdexec::set_value(static_cast<_Receiver&&>(__self.__rcvr_));
          } else {
            auto __token = stdexec::get_stop_token(stdexec::get_env(__self.__rcvr_));
            if (__token.stop_requested()) {
              stdexec::set_stopped(static_cast<_Receiver&&>(__self.__rcvr_));
            } else {
              stdexec::set_value(static_cast<_Receiver&&>(__self.__rcvr_));
            }
          }
        }
      };
    };

    template <class _Rcvr>
    using __stopped_means_break_t = __t<__stopped_means_break<__id<__decay_t<_Rcvr>>>>;

    template <class _Env>
    using __single_sender_completion_sigs = __if_c<
      unstoppable_token<stop_token_of_t<_Env>>,
      completion_signatures<set_value_t()>,
      completion_signatures<set_value_t(), set_stopped_t()>>;

    template <class _Sender, class _Receiver>
    concept __next_connectable_with_tag_invoke =
      receiver<_Receiver> &&                               //
      sender_in<_Sender, env_of_t<_Receiver>> &&           //
      !sequence_sender_in<_Sender, env_of_t<_Receiver>> && //
      sequence_receiver_of<_Receiver, completion_signatures_of_t<_Sender, env_of_t<_Receiver>>>
      &&                                                   //
      __receiver_from<__stopped_means_break_t<_Receiver>, __next_sender_of_t<_Receiver, _Sender>>
      &&                                                   //
      __connect::__connectable_with_tag_invoke<
        __next_sender_of_t<_Receiver, _Sender>&&,
        __stopped_means_break_t<_Receiver>>;


    template <class _Sender, class _Receiver>
    concept __sequence_connectable_with_tag_invoke =
      receiver<_Receiver> &&                              //
      sequence_sender_in<_Sender, env_of_t<_Receiver>> && //
      sequence_receiver_from<_Receiver, _Sender> &&       //
      tag_invocable<sequence_connect_t, _Sender, _Receiver>;

    struct sequence_connect_t {
      template <class _Sender, class _Receiver>
      static constexpr auto __select_impl() noexcept {
        // Report that 2300R5-style senders and receivers are deprecated:
        if constexpr (!enable_sender<__decay_t<_Sender>>)
          __connect::__update_sender_type_to_p2300r7_by_adding_enable_sender_trait<
            __decay_t<_Sender>>();

        if constexpr (!enable_receiver<__decay_t<_Receiver>>)
          __connect::__update_receiver_type_to_p2300r7_by_adding_enable_receiver_trait<
            __decay_t<_Receiver>>();

        if constexpr (__next_connectable_with_tag_invoke<_Sender, _Receiver>) {
          using _Result = tag_invoke_result_t<
            connect_t,
            __next_sender_of_t<_Receiver, _Sender>,
            __stopped_means_break_t<_Receiver>>;
          constexpr bool _Nothrow = nothrow_tag_invocable<
            connect_t,
            __next_sender_of_t<_Receiver, _Sender>,
            __stopped_means_break_t<_Receiver>>;
          return static_cast<_Result (*)() noexcept(_Nothrow)>(nullptr);
        } else if constexpr (__sequence_connectable_with_tag_invoke<_Sender, _Receiver>) {
          using _Result = tag_invoke_result_t<sequence_connect_t, _Sender, _Receiver>;
          constexpr bool _Nothrow = nothrow_tag_invocable<sequence_connect_t, _Sender, _Receiver>;
          return static_cast<_Result (*)() noexcept(_Nothrow)>(nullptr);
        } else {
          return static_cast<__debug::__debug_operation (*)() noexcept>(nullptr);
        }
      }

      template <class _Sender, class _Receiver>
      using __select_impl_t = decltype(__select_impl<_Sender, _Receiver>());

      template <sender _Sender, receiver _Receiver>
        requires __next_connectable_with_tag_invoke<_Sender, _Receiver>
              || __sequence_connectable_with_tag_invoke<_Sender, _Receiver>
              || __is_debug_env<env_of_t<_Receiver>>
      auto operator()(_Sender&& __sndr, _Receiver&& __rcvr) const
        noexcept(__nothrow_callable<__select_impl_t<_Sender, _Receiver>>)
          -> __call_result_t<__select_impl_t<_Sender, _Receiver>> {
        if constexpr (__next_connectable_with_tag_invoke<_Sender, _Receiver>) {
          static_assert(
            operation_state<
              tag_invoke_result_t<connect_t, __next_sender_of_t<_Receiver, _Sender>, _Receiver>>,
            "stdexec::connect(sender, receiver) must return a type that "
            "satisfies the operation_state concept");
          __next_sender_of_t<_Receiver, _Sender> __next = set_next(__rcvr, (_Sender&&) __sndr);
          return tag_invoke(
            connect_t{},
            (__next_sender_of_t<_Receiver, _Sender>&&) __next,
            __stopped_means_break_t<_Receiver>{(_Receiver&&) __rcvr});
        } else if constexpr (__sequence_connectable_with_tag_invoke<_Sender, _Receiver>) {
          static_assert(
            operation_state<tag_invoke_result_t<sequence_connect_t, _Sender, _Receiver>>,
            "stdexec::sequence_connect(sender, receiver) must return a type that "
            "satisfies the operation_state concept");
          return tag_invoke(sequence_connect_t{}, (_Sender&&) __sndr, (_Receiver&&) __rcvr);
        } else {
          // This should generate an instantiate backtrace that contains useful
          // debugging information.
          using __tag_invoke::tag_invoke;
          tag_invoke(*this, (_Sender&&) __sndr, (_Receiver&&) __rcvr);
        }
      }

      friend constexpr bool tag_invoke(forwarding_query_t, connect_t) noexcept {
        return false;
      }
    };

    template <class _Sender, class _Receiver>
    using sequence_connect_result_t = __call_result_t<sequence_connect_t, _Sender, _Receiver>;
  } // namespace __sequence_sndr

  using __sequence_sndr::__single_sender_completion_sigs;

  using __sequence_sndr::sequence_connect_t;
  inline constexpr sequence_connect_t sequence_connect;

  using __sequence_sndr::sequence_connect_result_t;

  template <class _Sender, class _Receiver>
  concept sequence_sender_to =
    sequence_receiver_from<_Receiver, _Sender> && //
    requires(_Sender&& __sndr, _Receiver&& __rcvr) {
      { sequence_connect((_Sender&&) __sndr, (_Receiver&&) __rcvr) };
    };
}