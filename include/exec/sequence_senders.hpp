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
#include "../stdexec/__detail/__concepts.hpp"
#include "../stdexec/__detail/__meta.hpp"

namespace exec {
  struct sequence_sender_t : stdexec::sender_t { };

  using sequence_tag [[deprecated("Renamed to exec::sequence_sender_t")]] = exec::sequence_sender_t;

  namespace __sequence_sndr {
    using namespace stdexec;

    template <class _Haystack>
    struct __mall_contained_in_impl {
      template <class... _Needles>
      using __f = __mand<__mapply<__mcontains<_Needles>, _Haystack>...>;
    };
    template <class _Needles, class _Haystack>
    using __mall_contained_in_t = __mapply<__mall_contained_in_impl<_Haystack>, _Needles>;

    template <class _Needles, class _Haystack>
    concept __all_contained_in = __v<__mall_contained_in_t<_Needles, _Haystack>>;
  } // namespace __sequence_sndr

  // This concept checks if a given sender satisfies the requirements to be returned from `set_next`.
  template <class _Sender, class _Env = stdexec::env<>>
  concept next_sender =
    stdexec::sender_in<_Sender, _Env>
    && __sequence_sndr::__all_contained_in<
      stdexec::completion_signatures_of_t<_Sender, _Env>,
      stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_stopped_t()>
    >;

  namespace __sequence_sndr {

    template <class _Receiver, class _Item>
    concept __has_set_next_member = requires(_Receiver& __rcvr, _Item&& __item) {
      __rcvr.set_next(static_cast<_Item&&>(__item));
    };

    // This is a sequence-receiver CPO that is used to apply algorithms on an input sender and it
    // returns a next-sender. `set_next` is usually called in a context where a sender will be
    // connected to a receiver. Since calling `set_next` usually involves constructing senders it
    // is allowed to throw an excpetion, which needs to be handled by a calling sequence-operation.
    // The returned object is a sender that can complete with `set_value_t()` or `set_stopped_t()`.
    struct set_next_t {
      template <receiver _Receiver, sender _Item>
        requires __has_set_next_member<_Receiver, _Item>
      auto operator()(_Receiver& __rcvr, _Item&& __item) const
        noexcept(noexcept(__rcvr.set_next(static_cast<_Item&&>(__item))))
          -> decltype(__rcvr.set_next(static_cast<_Item&&>(__item))) {
        return __rcvr.set_next(static_cast<_Item&&>(__item));
      }

      template <receiver _Receiver, sender _Item>
        requires(!__has_set_next_member<_Receiver, _Item>)
             && tag_invocable<set_next_t, _Receiver&, _Item>
      auto operator()(_Receiver& __rcvr, _Item&& __item) const
        noexcept(nothrow_tag_invocable<set_next_t, _Receiver&, _Item>)
          -> tag_invoke_result_t<set_next_t, _Receiver&, _Item> {
        static_assert(
          next_sender<tag_invoke_result_t<set_next_t, _Receiver&, _Item>>,
          "The sender returned from set_next is required to complete with set_value_t() or "
          "set_stopped_t()");
        return tag_invoke(*this, __rcvr, static_cast<_Item&&>(__item));
      }
    };
  } // namespace __sequence_sndr

  using __sequence_sndr::set_next_t;
  inline constexpr set_next_t set_next;

  template <class _Receiver, class _Sequence>
  using next_sender_of_t = decltype(exec::set_next(
    stdexec::__declval<stdexec::__decay_t<_Receiver>&>(),
    stdexec::__declval<_Sequence>()));

  namespace __sequence_sndr {

    template <class _ReceiverId>
    struct __stopped_means_break {
      struct __t {
        using receiver_concept = stdexec::receiver_t;
        using __id = __stopped_means_break;
        using _Receiver = stdexec::__t<_ReceiverId>;
        using __token_t = stop_token_of_t<env_of_t<_Receiver>>;
        STDEXEC_ATTRIBUTE(no_unique_address) _Receiver __rcvr_;

        auto get_env() const noexcept -> env_of_t<_Receiver> {
          return stdexec::get_env(__rcvr_);
        }

        void set_value() noexcept
          requires __callable<set_value_t, _Receiver>
        {
          return stdexec::set_value(static_cast<_Receiver&&>(__rcvr_));
        }

        void set_stopped() noexcept
          requires __callable<set_value_t, _Receiver>
                && (unstoppable_token<__token_t> || __callable<set_stopped_t, _Receiver>)
        {
          if constexpr (unstoppable_token<__token_t>) {
            stdexec::set_value(static_cast<_Receiver&&>(__rcvr_));
          } else {
            auto __token = stdexec::get_stop_token(stdexec::get_env(__rcvr_));
            if (__token.stop_requested()) {
              stdexec::set_stopped(static_cast<_Receiver&&>(__rcvr_));
            } else {
              stdexec::set_value(static_cast<_Receiver&&>(__rcvr_));
            }
          }
        }
      };
    };

    template <class _Rcvr>
    using __stopped_means_break_t = __t<__stopped_means_break<__id<__decay_t<_Rcvr>>>>;
  } // namespace __sequence_sndr

  template <class _Sequence>
  concept __enable_sequence_sender = requires {
    typename _Sequence::sender_concept;
  } && stdexec::derived_from<typename _Sequence::sender_concept, sequence_sender_t>;

  template <class _Sequence>
  inline constexpr bool enable_sequence_sender = __enable_sequence_sender<_Sequence>;

  template <class... _Senders>
  struct item_types { };

  template <class _Tp>
  concept __has_item_typedef = requires { typename _Tp::item_types; };

  namespace __debug {
    using namespace stdexec::__debug;

    struct __item_types { };
  } // namespace __debug

  namespace __errs {
    using namespace stdexec;
    inline constexpr __mstring __unrecognized_sequence_type_diagnostic =
      "The given type cannot be used as a sequence with the given environment "
      "because the attempt to compute the item types failed."_mstr;
  } // namespace __errs

  template <class _Sequence>
  struct _WITH_SEQUENCE_;

  template <class... _Sequences>
  struct _WITH_SEQUENCES_;

  template <stdexec::__mstring _Diagnostic = __errs::__unrecognized_sequence_type_diagnostic>
  struct _UNRECOGNIZED_SEQUENCE_TYPE_;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.seqtraits]
  namespace __sequence_sndr {
    struct get_item_types_t;

    template <class _Sequence, class... _Env>
    using __item_types_of_t = __call_result_t<get_item_types_t, _Sequence, _Env...>;

    template <class _Sequence, class... _Env>
    using __unrecognized_sequence_error_t = __mexception<
      _UNRECOGNIZED_SEQUENCE_TYPE_<>,
      _WITH_SEQUENCE_<_Sequence>,
      _WITH_ENVIRONMENT_<_Env>...
    >;

    template <class _Sequence, class _Env>
    using __member_result_t = decltype(__declval<_Sequence>().get_item_types(__declval<_Env>()));

    template <class _Sequence, class _Env>
    using __static_member_result_t = decltype(STDEXEC_REMOVE_REFERENCE(
      _Sequence)::get_item_types(__declval<_Sequence>(), __declval<_Env>()));

    template <class _Sequence, class _Env>
    using __tfx_sequence_t =
      transform_sender_result_t<__late_domain_of_t<_Sequence, _Env>, _Sequence, _Env>;

    template <class _Sequence, class _Env>
    concept __with_tag_invoke =
      tag_invocable<get_item_types_t, __tfx_sequence_t<_Sequence, _Env>, _Env>;

    template <class _Sequence, class _Env>
    using __member_alias_t = __decay_t<__tfx_sequence_t<_Sequence, _Env>>::item_types;

    template <class _Sequence, class _Env>
    concept __with_member_alias = __mvalid<__member_alias_t, _Sequence, _Env>;

    template <class _Sequence, class _Env>
    concept __with_static_member = __mvalid<__static_member_result_t, _Sequence, _Env>;

    template <class _Sequence, class... _Env>
    concept __with_member = __mvalid<__member_result_t, _Sequence, _Env...>;

    struct get_item_types_t {
      template <class _Sequence, class _Env>
      static auto __impl() {
        static_assert(sizeof(_Sequence), "Incomplete type used with get_item_types");
        static_assert(sizeof(_Env), "Incomplete type used with get_item_types");
        using __tfx_sequence_t = __tfx_sequence_t<_Sequence, _Env>;
        if constexpr (__merror<__tfx_sequence_t>) {
          // Computing the type of the transformed sender returned an error type. Propagate it.
          return static_cast<__tfx_sequence_t (*)()>(nullptr);
        } else if constexpr (__with_member_alias<__tfx_sequence_t, _Env>) {
          using __result_t = __member_alias_t<__tfx_sequence_t, _Env>;
          return static_cast<__result_t (*)()>(nullptr);
        } else if constexpr (__with_static_member<__tfx_sequence_t, _Env>) {
          using __result_t = __static_member_result_t<__tfx_sequence_t, _Env>;
          return static_cast<__result_t (*)()>(nullptr);
        } else if constexpr (__with_member<__tfx_sequence_t, _Env>) {
          using __result_t = decltype(__declval<__tfx_sequence_t>()
                                        .get_item_types(__declval<_Env>()));
          return static_cast<__result_t (*)()>(nullptr);
        } else if constexpr (__with_tag_invoke<__tfx_sequence_t, _Env>) {
          using __result_t = tag_invoke_result_t<get_item_types_t, __tfx_sequence_t, _Env>;
          return static_cast<__result_t (*)()>(nullptr);
        } else if constexpr (
          sender_in<__tfx_sequence_t, _Env>
          && !enable_sequence_sender<stdexec::__decay_t<__tfx_sequence_t>>) {
          using __result_t = item_types<stdexec::__decay_t<__tfx_sequence_t>>;
          return static_cast<__result_t (*)()>(nullptr);
        } else if constexpr (__is_debug_env<_Env>) {
          using __tag_invoke::tag_invoke;
          // This ought to cause a hard error that indicates where the problem is.
          using __item_types_t
            [[maybe_unused]] = tag_invoke_result_t<get_item_types_t, __tfx_sequence_t, _Env>;
          return static_cast<__debug::__item_types (*)()>(nullptr);
        } else {
          using __result_t = __unrecognized_sequence_error_t<_Sequence, _Env>;
          return static_cast<__result_t (*)()>(nullptr);
        }
      }

      template <class _Sequence, class _Env = env<>>
      constexpr auto operator()(_Sequence&&, _Env&& = {}) const noexcept
        -> decltype(__impl<_Sequence, _Env>()()) {
        return {};
      }
    };
  } // namespace __sequence_sndr

  using __sequence_sndr::get_item_types_t;
  inline constexpr get_item_types_t get_item_types{};

  template <class _Sequence, class... _Env>
  concept sequence_sender = stdexec::sender_in<_Sequence, _Env...>
                         && enable_sequence_sender<stdexec::__decay_t<_Sequence>>;

  template <class _Sequence, class... _Env>
  concept has_sequence_item_types = requires(_Sequence&& __sequence, _Env&&... __env) {
    { get_item_types(static_cast<_Sequence&&>(__sequence), static_cast<_Env&&>(__env)...) };
  };

  template <class _Sequence, class... _Env>
  concept sequence_sender_in = sequence_sender<_Sequence, _Env...>
                            && has_sequence_item_types<_Sequence, _Env...>;

  template <class _Sequence, class... _Env>
  using __item_types_of_t =
    decltype(get_item_types(stdexec::__declval<_Sequence>(), stdexec::__declval<_Env>()...));


  template <class _Item>
  struct _SEQUENCE_ITEM_IS_NOT_A_WELL_FORMED_SENDER_ { };

  template <class _Sequence, class _Item>
  auto __check_item(_Item*) -> stdexec::__mexception<
    _SEQUENCE_ITEM_IS_NOT_A_WELL_FORMED_SENDER_<_Item>,
    _WITH_SEQUENCE_<_Sequence>
  >;

  template <class _Sequence, class _Item>
    requires stdexec::__well_formed_sender<_Item>
  auto __check_item(_Item*) -> stdexec::__msuccess;

  template <class _Sequence, class _Items>
    requires stdexec::__merror<_Items>
  auto __check_items(_Items*) -> _Items;

  template <class _Item>
  struct _SEQUENCE_GET_ITEM_TYPES_RESULT_IS_NOT_WELL_FORMED_ { };

  template <class _Sequence, class _Items>
    requires(!stdexec::__merror<_Items>)
  auto __check_items(_Items*) -> stdexec::__mexception<
    _SEQUENCE_GET_ITEM_TYPES_RESULT_IS_NOT_WELL_FORMED_<_Items>,
    _WITH_SEQUENCE_<_Sequence>
  >;

  template <class _Sequence, class... _Items>
  auto __check_items(exec::item_types<_Items...>*) -> decltype((
    stdexec::__msuccess(),
    ...,
    exec::__check_item<_Sequence>(static_cast<_Items*>(nullptr))));

  template <class _Sequence>
    requires stdexec::__merror<_Sequence>
  auto __check_sequence(_Sequence*) -> _Sequence;

  struct _SEQUENCE_GET_ITEM_TYPES_IS_NOT_WELL_FORMED_ { };

  template <class _Sequence>
    requires(!stdexec::__merror<_Sequence>) && (!stdexec::__mvalid<__item_types_of_t, _Sequence>)
  auto __check_sequence(_Sequence*) -> stdexec::__mexception<
    _SEQUENCE_GET_ITEM_TYPES_IS_NOT_WELL_FORMED_,
    _WITH_SEQUENCE_<_Sequence>
  >;

  template <class _Sequence>
    requires(!stdexec::__merror<_Sequence>) && stdexec::__mvalid<__item_types_of_t, _Sequence>
  auto __check_sequence(_Sequence*) -> decltype(exec::__check_items<_Sequence>(
    static_cast<__item_types_of_t<_Sequence>*>(nullptr)));

  template <class _Sequence>
  concept __well_formed_item_senders = has_sequence_item_types<stdexec::__decay_t<_Sequence>>
                                    && requires(stdexec::__decay_t<_Sequence>* __sequence) {
                                         { exec::__check_sequence(__sequence) } -> stdexec::__ok;
                                       };

  template <class _Sequence>
  concept __well_formed_sequence_sender = stdexec::__well_formed_sender<_Sequence>
                                       && enable_sequence_sender<stdexec::__decay_t<_Sequence>>
                                       && __well_formed_item_senders<_Sequence>;

  template <class _Receiver>
  struct _WITH_RECEIVER_ { };

  template <class _Item>
  struct _MISSING_SET_NEXT_OVERLOAD_FOR_ITEM_ { };

  template <class _Receiver, class _Item>
  auto __try_item(_Item*) -> stdexec::__mexception<
    _MISSING_SET_NEXT_OVERLOAD_FOR_ITEM_<_Item>,
    _WITH_RECEIVER_<_Receiver>
  >;

  template <class _Receiver, class _Item>
    requires stdexec::__callable<set_next_t, _Receiver&, _Item>
  auto __try_item(_Item*) -> stdexec::__msuccess;

  template <class _Receiver, class... _Items>
  auto __try_items(exec::item_types<_Items...>*) -> decltype((
    stdexec::__msuccess(),
    ...,
    exec::__try_item<_Receiver>(static_cast<_Items*>(nullptr))));

  template <class _Receiver, class _Items>
  concept __sequence_receiver_of = requires(_Items* __items) {
    { exec::__try_items<stdexec::__decay_t<_Receiver>>(__items) } -> stdexec::__ok;
  };

  template <class _Receiver, class _SequenceItems>
  concept sequence_receiver_of = stdexec::receiver<_Receiver>
                              && __sequence_receiver_of<_Receiver, _SequenceItems>;

  template <class _Completions>
  using __to_sequence_completions_t = stdexec::__transform_completion_signatures<
    _Completions,
    stdexec::__mconst<stdexec::completion_signatures<stdexec::set_value_t()>>::__f,
    stdexec::__sigs::__default_set_error,
    stdexec::completion_signatures<stdexec::set_stopped_t()>,
    stdexec::__concat_completion_signatures
  >;

  template <class _Sender, class... _Env>
  using __item_completion_signatures_t = stdexec::transform_completion_signatures<
    stdexec::__completion_signatures_of_t<_Sender, _Env...>,
    stdexec::completion_signatures<stdexec::set_value_t()>,
    stdexec::__mconst<stdexec::completion_signatures<>>::__f
  >;

  template <class _Sequence, class... _Env>
  using __sequence_completion_signatures_t = stdexec::transform_completion_signatures<
    stdexec::__completion_signatures_of_t<_Sequence, _Env...>,
    stdexec::completion_signatures<stdexec::set_value_t()>,
    stdexec::__mconst<stdexec::completion_signatures<>>::__f
  >;

  template <class _Sequence, class... _Env>
  using __sequence_completion_signatures_of_t = stdexec::__mapply<
    stdexec::__mtransform<
      stdexec::__mbind_back_q<__item_completion_signatures_t, _Env...>,
      stdexec::__mbind_back<
        stdexec::__mtry_q<stdexec::__concat_completion_signatures>,
        __sequence_completion_signatures_t<_Sequence, _Env...>
      >
    >,
    __item_types_of_t<_Sequence, _Env...>
  >;

  template <class _Receiver, class _Sequence>
  concept sequence_receiver_from = stdexec::receiver<_Receiver>
                                && stdexec::sender_in<_Sequence, stdexec::env_of_t<_Receiver>>
                                && sequence_receiver_of<
                                     _Receiver,
                                     __item_types_of_t<_Sequence, stdexec::env_of_t<_Receiver>>
                                >
                                && ((sequence_sender_in<_Sequence, stdexec::env_of_t<_Receiver>>
                                     && stdexec::receiver_of<
                                       _Receiver,
                                       stdexec::completion_signatures_of_t<
                                         _Sequence,
                                         stdexec::env_of_t<_Receiver>
                                       >
                                     >)
                                    || (!sequence_sender_in<_Sequence, stdexec::env_of_t<_Receiver>> && stdexec::__receiver_from<__sequence_sndr::__stopped_means_break_t<_Receiver>, next_sender_of_t<_Receiver, _Sequence>>) );

  namespace __sequence_sndr {
    struct subscribe_t;

    struct _NO_USABLE_SUBSCRIBE_CUSTOMIZATION_FOUND_ {
      void operator()() const noexcept = delete;
    };

    template <class _Env>
    using __next_sender_completion_sigs_t = __if_c<
      unstoppable_token<stop_token_of_t<_Env>>,
      completion_signatures<set_value_t()>,
      completion_signatures<set_value_t(), set_stopped_t()>
    >;

    template <class _Sender, class _Receiver>
    concept __next_connectable =
      receiver<_Receiver> && sender_in<_Sender, env_of_t<_Receiver>>
      && !sequence_sender_in<_Sender, env_of_t<_Receiver>>
      && sequence_receiver_of<_Receiver, item_types<stdexec::__decay_t<_Sender>>>
      && sender_to<next_sender_of_t<_Receiver, _Sender>, __stopped_means_break_t<_Receiver>>;

    template <class _Sequence, class _Receiver>
    concept __subscribable_with_static_member =
      receiver<_Receiver> && sequence_sender_in<_Sequence, env_of_t<_Receiver>>
      && sequence_receiver_from<_Receiver, _Sequence>
      && requires(_Sequence&& __sequence, _Receiver&& __rcvr) {
           {
             STDEXEC_REMOVE_REFERENCE(_Sequence)
             ::subscribe(static_cast<_Sequence&&>(__sequence), static_cast<_Receiver&&>(__rcvr))
           };
         };

    template <class _Sequence, class _Receiver>
    concept __subscribable_with_member = receiver<_Receiver>
                                      && sequence_sender_in<_Sequence, env_of_t<_Receiver>>
                                      && sequence_receiver_from<_Receiver, _Sequence>
                                      && requires(_Sequence&& __sequence, _Receiver&& __rcvr) {
                                           {
                                             static_cast<_Sequence&&>(__sequence)
                                               .subscribe(static_cast<_Receiver&&>(__rcvr))
                                           };
                                         };

    template <class _Sequence, class _Receiver>
    concept __subscribable_with_tag_invoke = receiver<_Receiver>
                                          && sequence_sender_in<_Sequence, env_of_t<_Receiver>>
                                          && sequence_receiver_from<_Receiver, _Sequence>
                                          && tag_invocable<subscribe_t, _Sequence, _Receiver>;

    struct subscribe_t {
      template <class _Sequence, class _Receiver>
      using __tfx_sequence_t = __tfx_sequence_t<_Sequence, env_of_t<_Receiver>>;

      template <class _Sequence, class _Receiver>
      static constexpr auto __select_impl() noexcept {
        using __domain_t = __late_domain_of_t<_Sequence, env_of_t<_Receiver&>>;
        constexpr bool _NothrowTfxSequence =
          __nothrow_callable<transform_sender_t, __domain_t, _Sequence, env_of_t<_Receiver&>>;
        using __tfx_sequence_t = __tfx_sequence_t<_Sequence, _Receiver>;
        if constexpr (__next_connectable<__tfx_sequence_t, _Receiver>) {
          using __result_t = connect_result_t<
            next_sender_of_t<_Receiver, __tfx_sequence_t>,
            __stopped_means_break_t<_Receiver>
          >;
          static_assert(
            operation_state<__result_t>,
            "stdexec::connect(sender, receiver) must return a type that "
            "satisfies the operation_state concept");
          constexpr bool _Nothrow = __nothrow_connectable<
            next_sender_of_t<_Receiver, __tfx_sequence_t>,
            __stopped_means_break_t<_Receiver>
          >;
          return static_cast<__result_t (*)() noexcept(_Nothrow)>(nullptr);
        } else if constexpr (__subscribable_with_static_member<__tfx_sequence_t, _Receiver>) {
          using __result_t = decltype(STDEXEC_REMOVE_REFERENCE(
            __tfx_sequence_t)::subscribe(__declval<__tfx_sequence_t>(), __declval<_Receiver>()));
          static_assert(
            operation_state<__result_t>,
            "Sequence::subscribe(sender, receiver) must return a type that "
            "satisfies the operation_state concept");
          constexpr bool _Nothrow = _NothrowTfxSequence
                                 && noexcept(STDEXEC_REMOVE_REFERENCE(__tfx_sequence_t)::subscribe(
                                   __declval<__tfx_sequence_t>(), __declval<_Receiver>()));
          return static_cast<__result_t (*)() noexcept(_Nothrow)>(nullptr);
        } else if constexpr (__subscribable_with_member<__tfx_sequence_t, _Receiver>) {
          using __result_t = decltype(__declval<__tfx_sequence_t>()
                                        .subscribe(__declval<_Receiver>()));
          static_assert(
            operation_state<__result_t>,
            "Sequence::subscribe(sender, receiver) must return a type that "
            "satisfies the operation_state concept");
          constexpr bool _Nothrow = _NothrowTfxSequence
                                 && noexcept(__declval<__tfx_sequence_t>()
                                               .subscribe(__declval<_Receiver>()));
          return static_cast<__result_t (*)() noexcept(_Nothrow)>(nullptr);
        } else if constexpr (__subscribable_with_tag_invoke<__tfx_sequence_t, _Receiver>) {
          using __result_t = tag_invoke_result_t<subscribe_t, __tfx_sequence_t, _Receiver>;
          static_assert(
            operation_state<__result_t>,
            "exec::subscribe(sender, receiver) must return a type that "
            "satisfies the operation_state concept");
          constexpr bool _Nothrow = _NothrowTfxSequence
                                 && nothrow_tag_invocable<subscribe_t, __tfx_sequence_t, _Receiver>;
          return static_cast<__result_t (*)() noexcept(_Nothrow)>(nullptr);
        } else if constexpr (__is_debug_env<env_of_t<_Receiver>>) {
          using __result_t = __debug::__debug_operation;
          return static_cast<__result_t (*)() noexcept(_NothrowTfxSequence)>(nullptr);
        } else {
          return _NO_USABLE_SUBSCRIBE_CUSTOMIZATION_FOUND_();
        }
      }

      template <class _Sequence, class _Receiver>
      using __select_impl_t = decltype(__select_impl<_Sequence, _Receiver>());

      template <sender _Sequence, receiver _Receiver>
      auto operator()(_Sequence&& __sequence, _Receiver&& __rcvr) const
        noexcept(__nothrow_callable<__select_impl_t<_Sequence, _Receiver>>)
          -> __call_result_t<__select_impl_t<_Sequence, _Receiver>> {
        using __tfx_sequence_t = __tfx_sequence_t<_Sequence, _Receiver>;
        auto&& __env = stdexec::get_env(__rcvr);
        auto __domain = __get_late_domain(__sequence, __env);
        if constexpr (__next_connectable<__tfx_sequence_t, _Receiver>) {
          next_sender_of_t<_Receiver, __tfx_sequence_t> __next = set_next(
            __rcvr,
            stdexec::transform_sender(__domain, static_cast<_Sequence&&>(__sequence), __env));
          return stdexec::connect(
            static_cast<next_sender_of_t<_Receiver, __tfx_sequence_t>&&>(__next),
            __stopped_means_break_t<_Receiver>{static_cast<_Receiver&&>(__rcvr)});
          // NOLINTNEXTLINE(bugprone-branch-clone)
        } else if constexpr (__subscribable_with_static_member<__tfx_sequence_t, _Receiver>) {
          auto&& __tfx_sequence =
            transform_sender(__domain, static_cast<_Sequence&&>(__sequence), __env);
          return __tfx_sequence.subscribe(
            static_cast<__tfx_sequence_t&&>(__tfx_sequence), static_cast<_Receiver&&>(__rcvr));
        } else if constexpr (__subscribable_with_member<__tfx_sequence_t, _Receiver>) {
          return stdexec::transform_sender(__domain, static_cast<_Sequence&&>(__sequence), __env)
            .subscribe(static_cast<_Receiver&&>(__rcvr));
        } else if constexpr (__subscribable_with_tag_invoke<__tfx_sequence_t, _Receiver>) {
          return stdexec::tag_invoke(
            subscribe_t{},
            stdexec::transform_sender(__domain, static_cast<_Sequence&&>(__sequence), __env),
            static_cast<_Receiver&&>(__rcvr));
        } else if constexpr (enable_sequence_sender<stdexec::__decay_t<__tfx_sequence_t>>) {
          // This should generate an instantiate backtrace that contains useful
          // debugging information.
          auto&& __tfx_sequence =
            transform_sender(__domain, static_cast<_Sequence&&>(__sequence), __env);
          return __tfx_sequence.subscribe(
            static_cast<__tfx_sequence_t&&>(__tfx_sequence), static_cast<_Receiver&&>(__rcvr));
        } else {
          // This should generate an instantiate backtrace that contains useful
          // debugging information.
          next_sender_of_t<_Receiver, __tfx_sequence_t> __next = set_next(
            __rcvr,
            stdexec::transform_sender(__domain, static_cast<_Sequence&&>(__sequence), __env));
          return stdexec::connect(
            static_cast<next_sender_of_t<_Receiver, __tfx_sequence_t>&&>(__next),
            __stopped_means_break_t<_Receiver>{static_cast<_Receiver&&>(__rcvr)});
        }
      }

      static constexpr auto query(stdexec::forwarding_query_t) noexcept -> bool {
        return false;
      }
    };

    template <class _Sequence, class _Receiver>
    using subscribe_result_t = __call_result_t<subscribe_t, _Sequence, _Receiver>;
  } // namespace __sequence_sndr

  using __sequence_sndr::__next_sender_completion_sigs_t;

  using __sequence_sndr::subscribe_t;
  inline constexpr subscribe_t subscribe{};

  using __sequence_sndr::subscribe_result_t;

  template <class _Sequence, class _Receiver>
  concept sequence_sender_to =
    sequence_receiver_from<_Receiver, _Sequence>
    && requires(_Sequence&& __sequence, _Receiver&& __rcvr) {
         subscribe(static_cast<_Sequence&&>(__sequence), static_cast<_Receiver&&>(__rcvr));
       };

  template <class _Receiver>
  concept __stoppable_receiver = stdexec::__callable<stdexec::set_value_t, _Receiver>
                              && (stdexec::unstoppable_token<
                                    stdexec::stop_token_of_t<stdexec::env_of_t<_Receiver>>
                                  >
                                  || stdexec::__callable<stdexec::set_stopped_t, _Receiver>);

  template <class _Receiver>
    requires __stoppable_receiver<_Receiver>
  void __set_value_unless_stopped(_Receiver&& __rcvr) {
    using token_type = stdexec::stop_token_of_t<stdexec::env_of_t<_Receiver>>;
    if constexpr (stdexec::unstoppable_token<token_type>) {
      stdexec::set_value(static_cast<_Receiver&&>(__rcvr));
    } else {
      auto token = stdexec::get_stop_token(stdexec::get_env(__rcvr));
      if (!token.stop_requested()) {
        stdexec::set_value(static_cast<_Receiver&&>(__rcvr));
      } else {
        stdexec::set_stopped(static_cast<_Receiver&&>(__rcvr));
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
#define STDEXEC_ERROR_GET_ITEM_TYPES_RETURNED_AN_ERROR                                             \
  "\n"                                                                                             \
  "\n"                                                                                             \
  "Trying to compute the sequences's item types resulted in an error. See\n"                       \
  "the rest of the compiler diagnostic for clues. Look for the string \"_ERROR_\".\n"

#define STDEXEC_ERROR_GET_ITEM_TYPES_HAS_INVALID_RETURN_TYPE                                       \
  "\n"                                                                                             \
  "\n"                                                                                             \
  "The member function `get_item_types` of the sequence returned an\n"                             \
  "invalid type.\n"                                                                                \
  "\n"                                                                                             \
  "A sender's `get_item_types` function must return a specialization of\n"                         \
  "`exec::item_types<...>`, as follows:\n"                                                         \
  "\n"                                                                                             \
  "  class MySequence\n"                                                                           \
  "  {\n"                                                                                          \
  "  public:\n"                                                                                    \
  "    using sender_concept = exec::sequence_sender_t;\n"                                          \
  "\n"                                                                                             \
  "    template <class... _Env>\n"                                                                 \
  "    auto get_item_types(_Env&&...) -> exec::item_types<\n"                                      \
  "      // This sequence produces void items...\n"                                                \
  "      stdexec::__call_result_t<stdexec::just_t>>\n"                                             \
  "    {\n"                                                                                        \
  "    return {};\n"                                                                               \
  "    }\n"                                                                                        \
  "    ...\n"                                                                                      \
  "  };\n"

  // Used to report a meaningful error message when the sender_in<Sndr, Env>
  // concept check fails.
  template <class _Sequence, class... _Env>
  auto __diagnose_sequence_concept_failure() {
    if constexpr (!enable_sequence_sender<stdexec::__decay_t<_Sequence>>) {
      static_assert(enable_sequence_sender<_Sequence>, STDEXEC_ERROR_ENABLE_SENDER_IS_FALSE);
    } else if constexpr (!stdexec::__detail::__consistent_completion_domains<_Sequence>) {
      static_assert(
        stdexec::__detail::__consistent_completion_domains<_Sequence>,
        "The completion schedulers of the sequence do not have "
        "consistent domains. This is likely a "
        "bug in the sequence implementation.");
    } else if constexpr (!std::move_constructible<stdexec::__decay_t<_Sequence>>) {
      static_assert(
        std::move_constructible<stdexec::__decay_t<_Sequence>>,
        "The sequence type is not move-constructible.");
    } else if constexpr (!std::constructible_from<stdexec::__decay_t<_Sequence>, _Sequence>) {
      static_assert(
        std::constructible_from<stdexec::__decay_t<_Sequence>, _Sequence>,
        "The sequence cannot be decay-copied. Did you forget a std::move?");
    } else {
      using __items_t = __item_types_of_t<_Sequence, _Env...>;
      if constexpr (stdexec::__same_as<
                      __items_t,
                      __sequence_sndr::__unrecognized_sequence_error_t<_Sequence, _Env...>
                    >) {
        static_assert(
          stdexec::__mnever<__items_t>, STDEXEC_ERROR_CANNOT_COMPUTE_COMPLETION_SIGNATURES);
      } else if constexpr (stdexec::__merror<__items_t>) {
        static_assert(
          !stdexec::__merror<__items_t>, STDEXEC_ERROR_GET_ITEM_TYPES_RETURNED_AN_ERROR);
      } else if constexpr (!__well_formed_item_senders<_Sequence>) {
        static_assert(
          __well_formed_item_senders<_Sequence>,
          STDEXEC_ERROR_GET_ITEM_TYPES_HAS_INVALID_RETURN_TYPE);
      } else {
        stdexec::__diagnose_sender_concept_failure<_Sequence, _Env...>();
      }
    }
  }

  namespace __debug {

    template <class... _Items>
    struct __valid_next {
      template <class _Item>
        requires stdexec::__one_of<_Item, _Items...>
      STDEXEC_ATTRIBUTE(host, device)
      stdexec::__call_result_t<stdexec::just_t> set_next(_Item&&) noexcept {
        STDEXEC_TERMINATE();
        return stdexec::just();
      }
    };

    template <class _CvrefSequenceId, class _Env, class _Completions, class _ItemTypes>
    struct __debug_sequence_sender_receiver {
      using __t = __debug_sequence_sender_receiver;
      using __id = __debug_sequence_sender_receiver;
      using receiver_concept = stdexec::receiver_t;
    };

    template <class _CvrefSequenceId, class _Env, class... _Sigs, class... _Items>
    struct __debug_sequence_sender_receiver<
      _CvrefSequenceId,
      _Env,
      stdexec::completion_signatures<_Sigs...>,
      item_types<_Items...>
    >
      : __valid_completions<__normalize_sig_t<_Sigs>...>
      , __valid_next<_Items...> {
      using __t = __debug_sequence_sender_receiver;
      using __id = __debug_sequence_sender_receiver;
      using receiver_concept = stdexec::receiver_t;

      STDEXEC_ATTRIBUTE(host, device) auto get_env() const noexcept -> __debug_env_t<_Env> {
        STDEXEC_TERMINATE();
      }
    };

    template <class _Env = stdexec::env<>, class _Sequence>
    void __debug_sequence_sender(_Sequence&& __sequence, const _Env& = {}) {
      if constexpr (!__is_debug_env<_Env>) {
        if constexpr (sequence_sender_in<_Sequence, _Env>) {
          using __sigs_t = stdexec::__completion_signatures_of_t<_Sequence, __debug_env_t<_Env>>;
          using __item_types_t = __sequence_sndr::__item_types_of_t<_Sequence, __debug_env_t<_Env>>;
          using __receiver_t = __debug_sequence_sender_receiver<
            stdexec::__cvref_id<_Sequence>,
            _Env,
            __sigs_t,
            __item_types_t
          >;
          if constexpr (
            !std::same_as<__sigs_t, __debug::__completion_signatures>
            || !std::same_as<__item_types_t, __debug::__item_types>) {
            using __operation_t = exec::subscribe_result_t<_Sequence, __receiver_t>;
            //static_assert(receiver_of<_Receiver, _Sigs>);
            if constexpr (!std::same_as<__operation_t, __debug_operation>) {
              if (sizeof(_Sequence) == ~0ul) { // never true
                auto __op = subscribe(static_cast<_Sequence&&>(__sequence), __receiver_t{});
                stdexec::start(__op);
              }
            }
          }
        } else {
          __diagnose_sequence_concept_failure<_Sequence, _Env>();
        }
      }
    }
  } // namespace __debug
  using __debug::__debug_sequence_sender;

#if STDEXEC_ENABLE_EXTRA_TYPE_CHECKING()
  // __checked_completion_signatures is for catching logic bugs in a sender's metadata. If sender<S>
  // and sender_in<S, Ctx> are both true, then they had better report the same metadata. This
  // completion signatures wrapper enforces that at compile time.
  template <class _Sequence, class... _Env>
  auto __checked_item_types(_Sequence&& __sequence, _Env&&... __env) noexcept {
    using __completions_t =
      decltype(get_item_types(stdexec::__declval<_Sequence>(), stdexec::__declval<_Env>()...));
    // (void)__sequence;
    // [](auto&&...){}(__env...);
    exec::__debug_sequence_sender(static_cast<_Sequence&&>(__sequence), __env...);
    return __completions_t{};
  }

  template <class _Sequence, class... _Env>
    requires sequence_sender_in<_Sequence, _Env...>
  using item_types_of_t = decltype(exec::__checked_item_types(
    stdexec::__declval<_Sequence>(),
    stdexec::__declval<_Env>()...));
#else
  template <class _Sequence, class... _Env>
    requires sequence_sender_in<_Sequence, _Env...>
  using item_types_of_t = __item_types_of_t<_Sequence, _Env...>;
#endif
} // namespace exec
