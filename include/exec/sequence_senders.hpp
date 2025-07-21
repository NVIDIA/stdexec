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
    using __mall_contained_in = __mapply<__mall_contained_in_impl<_Haystack>, _Needles>;

    template <class _Needles, class _Haystack>
    concept __all_contained_in = __v<__mall_contained_in<_Needles, _Haystack>>;

    // This concept checks if a given sender satisfies the requirements to be returned from `set_next`.
    template <class _Sender, class _Env = env<>>
    concept next_sender = sender_in<_Sender, _Env>
                       && __all_contained_in<
                            completion_signatures_of_t<_Sender, _Env>,
                            completion_signatures<set_value_t(), set_stopped_t()>
                       >;

    template <class _Receiver, class _Item>
    concept __has_set_next_member = requires(_Receiver& __rcvr, _Item&& __item) {
      __rcvr.set_next(static_cast<_Item &&>(__item));
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

  template <class _Receiver, class _Sender>
  using next_sender_of_t = decltype(exec::set_next(
    stdexec::__declval<stdexec::__decay_t<_Receiver>&>(),
    stdexec::__declval<_Sender>()));

  namespace __sequence_sndr {

    template <class _ReceiverId>
    struct __stopped_means_break {
      struct __t {
        using receiver_concept = stdexec::receiver_t;
        using __id = __stopped_means_break;
        using _Receiver = stdexec::__t<_ReceiverId>;
        using _Token = stop_token_of_t<env_of_t<_Receiver>>;
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
                && (unstoppable_token<_Token> || __callable<set_stopped_t, _Receiver>)
        {
          if constexpr (unstoppable_token<_Token>) {
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

  template <class _Sender>
  concept __enable_sequence_sender = requires {
    typename _Sender::sender_concept;
  } && stdexec::same_as<typename _Sender::sender_concept, sequence_sender_t>;

  template <class _Sender>
  inline constexpr bool enable_sequence_sender = __enable_sequence_sender<_Sender>;

  template <class... _Senders>
  struct item_types { };

  template <class _Tp>
  concept __has_item_typedef = requires { typename _Tp::item_types; };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.sndtraits]
  namespace __sequence_sndr {
    struct get_item_types_t;
    template <class _Sender, class _Env>
    using __tfx_sender =
      transform_sender_result_t<__late_domain_of_t<_Sender, _Env>, _Sender, _Env>;

    template <class _Sender, class _Env>
    concept __with_tag_invoke = tag_invocable<get_item_types_t, __tfx_sender<_Sender, _Env>, _Env>;

    template <class _Sender, class _Env>
    using __member_alias_t = typename __decay_t<__tfx_sender<_Sender, _Env>>::item_types;

    template <class _Sender, class _Env>
    concept __with_member_alias = __mvalid<__member_alias_t, _Sender, _Env>;

    template <class _Sender, class _Env>
    concept __with_member = requires(__tfx_sender<_Sender, _Env>&& __sndr, _Env&& __env) {
      static_cast<__tfx_sender<_Sender, _Env> &&>(__sndr)
        .get_item_types(static_cast<_Env &&>(__env));
    };

    struct get_item_types_t {
      template <class _Sender, class _Env>
      static auto __impl() {
        static_assert(sizeof(_Sender), "Incomplete type used with get_item_types");
        static_assert(sizeof(_Env), "Incomplete type used with get_item_types");
        using _TfxSender = __tfx_sender<_Sender, _Env>;
        if constexpr (__merror<_TfxSender>) {
          // Computing the type of the transformed sender returned an error type. Propagate it.
          return static_cast<_TfxSender (*)()>(nullptr);
        } else if constexpr (__with_member_alias<_Sender, _Env>) {
          using _Result = __member_alias_t<_Sender, _Env>;
          return static_cast<_Result (*)()>(nullptr);
        } else if constexpr (__with_member<_Sender, _Env>) {
          using _Result = decltype(__declval<_TfxSender>().get_item_types(__declval<_Env>()));
          return static_cast<_Result (*)()>(nullptr);
        } else if constexpr (__with_tag_invoke<_Sender, _Env>) {
          using _Result = tag_invoke_result_t<get_item_types_t, _TfxSender, _Env>;
          return static_cast<_Result (*)()>(nullptr);
        } else if constexpr (
          sender_in<_TfxSender, _Env> && !enable_sequence_sender<stdexec::__decay_t<_TfxSender>>) {
          using _Result = item_types<stdexec::__decay_t<_TfxSender>>;
          return static_cast<_Result (*)()>(nullptr);
        } else if constexpr (__is_debug_env<_Env>) {
          // This ought to cause a hard error that indicates where the problem is.
          using _Completions
            [[maybe_unused]] = decltype(__declval<_TfxSender>().get_item_types(__declval<_Env>()));
          return static_cast<__debug::__completion_signatures (*)()>(nullptr);
        } else {
          using _Result = __mexception<
            _UNRECOGNIZED_SENDER_TYPE_<>,
            _WITH_SENDER_<_Sender>,
            _WITH_ENVIRONMENT_<_Env>
          >;
          return static_cast<_Result (*)()>(nullptr);
        }
      }

      template <class _Sender, class _Env = env<>>
      constexpr auto
        operator()(_Sender&&, _Env&& = {}) const noexcept -> decltype(__impl<_Sender, _Env>()()) {
        return {};
      }
    };
  } // namespace __sequence_sndr

  using __sequence_sndr::get_item_types_t;
  inline constexpr get_item_types_t get_item_types{};

  template <class _Sender, class... _Env>
  using item_types_of_t =
    decltype(get_item_types(stdexec::__declval<_Sender>(), stdexec::__declval<_Env>()...));

  template <class _Sender, class... _Env>
  concept sequence_sender = stdexec::sender_in<_Sender, _Env...>
                         && enable_sequence_sender<stdexec::__decay_t<_Sender>>;

  template <class _Sender, class... _Env>
  concept has_sequence_item_types = requires(_Sender&& __sndr, _Env&&... __env) {
    get_item_types(static_cast<_Sender &&>(__sndr), static_cast<_Env &&>(__env)...);
  };

  template <class _Sender, class... _Env>
  concept sequence_sender_in = stdexec::sender_in<_Sender, _Env...>
                            && has_sequence_item_types<_Sender, _Env...>
                            && sequence_sender<_Sender, _Env...>;

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
  using __item_completion_signatures = stdexec::transform_completion_signatures<
    stdexec::__completion_signatures_of_t<_Sender, _Env...>,
    stdexec::completion_signatures<stdexec::set_value_t()>,
    stdexec::__mconst<stdexec::completion_signatures<>>::__f
  >;

  template <class _Sequence, class... _Env>
  using __sequence_completion_signatures = stdexec::transform_completion_signatures<
    stdexec::__completion_signatures_of_t<_Sequence, _Env...>,
    stdexec::completion_signatures<stdexec::set_value_t()>,
    stdexec::__mconst<stdexec::completion_signatures<>>::__f
  >;

  template <class _Sequence, class... _Env>
  using __sequence_completion_signatures_of_t = stdexec::__mapply<
    stdexec::__mtransform<
      stdexec::__mbind_back_q<__item_completion_signatures, _Env...>,
      stdexec::__mbind_back<
        stdexec::__mtry_q<stdexec::__concat_completion_signatures>,
        __sequence_completion_signatures<_Sequence, _Env...>
      >
    >,
    item_types_of_t<_Sequence, _Env...>
  >;

  template <class _Receiver, class _Sender>
  concept sequence_receiver_from = stdexec::receiver<_Receiver>
                                && stdexec::sender_in<_Sender, stdexec::env_of_t<_Receiver>>
                                && sequence_receiver_of<
                                     _Receiver,
                                     item_types_of_t<_Sender, stdexec::env_of_t<_Receiver>>
                                >
                                && ((sequence_sender_in<_Sender, stdexec::env_of_t<_Receiver>>
                                     && stdexec::receiver_of<
                                       _Receiver,
                                       stdexec::completion_signatures_of_t<
                                         _Sender,
                                         stdexec::env_of_t<_Receiver>
                                       >
                                     >)
                                    || (!sequence_sender_in<_Sender, stdexec::env_of_t<_Receiver>> && stdexec::__receiver_from<__sequence_sndr::__stopped_means_break_t<_Receiver>, next_sender_of_t<_Receiver, _Sender>>) );

  namespace __sequence_sndr {
    struct subscribe_t;

    template <class _Env>
    using __single_sender_completion_sigs = __if_c<
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

    template <class _Sender, class _Receiver>
    concept __subscribeable_with_member = receiver<_Receiver>
                                       && sequence_sender_in<_Sender, env_of_t<_Receiver>>
                                       && sequence_receiver_from<_Receiver, _Sender>
                                       && requires(_Sender&& __sndr, _Receiver&& __rcvr) {
                                            {
                                              static_cast<_Sender &&>(__sndr)
                                                .subscribe(static_cast<_Receiver &&>(__rcvr))
                                            };
                                          };

    template <class _Sender, class _Receiver>
    concept __subscribeable_with_tag_invoke = receiver<_Receiver>
                                           && sequence_sender_in<_Sender, env_of_t<_Receiver>>
                                           && sequence_receiver_from<_Receiver, _Sender>
                                           && tag_invocable<subscribe_t, _Sender, _Receiver>;

    struct subscribe_t {
      template <class _Sender, class _Receiver>
      using __tfx_sndr = __tfx_sender<_Sender, env_of_t<_Receiver>>;

      template <class _Sender, class _Receiver>
      static constexpr auto __select_impl() noexcept {
        using _Domain = __late_domain_of_t<_Sender, env_of_t<_Receiver&>>;
        constexpr bool _NothrowTfxSender =
          __nothrow_callable<transform_sender_t, _Domain, _Sender, env_of_t<_Receiver&>>;
        using _TfxSender = __tfx_sndr<_Sender, _Receiver>;
        if constexpr (__next_connectable<_TfxSender, _Receiver>) {
          using _Result = connect_result_t<
            next_sender_of_t<_Receiver, _TfxSender>,
            __stopped_means_break_t<_Receiver>
          >;
          static_assert(
            operation_state<_Result>,
            "stdexec::connect(sender, receiver) must return a type that "
            "satisfies the operation_state concept");
          constexpr bool _Nothrow = __nothrow_connectable<
            next_sender_of_t<_Receiver, _TfxSender>,
            __stopped_means_break_t<_Receiver>
          >;
          return static_cast<_Result (*)() noexcept(_Nothrow)>(nullptr);
        } else if constexpr (__subscribeable_with_member<_TfxSender, _Receiver>) {
          using _Result = decltype(__declval<_TfxSender>().subscribe(__declval<_Receiver>()));
          static_assert(
            operation_state<_Result>,
            "Sender::subscribe(sender, receiver) must return a type that "
            "satisfies the operation_state concept");
          constexpr bool _Nothrow = _NothrowTfxSender
                                 && noexcept(__declval<_TfxSender>()
                                               .subscribe(__declval<_Receiver>()));
          return static_cast<_Result (*)() noexcept(_Nothrow)>(nullptr);
        } else if constexpr (__subscribeable_with_tag_invoke<_TfxSender, _Receiver>) {
          using _Result = tag_invoke_result_t<subscribe_t, _TfxSender, _Receiver>;
          static_assert(
            operation_state<_Result>,
            "exec::subscribe(sender, receiver) must return a type that "
            "satisfies the operation_state concept");
          constexpr bool _Nothrow = _NothrowTfxSender
                                 && nothrow_tag_invocable<subscribe_t, _TfxSender, _Receiver>;
          return static_cast<_Result (*)() noexcept(_Nothrow)>(nullptr);
        } else {
          return static_cast<__debug::__debug_operation (*)() noexcept>(nullptr);
        }
      }

      template <class _Sender, class _Receiver>
      using __select_impl_t = decltype(__select_impl<_Sender, _Receiver>());

      template <sender _Sender, receiver _Receiver>
        requires __next_connectable<__tfx_sndr<_Sender, _Receiver>, _Receiver>
              || __subscribeable_with_member<__tfx_sndr<_Sender, _Receiver>, _Receiver>
              || __subscribeable_with_tag_invoke<__tfx_sndr<_Sender, _Receiver>, _Receiver>
              || __is_debug_env<env_of_t<_Receiver>>
      auto operator()(_Sender&& __sndr, _Receiver&& __rcvr) const
        noexcept(__nothrow_callable<__select_impl_t<_Sender, _Receiver>>)
          -> __call_result_t<__select_impl_t<_Sender, _Receiver>> {
        using _TfxSender = __tfx_sndr<_Sender, _Receiver>;
        auto&& __env = stdexec::get_env(__rcvr);
        auto __domain = __get_late_domain(__sndr, __env);
        if constexpr (__next_connectable<_TfxSender, _Receiver>) {
          next_sender_of_t<_Receiver, _TfxSender> __next = set_next(
            __rcvr, stdexec::transform_sender(__domain, static_cast<_Sender&&>(__sndr), __env));
          return stdexec::connect(
            static_cast<next_sender_of_t<_Receiver, _TfxSender>&&>(__next),
            __stopped_means_break_t<_Receiver>{static_cast<_Receiver&&>(__rcvr)});
          // NOLINTNEXTLINE(bugprone-branch-clone)
        } else if constexpr (__subscribeable_with_member<_TfxSender, _Receiver>) {
          return stdexec::transform_sender(__domain, static_cast<_Sender&&>(__sndr), __env)
            .subscribe(static_cast<_Receiver&&>(__rcvr));
        } else if constexpr (__subscribeable_with_tag_invoke<_TfxSender, _Receiver>) {
          return stdexec::tag_invoke(
            subscribe_t{},
            stdexec::transform_sender(__domain, static_cast<_Sender&&>(__sndr), __env),
            static_cast<_Receiver&&>(__rcvr));
        } else if constexpr (enable_sequence_sender<stdexec::__decay_t<_TfxSender>>) {
          // This should generate an instantiate backtrace that contains useful
          // debugging information.
          return stdexec::transform_sender(__domain, static_cast<_Sender&&>(__sndr), __env)
            .subscribe(static_cast<_Receiver&&>(__rcvr));
        } else {
          // This should generate an instantiate backtrace that contains useful
          // debugging information.
          next_sender_of_t<_Receiver, _TfxSender> __next = set_next(
            __rcvr, stdexec::transform_sender(__domain, static_cast<_Sender&&>(__sndr), __env));
          return stdexec::connect(
            static_cast<next_sender_of_t<_Receiver, _TfxSender>&&>(__next),
            __stopped_means_break_t<_Receiver>{static_cast<_Receiver&&>(__rcvr)});
        }
      }

      static constexpr auto query(stdexec::forwarding_query_t) noexcept -> bool {
        return false;
      }
    };

    template <class _Sender, class _Receiver>
    using subscribe_result_t = __call_result_t<subscribe_t, _Sender, _Receiver>;
  } // namespace __sequence_sndr

  using __sequence_sndr::__single_sender_completion_sigs;

  using __sequence_sndr::subscribe_t;
  inline constexpr subscribe_t subscribe{};

  using __sequence_sndr::subscribe_result_t;

  template <class _Sender, class _Receiver>
  concept sequence_sender_to =
    sequence_receiver_from<_Receiver, _Sender> && requires(_Sender&& __sndr, _Receiver&& __rcvr) {
      subscribe(static_cast<_Sender &&>(__sndr), static_cast<_Receiver &&>(__rcvr));
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
} // namespace exec