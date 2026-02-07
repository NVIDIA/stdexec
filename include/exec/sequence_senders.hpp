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
#include "../stdexec/__detail/__diagnostics.hpp"
#include "../stdexec/__detail/__meta.hpp"
#include "completion_signatures.hpp"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(not_used_in_template_function_params)

////////////////////////////////////////////////////////////////////////////////
#define STDEXEC_ERROR_SEQUENCE_SENDER_DEFINITION                                                   \
  "A sequence sender must provide a `subscribe` member function that takes a receiver as an\n"     \
  "argument and returns an object whose type satisfies `STDEXEC::operation_state`,\n"              \
  "as shown below:\n"                                                                              \
  "\n"                                                                                             \
  "  class MySequenceSender\n"                                                                     \
  "  {\n"                                                                                          \
  "  public:\n"                                                                                    \
  "    using sender_concept        = exec::sequence_sender_t;\n"                                   \
  "    using item_types            = exec::item_types<>;\n"                                        \
  "    using completion_signatures = STDEXEC::completion_signatures<STDEXEC::set_value_t()>;\n"    \
  "\n"                                                                                             \
  "    template <class Receiver>\n"                                                                \
  "    struct MyOpState\n"                                                                         \
  "    {\n"                                                                                        \
  "      using operation_state_concept = STDEXEC::operation_state_t;\n"                            \
  "\n"                                                                                             \
  "      void start() noexcept\n"                                                                  \
  "      {\n"                                                                                      \
  "        // Start the operation, which will eventually complete and send its\n"                  \
  "        // result to rcvr_;\n"                                                                  \
  "      }\n"                                                                                      \
  "\n"                                                                                             \
  "      Receiver rcvr_;\n"                                                                        \
  "    };\n"                                                                                       \
  "\n"                                                                                             \
  "    template <STDEXEC::receiver Receiver>\n"                                                    \
  "    auto subscribe(Receiver rcvr) -> MyOpState<Receiver>\n"                                     \
  "    {\n"                                                                                        \
  "      return MyOpState<Receiver>{std::move(rcvr)};\n"                                           \
  "    }\n"                                                                                        \
  "\n"                                                                                             \
  "    ...\n"                                                                                      \
  "  };\n"

////////////////////////////////////////////////////////////////////////////////
#define STDEXEC_ERROR_CANNOT_SUBSCRIBE_SEQUENCE_TO_RECEIVER                                        \
  "\n"                                                                                             \
  "FAILURE: No usable subscribe customization was found.\n"                                        \
  "\n" STDEXEC_ERROR_SEQUENCE_SENDER_DEFINITION

////////////////////////////////////////////////////////////////////////////////
#define STDEXEC_ERROR_SUBSCRIBE_DOES_NOT_RETURN_OPERATION_STATE                                    \
  "\n"                                                                                             \
  "FAILURE: The subscribe customization did not return an `STDEXEC::operation_state`.\n"           \
  "\n" STDEXEC_ERROR_SEQUENCE_SENDER_DEFINITION

namespace exec {
  template <class _Sequence>
  struct _WITH_SEQUENCE_ { };

  template <class... _Sequences>
  struct _WITH_SEQUENCES_ { };

  template <class _Sequence>
  using _WITH_PRETTY_SEQUENCE_ = _WITH_SEQUENCE_<STDEXEC::__demangle_t<_Sequence>>;

  template <class... _Sequences>
  using _WITH_PRETTY_SEQUENCES_ = _WITH_SEQUENCES_<STDEXEC::__demangle_t<_Sequences>...>;

  struct sequence_sender_t : STDEXEC::sender_t { };

  using sequence_tag [[deprecated("Renamed to exec::sequence_sender_t")]] = exec::sequence_sender_t;

  namespace __sequence_sndr {
    using namespace STDEXEC;

    template <class _Haystack>
    struct __mall_contained_in_impl {
      template <class... _Needles>
      using __f = __mand<__mapply<__mcontains<_Needles>, _Haystack>...>;
    };
    template <class _Needles, class _Haystack>
    using __mall_contained_in_t = __mapply<__mall_contained_in_impl<_Haystack>, _Needles>;

    template <class _Needles, class _Haystack>
    concept __all_contained_in = __mall_contained_in_t<_Needles, _Haystack>::value;
  } // namespace __sequence_sndr

  // This concept checks if a given sender satisfies the requirements to be returned from `set_next`.
  template <class _Sender, class _Env = STDEXEC::env<>>
  concept next_sender =
    STDEXEC::sender_in<_Sender, _Env>
    && __sequence_sndr::__all_contained_in<
      STDEXEC::completion_signatures_of_t<_Sender, _Env>,
      STDEXEC::completion_signatures<STDEXEC::set_value_t(), STDEXEC::set_stopped_t()>
    >;

  namespace __sequence_sndr {
    template <class _Receiver, class _Item>
    concept __has_set_next_member = requires(_Receiver& __rcvr, _Item&& __item) {
      __rcvr.set_next(static_cast<_Item&&>(__item));
    };

    // This is a sequence-receiver CPO that is used to apply algorithms on an input sender and it
    // returns a next-sender. `set_next` is usually called in a context where a sender will be
    // connected to a receiver. Since calling `set_next` usually involves constructing senders it
    // is allowed to throw an exception, which needs to be handled by a calling sequence-operation.
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
        requires __has_set_next_member<_Receiver, _Item>
              || __tag_invocable<set_next_t, _Receiver&, _Item>
      [[deprecated("the use of tag_invoke for set_next is deprecated")]]
      auto operator()(_Receiver& __rcvr, _Item&& __item) const
        noexcept(__nothrow_tag_invocable<set_next_t, _Receiver&, _Item>)
          -> __tag_invoke_result_t<set_next_t, _Receiver&, _Item> {
        static_assert(
          next_sender<__tag_invoke_result_t<set_next_t, _Receiver&, _Item>>,
          "The sender returned from set_next is required to complete with set_value_t() or "
          "set_stopped_t()");
        return __tag_invoke(*this, __rcvr, static_cast<_Item&&>(__item));
      }
    };
  } // namespace __sequence_sndr

  using __sequence_sndr::set_next_t;
  inline constexpr set_next_t set_next{};

  template <class _Receiver, class _Sender>
  using next_sender_of_t = decltype(exec::set_next(
    STDEXEC::__declval<STDEXEC::__decay_t<_Receiver>&>(),
    STDEXEC::__declval<_Sender>()));

  namespace __sequence_sndr {
    template <class _Receiver>
    struct __stopped_means_break {
      using receiver_concept = STDEXEC::receiver_t;
      using __token_t = stop_token_of_t<env_of_t<_Receiver>>;

      constexpr void set_value() noexcept {
        return STDEXEC::set_value(static_cast<_Receiver&&>(__rcvr_));
      }

      constexpr void set_stopped() noexcept {
        if constexpr (unstoppable_token<__token_t>) {
          STDEXEC::set_value(static_cast<_Receiver&&>(__rcvr_));
        } else {
          auto __token = STDEXEC::get_stop_token(STDEXEC::get_env(__rcvr_));
          if (__token.stop_requested()) {
            STDEXEC::set_stopped(static_cast<_Receiver&&>(__rcvr_));
          } else {
            STDEXEC::set_value(static_cast<_Receiver&&>(__rcvr_));
          }
        }
      }

      constexpr auto get_env() const noexcept -> env_of_t<_Receiver> {
        return STDEXEC::get_env(__rcvr_);
      }

      STDEXEC_ATTRIBUTE(no_unique_address)
      _Receiver __rcvr_;
    };
  } // namespace __sequence_sndr

  template <class _Sequence>
  concept __enable_sequence_sender =
    requires { typename _Sequence::sender_concept; } //
    && STDEXEC::__std::derived_from<typename _Sequence::sender_concept, sequence_sender_t>;

  template <class _Sequence>
  inline constexpr bool enable_sequence_sender = __enable_sequence_sender<_Sequence>;

  template <class... _Senders>
  struct item_types {
    template <class _Fn, class _Continuation = STDEXEC::__qq<STDEXEC::__mlist>>
    static constexpr auto __transform(_Fn __fn, _Continuation __continuation = {}) {
      return __continuation(__fn.template operator()<_Senders>()...);
    }
  };

  template <class _Tp>
  concept __has_item_typedef = requires { typename _Tp::item_types; };

  namespace __debug {
    using namespace STDEXEC::__debug;

    struct __item_types { };
  } // namespace __debug

  struct _UNRECOGNIZED_SEQUENCE_TYPE_;

  struct _FAILED_TO_COMPUTE_SEQUENCE_ITEM_TYPES_ { };

#if STDEXEC_NO_STD_CONSTEXPR_EXCEPTIONS()

  template <class... _What, class... _Values>
  [[nodiscard]]
  consteval auto __invalid_item_types(_Values...) {
    return STDEXEC::__mexception<_What...>();
  }

#else // ^^^ no constexpr exceptions ^^^ / vvv constexpr exceptions vvv

  // C++26, https://wg21.link/p3068
  template <class _What, class... _More, class... _Values>
  [[noreturn, nodiscard]]
  consteval auto __invalid_item_types([[maybe_unused]] _Values... __values) -> item_types<> {
    if constexpr (sizeof...(_Values) == 1) {
      throw __sequence_type_check_failure<_Values..., _What, _More...>(__values...);
    } else {
      throw __sequence_type_check_failure<STDEXEC::__tuple<_Values...>, _What, _More...>(
        STDEXEC::__tuple{__values...});
    }
  }

#endif // ^^^ constexpr exceptions ^^^

  template <class... _What>
  [[nodiscard]]
  consteval auto __invalid_item_types(STDEXEC::__mexception<_What...>) {
    return exec::__invalid_item_types<_What...>();
  }

  template <class _Sequence, class... _Env>
  using __unrecognized_sequence_error_t = STDEXEC::__mexception<
    STDEXEC::_WHAT_(_UNRECOGNIZED_SEQUENCE_TYPE_),
    STDEXEC::_WHY_(_FAILED_TO_COMPUTE_SEQUENCE_ITEM_TYPES_),
    _WITH_PRETTY_SEQUENCE_<_Sequence>,
    STDEXEC::_WITH_ENVIRONMENT_(_Env)...
  >;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.seqtraits]
  namespace __sequence_sndr {
    struct get_item_types_t;

    template <class _Sequence, class... _Env>
    using __member_result_t = decltype(__declval<_Sequence>().get_item_types(__declval<_Env>()...));

    template <class _Sequence, class... _Env>
    using __static_member_result_t = decltype(STDEXEC_REMOVE_REFERENCE(
      _Sequence)::static_get_item_types(__declval<_Sequence>(), __declval<_Env>()...));

    template <class _Sequence, class... _Env>
    using __consteval_static_member_result_t = decltype(STDEXEC_REMOVE_REFERENCE(
      _Sequence)::template get_item_types<_Sequence, _Env...>());

    template <class _Sequence, class... _Env>
    using __member_alias_t =
      typename STDEXEC_REMOVE_REFERENCE(transform_sender_result_t<_Sequence, _Env...>)::item_types;

    template <class _Sequence, class... _Env>
    concept __with_member = __minvocable_q<__member_result_t, _Sequence, _Env...>;

    template <class _Sequence, class... _Env>
    concept __with_static_member = __minvocable_q<__static_member_result_t, _Sequence, _Env...>;

    template <class _Sequence, class... _Env>
    concept __with_member_alias = __minvocable_q<__member_alias_t, _Sequence, _Env...>;

    template <class _Sequence, class... _Env>
    concept __with_consteval_static_member =
      __minvocable_q<__consteval_static_member_result_t, _Sequence, _Env...>;

    template <class _Sequence, class... _Env>
    concept __with_tag_invoke = __tag_invocable<get_item_types_t, _Sequence, _Env...>;

    template <class _Sequence, class... _Env>
    [[nodiscard]]
    consteval auto __get_item_types_helper() {
      static_assert(sizeof(_Sequence), "Incomplete type used with get_item_types");
      static_assert((sizeof(_Env) && ...), "Incomplete type used with get_item_types");

      if constexpr (__with_static_member<_Sequence, _Env...>) {
        using __result_t = __static_member_result_t<_Sequence, _Env...>;
        return __result_t();
      } else if constexpr (__with_member<_Sequence, _Env...>) {
        using __result_t = decltype(__declval<_Sequence>().get_item_types(__declval<_Env>()...));
        return __result_t();
      } else if constexpr (__with_member_alias<_Sequence, _Env...>) {
        using __result_t = __member_alias_t<_Sequence, _Env...>;
        return __result_t();
      } else if constexpr (__with_consteval_static_member<_Sequence, _Env...>) {
        return STDEXEC_REMOVE_REFERENCE(_Sequence)::template get_item_types<_Sequence, _Env...>();
      } else if constexpr (__with_consteval_static_member<_Sequence>) {
        return STDEXEC_REMOVE_REFERENCE(_Sequence)::template get_item_types<_Sequence>();
      } else if constexpr (__with_tag_invoke<_Sequence, _Env...>) {
        using __result_t = __tag_invoke_result_t<get_item_types_t, _Sequence, _Env...>;
        return __result_t();
      } else if constexpr (
        sender_in<_Sequence, _Env...> && !enable_sequence_sender<STDEXEC::__decay_t<_Sequence>>) {
        using __result_t = item_types<STDEXEC::__decay_t<_Sequence>>;
        return __result_t();
      } else if constexpr (sizeof...(_Env) == 0) {
        return STDEXEC::__dependent_sender<_Sequence>();
      } else {
        return __unrecognized_sequence_error_t<_Sequence, _Env...>();
      }
    }

    struct get_item_types_t {
      template <class _Sequence>
      constexpr auto operator()(_Sequence&&) const noexcept {
        return __sequence_sndr::__get_item_types_helper<_Sequence>();
      }

      template <class _Sequence, class _Env>
      constexpr auto operator()(_Sequence&&, const _Env&) const noexcept {
        using __new_sequence_t = transform_sender_result_t<_Sequence, _Env>;
        static_assert(!STDEXEC::__merror<__new_sequence_t>);
        return __sequence_sndr::__get_item_types_helper<__new_sequence_t, _Env>();
      }
    };
  } // namespace __sequence_sndr

  using __sequence_sndr::get_item_types_t; // for backwards compatibility

  template <class _Sequence>
  [[nodiscard]]
  consteval auto get_item_types() {
    return __sequence_sndr::__get_item_types_helper<_Sequence>();
  }

  template <class _Sequence, class _Env>
  [[nodiscard]]
  consteval auto get_item_types() {
    using __new_sequence_t = STDEXEC::transform_sender_result_t<_Sequence, _Env>;
    static_assert(!STDEXEC::__merror<__new_sequence_t>);
    return __sequence_sndr::__get_item_types_helper<__new_sequence_t, _Env>();
  }

  // Legacy interface:
  template <class _Sequence, class... _Env>
  [[nodiscard]]
  constexpr auto get_item_types(_Sequence&&, const _Env&...) {
    return exec::get_item_types<_Sequence, _Env...>();
  }

  template <class _Sequence, class... _Env>
  using __item_types_of_t = decltype(exec::get_item_types<_Sequence, _Env...>());

  template <class _Sequence, class... _Env>
  concept has_sequence_item_types =
    STDEXEC::sender_in<_Sequence, _Env...> //
    && requires(_Sequence&& __sequence, _Env&&... __env) {
         get_item_types(static_cast<_Sequence&&>(__sequence), static_cast<_Env&&>(__env)...);
       };

  template <class _Sequence, class... _Env>
  concept sequence_sender = STDEXEC::sender_in<_Sequence, _Env...>
                         && enable_sequence_sender<STDEXEC::__decay_t<_Sequence>>;

  struct _SEQUENCE_ITEM_IS_NOT_A_WELL_FORMED_SENDER_ { };

  template <class _Sequence, class _Item>
  constexpr auto __check_item(_Item*) -> STDEXEC::__mexception<
    STDEXEC::_WHAT_(_SEQUENCE_ITEM_IS_NOT_A_WELL_FORMED_SENDER_),
    STDEXEC::_WITH_PRETTY_SENDER_<_Item>,
    _WITH_PRETTY_SEQUENCE_<_Sequence>
  >;

  template <class _Sequence, class _Item>
    requires STDEXEC::__well_formed_sender<_Item>
  auto __check_item(_Item*) -> STDEXEC::__msuccess;

  template <class _Sequence, class _Items>
    requires STDEXEC::__merror<_Items>
  auto __check_items(_Items*) -> _Items;

  struct _SEQUENCE_GET_ITEM_TYPES_RESULT_IS_NOT_WELL_FORMED_ { };

  template <template <class...> class _Class>
  struct _EXPECTED_A_SPECIALIZATION_OF_ { };

  template <class _Sequence, class _Items>
    requires(!STDEXEC::__merror<_Items>)
  auto __check_items(_Items*) -> STDEXEC::__mexception<
    STDEXEC::_WHAT_(_SEQUENCE_GET_ITEM_TYPES_RESULT_IS_NOT_WELL_FORMED_),
    STDEXEC::_WHY_(_EXPECTED_A_SPECIALIZATION_OF_<item_types>),
    STDEXEC::_WITH_TYPE_<_Items>,
    _WITH_PRETTY_SEQUENCE_<_Sequence>
  >;

  template <class _Sequence, class... _Items>
  constexpr auto __check_items(exec::item_types<_Items...>*) -> decltype((
    STDEXEC::__msuccess(),
    ...,
    exec::__check_item<_Sequence>(static_cast<_Items*>(nullptr))));

  template <class _ItemTypes, class _Sequence>
  concept __well_formed_item_types = requires(STDEXEC::__decay_t<_ItemTypes>* __item_types) {
    { exec::__check_items<_Sequence>(__item_types) } -> STDEXEC::__ok;
  };

  template <class _Sequence>
    requires STDEXEC::__merror<_Sequence>
  auto __check_sequence(_Sequence*) -> _Sequence;

  struct _ERROR_WHILE_COMPUTING_THE_SEQUENCE_ITEM_TYPES_ { };
  struct _THE_CALL_TO_GET_ITEM_TYPES_IS_ILL_FORMED_ { };

  template <class _Sequence>
    requires(!STDEXEC::__merror<_Sequence>)
         && (!STDEXEC::__minvocable_q<__item_types_of_t, _Sequence>)
  auto __check_sequence(_Sequence*) -> STDEXEC::__mexception<
    STDEXEC::_WHAT_(_ERROR_WHILE_COMPUTING_THE_SEQUENCE_ITEM_TYPES_),
    STDEXEC::_WHY_(_THE_CALL_TO_GET_ITEM_TYPES_IS_ILL_FORMED_),
    _WITH_PRETTY_SEQUENCE_<_Sequence>
  >;

  template <class _Sequence>
    requires(!STDEXEC::__merror<_Sequence>) && STDEXEC::__minvocable_q<__item_types_of_t, _Sequence>
  auto __check_sequence(_Sequence*) -> decltype(exec::__check_items<_Sequence>(
    static_cast<__item_types_of_t<_Sequence>*>(nullptr)));

  template <class _Sequence>
  concept __well_formed_item_senders = requires(STDEXEC::__decay_t<_Sequence>* __sequence) {
    { exec::__check_sequence(__sequence) } -> STDEXEC::__ok;
  };

  template <class _Sequence>
  concept __well_formed_sequence_sender = //
    STDEXEC::__well_formed_sender<_Sequence>
    && enable_sequence_sender<STDEXEC::__decay_t<_Sequence>>
    && (__well_formed_item_senders<_Sequence> || STDEXEC::dependent_sender<_Sequence>);

  template <class _Sequence, class... _Env>
  concept sequence_sender_in =
    sequence_sender<_Sequence, _Env...> && requires(_Sequence&& __sequence, _Env&&... __env) {
      {
        get_item_types(static_cast<_Sequence&&>(__sequence), static_cast<_Env&&>(__env)...)
      } -> __well_formed_item_types<_Sequence>;
    };

  namespace __debug {
    template <class _Env = STDEXEC::env<>, class _Sequence>
    constexpr void __debug_sequence_sender(_Sequence&& __sequence, const _Env& = {});
  } // namespace __debug
  using __debug::__debug_sequence_sender;

  template <class _Sequence, class... _Env>
  static constexpr auto __diagnose_sequence_sender_concept_failure();

#if STDEXEC_ENABLE_EXTRA_TYPE_CHECKING()
  // __checked_completion_signatures is for catching logic bugs in a sender's metadata. If sender<S>
  // and sender_in<S, Ctx> are both true, then they had better report the same metadata. This
  // completion signatures wrapper enforces that at compile time.
  template <class _Sequence, class... _Env>
  auto __checked_item_types(_Sequence&& __sequence, _Env&&... __env) noexcept {
    using __item_types_t = __item_types_of_t<_Sequence, _Env...>;
    static_assert(STDEXEC::__ok<__item_types_t>, "get_item_types returned an error");
    exec::__debug_sequence_sender(static_cast<_Sequence&&>(__sequence), __env...);
    return __item_types_t{};
  }

  template <class _Sequence, class... _Env>
    requires sequence_sender<_Sequence, _Env...>
  using item_types_of_t = decltype(exec::__checked_item_types(
    STDEXEC::__declval<_Sequence>(),
    STDEXEC::__declval<_Env>()...));
#else
  template <class _Sequence, class... _Env>
    requires sequence_sender_in<_Sequence, _Env...>
  using item_types_of_t = __item_types_of_t<_Sequence, _Env...>;
#endif

  template <class _Data, class... _What>
  struct __sequence_type_check_failure //
    : STDEXEC::__compile_time_error<__sequence_type_check_failure<_Data, _What...>> {
    static_assert(
      std::is_nothrow_move_constructible_v<_Data>,
      "The data member of sender_type_check_failure must be nothrow move constructible.");

    constexpr __sequence_type_check_failure() noexcept = default;

    constexpr explicit __sequence_type_check_failure(_Data data)
      : __data_(static_cast<_Data&&>(data)) {
    }

   private:
    friend struct STDEXEC::__compile_time_error<__sequence_type_check_failure>;

    [[nodiscard]]
    constexpr auto what() const noexcept -> const char* {
      return "This sequence sender is not well-formed. It does not meet the requirements of a "
             "sequence sender type.";
    }

    _Data __data_{};
  };

  struct _MISSING_SET_NEXT_OVERLOAD_FOR_ITEM_ { };

  template <class _Receiver, class _Item>
  constexpr auto __try_item(_Item*) -> STDEXEC::__mexception<
    STDEXEC::_WHAT_(_MISSING_SET_NEXT_OVERLOAD_FOR_ITEM_),
    STDEXEC::_WITH_PRETTY_SENDER_<_Item>,
    STDEXEC::_WITH_RECEIVER_(_Receiver)
  >;

  template <class _Receiver, class _Item>
    requires STDEXEC::__callable<set_next_t, _Receiver&, _Item>
  auto __try_item(_Item*) -> STDEXEC::__msuccess;

  template <class _Receiver, class... _Items>
  constexpr auto __try_items(exec::item_types<_Items...>*) -> decltype((
    STDEXEC::__msuccess(),
    ...,
    exec::__try_item<_Receiver>(static_cast<_Items*>(nullptr))));

  template <class _Receiver, class _Items>
  concept __sequence_receiver_of = requires(_Items* __items) {
    { exec::__try_items<STDEXEC::__decay_t<_Receiver>>(__items) } -> STDEXEC::__ok;
  };

  template <class _Receiver, class _SequenceItems>
  concept sequence_receiver_of = STDEXEC::receiver<_Receiver>
                              && __sequence_receiver_of<_Receiver, _SequenceItems>;

  template <class _Completions>
  consteval auto __to_sequence_completions(_Completions __completions) {
    return exec::concat_completion_signatures(
      STDEXEC::completion_signatures<STDEXEC::set_value_t()>(),
      exec::transform_completion_signatures(__completions, exec::ignore_completion()));
  }

  template <class _Completions>
  using __to_sequence_completions_t = decltype(exec::__to_sequence_completions(_Completions()));

  template <class _Sender, class... _Env>
  consteval auto __item_completion_signatures() {
    return exec::concat_completion_signatures(
      STDEXEC::completion_signatures<STDEXEC::set_value_t()>(),
      exec::transform_completion_signatures(
        STDEXEC::get_completion_signatures<_Sender, _Env...>(), exec::ignore_completion()));
  }

  template <class _Sequence, class... _Env>
  consteval auto __sequence_completion_signatures() {
    return exec::concat_completion_signatures(
      STDEXEC::completion_signatures<STDEXEC::set_value_t()>(),
      exec::transform_completion_signatures(
        STDEXEC::get_completion_signatures<_Sequence, _Env...>(), exec::ignore_completion()));
  }

  template <class... _Env>
  consteval auto __mk_item_transform() {
    return []<class _ItemSender>() {
      return exec::__item_completion_signatures<_ItemSender, _Env...>();
    };
  }

  template <class _Sequence, class... _Env>
  consteval auto __sequence_completion_signatures_of() {
    auto __item_types = get_item_types<_Sequence, _Env...>();
    if constexpr (STDEXEC::__merror<decltype(__item_types)>) {
      return exec::__invalid_item_types(__item_types);
    } else {
      return exec::concat_completion_signatures(
        exec::__sequence_completion_signatures<_Sequence, _Env...>(),
        __item_types
          .__transform(exec::__mk_item_transform<_Env...>(), exec::concat_completion_signatures));
    }
  }

  template <class _Sequence, class... _Env>
  using __sequence_completion_signatures_of_t =
    decltype(exec::__sequence_completion_signatures_of<_Sequence, _Env...>());

  template <class _Receiver, class _Sequence>
  concept __sequence_receiver_from =                            //
    sequence_sender_in<_Sequence, STDEXEC::env_of_t<_Receiver>> //
    && STDEXEC::receiver_of<
      _Receiver,
      STDEXEC::completion_signatures_of_t<_Sequence, STDEXEC::env_of_t<_Receiver>>
    >;

  template <class _Receiver, class _Sequence>
  concept __stopped_means_break_receiver_from = //
    !sequence_sender_in<_Sequence, STDEXEC::env_of_t<_Receiver>>
    && STDEXEC::__receiver_from<
      __sequence_sndr::__stopped_means_break<_Receiver>,
      next_sender_of_t<_Receiver, _Sequence>
    >;

  template <class _Receiver, class _Sequence>
  concept sequence_receiver_from =                                 //
    STDEXEC::receiver<_Receiver>                                   //
    && STDEXEC::sender_in<_Sequence, STDEXEC::env_of_t<_Receiver>> //
    && sequence_receiver_of<_Receiver, __item_types_of_t<_Sequence, STDEXEC::env_of_t<_Receiver>>> //
    && bool( // cast to bool to hide the disjunction
      __sequence_receiver_from<_Receiver, _Sequence> //
      || __stopped_means_break_receiver_from<_Receiver, _Sequence>);

  namespace __sequence_sndr {
    struct subscribe_t;

    struct _NO_USABLE_SUBSCRIBE_CUSTOMIZATION_FOUND_ {
      constexpr void operator()() const noexcept = delete;
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
      && sequence_receiver_of<_Receiver, item_types<STDEXEC::__decay_t<_Sender>>>
      && sender_to<next_sender_of_t<_Receiver, _Sender>, __stopped_means_break<_Receiver>>;

    template <class _Sequence, class _Receiver>
    using __subscribe_member_result_t = decltype(__declval<_Sequence>()
                                                   .subscribe(__declval<_Receiver>()));

    template <class _Sequence, class _Receiver>
    concept __subscribable_with_member =
      __minvocable_q<__subscribe_member_result_t, _Sequence, _Receiver>;

    template <class _Sequence, class _Receiver>
    using __subscribe_static_member_result_t = decltype(STDEXEC_REMOVE_REFERENCE(
      _Sequence)::subscribe(__declval<_Sequence>(), __declval<_Receiver>()));

    template <class _Sequence, class _Receiver>
    concept __subscribable_with_static_member =
      __minvocable_q<__subscribe_static_member_result_t, _Sequence, _Receiver>;

    template <class _Sequence, class _Receiver>
    concept __subscribable_with_tag_invoke = __tag_invocable<subscribe_t, _Sequence, _Receiver>;

    struct subscribe_t {
     private:
      template <class _Sequence, class _Receiver>
      STDEXEC_ATTRIBUTE(always_inline)
      static constexpr auto __type_check_arguments() -> bool {
        if constexpr (sequence_sender_in<_Sequence, env_of_t<_Receiver>>) {
          // Instantiate __debug_sender via completion_signatures_of_t and
          // item_types_of_t to check that the actual completions and item_types
          // match the expected completions and values.
          using __checked_signatures
            [[maybe_unused]] = completion_signatures_of_t<_Sequence, env_of_t<_Receiver>>;
          using __checked_item_types
            [[maybe_unused]] = item_types_of_t<_Sequence, env_of_t<_Receiver>>;
        } else {
          __diagnose_sequence_sender_concept_failure<_Sequence, env_of_t<_Receiver>>();
        }
        return true;
      }

      template <class _Sequence, class _Receiver>
      using __transform_sender_result_t = transform_sender_result_t<_Sequence, env_of_t<_Receiver>>;

      template <class _OpState>
      static constexpr void __check_operation_state() noexcept {
        static_assert(
          operation_state<_OpState>, STDEXEC_ERROR_SUBSCRIBE_DOES_NOT_RETURN_OPERATION_STATE);
      }

      template <class _Sequence, class _Receiver>
      static consteval auto __get_declfn() noexcept {
        constexpr bool __nothrow_tfx_seq =
          __noexcept_of<transform_sender, _Sequence, env_of_t<_Receiver>>;
        using __tfx_seq_t = __transform_sender_result_t<_Sequence, _Receiver>;

        static_assert(
          sequence_sender<_Sequence> || has_sequence_item_types<_Sequence, env_of_t<_Receiver>>,
          "The first argument to STDEXEC::subscribe must be a sequence sender");
        static_assert(
          receiver<_Receiver>, "The second argument to STDEXEC::subscribe must be a receiver");
#if STDEXEC_ENABLE_EXTRA_TYPE_CHECKING()
        static_assert(__type_check_arguments<__tfx_seq_t, _Receiver>());
#endif

        if constexpr (__next_connectable<__tfx_seq_t, _Receiver>) {
          using __result_t = connect_result_t<
            next_sender_of_t<_Receiver, __tfx_seq_t>,
            __stopped_means_break<_Receiver>
          >;
          __check_operation_state<__result_t>();
          constexpr bool __nothrow_subscribe = __nothrow_connectable<
            next_sender_of_t<_Receiver, __tfx_seq_t>,
            __stopped_means_break<_Receiver>
          >;
          return __declfn<__result_t, __nothrow_subscribe && __nothrow_tfx_seq>();
        } else if constexpr (__subscribable_with_static_member<__tfx_seq_t, _Receiver>) {
          using __result_t = decltype(STDEXEC_REMOVE_REFERENCE(
            __tfx_seq_t)::subscribe(__declval<__tfx_seq_t>(), __declval<_Receiver>()));
          __check_operation_state<__result_t>();
          constexpr bool __nothrow_subscribe = noexcept(STDEXEC_REMOVE_REFERENCE(
            __tfx_seq_t)::subscribe(__declval<__tfx_seq_t>(), __declval<_Receiver>()));
          return __declfn<__result_t, __nothrow_subscribe && __nothrow_tfx_seq>();
        } else if constexpr (__subscribable_with_member<__tfx_seq_t, _Receiver>) {
          using __result_t = decltype(__declval<__tfx_seq_t>().subscribe(__declval<_Receiver>()));
          __check_operation_state<__result_t>();
          constexpr bool __nothrow_subscribe = noexcept(__declval<__tfx_seq_t>()
                                                          .subscribe(__declval<_Receiver>()));
          return __declfn<__result_t, __nothrow_subscribe && __nothrow_tfx_seq>();
        } else if constexpr (__is_debug_env<env_of_t<_Receiver>>) {
          using __result_t = __debug::__opstate;
          return __declfn<__result_t, __nothrow_tfx_seq>();
        } else {
          static_assert(
            __subscribable_with_static_member<__tfx_seq_t, _Receiver>,
            STDEXEC_ERROR_CANNOT_SUBSCRIBE_SEQUENCE_TO_RECEIVER);
          return _NO_USABLE_SUBSCRIBE_CUSTOMIZATION_FOUND_();
        }
      }

     public:
      template <
        sender _Sequence,
        receiver _Receiver,
        auto _DeclFn = __get_declfn<_Sequence, _Receiver>()
      >
        requires STDEXEC::__callable<decltype(_DeclFn)>
      auto operator()(_Sequence&& __sequence, _Receiver __rcvr) const noexcept(noexcept(_DeclFn()))
        -> decltype(_DeclFn()) {
        using __tfx_seq_t = __transform_sender_result_t<_Sequence, _Receiver>;
        auto&& __env = STDEXEC::get_env(__rcvr);
        auto&& __tfx_seq = STDEXEC::transform_sender(static_cast<_Sequence&&>(__sequence), __env);

        if constexpr (__next_connectable<__tfx_seq_t, _Receiver>) {
          // sender as sequence of one
          next_sender_of_t<_Receiver, __tfx_seq_t> __next =
            set_next(__rcvr, static_cast<__tfx_seq_t&&>(__tfx_seq));
          return STDEXEC::connect(
            static_cast<next_sender_of_t<_Receiver, __tfx_seq_t>&&>(__next),
            __stopped_means_break<_Receiver>{static_cast<_Receiver&&>(__rcvr)});
          // NOLINTNEXTLINE(bugprone-branch-clone)
        } else if constexpr (__subscribable_with_static_member<__tfx_seq_t, _Receiver>) {
          return __tfx_seq
            .subscribe(static_cast<__tfx_seq_t&&>(__tfx_seq), static_cast<_Receiver&&>(__rcvr));
        } else if constexpr (__subscribable_with_member<__tfx_seq_t, _Receiver>) {
          return static_cast<__tfx_seq_t&&>(__tfx_seq).subscribe(static_cast<_Receiver&&>(__rcvr));
        } else if constexpr (enable_sequence_sender<STDEXEC::__decay_t<__tfx_seq_t>>) {
          // sequence sender fallback

          // This should generate an instantiate backtrace that contains useful
          // debugging information.
          return __tfx_seq
            .subscribe(static_cast<__tfx_seq_t&&>(__tfx_seq), static_cast<_Receiver&&>(__rcvr));
        } else {
          // sender as sequence of one fallback

          // This should generate an instantiate backtrace that contains useful
          // debugging information.
          next_sender_of_t<_Receiver, __tfx_seq_t> __next =
            set_next(__rcvr, static_cast<__tfx_seq_t&&>(__tfx_seq));
          return STDEXEC::connect(
            static_cast<next_sender_of_t<_Receiver, __tfx_seq_t>&&>(__next),
            __stopped_means_break<_Receiver>{static_cast<_Receiver&&>(__rcvr)});
        }
      }

      template <
        sender _Sequence,
        receiver _Receiver,
        auto _DeclFn = __get_declfn<_Sequence, _Receiver>()
      >
        requires STDEXEC::__callable<decltype(_DeclFn)>
              || STDEXEC::__tag_invocable<
                   subscribe_t,
                   __transform_sender_result_t<_Sequence, _Receiver>,
                   _Receiver
              >
      [[deprecated("the use of tag_invoke for subscribe is deprecated")]]
      auto operator()(_Sequence&& __sequence, _Receiver __rcvr) const noexcept(
        __nothrow_callable<transform_sender_t, _Sequence, env_of_t<_Receiver>>
        && STDEXEC::__nothrow_tag_invocable<
          subscribe_t,
          __transform_sender_result_t<_Sequence, _Receiver>,
          _Receiver
        >)
        -> STDEXEC::__tag_invoke_result_t<
          subscribe_t,
          __transform_sender_result_t<_Sequence, _Receiver>,
          _Receiver
        > {
        using __tfx_seq_t = __transform_sender_result_t<_Sequence, _Receiver>;
        using __result_t = __tag_invoke_result_t<subscribe_t, __tfx_seq_t, _Receiver>;
        __check_operation_state<__result_t>();
        auto&& __env = STDEXEC::get_env(__rcvr);
        auto&& __tfx_seq = STDEXEC::transform_sender(static_cast<_Sequence&&>(__sequence), __env);
        return STDEXEC::__tag_invoke(
          subscribe_t{}, static_cast<__tfx_seq_t&&>(__tfx_seq), static_cast<_Receiver&&>(__rcvr));
      }

      static constexpr auto query(STDEXEC::forwarding_query_t) noexcept -> bool {
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
  concept __nothrow_subscribable = STDEXEC::__nothrow_callable<subscribe_t, _Sequence, _Receiver>;

  template <class _Sequence, class _Receiver>
  concept sequence_sender_to =
    sequence_receiver_from<_Receiver, _Sequence>
    && requires(_Sequence&& __sequence, _Receiver&& __rcvr) {
         subscribe(static_cast<_Sequence&&>(__sequence), static_cast<_Receiver&&>(__rcvr));
       };

  template <class _Receiver>
  concept __stoppable_receiver = STDEXEC::__callable<STDEXEC::set_value_t, _Receiver>
                              && (STDEXEC::unstoppable_token<
                                    STDEXEC::stop_token_of_t<STDEXEC::env_of_t<_Receiver>>
                                  >
                                  || STDEXEC::__callable<STDEXEC::set_stopped_t, _Receiver>);

  template <class _Receiver>
    requires __stoppable_receiver<_Receiver>
  void __set_value_unless_stopped(_Receiver&& __rcvr) {
    using token_type = STDEXEC::stop_token_of_t<STDEXEC::env_of_t<_Receiver>>;
    if constexpr (STDEXEC::unstoppable_token<token_type>) { // NOLINT(bugprone-branch-clone)
      STDEXEC::set_value(static_cast<_Receiver&&>(__rcvr));
    } else if (!STDEXEC::get_stop_token(STDEXEC::get_env(__rcvr)).stop_requested()) {
      STDEXEC::set_value(static_cast<_Receiver&&>(__rcvr));
    } else {
      STDEXEC::set_stopped(static_cast<_Receiver&&>(__rcvr));
    }
  }

////////////////////////////////////////////////////////////////////////////////
#define STDEXEC_ERROR_ENABLE_SEQUENCE_SENDER_IS_FALSE                                              \
  "\n"                                                                                             \
  "\n"                                                                                             \
  "Trying to compute the sequences's item types resulted in an error. See\n"                       \
  "the rest of the compiler diagnostic for clues. Look for the string \"_ERROR_\".\n"

////////////////////////////////////////////////////////////////////////////////
#define STDEXEC_ERROR_GET_ITEM_TYPES_RETURNED_AN_ERROR                                             \
  "\n"                                                                                             \
  "\n"                                                                                             \
  "Trying to compute the sequences's item types resulted in an error. See\n"                       \
  "the rest of the compiler diagnostic for clues. Look for the string \"_ERROR_\".\n"

////////////////////////////////////////////////////////////////////////////////
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
  "      STDEXEC::__call_result_t<STDEXEC::just_t>>\n"                                             \
  "    {\n"                                                                                        \
  "    return {};\n"                                                                               \
  "    }\n"                                                                                        \
  "    ...\n"                                                                                      \
  "  };\n"

  // Used to report a meaningful error message when the sender_in<Sndr, Env>
  // concept check fails.
  template <class _Sequence, class... _Env>
  static constexpr auto __diagnose_sequence_sender_concept_failure() {
    if constexpr (
      !enable_sequence_sender<STDEXEC::__decay_t<_Sequence>>
      && !has_sequence_item_types<STDEXEC::__decay_t<_Sequence>, _Env...>) {
      static_assert(
        enable_sequence_sender<_Sequence>, STDEXEC_ERROR_ENABLE_SEQUENCE_SENDER_IS_FALSE);
    } else if constexpr (!std::move_constructible<STDEXEC::__decay_t<_Sequence>>) {
      static_assert(
        std::move_constructible<STDEXEC::__decay_t<_Sequence>>,
        "The sequence type is not move-constructible.");
    } else if constexpr (!std::constructible_from<STDEXEC::__decay_t<_Sequence>, _Sequence>) {
      static_assert(
        std::constructible_from<STDEXEC::__decay_t<_Sequence>, _Sequence>,
        "The sequence cannot be decay-copied. Did you forget a std::move?");
    } else {
      using __items_t = __item_types_of_t<_Sequence, _Env...>;
      if constexpr (STDEXEC::__same_as<
                      __items_t,
                      __unrecognized_sequence_error_t<_Sequence, _Env...>
                    >) {
        static_assert(
          STDEXEC::__mnever<__items_t>, STDEXEC_ERROR_CANNOT_COMPUTE_COMPLETION_SIGNATURES);
      } else if constexpr (STDEXEC::__merror<__items_t>) {
        static_assert(
          !STDEXEC::__merror<__items_t>, STDEXEC_ERROR_GET_ITEM_TYPES_RETURNED_AN_ERROR);
      } else if constexpr (!__well_formed_item_senders<_Sequence>) {
        static_assert(
          __well_formed_item_senders<_Sequence>,
          STDEXEC_ERROR_GET_ITEM_TYPES_HAS_INVALID_RETURN_TYPE);
      } else {
        STDEXEC::__diagnose_sender_concept_failure<STDEXEC::__demangle_t<_Sequence>, _Env...>();
      }
    }
  }

  struct THE_SEQUENCE_SENDER_CALLED_SET_NEXT_WITH_AN_ITEM_TYPE_NOT_LISTED_IN_ITS_GET_ITEM_TYPES_RESULT {
  };
  struct GET_ITEM_TYPES_RESULT { };

  namespace __debug {
    template <class... _Items>
    struct __valid_next {
      template <class _Item>
      STDEXEC_ATTRIBUTE(host, device)
      STDEXEC::__call_result_t<STDEXEC::just_t> set_next(_Item&&) noexcept {
        if constexpr (STDEXEC::__none_of<_Item, _Items...>) {
          using __what_t = STDEXEC::__mexception<
            STDEXEC::_WHAT_(
              THE_SEQUENCE_SENDER_CALLED_SET_NEXT_WITH_AN_ITEM_TYPE_NOT_LISTED_IN_ITS_GET_ITEM_TYPES_RESULT),
            STDEXEC::_WITH_TYPE_<_Item>,
            GET_ITEM_TYPES_RESULT(item_types<_Items...>)
          >;
          STDEXEC::__debug::_ATTENTION_<__what_t>();
        }
        STDEXEC_TERMINATE();
      }
    };

    template <class _CvSequence, class _Env, class _ItemTypes, class... _Sigs>
    struct __sequence_receiver;

    template <class _CvSequence, class _Env, class... _Items, class... _Sigs>
    struct __sequence_receiver<_CvSequence, _Env, item_types<_Items...>, _Sigs...>
      : STDEXEC::__debug::__receiver<_CvSequence, _Env, _Sigs...>
      , __valid_next<_Items...> {
      using receiver_concept = STDEXEC::receiver_t;

      STDEXEC_ATTRIBUTE(host, device)
      constexpr auto get_env() const noexcept -> STDEXEC::__debug::__env_t<_Env> {
        STDEXEC_TERMINATE();
      }
    };

    template <class _CvSequence, class _Env, class _Items, class _Sigs>
    using __sequence_receiver_t = STDEXEC::__mapply<
      STDEXEC::__mtransform<
        STDEXEC::__mcompose<
          STDEXEC::__q1<std::remove_pointer_t>,
          STDEXEC::__q1<STDEXEC::__cmplsigs::__normalize_sig_t>
        >,
        STDEXEC::__mbind_front_q<__sequence_receiver, _CvSequence, _Env, _Items>
      >,
      _Sigs
    >;

    template <class _CvSequence, class _Env>
    constexpr void __debug_sequence_sender(_CvSequence&& __sequence, const _Env&) {
      if constexpr (!STDEXEC::__is_debug_env<_Env>) {
        if constexpr (sequence_sender_in<_CvSequence, _Env>) {
          using __completions_t =
            STDEXEC::__completion_signatures_of_t<_CvSequence, STDEXEC::__debug::__env_t<_Env>>;
          using __item_types_t = __item_types_of_t<_CvSequence, STDEXEC::__debug::__env_t<_Env>>;
          using __receiver_t =
            __sequence_receiver_t<_CvSequence, _Env, __item_types_t, __completions_t>;

          if constexpr (STDEXEC::__not_same_as<__item_types_t, __debug::__item_types>) {
            using __operation_t = exec::subscribe_result_t<_CvSequence, __receiver_t>;
            //static_assert(receiver_of<__receiver_t, __sigs_t>);
            if constexpr (STDEXEC::__not_same_as<__operation_t, STDEXEC::__debug::__opstate>) {
              if (STDEXEC::__mnever<_CvSequence>) { // never true
                auto __op = subscribe(static_cast<_CvSequence&&>(__sequence), __receiver_t{});
                STDEXEC::start(__op);
              }
            }
          }
        } else {
          __diagnose_sequence_sender_concept_failure<_CvSequence, _Env>();
        }
      }
    }
  } // namespace __debug
} // namespace exec

STDEXEC_PRAGMA_POP()
