/*
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
  namespace __sequence_sender {
    using namespace stdexec;

    struct set_next_t {
      template <receiver _Receiver, sender _Item>
        requires tag_invocable<set_next_t, _Receiver&, _Item>
      auto operator()(_Receiver& __rcvr, _Item&& __item) const noexcept
        -> tag_invoke_result_t<set_next_t, _Receiver&, _Item> {
        static_assert(nothrow_tag_invocable<set_next_t, _Receiver&, _Item>);
        return tag_invoke(*this, __rcvr, (_Item&&) __item);
      }
    };
  } // namespace __sequence_sender

  using __sequence_sender::set_next_t;
  inline constexpr set_next_t set_next;

  namespace __sequence_sender {
    template <class _Signature>
    struct _MISSING_NEXT_SIGNATURE_;

    template <class _Item>
    struct _MISSING_NEXT_SIGNATURE_<set_next_t(_Item)> {
      template <class _Receiver>
      struct _WITH_RECEIVER_ : std::false_type { };

      struct _ {
        _(...) {
        }
      };

      friend auto operator,(_MISSING_NEXT_SIGNATURE_, _) -> _MISSING_NEXT_SIGNATURE_ {
        return {};
      }
    };

    struct __found_next_signature {
      template <class _Receiver>
      using _WITH_RECEIVER_ = std::true_type;
    };

    template <class... _Args>
    using __just_t = decltype(just(__declval<_Args>()...));

    template <class _Receiver, class... _Args>
    using __missing_next_signature_t = __if<
      __mbool<nothrow_tag_invocable<set_next_t, _Receiver&, __just_t<_Args...>>>,
      __found_next_signature,
      _MISSING_NEXT_SIGNATURE_<set_next_t(__just_t<_Args...>)>>;

    template <class _Receiver, class... _Args>
    auto __has_sequence_signature(set_value_t (*)(_Args...))
      -> __missing_next_signature_t<_Receiver, _Args...>;

    template < class _Receiver>
    auto __has_sequence_signature(set_stopped_t (*)())
      -> __receiver_concepts::__missing_completion_signal_t<_Receiver, set_stopped_t>;

    template <class _Receiver, class Error>
    auto __has_sequence_signature(set_error_t (*)(Error))
      -> __receiver_concepts::__missing_completion_signal_t<_Receiver, set_error_t, Error>;

    template <class _Receiver, class... _Sigs>
    auto __has_sequence_signatures(completion_signatures<_Sigs...>*)
      -> decltype((__has_sequence_signature<_Receiver>(static_cast<_Sigs*>(nullptr)), ...));

    template <class _Signatures, class _Receiver>
    concept __is_valid_next_completions = _Signatures::template _WITH_RECEIVER_<_Receiver>::value;
  }

  template <class _Receiver, class Signatures>
  concept sequence_receiver_of =
    stdexec::receiver_of<_Receiver, stdexec::completion_signatures<stdexec::set_value_t()>>
    && requires(Signatures* sigs) {
         {
           __sequence_sender::__has_sequence_signatures<stdexec::__decay_t<_Receiver>>(sigs)
         } -> __sequence_sender::__is_valid_next_completions<stdexec::__decay_t<_Receiver>>;
       };

  template <class _Receiver, class _Sender>
  concept sequence_receiver_from = sequence_receiver_of<
    _Receiver,
    stdexec::completion_signatures_of_t<_Sender, stdexec::env_of_t<_Receiver>>>;

  namespace __sequence_sender {
    struct sequence_connect_t;

    template <class _Sender, class _Receiver>
    concept __sequence_connectable_with_tag_invoke =
      receiver<_Receiver> &&                        //
      sender_in<_Sender, env_of_t<_Receiver>> &&    //
      sequence_receiver_from<_Receiver, _Sender> && //
      tag_invocable<sequence_connect_t, _Sender, _Receiver>;

    struct sequence_connect_t {
      template <class _Sender, class _Receiver>
        requires __sequence_connectable_with_tag_invoke<_Sender, _Receiver>
      auto operator()(_Sender&& __sender, _Receiver&& __rcvr) const
        noexcept(nothrow_tag_invocable<sequence_connect_t, _Sender, _Receiver>)
          -> tag_invoke_result_t<sequence_connect_t, _Sender, _Receiver> {
        static_assert(
          operation_state<tag_invoke_result_t<sequence_connect_t, _Sender, _Receiver>>,
          "exec::sequence_connect(sender, receiver) must return a type that "
          "satisfies the operation_state concept");
        return tag_invoke(*this, (_Sender&&) __sender, (_Receiver&&) __rcvr);
      }
    };

    template <class _Sender, class _Receiver>
    using sequence_connect_result_t = __call_result_t<sequence_connect_t, _Sender, _Receiver>;
  } // namespace __sequence_sender

  using __sequence_sender::sequence_connect_t;
  inline constexpr sequence_connect_t sequence_connect;

  using __sequence_sender::sequence_connect_result_t;

  template <class _Sender, class _Receiver>
  concept sequence_sender_to =
    stdexec::sender_in<_Sender, stdexec::env_of_t<_Receiver>>
    && sequence_receiver_from<_Receiver, _Sender>
    && requires(_Sender&& __sndr, _Receiver&& __rcvr) {
         { sequence_connect((_Sender&&) __sndr, (_Receiver&&) __rcvr) } -> stdexec::operation_state;
       };
}