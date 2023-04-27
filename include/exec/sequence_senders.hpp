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

  namespace __sequence_sender {
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
  } // namespace __sequence_sender

  using __sequence_sender::set_next_t;
  inline constexpr set_next_t set_next;

  namespace __sequence_sender {
    struct __nop_operation {
      friend void tag_invoke(start_t, __nop_operation&) noexcept {
      }
    };

    template <__is_completion_signatures _Sigs>
    struct __some_sender_of {
      using is_sender = void;
      using completion_signatures = _Sigs;
      using __id = __some_sender_of;
      using __t = __some_sender_of;

      template <class R>
      friend __nop_operation tag_invoke(connect_t, __some_sender_of, R&&) {
        return {};
      }
    };
  } // namespace __sequence_sender

  template <class _Sender>
  concept __enable_sequence_sender =             //
    requires { typename _Sender::is_sender; } && //
    stdexec::same_as<typename _Sender::is_sender, sequence_tag>;

  template <class _Sender>
  inline constexpr bool enable_sequence_sender = __enable_sequence_sender<_Sender>;

  template <class _Signatures, class _Env = stdexec::empty_env>
  using __sequence_to_sender_sigs_t = stdexec::__try_make_completion_signatures<
    __sequence_sender::__some_sender_of<_Signatures>,
    _Env,
    stdexec::completion_signatures<stdexec::set_value_t()>,
    stdexec::__mconst<stdexec::completion_signatures<>>>;

  template <class _Receiver, class _Signatures>
  concept sequence_receiver_of =
    stdexec::receiver<_Receiver>
    && stdexec::receiver_of<
      _Receiver,
      __sequence_to_sender_sigs_t<_Signatures, stdexec::env_of_t<_Receiver>>>
    && stdexec::__callable<
      set_next_t,
      stdexec::__decay_t<_Receiver>&,
      __sequence_sender::__some_sender_of<_Signatures>>;

  template <class _Receiver, class _Sender>
  concept sequence_receiver_from =
    stdexec::sender_in<_Sender, stdexec::env_of_t<_Receiver>>
    && sequence_receiver_of<
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

  template <class _Sender>
  concept sequence_sender =     //
    stdexec::sender<_Sender> && //
    enable_sequence_sender<stdexec::__decay_t<_Sender>>;

  template <class _Sender, class _Receiver>
  concept sequence_sender_to =
    sequence_sender<_Sender> && stdexec::sender_in<_Sender, stdexec::env_of_t<_Receiver>>
    && sequence_receiver_from<_Receiver, _Sender>
    && requires(_Sender&& __sndr, _Receiver&& __rcvr) {
         { sequence_connect((_Sender&&) __sndr, (_Receiver&&) __rcvr) };
       };

  template <class _Receiver, class _Sender>
  using __next_sender_of_t =
    decltype(exec::set_next(stdexec::__declval<_Receiver>(), stdexec::__declval<_Sender>()));
}