/*
 * Copyright (c) 2021-2025 NVIDIA Corporation
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

#include "__senders_core.hpp"
#include "__operation_states.hpp"

namespace stdexec {
  namespace __submit {
    template <class _Sender, class _Receiver>
    concept __has_memfn =
      requires(_Sender&& (*__sndr)(), _Receiver&& (*__rcvr)()) { __sndr().submit(__rcvr()); };

    template <class _Sender, class _Receiver>
    concept __has_static_memfn = requires(_Sender&& (*__sndr)(), _Receiver&& (*__rcvr)()) {
      __decay_t<_Sender>::submit(__sndr(), __rcvr());
    };

    template <class _CvSenderId, class _ReceiverId>
    struct __data {
      using _CvSender = __cvref_t<_CvSenderId>;
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using operation_state_concept = operation_state_t;
        using __id = __data;

        STDEXEC_IMMOVABLE(__t);
        explicit __t(_CvSender&& __sndr, _Receiver __rcvr) noexcept(
          __nothrow_connectable<_CvSender, _Receiver>)
          : __op_(connect(static_cast<_CvSender&&>(__sndr), static_cast<_Receiver&&>(__rcvr))) {
          start(__op_);
        }

        connect_result_t<_CvSender, _Receiver> __op_;
      };
    };

    template <class _CvSender, class _Receiver>
    using __op_data = __t<__data<__cvref_id<_CvSender>, __id<_Receiver>>>;
  } // namespace __submit

  // submit is a combination of connect and start. it is customizable for times when it
  // can be done more efficiently than by calling connect and start directly.
  struct __submit_t {
    struct __void { };

    // The default implementation of submit is to call connect and start.
    template <class _Sender, class _Receiver, class _Default = __void>
      requires sender_to<_Sender, _Receiver>
    auto operator()(_Sender&& __sndr, _Receiver __rcvr, _Default = {}) const
      noexcept(__nothrow_connectable<_Sender, _Receiver>)
        -> __submit::__op_data<_Sender, _Receiver> {
      return __submit::__op_data<_Sender, _Receiver>{
        static_cast<_Sender&&>(__sndr), static_cast<_Receiver&&>(__rcvr)};
    }

    // This implementation is used if the sender has a non-static submit member function.
    template <class _Sender, class _Receiver, class _Default = __void>
      requires sender_to<_Sender, _Receiver> && __submit::__has_memfn<_Sender, _Receiver>
    auto operator()(_Sender&& __sndr, _Receiver __rcvr, [[maybe_unused]] _Default __def = {}) const
      noexcept(noexcept(static_cast<_Sender&&>(__sndr).submit(static_cast<_Receiver&&>(__rcvr)))) {
      using __result_t =
        decltype(static_cast<_Sender&&>(__sndr).submit(static_cast<_Receiver&&>(__rcvr)));
      if constexpr (__same_as<__result_t, void> && !__same_as<_Default, __void>) {
        static_cast<_Sender&&>(__sndr).submit(static_cast<_Receiver&&>(__rcvr));
        return __def;
      } else {
        return static_cast<_Sender&&>(__sndr).submit(static_cast<_Receiver&&>(__rcvr));
      }
    }

    // This implementation is used if the sender has a static submit member function.
    template <class _Sender, class _Receiver, class _Default = __void>
      requires sender_to<_Sender, _Receiver> && __submit::__has_static_memfn<_Sender, _Receiver>
    auto operator()(_Sender&& __sndr, _Receiver __rcvr, [[maybe_unused]] _Default __def = {}) const
      noexcept(
        noexcept(__sndr.submit(static_cast<_Sender&&>(__sndr), static_cast<_Receiver&&>(__rcvr)))) {
      using __result_t =
        decltype(__sndr.submit(static_cast<_Sender&&>(__sndr), static_cast<_Receiver&&>(__rcvr)));
      if constexpr (__same_as<__result_t, void> && !__same_as<_Default, __void>) {
        __sndr.submit(static_cast<_Sender&&>(__sndr), static_cast<_Receiver&&>(__rcvr));
        return __def;
      } else {
        return __sndr.submit(static_cast<_Sender&&>(__sndr), static_cast<_Receiver&&>(__rcvr));
      }
    }
  };

  inline constexpr __submit_t submit{};

  template <class _Sender, class _Receiver, class _Default = __submit_t::__void>
  using submit_result_t = __call_result_t<__submit_t, _Sender, _Receiver, _Default>;
} // namespace stdexec
