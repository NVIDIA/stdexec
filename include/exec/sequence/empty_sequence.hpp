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

#include "../sequence_senders.hpp"

namespace exec {
  namespace __empty_sequence {

    using namespace stdexec;

    template <class _ReceiverId>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using __id = __operation;
        STDEXEC_ATTRIBUTE((no_unique_address)) _Receiver __rcvr_;

        friend void tag_invoke(start_t, __t& __self) noexcept {
          stdexec::set_value(static_cast<_Receiver&&>(__self.__rcvr_));
        }
      };
    };

    struct __sender {
      struct __t {
        using __id = __sender;
        using sender_concept = sequence_sender_t;
        using completion_signatures = stdexec::completion_signatures<stdexec::set_value_t()>;
        using item_types = exec::item_types<>;

        template <__decays_to<__t> _Self, receiver_of<completion_signatures> _Rcvr>
        friend auto tag_invoke(subscribe_t, _Self&&, _Rcvr&& __rcvr) noexcept(
          __nothrow_decay_copyable<_Rcvr>) {
          return stdexec::__t<__operation<stdexec::__id<__decay_t<_Rcvr>>>>{
            static_cast<_Rcvr&&>(__rcvr)};
        }
      };
    };

    struct empty_sequence_t {
      __t<__sender> operator()() const noexcept {
        return {};
      }
    };

  } // namespace __empty_sequence

  using __empty_sequence::empty_sequence_t;
  inline constexpr empty_sequence_t empty_sequence{};

} // namespace exec
