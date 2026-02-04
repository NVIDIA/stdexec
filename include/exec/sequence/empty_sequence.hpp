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

#include "../../stdexec/execution.hpp"
#include "../sequence_senders.hpp"

namespace exec {
  namespace __empty_sequence {

    using namespace STDEXEC;

    template <class _Receiver>
    struct __operation {
      void start() & noexcept {
        STDEXEC::set_value(static_cast<_Receiver&&>(__rcvr_));
      }

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Receiver __rcvr_;
    };

    struct __sender {
      using sender_concept = sequence_sender_t;
      using completion_signatures = STDEXEC::completion_signatures<STDEXEC::set_value_t()>;
      using item_types = exec::item_types<>;

      template <receiver_of<completion_signatures> _Rcvr>
      auto subscribe(_Rcvr __rcvr) const noexcept {
        return __operation<_Rcvr>{static_cast<_Rcvr&&>(__rcvr)};
      }
    };

    struct empty_sequence_t {
      auto operator()() const noexcept -> __sender {
        return {};
      }
    };

  } // namespace __empty_sequence

  using __empty_sequence::empty_sequence_t;
  inline constexpr empty_sequence_t empty_sequence{};

} // namespace exec
