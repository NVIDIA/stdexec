/*
 * Copyright (c) 2026 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions
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

#include "stdexec/execution.hpp"

namespace repeat_receiver_lifetime_test
{
  namespace ex = STDEXEC;

  template <class Completion>
  struct invalidate_on_destroy_sender
  {
    using sender_concept        = ex::sender_tag;
    using completion_signatures = ex::completion_signatures<typename Completion::signature>;

    template <class Receiver>
    struct operation
    {
      operation(Receiver rcvr, Completion completion, bool *invalidated) noexcept
        : rcvr_(static_cast<Receiver &&>(rcvr))
        , completion_(static_cast<Completion &&>(completion))
        , invalidated_(invalidated)
      {}

      ~operation()
      {
        if constexpr (requires { rcvr_.__self_->__rcvr_.__state_; })
        {
          if (started_)
          {
            rcvr_.__self_->__rcvr_.__state_ = nullptr;
            *invalidated_                   = true;
          }
        }
      }

      void start() & noexcept
      {
        started_ = true;
        completion_(static_cast<Receiver &&>(rcvr_));
      }

      Receiver   rcvr_;
      Completion completion_;
      bool      *invalidated_;
      bool       started_ = false;
    };

    template <ex::receiver_of<completion_signatures> Receiver>
    auto connect(Receiver rcvr) const -> operation<Receiver>
    {
      return {static_cast<Receiver &&>(rcvr), completion_, invalidated_};
    }

    Completion completion_;
    bool      *invalidated_;
  };

  template <class Completion>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
  invalidate_on_destroy_sender(Completion, bool *) -> invalidate_on_destroy_sender<Completion>;
}  // namespace repeat_receiver_lifetime_test
