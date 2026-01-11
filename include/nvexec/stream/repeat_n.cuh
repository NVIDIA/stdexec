/*
 * Copyright (c) 2025 NVIDIA Corporation
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

// clang-format Language: Cpp

#pragma once

#include "../../stdexec/execution.hpp"
#include "../../exec/repeat_n.hpp"

#include "algorithm_base.cuh" // IWYU pragma: keep

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(expr_has_no_effect)
STDEXEC_PRAGMA_IGNORE_GNU("-Wunused-value")

namespace nvexec::_strm {
  namespace repeat_n {
    template <class OpState>
    struct receiver_t : stream_receiver_base {
      receiver_t(OpState& opstate) noexcept
        : opstate_(opstate) {
      }

      void set_value() noexcept {
        opstate_._complete(stdexec::set_value);
      }

      template <class Error>
      void set_error(Error&& err) noexcept {
        opstate_._complete(stdexec::set_error, static_cast<Error&&>(err));
      }

      void set_stopped() noexcept {
        opstate_._complete(stdexec::set_stopped);
      }

      auto get_env() const noexcept -> OpState::env_t {
        return opstate_.make_env();
      }

      OpState& opstate_;
    };

    template <class CvrefSenderId, class ReceiverId>
    struct operation_state_t : operation_state_base_t<ReceiverId> {
      using operation_state_concept = stdexec::operation_state_t;
      using __t = operation_state_t;
      using __id = operation_state_t;

      using CvrefSender = stdexec::__cvref_t<CvrefSenderId>;
      using Sender = stdexec::__decay_t<CvrefSender>;
      using Receiver = stdexec::__t<ReceiverId>;
      using Scheduler = stdexec::__result_of<
        stdexec::get_completion_scheduler<stdexec::set_value_t>,
        stdexec::env_of_t<Sender>,
        stdexec::env_of_t<Receiver>
      >;

      using inner_sender_t =
        stdexec::__result_of<exec::sequence, stdexec::schedule_result_t<Scheduler&>, Sender&>;
      using inner_opstate_t =
        stdexec::connect_result_t<inner_sender_t, receiver_t<operation_state_t>>;

      explicit operation_state_t(
        CvrefSender&& sndr,
        Receiver&& rcvr,
        std::size_t count,
        Scheduler sched)
        : operation_state_base_t<ReceiverId>(static_cast<Receiver&&>(rcvr), sched.context_state_)
        , sndr_(static_cast<CvrefSender&&>(sndr))
        , sched_(std::move(sched))
        , count_(count) {
        _connect();
      }

      void _connect() {
        inner_opstate_.__emplace_from(
          stdexec::connect, exec::sequence(stdexec::schedule(sched_), sndr_), receiver_t{*this});
      }

      template <class Tag, class... Args>
      void _complete(Tag, Args&&... args) noexcept {
        static_assert(sizeof...(Args) <= 1);
        static_assert(sizeof...(Args) == 0 || std::is_same_v<Tag, stdexec::set_error_t>);
        STDEXEC_ASSERT(count_ > 0);

        STDEXEC_TRY {
          auto arg_copy = (0, ..., static_cast<Args&&>(args)); // copy any arg...
          inner_opstate_.reset(); // ... because this could potentially invalidate it.

          if constexpr (same_as<Tag, stdexec::set_value_t>) {
            if (--count_ == 0) {
              this->propagate_completion_signal(stdexec::set_value);
            } else {
              _connect();
              stdexec::start(*inner_opstate_);
            }
          } else {
            this->propagate_completion_signal(Tag{}, static_cast<__decay_t<Args>&&>(arg_copy)...);
          }
        }
        STDEXEC_CATCH_ALL {
          this->propagate_completion_signal(Tag{}, std::current_exception());
        }
      }

      void start() noexcept {
        if (this->stream_provider_.status_ != cudaSuccess) {
          // Couldn't allocate memory for operation state, complete with error
          this->propagate_completion_signal(
            stdexec::set_error, cudaError_t(this->stream_provider_.status_));
        } else if (count_ == 0) {
          this->propagate_completion_signal(stdexec::set_value);
        } else {
          stdexec::start(*inner_opstate_);
        }
      }

      Sender sndr_;
      Scheduler sched_;
      stdexec::__optional<inner_opstate_t> inner_opstate_;
      std::size_t count_{};
    };

    template <class CvrefSenderId>
    struct sender_t {
      using sender_concept = stdexec::sender_t;
      using __t = sender_t;
      using __id = sender_t;
      using CvrefSender = stdexec::__cvref_t<CvrefSenderId>;

      using completion_signatures = stdexec::completion_signatures<
        stdexec::set_value_t(),
        stdexec::set_stopped_t(),
        stdexec::set_error_t(std::exception_ptr),
        stdexec::set_error_t(cudaError_t)
      >;

      template <stdexec::receiver Receiver>
      auto connect(Receiver rcvr) && //
        -> repeat_n::operation_state_t<CvrefSenderId, stdexec::__id<Receiver>> {
        auto sched = stdexec::get_completion_scheduler<stdexec::set_value_t>(
          stdexec::get_env(sndr_), stdexec::get_env(rcvr));
        return repeat_n::operation_state_t<CvrefSenderId, stdexec::__id<Receiver>>(
          static_cast<CvrefSender&&>(sndr_),
          static_cast<Receiver&&>(rcvr),
          count_,
          std::move(sched));
      }

      [[nodiscard]]
      auto get_env() const noexcept -> stdexec::env_of_t<CvrefSender> {
        return stdexec::get_env(sndr_);
      }

      CvrefSender sndr_; // could be a value or a reference
      std::size_t count_;
    };
  } // namespace repeat_n

  template <class Env>
  struct transform_sender_for<exec::repeat_n_t, Env> {
    template <class CvrefSender>
    auto operator()(stdexec::__ignore, size_t count, CvrefSender&& sndr) const {
      static_assert(sizeof(sndr) == 0);
      using sender_t = repeat_n::sender_t<stdexec::__cvref_id<CvrefSender>>;
      return sender_t{static_cast<CvrefSender&&>(sndr), count};
    }

    const Env& env_;
  };
} // namespace nvexec::_strm

STDEXEC_PRAGMA_POP()
