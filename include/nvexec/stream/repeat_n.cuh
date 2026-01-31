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

#include "../../exec/repeat_n.hpp"
#include "../../stdexec/execution.hpp"

#include "algorithm_base.cuh" // IWYU pragma: keep

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(expr_has_no_effect)
STDEXEC_PRAGMA_IGNORE_GNU("-Wunused-value")

namespace nvexec::_strm {
  namespace repeat_n {
    template <class OpState>
    struct receiver : stream_receiver_base {
      receiver(OpState& opstate) noexcept
        : opstate_(opstate) {
      }

      void set_value() noexcept {
        opstate_._complete(STDEXEC::set_value);
      }

      template <class Error>
      void set_error(Error&& err) noexcept {
        opstate_._complete(STDEXEC::set_error, static_cast<Error&&>(err));
      }

      void set_stopped() noexcept {
        opstate_._complete(STDEXEC::set_stopped);
      }

      auto get_env() const noexcept -> OpState::env_t {
        return opstate_.make_env();
      }

      OpState& opstate_;
    };

    template <class CvSender, class Receiver>
    struct opstate : _strm::opstate_base<Receiver> {
      using operation_state_concept = STDEXEC::operation_state_t;

      using sender_t = STDEXEC::__decay_t<CvSender>;
      using scheduler_t = STDEXEC::__result_of<
        STDEXEC::get_completion_scheduler<STDEXEC::set_value_t>,
        STDEXEC::env_of_t<sender_t>,
        STDEXEC::env_of_t<Receiver>
      >;

      using inner_sender_t =
        STDEXEC::__result_of<exec::sequence, STDEXEC::schedule_result_t<scheduler_t&>, sender_t&>;
      using inner_opstate_t = STDEXEC::connect_result_t<inner_sender_t, receiver<opstate>>;

      explicit opstate(CvSender&& sndr, Receiver rcvr, std::size_t count, scheduler_t sched)
        : _strm::opstate_base<Receiver>(static_cast<Receiver&&>(rcvr), sched.ctx_)
        , sndr_(static_cast<CvSender&&>(sndr))
        , sched_(std::move(sched))
        , count_(count) {
        _connect();
      }

      void _connect() {
        inner_opstate_.__emplace_from(
          STDEXEC::connect, exec::sequence(STDEXEC::schedule(sched_), sndr_), receiver{*this});
      }

      template <class Tag, class... Args>
      void _complete(Tag, Args&&... args) noexcept {
        static_assert(sizeof...(Args) <= 1);
        static_assert(sizeof...(Args) == 0 || std::is_same_v<Tag, STDEXEC::set_error_t>);
        STDEXEC_ASSERT(count_ > 0);

        STDEXEC_TRY {
          auto arg_copy = (0, ..., static_cast<Args&&>(args)); // copy any arg...
          inner_opstate_.reset(); // ... because this could potentially invalidate it.

          if constexpr (__std::same_as<Tag, STDEXEC::set_value_t>) {
            if (--count_ == 0) {
              this->propagate_completion_signal(STDEXEC::set_value);
            } else {
              _connect();
              STDEXEC::start(*inner_opstate_);
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
            STDEXEC::set_error, cudaError_t(this->stream_provider_.status_));
        } else if (count_ == 0) {
          this->propagate_completion_signal(STDEXEC::set_value);
        } else {
          STDEXEC::start(*inner_opstate_);
        }
      }

      sender_t sndr_;
      scheduler_t sched_;
      STDEXEC::__optional<inner_opstate_t> inner_opstate_;
      std::size_t count_{};
    };

    template <class CvSender>
    struct sender {
      using sender_concept = STDEXEC::sender_t;

      using completion_signatures = STDEXEC::completion_signatures<
        STDEXEC::set_value_t(),
        STDEXEC::set_stopped_t(),
        STDEXEC::set_error_t(std::exception_ptr),
        STDEXEC::set_error_t(cudaError_t)
      >;

      template <STDEXEC::receiver Receiver>
      auto connect(Receiver rcvr) && //
        -> repeat_n::opstate<CvSender, Receiver> {
        auto sched = STDEXEC::get_completion_scheduler<STDEXEC::set_value_t>(
          STDEXEC::get_env(sndr_), STDEXEC::get_env(rcvr));
        return repeat_n::opstate<CvSender, Receiver>(
          static_cast<CvSender&&>(sndr_), static_cast<Receiver&&>(rcvr), count_, std::move(sched));
      }

      [[nodiscard]]
      auto get_env() const noexcept -> STDEXEC::env_of_t<CvSender> {
        return STDEXEC::get_env(sndr_);
      }

      CvSender sndr_; // could be a value or a reference
      std::size_t count_;
    };
  } // namespace repeat_n

  template <class Env>
  struct transform_sender_for<exec::repeat_n_t, Env> {
    template <class CvSender>
    auto operator()(STDEXEC::__ignore, size_t count, CvSender&& sndr) const {
      using sender_t = repeat_n::sender<CvSender>;
      return sender_t{static_cast<CvSender&&>(sndr), count};
    }

    const Env& env_;
  };
} // namespace nvexec::_strm

STDEXEC_PRAGMA_POP()
