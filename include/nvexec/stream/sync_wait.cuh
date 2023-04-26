/*
 * Copyright (c) 2022 NVIDIA Corporation
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
#include <type_traits>

#include "common.cuh"

namespace nvexec::STDEXEC_STREAM_DETAIL_NS { namespace _sync_wait {
  struct __env {
    run_loop::__scheduler __sched_;

    friend auto tag_invoke(get_scheduler_t, const __env& __self) noexcept -> run_loop::__scheduler {
      return __self.__sched_;
    }

    friend auto tag_invoke(get_delegatee_scheduler_t, const __env& __self) noexcept
      -> run_loop::__scheduler {
      return __self.__sched_;
    }
  };

  // What should sync_wait(just_stopped()) return?
  template <class Sender>
    requires sender_in<Sender, __env>
  using sync_wait_result_t = value_types_of_t< Sender, __env, __decayed_tuple, __msingle>;

  template <class SenderId>
  struct state_t;

  template <class SenderId>
  struct receiver_t {
    using Sender = stdexec::__t<SenderId>;

    struct __t : public stream_receiver_base {
      using __id = receiver_t;

      state_t<SenderId>* state_;
      run_loop* loop_;

      template <class Error>
      void set_error(Error err) noexcept {
        if constexpr (__decays_to<Error, cudaError_t>) {
          state_->data_.template emplace<2>((Error&&) err);
        } else {
          // What is `exception_ptr` but death pending
          state_->data_.template emplace<2>(cudaErrorUnknown);
        }
        loop_->finish();
      }

      template <same_as<set_value_t> _Tag, class Sender2 = Sender, class... As>
        requires std::constructible_from<sync_wait_result_t<Sender2>, As...>
      friend void tag_invoke(_Tag, __t&& rcvr, As&&... as) noexcept {
        try {
          if (cudaError_t status = STDEXEC_DBG_ERR(cudaStreamSynchronize(rcvr.state_->stream_));
              status == cudaSuccess) {
            rcvr.state_->data_.template emplace<1>((As&&) as...);
          } else {
            rcvr.set_error(status);
          }
          rcvr.loop_->finish();
        } catch (...) {
          rcvr.set_error(std::current_exception());
        }
      }

      template <same_as<set_error_t> _Tag, class Error>
      friend void tag_invoke(_Tag, __t&& rcvr, Error err) noexcept {
        if (cudaError_t status = STDEXEC_DBG_ERR(cudaStreamSynchronize(rcvr.state_->stream_));
            status == cudaSuccess) {
          rcvr.set_error((Error&&) err);
        } else {
          rcvr.set_error(status);
        }
      }

      friend void tag_invoke(set_stopped_t __d, __t&& rcvr) noexcept {
        if (cudaError_t status = STDEXEC_DBG_ERR(cudaStreamSynchronize(rcvr.state_->stream_));
            status == cudaSuccess) {
          rcvr.state_->data_.template emplace<3>(__d);
        } else {
          rcvr.set_error(status);
        }
        rcvr.loop_->finish();
      }

      friend empty_env tag_invoke(get_env_t, const __t& rcvr) noexcept {
        return {};
      }
    };
  };

  template <class SenderId>
  struct state_t {
    using _Tuple = sync_wait_result_t<stdexec::__t<SenderId>>;

    cudaStream_t stream_{};
    std::variant<std::monostate, _Tuple, cudaError_t, set_stopped_t> data_{};
  };

  struct sync_wait_t {
    template <class Sender>
    using receiver_t = stdexec::__t<receiver_t<stdexec::__id<Sender>>>;

    template <__single_value_variant_sender<__env> Sender>
      requires(!__tag_invocable_with_completion_scheduler< sync_wait_t, set_value_t, Sender>)
           && (!tag_invocable<sync_wait_t, Sender>) && sender<Sender, __env>
           && __receiver_from<receiver_t<Sender>, Sender>
    auto operator()(context_state_t context_state, Sender&& __sndr) const
      -> std::optional<sync_wait_result_t<Sender>> {
      using state_t = state_t<stdexec::__id<Sender>>;
      state_t state{};
      run_loop loop;

      exit_operation_state_t<Sender, receiver_t<Sender>> __op_state = exit_op_state(
        (Sender&&) __sndr, receiver_t<Sender>{{}, &state, &loop}, context_state);
      state.stream_ = __op_state.get_stream();

      start(__op_state);

      // Wait for the variant to be filled in.
      loop.run();

      if (state.data_.index() == 2)
        std::rethrow_exception(std::make_exception_ptr(std::get<2>(state.data_)));

      if (state.data_.index() == 3)
        return std::nullopt;

      return std::move(std::get<1>(state.data_));
    }
  };
} // namespace _sync_wait
} // namespace nvexec::STDEXEC_STREAM_DETAIL_NS
