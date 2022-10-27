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

namespace nvexec::STDEXEC_STREAM_DETAIL_NS {
  namespace sync_wait {
    namespace __impl {
      struct __env {
        stdexec::run_loop::__scheduler __sched_;

        friend auto tag_invoke(stdexec::get_scheduler_t, const __env& __self) noexcept
          -> stdexec::run_loop::__scheduler {
          return __self.__sched_;
        }

        friend auto tag_invoke(stdexec::get_delegatee_scheduler_t, const __env& __self) noexcept
          -> stdexec::run_loop::__scheduler {
          return __self.__sched_;
        }
      };

      // What should sync_wait(just_stopped()) return?
      template <class Sender>
          requires stdexec::sender<Sender, __env>
        using sync_wait_result_t =
          stdexec::value_types_of_t<
            Sender,
            __env,
            stdexec::__decayed_tuple,
            stdexec::__single>;

      template <class SenderId>
        struct state_t;

      template <class SenderId>
        struct receiver_t : public stream_receiver_base {
          using Sender = stdexec::__t<SenderId>;

          state_t<SenderId>* state_;
          stdexec::run_loop* loop_;

          template <class Error>
            void set_error(Error err) noexcept {
              if constexpr (stdexec::__decays_to<Error, cudaError_t>) {
                state_->data_.template emplace<2>((Error&&) err);
              } else {
                // What is `exception_ptr` but death pending
                state_->data_.template emplace<2>(cudaErrorUnknown);
              }
              loop_->finish();
            }

          template <class Sender2 = Sender, class... As _NVCXX_CAPTURE_PACK(As)>
              requires std::constructible_from<sync_wait_result_t<Sender2>, As...>
            friend void tag_invoke(stdexec::set_value_t, receiver_t&& rcvr, As&&... as) noexcept try {
              if (cudaError_t status = STDEXEC_DBG_ERR(cudaStreamSynchronize(rcvr.state_->stream_));
                  status == cudaSuccess) {
                _NVCXX_EXPAND_PACK(As, as,
                  rcvr.state_->data_.template emplace<1>((As&&) as...);
                )
              } else {
                rcvr.set_error(status);
              }
              rcvr.loop_->finish();
            } catch(...) {
              rcvr.set_error(std::current_exception());
            }

          template <class Error>
            friend void tag_invoke(stdexec::set_error_t, receiver_t&& rcvr, Error err) noexcept {
              if (cudaError_t status = STDEXEC_DBG_ERR(cudaStreamSynchronize(rcvr.state_->stream_));
                  status == cudaSuccess) {
                rcvr.set_error((Error &&) err);
              } else {
                rcvr.set_error(status);
              }
            }

          friend void tag_invoke(stdexec::set_stopped_t __d, receiver_t&& rcvr) noexcept {
            if (cudaError_t status = STDEXEC_DBG_ERR(cudaStreamSynchronize(rcvr.state_->stream_));
                status == cudaSuccess) {
              rcvr.state_->data_.template emplace<3>(__d);
            } else {
              rcvr.set_error(status);
            }
            rcvr.loop_->finish();
          }

          friend stdexec::__empty_env
          tag_invoke(stdexec::get_env_t, const receiver_t& rcvr) noexcept {
            return {};
          }
        };

      template <class SenderId>
        struct state_t {
          using _Tuple = sync_wait_result_t<stdexec::__t<SenderId>>;

          cudaStream_t stream_{};
          std::variant<std::monostate, _Tuple, cudaError_t, stdexec::set_stopped_t> data_{};
        };
    } // namespace __impl

    struct sync_wait_t {
      template <stdexec::__single_value_variant_sender<__impl::__env> Sender>
        requires
          (!stdexec::__tag_invocable_with_completion_scheduler<
            sync_wait_t, stdexec::set_value_t, Sender>) &&
          (!std::tag_invocable<sync_wait_t, Sender>) &&
          stdexec::sender<Sender, __impl::__env> &&
          stdexec::sender_to<Sender, __impl::receiver_t<stdexec::__x<Sender>>>
      auto operator()(queue::task_hub_t* hub, Sender&& __sndr) const
        -> std::optional<__impl::sync_wait_result_t<Sender>> {
        using state_t = __impl::state_t<stdexec::__x<Sender>>;
        state_t state {};
        stdexec::run_loop loop;

        exit_operation_state_t<Sender, __impl::receiver_t<stdexec::__x<Sender>>> __op_state =
          exit_op_state(hub, (Sender&&)__sndr, __impl::receiver_t<stdexec::__x<Sender>>{{}, &state, &loop});
        state.stream_ = __op_state.get_stream();

        stdexec::start(__op_state);

        // Wait for the variant to be filled in.
        loop.run();

        if (state.data_.index() == 2)
          std::rethrow_exception(std::make_exception_ptr(std::get<2>(state.data_)));

        if (state.data_.index() == 3)
          return std::nullopt;

        return std::move(std::get<1>(state.data_));
      }
    };
  } // namespace stream_sync_wait
}

