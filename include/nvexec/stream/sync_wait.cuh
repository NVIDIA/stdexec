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

// clang-format Language: Cpp

#pragma once

#include "../../stdexec/execution.hpp"
#include <concepts>
#include <exception>
#include <new>
#include <optional>
#include <tuple> // IWYU pragma: keep
#include <utility>
#include <variant>

#include "common.cuh"

namespace nvexec::_strm {
  namespace _sync_wait {
    struct __env {
      using __t = __env;
      using __id = __env;

      run_loop::__scheduler __sched_;

      [[nodiscard]]
      auto query(get_scheduler_t) const noexcept -> run_loop::__scheduler {
        return __sched_;
      }

      [[nodiscard]]
      auto query(get_delegation_scheduler_t) const noexcept -> run_loop::__scheduler {
        return __sched_;
      }
    };

    // What should sync_wait(just_stopped()) return?
    template <class Sender>
      requires sender_in<Sender, __env>
    using sync_wait_result_t = value_types_of_t<Sender, __env, __decayed_std_tuple, __msingle>;

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
        void set_error_(Error err) noexcept {
          if constexpr (__decays_to<Error, cudaError_t>) {
            state_->data_.template emplace<2>(static_cast<Error&&>(err));
          } else {
            // What is `exception_ptr` but death pending
            state_->data_.template emplace<2>(cudaErrorUnknown);
          }
          loop_->finish();
        }

        template <class T>
        static void prefetch(T&& ref, cudaStream_t stream) {
          using decay_type = __decay_t<T>;
          decay_type* ptr = &ref;
          cudaPointerAttributes attributes{};
          if (cudaError_t err = cudaPointerGetAttributes(&attributes, ptr); err == cudaSuccess) {
            if (attributes.type == cudaMemoryTypeManaged) {
              [[maybe_unused]]
              auto _ign = STDEXEC_LOG_CUDA_API(
                cudaMemPrefetchAsync(ptr, sizeof(decay_type), cudaCpuDeviceId, stream));
            }
          }
        }

        template <class... As>
        void set_value(As&&... as) noexcept {
          static_assert(std::constructible_from<sync_wait_result_t<Sender>, As...>);
          STDEXEC_TRY {
            int dev_id{};
            cudaStream_t stream = state_->stream_;

            if constexpr (sizeof...(As)) {
              if (STDEXEC_LOG_CUDA_API(cudaGetDevice(&dev_id)) == cudaSuccess) {
                int concurrent_managed_access{};
                if (
                  STDEXEC_LOG_CUDA_API(cudaDeviceGetAttribute(
                    &concurrent_managed_access, cudaDevAttrConcurrentManagedAccess, dev_id))
                  == cudaSuccess) {
                  // Avoid launching the destruction kernel if the memory targeting host
                  (prefetch(static_cast<As&&>(as), stream), ...);
                }
              }
            }

            if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaStreamSynchronize(stream));
                status == cudaSuccess) {
              state_->data_.template emplace<1>(static_cast<As&&>(as)...);
            } else {
              set_error_(status);
            }
            loop_->finish();
          }
          STDEXEC_CATCH_ALL {
            set_error_(std::current_exception());
          }
        }

        template <class Error>
        void set_error(Error err) noexcept {
          if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaStreamSynchronize(state_->stream_));
              status == cudaSuccess) {
            set_error_(static_cast<Error&&>(err));
          } else {
            set_error_(status);
          }
        }

        void set_stopped() noexcept {
          if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaStreamSynchronize(state_->stream_));
              status == cudaSuccess) {
            state_->data_.template emplace<3>(set_stopped_t());
          } else {
            set_error_(status);
          }
          loop_->finish();
        }

        [[nodiscard]]
        auto get_env() const noexcept -> __env {
          return {loop_->get_scheduler()};
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
        requires sender_in<Sender, __env> && __receiver_from<receiver_t<Sender>, Sender>
      auto operator()(context_state_t context_state, Sender&& __sndr) const
        -> std::optional<sync_wait_result_t<Sender>> {
        using state_t = state_t<stdexec::__id<Sender>>;
        state_t state{};
        run_loop loop;

        cudaError_t status = cudaSuccess;
        auto __op_state = make_host<exit_operation_state_t<Sender, receiver_t<Sender>>>(
          status, context_state.pinned_resource_, __emplace_from{[&] {
            return exit_op_state(
              static_cast<Sender&&>(__sndr), receiver_t<Sender>{{}, &state, &loop}, context_state);
          }});
        if (status != cudaSuccess) {
          throw std::bad_alloc{};
        }

        state.stream_ = __op_state->get_stream();

        start(*__op_state);

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

  template <>
  struct apply_sender_for<sync_wait_t> {
    template <stream_completing_sender Sender>
    auto operator()(Sender&& sndr) const {
      auto sched = get_completion_scheduler<set_value_t>(get_env(sndr));
      return _sync_wait::sync_wait_t{}(sched.context_state_, static_cast<Sender&&>(sndr));
    }
  };
} // namespace nvexec::_strm
