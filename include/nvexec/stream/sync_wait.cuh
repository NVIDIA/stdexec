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
    using __t = __env;
    using __id = __env;

    run_loop::__scheduler __sched_;

    auto query(get_scheduler_t) const noexcept -> run_loop::__scheduler {
      return __sched_;
    }

    auto query(get_delegatee_scheduler_t) const noexcept -> run_loop::__scheduler {
      return __sched_;
    }
  };

  // What should sync_wait(just_stopped()) return?
  template <class Sender>
    requires sender_in<Sender, __env>
  using sync_wait_result_t = value_types_of_t<Sender, __env, __decayed_tuple, __msingle>;

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
            STDEXEC_DBG_ERR(cudaMemPrefetchAsync(ptr, sizeof(decay_type), cudaCpuDeviceId, stream));
          }
        }
      }

      template <class Sender2 = Sender, class... As>
        requires std::constructible_from<sync_wait_result_t<Sender2>, As...>
      STDEXEC_MEMFN_DECL(
        void set_value)(this __t&& rcvr, As&&... as) noexcept {
        try {
          int dev_id{};
          cudaStream_t stream = rcvr.state_->stream_;

          if constexpr (sizeof...(As)) {
            if (STDEXEC_DBG_ERR(cudaGetDevice(&dev_id)) == cudaSuccess) {
              int concurrent_managed_access{};
              if (
                STDEXEC_DBG_ERR(cudaDeviceGetAttribute(
                  &concurrent_managed_access, cudaDevAttrConcurrentManagedAccess, dev_id))
                == cudaSuccess) {
                // Avoid launching the destruction kernel if the memory targeting host
                (prefetch(static_cast<As&&>(as), stream), ...);
              }
            }
          }

          if (cudaError_t status = STDEXEC_DBG_ERR(cudaStreamSynchronize(stream));
              status == cudaSuccess) {
            rcvr.state_->data_.template emplace<1>(static_cast<As&&>(as)...);
          } else {
            rcvr.set_error(status);
          }
          rcvr.loop_->finish();
        } catch (...) {
          rcvr.set_error(std::current_exception());
        }
      }

      template <class Error>
      STDEXEC_MEMFN_DECL(void set_error)(this __t&& rcvr, Error err) noexcept {
        if (cudaError_t status = STDEXEC_DBG_ERR(cudaStreamSynchronize(rcvr.state_->stream_));
            status == cudaSuccess) {
          rcvr.set_error(static_cast<Error&&>(err));
        } else {
          rcvr.set_error(status);
        }
      }

      STDEXEC_MEMFN_DECL(void set_stopped)(this __t&& rcvr) noexcept {
        if (cudaError_t status = STDEXEC_DBG_ERR(cudaStreamSynchronize(rcvr.state_->stream_));
            status == cudaSuccess) {
          rcvr.state_->data_.template emplace<3>(set_stopped_t());
        } else {
          rcvr.set_error(status);
        }
        rcvr.loop_->finish();
      }

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
        status, context_state.pinned_resource_, __conv{[&] {
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

#if STDEXEC_NVHPC()
    // For reporting better diagnostics with nvc++
    template <class _Sender, class _Error = stdexec::__sync_wait::__error_description_t<_Sender>>
    auto operator()(
      context_state_t context_state,
      _Sender&&,
      [[maybe_unused]] _Error __diagnostic = {}) const -> std::optional<std::tuple<int>> = delete;
#endif
  };
}} // namespace nvexec::STDEXEC_STREAM_DETAIL_NS::_sync_wait
