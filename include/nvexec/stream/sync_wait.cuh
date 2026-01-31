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
#include "common.cuh"

#include <concepts>
#include <exception>
#include <new>
#include <optional>
#include <tuple> // IWYU pragma: keep
#include <utility>
#include <variant>

namespace nvexec::_strm {
  namespace _sync_wait {
    struct env {
      [[nodiscard]]
      constexpr auto query(get_scheduler_t) const noexcept -> run_loop::scheduler {
        return __sched_;
      }

      [[nodiscard]]
      constexpr auto query(get_delegation_scheduler_t) const noexcept -> run_loop::scheduler {
        return __sched_;
      }

      run_loop::scheduler __sched_;
    };

    // What should sync_wait(just_stopped()) return?
    template <class Sender>
      requires sender_in<Sender, env>
    using sync_wait_result_t = value_types_of_t<Sender, env, __decayed_std_tuple, __msingle>;

    template <class Sender>
    struct state;

    template <class Sender>
    struct receiver : public stream_receiver_base {
      template <class Error>
      void _set_error(Error err) noexcept {
        if constexpr (__decays_to<Error, cudaError_t>) {
          state_->data_.template emplace<2>(static_cast<Error&&>(err));
        } else {
          // What is `exception_ptr` but death pending
          state_->data_.template emplace<2>(cudaErrorUnknown);
        }
        loop_->finish();
      }

      template <class Type>
      static void prefetch(Type&& ref, cudaStream_t stream) {
        using type_t = __decay_t<Type>;
        type_t* ptr = &ref;
        cudaPointerAttributes attributes{};
        if (cudaError_t err = cudaPointerGetAttributes(&attributes, ptr); err == cudaSuccess) {
          if (attributes.type == cudaMemoryTypeManaged) {
            [[maybe_unused]]
            auto _ign = STDEXEC_LOG_CUDA_API(
              cudaMemPrefetchAsync(ptr, sizeof(type_t), cudaCpuDeviceId, stream));
          }
        }
      }

      template <class... Args>
      void set_value(Args&&... args) noexcept {
        static_assert(std::constructible_from<sync_wait_result_t<Sender>, Args...>);
        STDEXEC_TRY {
          int dev_id{};
          cudaStream_t stream = state_->stream_;

          if constexpr (sizeof...(Args)) {
            if (STDEXEC_LOG_CUDA_API(cudaGetDevice(&dev_id)) == cudaSuccess) {
              int concurrent_managed_access{};
              if (
                STDEXEC_LOG_CUDA_API(cudaDeviceGetAttribute(
                  &concurrent_managed_access, cudaDevAttrConcurrentManagedAccess, dev_id))
                == cudaSuccess) {
                // Avoid launching the destruction kernel if the memory targeting host
                (prefetch(static_cast<Args&&>(args), stream), ...);
              }
            }
          }

          if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaStreamSynchronize(stream));
              status == cudaSuccess) {
            state_->data_.template emplace<1>(static_cast<Args&&>(args)...);
          } else {
            _set_error(status);
          }
          loop_->finish();
        }
        STDEXEC_CATCH_ALL {
          _set_error(std::current_exception());
        }
      }

      template <class Error>
      void set_error(Error err) noexcept {
        if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaStreamSynchronize(state_->stream_));
            status == cudaSuccess) {
          _set_error(static_cast<Error&&>(err));
        } else {
          _set_error(status);
        }
      }

      void set_stopped() noexcept {
        if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaStreamSynchronize(state_->stream_));
            status == cudaSuccess) {
          state_->data_.template emplace<3>(set_stopped_t());
        } else {
          _set_error(status);
        }
        loop_->finish();
      }

      [[nodiscard]]
      auto get_env() const noexcept -> env {
        return {loop_->get_scheduler()};
      }

      state<Sender>* state_;
      run_loop* loop_;
    };

    template <class Sender>
    struct state {
      using tuple_t = sync_wait_result_t<Sender>;

      cudaStream_t stream_{};
      std::variant<std::monostate, tuple_t, cudaError_t, set_stopped_t> data_{};
    };

    struct sync_wait_t {
      template <sender_in<env> Sender>
        requires __single_value_variant_sender<Sender, env>
      auto operator()(context ctx, Sender&& sndr) const //
        -> std::optional<sync_wait_result_t<Sender>> {
        using state_t = _sync_wait::state<Sender>;
        state_t state{};
        run_loop loop;

        cudaError_t status = cudaSuccess;
        auto opstate = host_allocate<exit_opstate_t<Sender, receiver<Sender>>>(
          status, ctx.pinned_resource_, __emplace_from{[&] {
            return exit_opstate(
              static_cast<Sender&&>(sndr), receiver<Sender>{{}, &state, &loop}, ctx);
          }});
        if (status != cudaSuccess) {
          throw std::bad_alloc{};
        }

        state.stream_ = opstate->get_stream();

        STDEXEC::start(*opstate);

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
    template <stream_completing_sender<STDEXEC::env<>> Sender>
    auto operator()(Sender&& sndr) const {
      auto sched = get_completion_scheduler<set_value_t>(get_env(sndr), STDEXEC::env{});
      return _sync_wait::sync_wait_t{}(sched.ctx_, static_cast<Sender&&>(sndr));
    }
  };
} // namespace nvexec::_strm
