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

#include "../stdexec/execution.hpp"

#include "stream_context.cuh"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

namespace nvexec {
  namespace _strm {
    struct multi_gpu_stream_scheduler : private stream_scheduler_env {
      using __t = multi_gpu_stream_scheduler;
      using __id = multi_gpu_stream_scheduler;

      multi_gpu_stream_scheduler(int num_devices, context_state_t context_state)
        : num_devices_(num_devices)
        , context_state_(context_state) {
      }

      auto operator==(const multi_gpu_stream_scheduler& other) const noexcept -> bool {
        return context_state_.hub_ == other.context_state_.hub_;
      }

      [[nodiscard]]
      STDEXEC_ATTRIBUTE(host, device) auto schedule() const noexcept {
        return sender_t{num_devices_, context_state_};
      }

      using stream_scheduler_env::query;

     private:
      template <class ReceiverId>
      struct operation_state_t : stream_op_state_base {
        using Receiver = stdexec::__t<ReceiverId>;

        explicit operation_state_t(Receiver rcvr)
          : rcvr_(static_cast<Receiver&&>(rcvr)) {
          status_ = STDEXEC_LOG_CUDA_API(cudaStreamCreate(&stream_));
        }

        ~operation_state_t() {
          STDEXEC_ASSERT_CUDA_API(cudaStreamDestroy(stream_));
        }

        [[nodiscard]]
        auto get_stream() -> cudaStream_t {
          return stream_;
        }

        void start() & noexcept {
          if constexpr (stream_receiver<Receiver>) {
            if (status_ == cudaSuccess) {
              stdexec::set_value(static_cast<Receiver&&>(rcvr_));
            } else {
              stdexec::set_error(static_cast<Receiver&&>(rcvr_), std::move(status_));
            }
          } else {
            if (status_ == cudaSuccess) {
              continuation_kernel<<<1, 1, 0, stream_>>>(std::move(rcvr_), stdexec::set_value);
            } else {
              continuation_kernel<<<1, 1, 0, stream_>>>(
                std::move(rcvr_), stdexec::set_error, std::move(status_));
            }
          }
        }

       private:
        friend stream_context;

        Receiver rcvr_;
        cudaStream_t stream_{};
        cudaError_t status_{cudaSuccess};
      };

      struct sender_t : stream_sender_base {
        using __t = sender_t;
        using __id = sender_t;

        using completion_signatures =
          stdexec::completion_signatures<set_value_t(), set_error_t(cudaError_t)>;

        STDEXEC_ATTRIBUTE(host, device)
        explicit sender_t(int num_devices, context_state_t context_state) noexcept
          : env_{.num_devices_ = num_devices, .context_state_ = context_state} {
        }

        template <class Receiver>
        [[nodiscard]]
        auto connect(Receiver rcvr) const & noexcept(__nothrow_move_constructible<Receiver>)
          -> operation_state_t<stdexec::__id<Receiver>> {
          return operation_state_t<stdexec::__id<Receiver>>(static_cast<Receiver&&>(rcvr));
        }

        [[nodiscard]]
        auto get_env() const noexcept -> decltype(auto) {
          return (env_);
        }

       private:
        struct env {
          using __t = env;
          using __id = env;

          int num_devices_;
          context_state_t context_state_;

          template <class CPO>
          [[nodiscard]]
          auto query(get_completion_scheduler_t<CPO>) const noexcept -> multi_gpu_stream_scheduler {
            return multi_gpu_stream_scheduler{num_devices_, context_state_};
          }
        };

        env env_;
      };

     public:
      // private: TODO
      int num_devices_{};
      context_state_t context_state_;
    };
  } // namespace _strm

  using _strm::multi_gpu_stream_scheduler;

  struct multi_gpu_stream_context {
    multi_gpu_stream_context()
      : dev_id_(_get_device())
      , hub_(dev_id_, pinned_resource_.get()) {
      // TODO Manage errors
      cudaGetDeviceCount(&num_devices_);

      for (int dev_id = 0; dev_id < num_devices_; dev_id++) {
        cudaSetDevice(dev_id);
        for (int peer_id = 0; peer_id < num_devices_; peer_id++) {
          if (peer_id != dev_id) {
            int can_access{};
            cudaDeviceCanAccessPeer(&can_access, dev_id, peer_id);

            if (can_access) {
              cudaDeviceEnablePeerAccess(peer_id, 0);
            }
          }
        }
      }
      cudaSetDevice(dev_id_);
    }

    [[nodiscard]]
    auto get_scheduler(stream_priority priority = stream_priority::normal)
      -> multi_gpu_stream_scheduler {
      return {
        num_devices_,
        _strm::context_state_t(
          pinned_resource_.get(), managed_resource_.get(), &stream_pools_, &hub_, priority)};
    }

   private:
    static auto _get_device() -> int {
      int dev_id{};
      cudaGetDevice(&dev_id);
      return dev_id;
    }

    int num_devices_{};

    _strm::resource_storage<_strm::pinned_resource> pinned_resource_{};
    _strm::resource_storage<_strm::managed_resource> managed_resource_{};
    _strm::stream_pools_t stream_pools_{};

    int dev_id_{};
    _strm::queue::task_hub_t hub_;
  };
} // namespace nvexec

STDEXEC_PRAGMA_POP()
