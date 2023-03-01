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

  namespace bulk {
    template <int BlockThreads, std::integral Shape, class Fun, class... As>
    __launch_bounds__(BlockThreads) __global__ void kernel(Shape shape, Fun fn, As... as) {
      const int tid = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);

      if (tid < static_cast<int>(shape)) {
        fn(tid, as...);
      }
    }

    template <class ReceiverId, std::integral Shape, class Fun>
    struct receiver_t {
      class __t : public stream_receiver_base {
        using Receiver = stdexec::__t<ReceiverId>;
        using Env = typename operation_state_base_t<ReceiverId>::env_t;

        Shape shape_;
        Fun f_;

        operation_state_base_t<ReceiverId>& op_state_;

       public:
        using __id = receiver_t;

        template <class... As>
        friend void tag_invoke(stdexec::set_value_t, __t&& self, As&&... as) noexcept
          requires stdexec::__callable<Fun, Shape, As...>
        {
          operation_state_base_t<ReceiverId>& op_state = self.op_state_;

          if (self.shape_) {
            cudaStream_t stream = op_state.get_stream();
            constexpr int block_threads = 256;
            const int grid_blocks = (static_cast<int>(self.shape_) + block_threads - 1)
                                  / block_threads;
            kernel<block_threads, Shape, Fun, As...>
              <<<grid_blocks, block_threads, 0, stream>>>(self.shape_, self.f_, (As&&) as...);
             STDEXEC_CHECK_CUDA_ERROR(cudaGetLastError());
          }

          if (cudaError_t status = STDEXEC_CHECK_CUDA_ERROR(cudaPeekAtLastError()); status == cudaSuccess) {
            op_state.propagate_completion_signal(stdexec::set_value, (As&&) as...);
          } else {
            op_state.propagate_completion_signal(stdexec::set_error, std::move(status));
          }
        }

        template <stdexec::__one_of<stdexec::set_error_t, stdexec::set_stopped_t> Tag, class... As>
        friend void tag_invoke(Tag tag, __t&& self, As&&... as) noexcept {
          self.op_state_.propagate_completion_signal(tag, (As&&) as...);
        }

        friend Env tag_invoke(stdexec::get_env_t, const __t& self) noexcept {
          return self.op_state_.make_env();
        }

        explicit __t(Shape shape, Fun fun, operation_state_base_t<ReceiverId>& op_state)
          : shape_(shape)
          , f_((Fun&&) fun)
          , op_state_(op_state) {
        }
      };
    };
  }

  template <class SenderId, std::integral Shape, class Fun>
  struct bulk_sender_t {
    using Sender = stdexec::__t<SenderId>;

    struct __t : stream_sender_base {
      using __id = bulk_sender_t;
      Sender sndr_;
      Shape shape_;
      Fun fun_;

      using set_error_t = stdexec::completion_signatures< stdexec::set_error_t(cudaError_t)>;

      template <class Receiver>
      using receiver_t = stdexec::__t<bulk::receiver_t<stdexec::__id<Receiver>, Shape, Fun>>;

      template <class... Tys>
      using set_value_t = stdexec::completion_signatures< stdexec::set_value_t(Tys...)>;

      template <class Self, class Env>
      using completion_signatures = //
        stdexec::__make_completion_signatures<
          stdexec::__copy_cvref_t<Self, Sender>,
          Env,
          set_error_t,
          stdexec::__q<set_value_t>>;

      template <stdexec::__decays_to<__t> Self, stdexec::receiver Receiver>
        requires stdexec::receiver_of<
          Receiver,
          completion_signatures<Self, stdexec::env_of_t<Receiver>>>
      friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr)
        -> stream_op_state_t<stdexec::__copy_cvref_t<Self, Sender>, receiver_t<Receiver>, Receiver> {
        return stream_op_state<stdexec::__copy_cvref_t<Self, Sender>>(
          ((Self&&) self).sndr_,
          (Receiver&&) rcvr,
          [&](operation_state_base_t<stdexec::__id<Receiver>>& stream_provider)
            -> receiver_t<Receiver> {
            return receiver_t<Receiver>(self.shape_, self.fun_, stream_provider);
          });
      }

      template <stdexec::__decays_to<__t> Self, class Env>
      friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
        -> stdexec::dependent_completion_signatures<Env>;

      template <stdexec::__decays_to<__t> Self, class Env>
      friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
        -> completion_signatures<Self, Env>
        requires true;

      friend auto tag_invoke(stdexec::get_env_t, const __t& self) //
        noexcept(stdexec::__nothrow_callable<stdexec::get_env_t, const Sender&>)
          -> stdexec::__call_result_t<stdexec::get_env_t, const Sender&> {
        return stdexec::get_env(self.sndr_);
      }
    };
  };

  namespace multi_gpu_bulk {
    template <int BlockThreads, std::integral Shape, class Fun, class... As>
    __launch_bounds__(BlockThreads) __global__
      void kernel(Shape begin, Shape end, Fun fn, As... as) {
      const Shape i = begin + static_cast<Shape>(threadIdx.x + blockIdx.x * blockDim.x);

      if (i < end) {
        fn(i, as...);
      }
    }

    template <class CvrefSenderId, class ReceiverId, class Shape, class Fun>
    struct operation_t;

    template <class CvrefSenderId, class ReceiverId, std::integral Shape, class Fun>
    struct receiver_t {
      using Receiver = stdexec::__t<ReceiverId>;

      class __t : public stream_receiver_base {
        Shape shape_;
        Fun f_;

        operation_t<CvrefSenderId, ReceiverId, Shape, Fun>& op_state_;

        static std::pair<Shape, Shape>
          even_share(Shape n, std::uint32_t rank, std::uint32_t size) noexcept {
          const auto avg_per_thread = n / size;
          const auto n_big_share = avg_per_thread + 1;
          const auto big_shares = n % size;
          const auto is_big_share = rank < big_shares;
          const auto begin = is_big_share
                             ? n_big_share * rank
                             : n_big_share * big_shares + (rank - big_shares) * avg_per_thread;
          const auto end = begin + (is_big_share ? n_big_share : avg_per_thread);

          return std::make_pair(begin, end);
        }

       public:
        using __id = receiver_t;

        template <class... As>
        friend void tag_invoke(stdexec::set_value_t, __t&& self, As&&... as) noexcept
          requires stdexec::__callable<Fun, Shape, As...>
        {
          operation_t<CvrefSenderId, ReceiverId, Shape, Fun>& op_state = self.op_state_;

          // TODO Manage errors
          // TODO Usual logic when there's only a single GPU
          cudaStream_t baseline_stream = op_state.get_stream();
          STDEXEC_CHECK_CUDA_ERROR(cudaEventRecord(op_state.ready_to_launch_, baseline_stream));

          if (self.shape_) {
            constexpr int block_threads = 256;
            for (int dev = 0; dev < op_state.num_devices_; dev++) {
              if (op_state.current_device_ != dev) {
                cudaStream_t stream = op_state.streams_[dev];
                auto [begin, end] = even_share(self.shape_, dev, op_state.num_devices_);
                auto shape = static_cast<int>(end - begin);
                const int grid_blocks = (shape + block_threads - 1) / block_threads;

                if (begin < end) {
                  STDEXEC_CHECK_CUDA_ERROR(cudaSetDevice(dev));
                  STDEXEC_CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream, op_state.ready_to_launch_));
                  kernel<block_threads, Shape, Fun, As...>
                    <<<grid_blocks, block_threads, 0, stream>>>(begin, end, self.f_, (As&&) as...);
                  STDEXEC_CHECK_CUDA_ERROR(cudaGetLastError());
                  STDEXEC_CHECK_CUDA_ERROR(cudaEventRecord(op_state.ready_to_complete_[dev], op_state.streams_[dev]));
                }
              }
            }

            {
              const int dev = op_state.current_device_;
              STDEXEC_CHECK_CUDA_ERROR(cudaSetDevice(dev));
              auto [begin, end] = even_share(self.shape_, dev, op_state.num_devices_);
              auto shape = static_cast<int>(end - begin);
              const int grid_blocks = (shape + block_threads - 1) / block_threads;

              if (begin < end) {
                kernel<block_threads, Shape, Fun, As...>
                  <<<grid_blocks, block_threads, 0, baseline_stream>>>(
                    begin, end, self.f_, (As&&) as...);
              }
            }

            for (int dev = 0; dev < op_state.num_devices_; dev++) {
              if (dev != op_state.current_device_) {
                STDEXEC_CHECK_CUDA_ERROR(cudaStreamWaitEvent(baseline_stream, op_state.ready_to_complete_[dev]));
              }
            }
          }

          if (cudaError_t status = STDEXEC_CHECK_CUDA_ERROR(cudaPeekAtLastError()); status == cudaSuccess) {
            op_state.propagate_completion_signal(stdexec::set_value, (As&&) as...);
          } else {
            op_state.propagate_completion_signal(stdexec::set_error, std::move(status));
          }
        }

        template <stdexec::__one_of<stdexec::set_error_t, stdexec::set_stopped_t> Tag, class... As>
        friend void tag_invoke(Tag tag, __t&& self, As&&... as) noexcept {
          self.op_state_.propagate_completion_signal(tag, (As&&) as...);
        }

        friend stdexec::env_of_t<Receiver> tag_invoke(stdexec::get_env_t, const __t& self) {
          return stdexec::get_env(self.op_state_.receiver_);
        }

        explicit __t(
          Shape shape,
          Fun fun,
          operation_t<CvrefSenderId, ReceiverId, Shape, Fun>& op_state)
          : shape_(shape)
          , f_((Fun&&) fun)
          , op_state_(op_state) {
        }
      };
    };

    template <class SenderId, class ReceiverId, class Shape, class Fun>
    using operation_base_t =
      operation_state_t< SenderId, receiver_t<SenderId, ReceiverId, Shape, Fun>, ReceiverId>;

    template <class CvrefSenderId, class ReceiverId, class Shape, class Fun>
    struct operation_t : operation_base_t<CvrefSenderId, ReceiverId, Shape, Fun> {
      using Sender = stdexec::__cvref_t<CvrefSenderId>;
      using Receiver = stdexec::__t<ReceiverId>;

      template <class _Receiver2>
      operation_t(
        int num_devices,
        Sender&& __sndr,
        _Receiver2&& __rcvr,
        Shape shape,
        Fun fun,
        context_state_t context_state)
        : operation_base_t<CvrefSenderId, ReceiverId, Shape, Fun>(
          (Sender&&) __sndr,
          (_Receiver2&&) __rcvr,
          [&](operation_state_base_t<stdexec::__id<_Receiver2>>&)
            -> stdexec::__t<receiver_t<CvrefSenderId, ReceiverId, Shape, Fun>> {
            return stdexec::__t<receiver_t<CvrefSenderId, ReceiverId, Shape, Fun>>(
              shape, fun, *this);
          },
          context_state)
        , num_devices_(num_devices)
        , streams_(new cudaStream_t[num_devices_])
        , ready_to_complete_(new cudaEvent_t[num_devices_]) {
        // TODO Manage errors
        STDEXEC_CHECK_CUDA_ERROR(cudaGetDevice(&current_device_));
        STDEXEC_CHECK_CUDA_ERROR(cudaEventCreate(&ready_to_launch_));
        for (int dev = 0; dev < num_devices_; dev++) {
          STDEXEC_CHECK_CUDA_ERROR(cudaSetDevice(dev));
          STDEXEC_CHECK_CUDA_ERROR(cudaStreamCreate(streams_.get() + dev));
          STDEXEC_CHECK_CUDA_ERROR(cudaEventCreate(ready_to_complete_.get() + dev));
        }
        STDEXEC_CHECK_CUDA_ERROR(cudaSetDevice(current_device_));
      }

      ~operation_t() {
        // TODO Manage errors
        for (int dev = 0; dev < num_devices_; dev++) {
          STDEXEC_CHECK_CUDA_ERROR(cudaSetDevice(dev));
          STDEXEC_CHECK_CUDA_ERROR(cudaStreamDestroy(streams_[dev]));
          STDEXEC_CHECK_CUDA_ERROR(cudaEventDestroy(ready_to_complete_[dev]));
        }
        STDEXEC_CHECK_CUDA_ERROR(cudaSetDevice(current_device_));
        STDEXEC_CHECK_CUDA_ERROR(cudaEventDestroy(ready_to_launch_));
      }

      STDEXEC_IMMOVABLE(operation_t);

      int num_devices_{};
      int current_device_{};
      std::unique_ptr<cudaStream_t[]> streams_;
      std::unique_ptr<cudaEvent_t[]> ready_to_complete_;
      cudaEvent_t ready_to_launch_;
    };
  }

  template <class SenderId, std::integral Shape, class Fun>
  struct multi_gpu_bulk_sender_t {
    using is_sender = void;
    using Sender = stdexec::__t<SenderId>;

    struct __t : stream_sender_base {
      using __id = multi_gpu_bulk_sender_t;
      int num_devices_;
      Sender sndr_;
      Shape shape_;
      Fun fun_;

      using set_error_t = stdexec::completion_signatures< stdexec::set_error_t(cudaError_t)>;

      template <class... Tys>
      using set_value_t = stdexec::completion_signatures< stdexec::set_value_t(Tys...)>;

      template <class Self, class Env>
      using completion_signatures = //
        stdexec::__make_completion_signatures<
          stdexec::__copy_cvref_t<Self, Sender>,
          Env,
          set_error_t,
          stdexec::__q<set_value_t>>;

      template <stdexec::__decays_to<__t> Self, stdexec::receiver Receiver>
        requires stdexec::
          receiver_of<Receiver, completion_signatures<Self, stdexec::env_of_t<Receiver>>>
        friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr) -> multi_gpu_bulk::
          operation_t<stdexec::__cvref_id<Self, Sender>, stdexec::__id<Receiver>, Shape, Fun> {
        auto sch = stdexec::get_completion_scheduler<stdexec::set_value_t>(
          stdexec::get_env(self.sndr_));
        context_state_t context_state = sch.context_state_;
        return multi_gpu_bulk::
          operation_t<stdexec::__cvref_id<Self, Sender>, stdexec::__id<Receiver>, Shape, Fun>(
            self.num_devices_,
            ((Self&&) self).sndr_,
            (Receiver&&) rcvr,
            self.shape_,
            self.fun_,
            context_state);
      }

      template <stdexec::__decays_to<__t> Self, class Env>
      friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
        -> stdexec::dependent_completion_signatures<Env>;

      template <stdexec::__decays_to<__t> Self, class Env>
      friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
        -> completion_signatures<Self, Env>
        requires true;

      friend auto tag_invoke(stdexec::get_env_t, const __t& self) //
        noexcept(stdexec::__nothrow_callable<stdexec::get_env_t, const Sender&>)
          -> stdexec::__call_result_t<stdexec::get_env_t, const Sender&> {
        return stdexec::get_env(self.sndr_);
      }
    };
  };
}
