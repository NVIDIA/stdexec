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

#include "../stdexec/execution.hpp"
#include <type_traits>

#include "stream_context.cuh"

namespace nvexec {
  namespace STDEXEC_STREAM_DETAIL_NS {
    template <stdexec::sender Sender, std::integral Shape, class Fun>
    using multi_gpu_bulk_sender_th =
      stdexec::__t<multi_gpu_bulk_sender_t<stdexec::__id<std::decay_t<Sender>>, Shape, Fun>>;

    struct multi_gpu_stream_scheduler {
      friend stream_context;

      template <stdexec::sender Sender>
      using schedule_from_sender_th =
        stdexec::__t<schedule_from_sender_t<stream_scheduler, stdexec::__id<std::decay_t<Sender>>>>;

      template <class RId>
      struct operation_state_t : stream_op_state_base {
        using R = stdexec::__t<RId>;

        R rec_;
        cudaStream_t stream_{0};
        cudaError_t status_{cudaSuccess};

        template <stdexec::__decays_to<R> Receiver>
        operation_state_t(Receiver&& rec)
          : rec_((Receiver&&) rec) {
          status_ = STDEXEC_CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
        }

        ~operation_state_t() {
          STDEXEC_CHECK_CUDA_ERROR(cudaStreamDestroy(stream_));
        }

        cudaStream_t get_stream() {
          return stream_;
        }

        friend void tag_invoke(stdexec::start_t, operation_state_t& op) noexcept {
          if constexpr (stream_receiver<R>) {
            if (op.status_ == cudaSuccess) {
              stdexec::set_value((R&&) op.rec_);
            } else {
              stdexec::set_error((R&&) op.rec_, std::move(op.status_));
            }
          } else {
            if (op.status_ == cudaSuccess) {
              continuation_kernel<std::decay_t<R>, stdexec::set_value_t>
                <<<1, 1, 0, op.stream_>>>(op.rec_, stdexec::set_value);
              STDEXEC_CHECK_CUDA_ERROR(cudaGetLastError());
            } else {
              continuation_kernel<std::decay_t<R>, stdexec::set_error_t, cudaError_t>
                <<<1, 1, 0, op.stream_>>>(op.rec_, stdexec::set_error, op.status_);
              STDEXEC_CHECK_CUDA_ERROR(cudaGetLastError());
            }
          }
        }
      };

      struct sender_t : stream_sender_base {

        struct env {
          int num_devices_;
          context_state_t context_state_;

          template <class CPO>
          friend multi_gpu_stream_scheduler
            tag_invoke(stdexec::get_completion_scheduler_t<CPO>, const env& self) noexcept {
            return self.make_scheduler();
          }

          multi_gpu_stream_scheduler make_scheduler() const {
            return multi_gpu_stream_scheduler{num_devices_, context_state_};
          }
        };

        using completion_signatures =
          stdexec::completion_signatures< stdexec::set_value_t(), stdexec::set_error_t(cudaError_t)>;

        template <class R>
        friend auto tag_invoke(stdexec::connect_t, sender_t, R&& rec) //
          noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<R>, R>)
            -> operation_state_t<stdexec::__id<std::remove_cvref_t<R>>> {
          return operation_state_t<stdexec::__id<std::remove_cvref_t<R>>>((R&&) rec);
        }

        friend const env& tag_invoke(stdexec::get_env_t, const sender_t& self) noexcept {
          return self.env_;
        }

        sender_t(int num_devices, context_state_t context_state) noexcept
          : env_{num_devices, context_state} {
        }

        env env_;
      };

      template <stdexec::sender S>
      friend schedule_from_sender_th<S>
        tag_invoke(stdexec::schedule_from_t, const multi_gpu_stream_scheduler& sch, S&& sndr) //
        noexcept {
        return schedule_from_sender_th<S>(sch.context_state_, (S&&) sndr);
      }

      template <stdexec::sender S, std::integral Shape, class Fn>
      friend multi_gpu_bulk_sender_th<S, Shape, Fn> tag_invoke( //
        stdexec::bulk_t,                                        //
        const multi_gpu_stream_scheduler& sch,                  //
        S&& sndr,                                               //
        Shape shape,                                            //
        Fn fun)                                                 //
        noexcept {
        return multi_gpu_bulk_sender_th<S, Shape, Fn>{
          {}, sch.num_devices_, (S&&) sndr, shape, (Fn&&) fun};
      }

      template <stdexec::sender S, class Fn>
      friend then_sender_th<S, Fn>
        tag_invoke(stdexec::then_t, const multi_gpu_stream_scheduler& sch, S&& sndr, Fn fun) //
        noexcept {
        return then_sender_th<S, Fn>{{}, (S&&) sndr, (Fn&&) fun};
      }

      template <
        stdexec::__one_of<stdexec::let_value_t, stdexec::let_stopped_t, stdexec::let_error_t> Let,
        stdexec::sender S,
        class Fn>
      friend let_xxx_th<Let, S, Fn>
        tag_invoke(Let, const multi_gpu_stream_scheduler& sch, S&& sndr, Fn fun) noexcept {
        return let_xxx_th<Let, S, Fn>{{}, (S&&) sndr, (Fn&&) fun};
      }

      template <stdexec::sender S, class Fn>
      friend upon_error_sender_th<S, Fn> tag_invoke(
        stdexec::upon_error_t,
        const multi_gpu_stream_scheduler& sch,
        S&& sndr,
        Fn fun) noexcept {
        return upon_error_sender_th<S, Fn>{{}, (S&&) sndr, (Fn&&) fun};
      }

      template <stdexec::sender S, class Fn>
      friend upon_stopped_sender_th<S, Fn> tag_invoke(
        stdexec::upon_stopped_t,
        const multi_gpu_stream_scheduler& sch,
        S&& sndr,
        Fn fun) noexcept {
        return upon_stopped_sender_th<S, Fn>{{}, (S&&) sndr, (Fn&&) fun};
      }

      template <stream_completing_sender... Senders>
      friend auto tag_invoke(                  //
        stdexec::transfer_when_all_t,
        const multi_gpu_stream_scheduler& sch, //
        Senders&&... sndrs) noexcept {
        return transfer_when_all_sender_th<multi_gpu_stream_scheduler, Senders...>(
          sch.context_state_, (Senders&&) sndrs...);
      }

      template <stream_completing_sender... Senders>
      friend auto tag_invoke(                      //
        stdexec::transfer_when_all_with_variant_t, //
        const multi_gpu_stream_scheduler& sch,     //
        Senders&&... sndrs) noexcept {
        return transfer_when_all_sender_th<
          multi_gpu_stream_scheduler,
          stdexec::tag_invoke_result_t<stdexec::into_variant_t, Senders>...>(
          sch.context_state_, stdexec::into_variant((Senders&&) sndrs)...);
      }

      template <stdexec::sender S, stdexec::scheduler Sch>
      friend auto tag_invoke(                  //
        stdexec::transfer_t,                   //
        const multi_gpu_stream_scheduler& sch, //
        S&& sndr,                              //
        Sch&& scheduler) noexcept {
        return stdexec::schedule_from(
          (Sch&&) scheduler, transfer_sender_th<S>(sch.context_state_, (S&&) sndr));
      }

      template <stdexec::sender S>
      friend split_sender_th<S>
        tag_invoke(stdexec::split_t, const multi_gpu_stream_scheduler& sch, S&& sndr) noexcept {
        return split_sender_th<S>((S&&) sndr, sch.context_state_);
      }

      template <stdexec::sender S>
      friend ensure_started_th<S>
        tag_invoke(stdexec::ensure_started_t, const multi_gpu_stream_scheduler& sch, S&& sndr) //
        noexcept {
        return ensure_started_th<S>((S&&) sndr, sch.context_state_);
      }

      friend sender_t
        tag_invoke(stdexec::schedule_t, const multi_gpu_stream_scheduler& self) noexcept {
        return {self.num_devices_, self.context_state_};
      }

      template <stdexec::sender S>
      friend auto
        tag_invoke(stdexec::sync_wait_t, const multi_gpu_stream_scheduler& self, S&& sndr) {
        return sync_wait::sync_wait_t{}(self.context_state_, (S&&) sndr);
      }

      friend stdexec::forward_progress_guarantee
        tag_invoke(stdexec::get_forward_progress_guarantee_t, const multi_gpu_stream_scheduler&) //
        noexcept {
        return stdexec::forward_progress_guarantee::weakly_parallel;
      }

      bool operator==(const multi_gpu_stream_scheduler& other) const noexcept {
        return context_state_.hub_ == other.context_state_.hub_;
      }

      multi_gpu_stream_scheduler(int num_devices, context_state_t context_state)
        : num_devices_(num_devices)
        , context_state_(context_state) {
      }

      // private: TODO
      int num_devices_{};
      context_state_t context_state_;
    };
  }

  using STDEXEC_STREAM_DETAIL_NS::multi_gpu_stream_scheduler;

  struct multi_gpu_stream_context {
    int num_devices_{};

    STDEXEC_STREAM_DETAIL_NS::resource_storage<STDEXEC_STREAM_DETAIL_NS::pinned_resource>
      pinned_resource_{};
    STDEXEC_STREAM_DETAIL_NS::resource_storage<STDEXEC_STREAM_DETAIL_NS::managed_resource>
      managed_resource_{};

    int dev_id_{};
    STDEXEC_STREAM_DETAIL_NS::queue::task_hub_t hub_;

    static int get_device() {
      int dev_id{};
      cudaGetDevice(&dev_id);
      return dev_id;
    }

    multi_gpu_stream_context()
      : dev_id_(get_device())
      , hub_(dev_id_, pinned_resource_.get()) {
      // TODO Manage errors
      STDEXEC_CHECK_CUDA_ERROR(cudaGetDeviceCount(&num_devices_));

      for (int dev_id = 0; dev_id < num_devices_; dev_id++) {
        STDEXEC_CHECK_CUDA_ERROR(cudaSetDevice(dev_id));
        for (int peer_id = 0; peer_id < num_devices_; peer_id++) {
          if (peer_id != dev_id) {
            int can_access{0};
            STDEXEC_CHECK_CUDA_ERROR(cudaDeviceCanAccessPeer(&can_access, dev_id, peer_id));

            if (can_access) {
              STDEXEC_CHECK_CUDA_ERROR(cudaDeviceEnablePeerAccess(peer_id, 0));
            }
          }
        }
      }
      STDEXEC_CHECK_CUDA_ERROR(cudaSetDevice(dev_id_));
    }

    multi_gpu_stream_scheduler get_scheduler(stream_priority priority = stream_priority::normal) {
      return {
        num_devices_,
        STDEXEC_STREAM_DETAIL_NS::context_state_t(
          pinned_resource_.get(), managed_resource_.get(), &hub_, priority)};
    }
  };
}
