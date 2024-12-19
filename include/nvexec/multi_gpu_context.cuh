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
    template <sender Sender, std::integral Shape, class Fun>
    using multi_gpu_bulk_sender_th =
      stdexec::__t<multi_gpu_bulk_sender_t<stdexec::__id<__decay_t<Sender>>, Shape, Fun>>;

    struct multi_gpu_stream_scheduler {
      using __t = multi_gpu_stream_scheduler;
      using __id = multi_gpu_stream_scheduler;
      friend stream_context;

      template <sender Sender>
      using schedule_from_sender_th =
        stdexec::__t<schedule_from_sender_t<stream_scheduler, stdexec::__id<__decay_t<Sender>>>>;

      template <class RId>
      struct operation_state_t : stream_op_state_base {
        using R = stdexec::__t<RId>;

        R rec_;
        cudaStream_t stream_{0};
        cudaError_t status_{cudaSuccess};

        template <__decays_to<R> Receiver>
        operation_state_t(Receiver&& rec)
          : rec_(static_cast<Receiver&&>(rec)) {
          status_ = STDEXEC_DBG_ERR(cudaStreamCreate(&stream_));
        }

        ~operation_state_t() {
          STDEXEC_DBG_ERR(cudaStreamDestroy(stream_));
        }

        cudaStream_t get_stream() {
          return stream_;
        }

        void start() & noexcept {
          if constexpr (stream_receiver<R>) {
            if (status_ == cudaSuccess) {
              stdexec::set_value(static_cast<R&&>(rec_));
            } else {
              stdexec::set_error(static_cast<R&&>(rec_), std::move(status_));
            }
          } else {
            if (status_ == cudaSuccess) {
              continuation_kernel<<<1, 1, 0, stream_>>>(std::move(rec_), stdexec::set_value);
            } else {
              continuation_kernel<<<1, 1, 0, stream_>>>(
                std::move(rec_), stdexec::set_error, std::move(status_));
            }
          }
        }
      };

      struct sender_t : stream_sender_base {

        struct env {
          int num_devices_;
          context_state_t context_state_;

          template <class CPO>
          multi_gpu_stream_scheduler query(get_completion_scheduler_t<CPO>) const noexcept {
            return multi_gpu_stream_scheduler{num_devices_, context_state_};
          }
        };

        using completion_signatures =
          completion_signatures<set_value_t(), set_error_t(cudaError_t)>;

        template <class R>
        auto connect(R rec) const & noexcept(__nothrow_move_constructible<R>) //
          -> operation_state_t<stdexec::__id<__decay_t<R>>> {
          return operation_state_t<stdexec::__id<__decay_t<R>>>(static_cast<R&&>(rec));
        }

        auto get_env() const noexcept -> const env& {
          return env_;
        }

        sender_t(int num_devices, context_state_t context_state) noexcept
          : env_{num_devices, context_state} {
        }

        env env_;
      };

      template <sender S>
      STDEXEC_MEMFN_DECL(schedule_from_sender_th<S> schedule_from)(
        this const multi_gpu_stream_scheduler& sch,
        S&& sndr) //
        noexcept {
        return schedule_from_sender_th<S>(sch.context_state_, static_cast<S&&>(sndr));
      }

      template <sender S, std::integral Shape, class Fn>
      STDEXEC_MEMFN_DECL(multi_gpu_bulk_sender_th<S, Shape, Fn> bulk)(
        this const multi_gpu_stream_scheduler& sch, //
        S&& sndr,                                   //
        Shape shape,                                //
        Fn fun)                                     //
        noexcept {
        return multi_gpu_bulk_sender_th<S, Shape, Fn>{
          {}, sch.num_devices_, static_cast<S&&>(sndr), shape, static_cast<Fn&&>(fun)};
      }

      template <sender S, class Fn>
      STDEXEC_MEMFN_DECL(then_sender_th<S, Fn> then)(
        this const multi_gpu_stream_scheduler& sch,
        S&& sndr,
        Fn fun) //
        noexcept {
        return then_sender_th<S, Fn>{{}, static_cast<S&&>(sndr), static_cast<Fn&&>(fun)};
      }

      template <__one_of<let_value_t, let_stopped_t, let_error_t> Let, sender S, class Fn>
      friend let_xxx_th<Let, S, Fn>
        tag_invoke(Let, const multi_gpu_stream_scheduler& sch, S&& sndr, Fn fun) noexcept {
        return let_xxx_th<Let, S, Fn>{{}, static_cast<S&&>(sndr), static_cast<Fn&&>(fun)};
      }

      template <sender S, class Fn>
      STDEXEC_MEMFN_DECL(
        upon_error_sender_th<S, Fn>
        upon_error)(this const multi_gpu_stream_scheduler& sch, S&& sndr, Fn fun) noexcept {
        return upon_error_sender_th<S, Fn>{{}, static_cast<S&&>(sndr), static_cast<Fn&&>(fun)};
      }

      template <sender S, class Fn>
      STDEXEC_MEMFN_DECL(
        upon_stopped_sender_th<S, Fn>
        upon_stopped)(this const multi_gpu_stream_scheduler& sch, S&& sndr, Fn fun) noexcept {
        return upon_stopped_sender_th<S, Fn>{{}, static_cast<S&&>(sndr), static_cast<Fn&&>(fun)};
      }

      template <stream_completing_sender... Senders>
      STDEXEC_MEMFN_DECL(auto transfer_when_all)(
        this const multi_gpu_stream_scheduler& sch, //
        Senders&&... sndrs) noexcept {
        return transfer_when_all_sender_th<multi_gpu_stream_scheduler, Senders...>(
          sch.context_state_, static_cast<Senders&&>(sndrs)...);
      }

      template <stream_completing_sender... Senders>
      STDEXEC_MEMFN_DECL(auto transfer_when_all_with_variant)(
        this const multi_gpu_stream_scheduler& sch, //
        Senders&&... sndrs) noexcept {
        return transfer_when_all_sender_th<
          multi_gpu_stream_scheduler,
          __result_of<into_variant, Senders>...>(
          sch.context_state_, into_variant(static_cast<Senders&&>(sndrs))...);
      }

      template <sender S, scheduler Sch>
      STDEXEC_MEMFN_DECL(auto continues_on)(
        this const multi_gpu_stream_scheduler& sch, //
        S&& sndr,                                   //
        Sch&& scheduler) noexcept {
        return schedule_from(
          static_cast<Sch&&>(scheduler),
          continues_on_sender_th<S>(sch.context_state_, static_cast<S&&>(sndr)));
      }

      template <sender S>
      STDEXEC_MEMFN_DECL(
        split_sender_th<S> split)(this const multi_gpu_stream_scheduler& sch, S&& sndr) noexcept {
        return split_sender_th<S>(static_cast<S&&>(sndr), sch.context_state_);
      }

      template <sender S>
      STDEXEC_MEMFN_DECL(ensure_started_th<S> ensure_started)(
        this const multi_gpu_stream_scheduler& sch,
        S&& sndr) //
        noexcept {
        return ensure_started_th<S>(static_cast<S&&>(sndr), sch.context_state_);
      }

      sender_t schedule() const noexcept {
        return {num_devices_, context_state_};
      }

      template <sender S>
      STDEXEC_MEMFN_DECL(auto sync_wait)(this const multi_gpu_stream_scheduler& self, S&& sndr) {
        return _sync_wait::sync_wait_t{}(self.context_state_, static_cast<S&&>(sndr));
      }

      forward_progress_guarantee query(get_forward_progress_guarantee_t) const noexcept {
        return forward_progress_guarantee::weakly_parallel;
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
  } // namespace STDEXEC_STREAM_DETAIL_NS

  using STDEXEC_STREAM_DETAIL_NS::multi_gpu_stream_scheduler;

  struct multi_gpu_stream_context {
    int num_devices_{};

    STDEXEC_STREAM_DETAIL_NS::resource_storage<STDEXEC_STREAM_DETAIL_NS::pinned_resource>
      pinned_resource_{};
    STDEXEC_STREAM_DETAIL_NS::resource_storage<STDEXEC_STREAM_DETAIL_NS::managed_resource>
      managed_resource_{};
    STDEXEC_STREAM_DETAIL_NS::stream_pools_t stream_pools_{};

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

    multi_gpu_stream_scheduler get_scheduler(stream_priority priority = stream_priority::normal) {
      return {
        num_devices_,
        STDEXEC_STREAM_DETAIL_NS::context_state_t(
          pinned_resource_.get(), managed_resource_.get(), &stream_pools_, &hub_, priority)};
    }
  };
} // namespace nvexec
