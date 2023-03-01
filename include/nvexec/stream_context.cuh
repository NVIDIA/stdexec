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
#include <memory_resource>

#include "detail/config.cuh"
#include "stream/sync_wait.cuh"
#include "stream/bulk.cuh"
#include "stream/let_xxx.cuh"
#include "stream/schedule_from.cuh"
#include "stream/start_detached.cuh"
#include "stream/submit.cuh"
#include "stream/split.cuh"
#include "stream/then.cuh"
#include "stream/transfer.cuh"
#include "stream/upon_error.cuh"
#include "stream/upon_stopped.cuh"
#include "stream/when_all.cuh"
#include "stream/reduce.cuh"
#include "stream/ensure_started.cuh"

#include "stream/common.cuh"
#include "detail/queue.cuh"

namespace nvexec {
  namespace STDEXEC_STREAM_DETAIL_NS {
    template <stdexec::sender Sender, std::integral Shape, class Fun>
    using bulk_sender_th =
      stdexec::__t<bulk_sender_t<stdexec::__id<std::decay_t<Sender>>, Shape, Fun>>;

    template <stdexec::sender Sender>
    using split_sender_th = stdexec::__t<split_sender_t<stdexec::__id<std::decay_t<Sender>>>>;

    template <stdexec::sender Sender, class Fun>
    using then_sender_th = stdexec::__t<then_sender_t<stdexec::__id<std::decay_t<Sender>>, Fun>>;

    template <class Scheduler, stdexec::sender... Senders>
    using when_all_sender_th =
      stdexec::__t<when_all_sender_t<false, Scheduler, stdexec::__id<std::decay_t<Senders>>...>>;

    template <class Scheduler, stdexec::sender... Senders>
    using transfer_when_all_sender_th =
      stdexec::__t<when_all_sender_t<true, Scheduler, stdexec::__id<std::decay_t<Senders>>...>>;

    template <stdexec::sender Sender, class Fun>
    using upon_error_sender_th =
      stdexec::__t<upon_error_sender_t<stdexec::__id<std::decay_t<Sender>>, Fun>>;

    template <stdexec::sender Sender, class Fun>
    using upon_stopped_sender_th =
      stdexec::__t<upon_stopped_sender_t<stdexec::__id<std::decay_t<Sender>>, Fun>>;

    template <class Set, stdexec::sender Sender, class Fun>
    using let_xxx_th =
      stdexec::__t<let_sender_t<stdexec::__id<std::decay_t<Sender>>, Fun, stdexec::__x<Set>>>;

    template <stdexec::sender Sender>
    using transfer_sender_th = stdexec::__t<transfer_sender_t<stdexec::__id<std::decay_t<Sender>>>>;

    template <stdexec::sender Sender>
    using ensure_started_th = stdexec::__t<ensure_started_sender_t<stdexec::__id<Sender>>>;

    struct stream_scheduler {
      friend stream_context;

      template <stdexec::sender Sender>
      using schedule_from_sender_th =
        stdexec::__t<schedule_from_sender_t<stream_scheduler, stdexec::__id<std::decay_t<Sender>>>>;

      template <class ReceiverId>
      struct operation_state_ {
        using Receiver = stdexec::__t<ReceiverId>;

        struct __t : operation_state_base_t<ReceiverId> {
          using __id = operation_state_;

          cudaStream_t stream_{0};
          cudaError_t status_{cudaSuccess};

          __t(Receiver&& receiver, context_state_t context_state)
            : operation_state_base_t<ReceiverId>((Receiver&&) receiver, context_state, false) {
          }

          friend void tag_invoke(stdexec::start_t, __t& op) noexcept {
            op.propagate_completion_signal(stdexec::set_value);
          }
        };
      };

      template <class ReceiverId>
      using operation_state_t = stdexec::__t<operation_state_<ReceiverId>>;

      struct env {
        context_state_t context_state_;

        stream_scheduler make_scheduler() const {
          return stream_scheduler{context_state_};
        }

        template <class CPO>
        friend stream_scheduler
          tag_invoke(stdexec::get_completion_scheduler_t<CPO>, const env& self) noexcept {
          return self.make_scheduler();
        }
      };

      struct sender_ {
        struct __t : stream_sender_base {
          using __id = sender_;
          using completion_signatures = stdexec::
            completion_signatures< stdexec::set_value_t(), stdexec::set_error_t(cudaError_t)>;

          template <class R>
          friend auto tag_invoke(stdexec::connect_t, const __t& self, R&& rec) //
            noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<R>, R>)
              -> operation_state_t<stdexec::__id<std::remove_cvref_t<R>>> {
            return operation_state_t<stdexec::__id<std::remove_cvref_t<R>>>(
              (R&&) rec, self.env_.context_state_);
          }

          friend const env& tag_invoke(stdexec::get_env_t, const __t& self) noexcept {
            return self.env_;
          };

          STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
            inline __t(context_state_t context_state) noexcept
            : env_{context_state} {
          }

          env env_;
        };
      };

      using sender_t = stdexec::__t<sender_>;

      template <stdexec::sender S>
      friend schedule_from_sender_th<S>
        tag_invoke(stdexec::schedule_from_t, const stream_scheduler& sch, S&& sndr) noexcept {
        return schedule_from_sender_th<S>(sch.context_state_, (S&&) sndr);
      }

      template <stdexec::sender S, std::integral Shape, class Fn>
      friend bulk_sender_th<S, Shape, Fn>
        tag_invoke(stdexec::bulk_t, const stream_scheduler& sch, S&& sndr, Shape shape, Fn fun) //
        noexcept {
        return bulk_sender_th<S, Shape, Fn>{{}, (S&&) sndr, shape, (Fn&&) fun};
      }

      template <stdexec::sender S, class Fn>
      friend then_sender_th<S, Fn>
        tag_invoke(stdexec::then_t, const stream_scheduler& sch, S&& sndr, Fn fun) noexcept {
        return then_sender_th<S, Fn>{{}, (S&&) sndr, (Fn&&) fun};
      }

      template <stdexec::sender S>
      friend ensure_started_th<S>
        tag_invoke(stdexec::ensure_started_t, const stream_scheduler& sch, S&& sndr) noexcept {
        return ensure_started_th<S>(sch.context_state_, (S&&) sndr);
      }

      template <stdexec::sender S, class Fn>
      friend let_xxx_th<stdexec::set_value_t, S, Fn>
        tag_invoke(stdexec::let_value_t, const stream_scheduler& sch, S&& sndr, Fn fun) noexcept {
        return let_xxx_th<stdexec::set_value_t, S, Fn>{{}, (S&&) sndr, (Fn&&) fun};
      }

      template <stdexec::sender S, class Fn>
      friend let_xxx_th<stdexec::set_error_t, S, Fn>
        tag_invoke(stdexec::let_error_t, const stream_scheduler& sch, S&& sndr, Fn fun) noexcept {
        return let_xxx_th<stdexec::set_error_t, S, Fn>{{}, (S&&) sndr, (Fn&&) fun};
      }

      template <stdexec::sender S, class Fn>
      friend let_xxx_th<stdexec::set_stopped_t, S, Fn>
        tag_invoke(stdexec::let_stopped_t, const stream_scheduler& sch, S&& sndr, Fn fun) noexcept {
        return let_xxx_th<stdexec::set_stopped_t, S, Fn>{{}, (S&&) sndr, (Fn&&) fun};
      }

      template <stdexec::sender S, class Fn>
      friend upon_error_sender_th<S, Fn>
        tag_invoke(stdexec::upon_error_t, const stream_scheduler& sch, S&& sndr, Fn fun) noexcept {
        return upon_error_sender_th<S, Fn>{{}, (S&&) sndr, (Fn&&) fun};
      }

      template <stdexec::sender S, class Fn>
      friend upon_stopped_sender_th<S, Fn>
        tag_invoke(stdexec::upon_stopped_t, const stream_scheduler& sch, S&& sndr, Fn fun) //
        noexcept {
        return upon_stopped_sender_th<S, Fn>{{}, (S&&) sndr, (Fn&&) fun};
      }

      template <stream_completing_sender... Senders>
      friend auto
        tag_invoke(stdexec::transfer_when_all_t, const stream_scheduler& sch, Senders&&... sndrs) //
        noexcept {
        return transfer_when_all_sender_th<stream_scheduler, Senders...>(
          sch.context_state_, (Senders&&) sndrs...);
      }

      template <stream_completing_sender... Senders>
      friend auto tag_invoke(                      //
        stdexec::transfer_when_all_with_variant_t, //
        const stream_scheduler& sch,               //
        Senders&&... sndrs) noexcept {
        return transfer_when_all_sender_th<
          stream_scheduler,
          stdexec::tag_invoke_result_t<stdexec::into_variant_t, Senders>...>(
          sch.context_state_, stdexec::into_variant((Senders&&) sndrs)...);
      }

      template <stdexec::sender S, stdexec::scheduler Sch>
      friend auto
        tag_invoke(stdexec::transfer_t, const stream_scheduler& sch, S&& sndr, Sch&& scheduler) //
        noexcept {
        return stdexec::schedule_from(
          (Sch&&) scheduler, transfer_sender_th<S>(sch.context_state_, (S&&) sndr));
      }

      template <stdexec::sender S>
      friend split_sender_th<S>
        tag_invoke(stdexec::split_t, const stream_scheduler& sch, S&& sndr) noexcept {
        return split_sender_th<S>(sch.context_state_, (S&&) sndr);
      }

      STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
        friend inline sender_t
        tag_invoke(stdexec::schedule_t, const stream_scheduler& self) noexcept {
        return {self.context_state_};
      }

      friend std::true_type tag_invoke(            //
        stdexec::__has_algorithm_customizations_t, //
        const stream_scheduler& self) noexcept {
        return {};
      }

      template <stdexec::sender S>
      friend auto
        tag_invoke(stdexec::this_thread::sync_wait_t, const stream_scheduler& self, S&& sndr) {
        return sync_wait::sync_wait_t{}(self.context_state_, (S&&) sndr);
      }

      friend stdexec::forward_progress_guarantee
        tag_invoke(stdexec::get_forward_progress_guarantee_t, const stream_scheduler&) noexcept {
        return stdexec::forward_progress_guarantee::weakly_parallel;
      }

      bool operator==(const stream_scheduler& other) const noexcept {
        return context_state_.hub_ == other.context_state_.hub_;
      }

      stream_scheduler(context_state_t context_state)
        : context_state_(context_state) {
      }

      // private: TODO
      context_state_t context_state_;
    };

    template <stream_completing_sender Sender>
    void tag_invoke(stdexec::start_detached_t, Sender&& sndr) noexcept(false) {
      submit::submit_t{}((Sender&&) sndr, start_detached::detached_receiver_t{});
    }

    template <stream_completing_sender... Senders>
    when_all_sender_th<stream_scheduler, Senders...>
      tag_invoke(stdexec::when_all_t, Senders&&... sndrs) noexcept {
      return when_all_sender_th<stream_scheduler, Senders...>{
        context_state_t{nullptr, nullptr, nullptr},
        (Senders&&) sndrs...
      };
    }

    template <stream_completing_sender... Senders>
    when_all_sender_th<
      stream_scheduler,
      stdexec::tag_invoke_result_t<stdexec::into_variant_t, Senders>...>
      tag_invoke(stdexec::when_all_with_variant_t, Senders&&... sndrs) noexcept {
      return when_all_sender_th<
        stream_scheduler,
        stdexec::tag_invoke_result_t<stdexec::into_variant_t, Senders>...>{
        context_state_t{nullptr, nullptr, nullptr},
        stdexec::into_variant((Senders&&) sndrs)...
      };
    }

    template <stdexec::sender S, class Fn>
    upon_error_sender_th<S, Fn> tag_invoke(stdexec::upon_error_t, S&& sndr, Fn fun) noexcept {
      return upon_error_sender_th<S, Fn>{{}, (S&&) sndr, (Fn&&) fun};
    }

    template <stdexec::sender S, class Fn>
    upon_stopped_sender_th<S, Fn> tag_invoke(stdexec::upon_stopped_t, S&& sndr, Fn fun) noexcept {
      return upon_stopped_sender_th<S, Fn>{{}, (S&&) sndr, (Fn&&) fun};
    }

    struct pinned_resource : public std::pmr::memory_resource {
      void* do_allocate(size_t bytes, size_t /* alignment */) override {
        void* ret;

        if (cudaError_t status = STDEXEC_CHECK_CUDA_ERROR(cudaMallocHost(&ret, bytes));
            status != cudaSuccess) {
          throw std::bad_alloc();
        }

        return ret;
      }

      void do_deallocate(void* ptr, size_t /* bytes */, size_t /* alignment */) override {
        STDEXEC_CHECK_CUDA_ERROR(cudaFreeHost(ptr));
      }

      bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
      }
    };

    struct gpu_resource : public std::pmr::memory_resource {
      void* do_allocate(size_t bytes, size_t /* alignment */) override {
        void* ret;

        if (cudaError_t status = STDEXEC_CHECK_CUDA_ERROR(cudaMalloc(&ret, bytes)); status != cudaSuccess) {
          throw std::bad_alloc();
        }

        return ret;
      }

      void do_deallocate(void* ptr, size_t /* bytes */, size_t /* alignment */) override {
        STDEXEC_CHECK_CUDA_ERROR(cudaFree(ptr));
      }

      bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
      }
    };

    struct managed_resource : public std::pmr::memory_resource {
      void* do_allocate(size_t bytes, size_t /* alignment */) override {
        void* ret;

        if (cudaError_t status = STDEXEC_CHECK_CUDA_ERROR(cudaMallocManaged(&ret, bytes));
            status != cudaSuccess) {
          throw std::bad_alloc();
        }

        return ret;
      }

      void do_deallocate(void* ptr, size_t /* bytes */, size_t /* alignment */) override {
        STDEXEC_CHECK_CUDA_ERROR(cudaFree(ptr));
      }

      bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
      }
    };

    template <class UnderlyingResource>
    class resource_storage {
      UnderlyingResource underlying_resource_{};
      std::pmr::monotonic_buffer_resource monotonic_resource_{512 * 1024, &underlying_resource_};
      std::pmr::synchronized_pool_resource resource_{&monotonic_resource_};

     public:
      std::pmr::memory_resource* get() {
        return &resource_;
      }
    };
  }

  using STDEXEC_STREAM_DETAIL_NS::stream_scheduler;

  struct stream_context {
    STDEXEC_STREAM_DETAIL_NS::resource_storage<STDEXEC_STREAM_DETAIL_NS::pinned_resource>
      pinned_resource_{};
    STDEXEC_STREAM_DETAIL_NS::resource_storage<STDEXEC_STREAM_DETAIL_NS::managed_resource>
      managed_resource_{};

    // STDEXEC_STREAM_DETAIL_NS::resource_storage<STDEXEC_STREAM_DETAIL_NS::gpu_resource> gpu_resource_{};

    static int get_device() {
      int dev_id{};
      cudaGetDevice(&dev_id);
      return dev_id;
    }

    int dev_id_{};
    STDEXEC_STREAM_DETAIL_NS::queue::task_hub_t hub_;

    stream_context()
      : dev_id_(get_device())
      , hub_(dev_id_, pinned_resource_.get()) {
    }

    stream_scheduler get_scheduler(stream_priority priority = stream_priority::normal) {
      return {STDEXEC_STREAM_DETAIL_NS::context_state_t(
        pinned_resource_.get(),
        managed_resource_.get(),
        // gpu_resource_.get(),
        &hub_,
        priority)};
    }
  };
}
