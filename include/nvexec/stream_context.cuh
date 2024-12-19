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
#include "detail/memory.cuh"
#include "stream/sync_wait.cuh"
#include "stream/bulk.cuh"
#include "stream/let_xxx.cuh"
#include "stream/schedule_from.cuh"
#include "stream/start_detached.cuh"
#include "stream/submit.cuh"
#include "stream/split.cuh"
#include "stream/then.cuh"
#include "stream/continues_on.cuh"
#include "stream/launch.cuh"
#include "stream/upon_error.cuh"
#include "stream/upon_stopped.cuh"
#include "stream/when_all.cuh"
#include "stream/reduce.cuh"
#include "stream/ensure_started.cuh"

#include "stream/common.cuh"
#include "detail/queue.cuh"
#include "detail/throw_on_cuda_error.cuh"

namespace nvexec {
  namespace STDEXEC_STREAM_DETAIL_NS {
    template <sender Sender, std::integral Shape, class Fun>
    using bulk_sender_th = __t<bulk_sender_t<__id<__decay_t<Sender>>, Shape, Fun>>;

    template <sender Sender>
    using split_sender_th = __t<split_sender_t<__id<__decay_t<Sender>>>>;

    template <sender Sender, class Fun>
    using then_sender_th = __t<then_sender_t<__id<__decay_t<Sender>>, Fun>>;

    template <class Scheduler, sender... Senders>
    using when_all_sender_th =
      __t<when_all_sender_t<false, Scheduler, __id<__decay_t<Senders>>...>>;

    template <class Scheduler, sender... Senders>
    using transfer_when_all_sender_th =
      __t<when_all_sender_t<true, Scheduler, __id<__decay_t<Senders>>...>>;

    template <sender Sender, class Fun>
    using upon_error_sender_th = __t<upon_error_sender_t<__id<__decay_t<Sender>>, Fun>>;

    template <sender Sender, class Fun>
    using upon_stopped_sender_th = __t<upon_stopped_sender_t<__id<__decay_t<Sender>>, Fun>>;

    template <class Set, sender Sender, class Fun>
    using let_xxx_th = __t<let_sender_t<__id<__decay_t<Sender>>, Fun, Set>>;

    template <sender Sender>
    using continues_on_sender_th = __t<continues_on_sender_t<__id<__decay_t<Sender>>>>;

    template <sender Sender>
    using ensure_started_th = __t<ensure_started_sender_t<__id<Sender>>>;

    struct stream_scheduler {
      using __t = stream_scheduler;
      using __id = stream_scheduler;
      friend stream_context;

      template <sender Sender>
      using schedule_from_sender_th =
        stdexec::__t<schedule_from_sender_t<stream_scheduler, stdexec::__id<__decay_t<Sender>>>>;

      template <class ReceiverId>
      struct operation_state_ {
        using Receiver = stdexec::__t<ReceiverId>;

        struct __t : operation_state_base_t<ReceiverId> {
          using __id = operation_state_;

          cudaStream_t stream_{0};
          cudaError_t status_{cudaSuccess};

          __t(Receiver&& rcvr, context_state_t context_state)
            : operation_state_base_t<ReceiverId>(static_cast<Receiver&&>(rcvr), context_state) {
          }

          void start() & noexcept {
            this->propagate_completion_signal(set_value);
          }
        };
      };

      template <class ReceiverId>
      using operation_state_t = stdexec::__t<operation_state_<ReceiverId>>;

      struct env {
        context_state_t context_state_;

        template <class CPO>
        stream_scheduler query(get_completion_scheduler_t<CPO>) const noexcept {
          return stream_scheduler{context_state_};
        }
      };

      struct sender_ {
        struct __t : stream_sender_base {
          using __id = sender_;
          using completion_signatures =
            completion_signatures<set_value_t(), set_error_t(cudaError_t)>;

          template <class R>
          auto connect(R rec) const & noexcept(__nothrow_move_constructible<R>) //
            -> operation_state_t<stdexec::__id<R>> {
            return operation_state_t<stdexec::__id<R>>(static_cast<R&&>(rec), env_.context_state_);
          }

          auto get_env() const noexcept -> const env& {
            return env_;
          };

          STDEXEC_ATTRIBUTE((host, device)) inline __t(context_state_t context_state) noexcept
            : env_{context_state} {
          }

          env env_;
        };
      };

      using sender_t = stdexec::__t<sender_>;

      template <sender S>
      STDEXEC_MEMFN_DECL(auto schedule_from)(this const stream_scheduler& sch, S&& sndr) noexcept
        -> schedule_from_sender_th<S> {
        return schedule_from_sender_th<S>(sch.context_state_, static_cast<S&&>(sndr));
      }

      template <sender S, std::integral Shape, class Fn>
      STDEXEC_MEMFN_DECL(
        auto bulk)(this const stream_scheduler& sch, S&& sndr, Shape shape, Fn fun) noexcept
        -> bulk_sender_th<S, Shape, Fn> {
        return bulk_sender_th<S, Shape, Fn>{
          {}, static_cast<S&&>(sndr), shape, static_cast<Fn&&>(fun)};
      }

      template <sender S, class Fn>
      STDEXEC_MEMFN_DECL(
        auto
        then)(this const stream_scheduler& sch, S&& sndr, Fn fun) noexcept -> then_sender_th<S, Fn> {
        return then_sender_th<S, Fn>{{}, static_cast<S&&>(sndr), static_cast<Fn&&>(fun)};
      }

      template <sender S>
      STDEXEC_MEMFN_DECL(auto ensure_started)(this const stream_scheduler& sch, S&& sndr) noexcept
        -> ensure_started_th<S> {
        return ensure_started_th<S>(sch.context_state_, static_cast<S&&>(sndr));
      }

      template <sender S, class Fn>
      STDEXEC_MEMFN_DECL(auto let_value)(this const stream_scheduler& sch, S&& sndr, Fn fun) noexcept
        -> let_xxx_th<set_value_t, S, Fn> {
        return let_xxx_th<set_value_t, S, Fn>{{}, static_cast<S&&>(sndr), static_cast<Fn&&>(fun)};
      }

      template <sender S, class Fn>
      STDEXEC_MEMFN_DECL(auto let_error)(this const stream_scheduler& sch, S&& sndr, Fn fun) noexcept
        -> let_xxx_th<set_error_t, S, Fn> {
        return let_xxx_th<set_error_t, S, Fn>{{}, static_cast<S&&>(sndr), static_cast<Fn&&>(fun)};
      }

      template <sender S, class Fn>
      STDEXEC_MEMFN_DECL(
        auto let_stopped)(this const stream_scheduler& sch, S&& sndr, Fn fun) noexcept
        -> let_xxx_th<set_stopped_t, S, Fn> {
        return let_xxx_th<set_stopped_t, S, Fn>{{}, static_cast<S&&>(sndr), static_cast<Fn&&>(fun)};
      }

      template <sender S, class Fn>
      STDEXEC_MEMFN_DECL(
        auto upon_error)(this const stream_scheduler& sch, S&& sndr, Fn fun) noexcept
        -> upon_error_sender_th<S, Fn> {
        return upon_error_sender_th<S, Fn>{{}, static_cast<S&&>(sndr), static_cast<Fn&&>(fun)};
      }

      template <sender S, class Fn>
      STDEXEC_MEMFN_DECL(
        auto upon_stopped)(this const stream_scheduler& sch, S&& sndr, Fn fun) noexcept
        -> upon_stopped_sender_th<S, Fn> {
        return upon_stopped_sender_th<S, Fn>{{}, static_cast<S&&>(sndr), static_cast<Fn&&>(fun)};
      }

      template <stream_completing_sender... Senders>
      STDEXEC_MEMFN_DECL(auto transfer_when_all)(
        this const stream_scheduler& sch,
        Senders&&... sndrs) //
        noexcept -> transfer_when_all_sender_th<stream_scheduler, Senders...> {
        return transfer_when_all_sender_th<stream_scheduler, Senders...>(
          sch.context_state_, static_cast<Senders&&>(sndrs)...);
      }

      template <stream_completing_sender... Senders>
      STDEXEC_MEMFN_DECL(auto transfer_when_all_with_variant)(
        this const stream_scheduler& sch, //
        Senders&&... sndrs) noexcept      //
        -> transfer_when_all_sender_th<stream_scheduler, __result_of<into_variant, Senders>...> {
        return transfer_when_all_sender_th<stream_scheduler, __result_of<into_variant, Senders>...>(
          sch.context_state_, into_variant(static_cast<Senders&&>(sndrs))...);
      }

      template <sender S, scheduler Sch>
      STDEXEC_MEMFN_DECL(
        auto continues_on)(this const stream_scheduler& sch, S&& sndr, Sch&& scheduler) //
        noexcept -> __result_of<schedule_from, Sch, continues_on_sender_th<S>> {
        return schedule_from(
          static_cast<Sch&&>(scheduler),
          continues_on_sender_th<S>(sch.context_state_, static_cast<S&&>(sndr)));
      }

      template <sender S>
      STDEXEC_MEMFN_DECL(auto split)(this const stream_scheduler& sch, S&& sndr) noexcept //
        -> split_sender_th<S> {
        return split_sender_th<S>(sch.context_state_, static_cast<S&&>(sndr));
      }

      STDEXEC_ATTRIBUTE((host, device)) inline auto schedule() const noexcept -> sender_t {
        return {context_state_};
      }

      STDEXEC_MEMFN_DECL(auto __has_algorithm_customizations)(this const stream_scheduler& self) //
        noexcept -> std::true_type {
        return {};
      }

      template <sender S>
      STDEXEC_MEMFN_DECL(auto sync_wait)(this const stream_scheduler& self, S&& sndr) //
        -> __call_result_t<_sync_wait::sync_wait_t, context_state_t, S> {
        return _sync_wait::sync_wait_t{}(self.context_state_, static_cast<S&&>(sndr));
      }

      auto query(get_forward_progress_guarantee_t) const noexcept -> forward_progress_guarantee {
        return forward_progress_guarantee::weakly_parallel;
      }

      bool operator==(const stream_scheduler& other) const noexcept {
        return context_state_.hub_ == other.context_state_.hub_;
      }

      explicit stream_scheduler(context_state_t context_state) noexcept
        : context_state_(context_state) {
      }

      // private: TODO
      context_state_t context_state_;
    };

    template <stream_completing_sender Sender>
    void tag_invoke(start_detached_t, Sender&& sndr) noexcept(false) {
      _submit::submit_t{}(static_cast<Sender&&>(sndr), _start_detached::detached_receiver_t{});
    }

    template <stream_completing_sender... Senders>
    when_all_sender_th<stream_scheduler, Senders...>
      tag_invoke(when_all_t, Senders&&... sndrs) noexcept {
      return when_all_sender_th<stream_scheduler, Senders...>{
        context_state_t{nullptr, nullptr, nullptr, nullptr},
        static_cast<Senders&&>(sndrs)...
      };
    }

    template <stream_completing_sender... Senders>
    when_all_sender_th<stream_scheduler, __result_of<into_variant, Senders>...>
      tag_invoke(when_all_with_variant_t, Senders&&... sndrs) noexcept {
      return when_all_sender_th<stream_scheduler, __result_of<into_variant, Senders>...>{
        context_state_t{nullptr, nullptr, nullptr, nullptr},
        into_variant(static_cast<Senders&&>(sndrs))...
      };
    }

    template <sender S, class Fn>
    upon_error_sender_th<S, Fn> tag_invoke(upon_error_t, S&& sndr, Fn fun) noexcept {
      return upon_error_sender_th<S, Fn>{{}, static_cast<S&&>(sndr), static_cast<Fn&&>(fun)};
    }

    template <sender S, class Fn>
    upon_stopped_sender_th<S, Fn> tag_invoke(upon_stopped_t, S&& sndr, Fn fun) noexcept {
      return upon_stopped_sender_th<S, Fn>{{}, static_cast<S&&>(sndr), static_cast<Fn&&>(fun)};
    }

  } // namespace STDEXEC_STREAM_DETAIL_NS

  using STDEXEC_STREAM_DETAIL_NS::stream_scheduler;

  struct stream_context {
    STDEXEC_STREAM_DETAIL_NS::resource_storage<STDEXEC_STREAM_DETAIL_NS::pinned_resource>
      pinned_resource_{};
    STDEXEC_STREAM_DETAIL_NS::resource_storage<STDEXEC_STREAM_DETAIL_NS::managed_resource>
      managed_resource_{};
    STDEXEC_STREAM_DETAIL_NS::stream_pools_t stream_pools_{};

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
      return stream_scheduler{STDEXEC_STREAM_DETAIL_NS::context_state_t(
        pinned_resource_.get(), managed_resource_.get(), &stream_pools_, &hub_, priority)};
    }
  };
} // namespace nvexec
