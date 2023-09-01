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
#include "stream/transfer.cuh"
#include "stream/upon_error.cuh"
#include "stream/upon_stopped.cuh"
#include "stream/when_all.cuh"
#include "stream/reduce.cuh"
#include "stream/ensure_started.cuh"

#include "stream/common.cuh"
#include "detail/queue.cuh"
#include "detail/throw_on_cuda_error.cuh"

template <typename T>
struct type_printer;

namespace nvexec {
  namespace STDEXEC_STREAM_DETAIL_NS {
    template <sender Sender, std::integral Shape, class Fun>
    using bulk_sender_th = __t<bulk_sender_t<__id<__decay_t<Sender>>, Shape, Fun>>;

    template <stdexec::sender Sender, class InitT, class Fun>
    using reduce_sender_t = //
      stdexec::__t<
        reduce_::
          sender_t< stdexec::__id<stdexec::__decay_t<Sender>>, InitT, stdexec::__decay_t<Fun>>>;

    template <class Fun, class InitT, class... Args>
      requires stdexec::__callable<Fun, InitT, Args&...>
    using reduce_non_throwing = stdexec::__mbool<
      stdexec::__nothrow_callable<Fun, InitT, Args&...>
      && noexcept(stdexec::__decayed_tuple<Args...>(std::declval<Args>()...)) >;

    template <class SenderId, class ReceiverId, class Shape, class Fn, bool MayThrow>
    struct reduce_receiver;

    template <sender Sender>
    using split_sender_th = __t<split_sender_t<__id<__decay_t<Sender>>>>;

    template <sender Sender, class Fun>
    using then_sender_th = __t<then_sender_t<__id<__decay_t<Sender>>, Fun>>;

    template <class Scheduler, sender... Senders>
    using when_all_sender_th =
      __t< when_all_sender_t<false, Scheduler, __id<__decay_t<Senders>>...>>;

    template <class Scheduler, sender... Senders>
    using transfer_when_all_sender_th =
      __t< when_all_sender_t<true, Scheduler, __id<__decay_t<Senders>>...>>;

    template <sender Sender, class Fun>
    using upon_error_sender_th = __t<upon_error_sender_t<__id<__decay_t<Sender>>, Fun>>;

    template <sender Sender, class Fun>
    using upon_stopped_sender_th = __t<upon_stopped_sender_t<__id<__decay_t<Sender>>, Fun>>;

    template <class Set, sender Sender, class Fun>
    using let_xxx_th = __t<let_sender_t<__id<__decay_t<Sender>>, Fun, Set>>;

    template <sender Sender>
    using transfer_sender_th = __t<transfer_sender_t<__id<__decay_t<Sender>>>>;

    template <sender Sender>
    using ensure_started_th = __t<ensure_started_sender_t<__id<Sender>>>;

    struct stream_domain {

      template < __lazy_sender_for<reduce_t> Sender, typename... Senders, typename InitT, typename Fun>
      auto transform_sender_recursive(Sender&& head, Senders&&... tail, InitT&& initT, Fun&& fun) const {
        static_assert(
            std::is_constructible<reduce_sender_t<Sender, InitT, Fun>, Sender, ArgInit2Type, Arg3Type>::value,
            "reduce_sender_t cannot be constructed with the provided argument types!"
        );
        if constexpr (sizeof...(tail) == 0) {
          return reduce_sender_t<Sender, InitT, Fun>{(Sender&&) head, initT, fun};
        } else {
          return transform_sender_recursive(
            reduce_sender_t<Sender, InitT, Fun>{(Sender&&) head, initT, std::move(fun)},
            (Senders&&) tail...,
            initT,
            fun);
        }
      }

      template <class _Sender, class _Env = empty_env>
      _Sender transform_sender(_Sender&& __sndr, const _Env& = {}) const
        // BUGBUG fix me:
        noexcept(std::is_reference_v<_Sender>)
      // noexcept(__nothrow_constructible_from<_Sender, _Sender&&>)
      {
        return static_cast<_Sender&&>(__sndr);
      }


      template <stdexec::__lazy_sender_for<reduce_t> Sender, class Env = empty_env>
      auto transform_sender(Sender&& sndr, const Env& env = {}) const noexcept {
        
        return stdexec::__sender_apply(
          (Sender&&) sndr,
          [&]<class Tag, class Data, class First, class... Rest>(
            Tag, Data&& data, First first, Rest... rest) {
              stdexec::print(rest...);
            auto initT = stdexec::__nth_member<0>()((Data&&) data);
            auto fun = stdexec::__nth_member<1>()((Data&&) data);

            return transform_sender_recursive((Sender&&) first,
              rest..., initT, std::move(fun));
          });
      }
    };

    struct stream_scheduler {
      using __t = stream_scheduler;
      using __id = stream_scheduler;
      friend stream_context;

      template <sender Sender>
      using schedule_from_sender_th =
        stdexec::__t< schedule_from_sender_t<stream_scheduler, stdexec::__id<__decay_t<Sender>>>>;

      template <class ReceiverId>
      struct operation_state_ {
        using Receiver = stdexec::__t<ReceiverId>;

        struct __t : operation_state_base_t<ReceiverId> {
          using __id = operation_state_;

          cudaStream_t stream_{0};
          cudaError_t status_{cudaSuccess};

          __t(Receiver&& rcvr, context_state_t context_state)
            : operation_state_base_t<ReceiverId>((Receiver&&) rcvr, context_state) {
          }

          friend void tag_invoke(start_t, __t& op) noexcept {
            op.propagate_completion_signal(set_value);
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
          tag_invoke(get_completion_scheduler_t<CPO>, const env& self) noexcept {
          return self.make_scheduler();
        }
      };

      struct sender_ {
        using is_sender = struct __t : stream_sender_base {
          using __id = sender_;
          using completion_signatures =
            completion_signatures< set_value_t(), set_error_t(cudaError_t)>;

          template <class R>
          friend auto tag_invoke(connect_t, const __t& self, R&& rec) //
            noexcept(__nothrow_constructible_from<__decay_t<R>, R>)
              -> operation_state_t<stdexec::__id<__decay_t<R>>> {
            return operation_state_t<stdexec::__id<__decay_t<R>>>(
              (R&&) rec, self.env_.context_state_);
          }

          friend const env& tag_invoke(get_env_t, const __t& self) noexcept {

            return self.env_;
          };

          STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
            inline __t(context_state_t context_state) noexcept
            : env_{context_state} {
          }

          env env_;
        };
      };

      friend stream_domain tag_invoke(get_domain_t, const stream_scheduler& sch) noexcept {
        return {};
      }

      using sender_t = stdexec::__t<sender_>;

      template <sender S>
      friend schedule_from_sender_th<S>
        tag_invoke(schedule_from_t, const stream_scheduler& sch, S&& sndr) noexcept {
        return schedule_from_sender_th<S>(sch.context_state_, (S&&) sndr);
      }

      template <sender S, std::integral Shape, class Fn>
      friend bulk_sender_th<S, Shape, Fn>
        tag_invoke(bulk_t, const stream_scheduler& sch, S&& sndr, Shape shape, Fn fun) //
        noexcept {
        return bulk_sender_th<S, Shape, Fn>{{}, (S&&) sndr, shape, (Fn&&) fun};
      }

      template <sender S, std::integral InitT, class Fn>
      friend reduce_sender_t<S, InitT, Fn>
        tag_invoke(reduce_t, const stream_scheduler& sch, S&& sndr, InitT initT, Fn fun) //
        noexcept {
        return reduce_sender_t<S, InitT, Fn>{{}, (S&&) sndr, initT, (Fn&&) fun};
      }

      template <sender S, class Fn>
      friend then_sender_th<S, Fn>
        tag_invoke(then_t, const stream_scheduler& sch, S&& sndr, Fn fun) noexcept {
        return then_sender_th<S, Fn>{{}, (S&&) sndr, (Fn&&) fun};
      }

      template <sender S>
      friend ensure_started_th<S>
        tag_invoke(ensure_started_t, const stream_scheduler& sch, S&& sndr) noexcept {
        return ensure_started_th<S>(sch.context_state_, (S&&) sndr);
      }

      template <sender S, class Fn>
      friend let_xxx_th<set_value_t, S, Fn>
        tag_invoke(let_value_t, const stream_scheduler& sch, S&& sndr, Fn fun) noexcept {
        return let_xxx_th<set_value_t, S, Fn>{{}, (S&&) sndr, (Fn&&) fun};
      }

      template <sender S, class Fn>
      friend let_xxx_th<set_error_t, S, Fn>
        tag_invoke(let_error_t, const stream_scheduler& sch, S&& sndr, Fn fun) noexcept {
        return let_xxx_th<set_error_t, S, Fn>{{}, (S&&) sndr, (Fn&&) fun};
      }

      template <sender S, class Fn>
      friend let_xxx_th<set_stopped_t, S, Fn>
        tag_invoke(let_stopped_t, const stream_scheduler& sch, S&& sndr, Fn fun) noexcept {
        return let_xxx_th<set_stopped_t, S, Fn>{{}, (S&&) sndr, (Fn&&) fun};
      }

      template <sender S, class Fn>
      friend upon_error_sender_th<S, Fn>
        tag_invoke(upon_error_t, const stream_scheduler& sch, S&& sndr, Fn fun) noexcept {
        return upon_error_sender_th<S, Fn>{{}, (S&&) sndr, (Fn&&) fun};
      }

      template <sender S, class Fn>
      friend upon_stopped_sender_th<S, Fn>
        tag_invoke(upon_stopped_t, const stream_scheduler& sch, S&& sndr, Fn fun) //
        noexcept {
        return upon_stopped_sender_th<S, Fn>{{}, (S&&) sndr, (Fn&&) fun};
      }

      template <stream_completing_sender... Senders>
      friend auto
        tag_invoke(transfer_when_all_t, const stream_scheduler& sch, Senders&&... sndrs) //
        noexcept {
        return transfer_when_all_sender_th<stream_scheduler, Senders...>(
          sch.context_state_, (Senders&&) sndrs...);
      }

      template <stream_completing_sender... Senders>
      friend auto tag_invoke(             //
        transfer_when_all_with_variant_t, //
        const stream_scheduler& sch,      //
        Senders&&... sndrs) noexcept {
        return transfer_when_all_sender_th<
          stream_scheduler,
          tag_invoke_result_t<into_variant_t, Senders>...>(
          sch.context_state_, into_variant((Senders&&) sndrs)...);
      }

      template <sender S, scheduler Sch>
      friend auto tag_invoke(transfer_t, const stream_scheduler& sch, S&& sndr, Sch&& scheduler) //
        noexcept {
        return schedule_from(
          (Sch&&) scheduler, transfer_sender_th<S>(sch.context_state_, (S&&) sndr));
      }

      template <sender S>
      friend split_sender_th<S>
        tag_invoke(split_t, const stream_scheduler& sch, S&& sndr) noexcept {
        return split_sender_th<S>(sch.context_state_, (S&&) sndr);
      }

      STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
        friend inline sender_t
        tag_invoke(schedule_t, const stream_scheduler& self) noexcept {
        return {self.context_state_};
      }

      template <__decays_to<__t> _Self, class _Receivr>
      friend auto tag_invoke(connect_t, _Self&& __self, _Receivr __rcvr)
        -> operation_state_t<stdexec::__id<__decay_t<_Receivr>>> {
        return {((_Self&&) __self).sender, (_Receivr&&) __rcvr};
      }

      friend std::true_type tag_invoke(
        __has_algorithm_customizations_t, //
        const stream_scheduler& self) noexcept {
        return {};
      }

      template <sender S>
      friend auto tag_invoke(sync_wait_t, const stream_scheduler& self, S&& sndr) {
        return _sync_wait::sync_wait_t{}(self.context_state_, (S&&) sndr);
      }

      template <class S>
      friend auto tag_invoke(sync_wait_t, const stream_scheduler& self, S&& sndr) {
        return _sync_wait::sync_wait_t{}(self.context_state_, (S&&) sndr);
      }

      friend forward_progress_guarantee
        tag_invoke(get_forward_progress_guarantee_t, const stream_scheduler&) noexcept {
        return forward_progress_guarantee::weakly_parallel;
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
    void tag_invoke(start_detached_t, Sender&& sndr) noexcept(false) {
      _submit::submit_t{}((Sender&&) sndr, _start_detached::detached_receiver_t{});
    }

    template <stream_completing_sender... Senders>
    when_all_sender_th<stream_scheduler, Senders...>
      tag_invoke(when_all_t, Senders&&... sndrs) noexcept {
      return when_all_sender_th<stream_scheduler, Senders...>{
        context_state_t{nullptr, nullptr, nullptr, nullptr},
        (Senders&&) sndrs...
      };
    }

    template <stream_completing_sender... Senders>
    when_all_sender_th< stream_scheduler, tag_invoke_result_t<into_variant_t, Senders>...>
      tag_invoke(when_all_with_variant_t, Senders&&... sndrs) noexcept {
      return when_all_sender_th< stream_scheduler, tag_invoke_result_t<into_variant_t, Senders>...>{
        context_state_t{nullptr, nullptr, nullptr, nullptr},
        into_variant((Senders&&) sndrs)...
      };
    }

    template <sender S, class Fn>
    upon_error_sender_th<S, Fn> tag_invoke(upon_error_t, S&& sndr, Fn fun) noexcept {
      return upon_error_sender_th<S, Fn>{{}, (S&&) sndr, (Fn&&) fun};
    }

    template <sender S, class Fn>
    upon_stopped_sender_th<S, Fn> tag_invoke(upon_stopped_t, S&& sndr, Fn fun) noexcept {
      return upon_stopped_sender_th<S, Fn>{{}, (S&&) sndr, (Fn&&) fun};
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
      return {STDEXEC_STREAM_DETAIL_NS::context_state_t(
        pinned_resource_.get(), managed_resource_.get(), &stream_pools_, &hub_, priority)};
    }
  };
} // namespace nvexec
