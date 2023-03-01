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

#include <atomic>
#include "../../stdexec/execution.hpp"
#include "../../exec/env.hpp"
#include <type_traits>

#include "common.cuh"

namespace nvexec::STDEXEC_STREAM_DETAIL_NS {
  namespace split {
    using env_t = //
      make_stream_env_t< stdexec::__make_env_t<
        stdexec::__with<stdexec::get_stop_token_t, stdexec::in_place_stop_token>>>;

    template <class Tag, class Variant, class... As>
    __launch_bounds__(1) __global__ void copy_kernel(Variant* var, As&&... as) {
      using tuple_t = decayed_tuple<Tag, As...>;
      var->template emplace<tuple_t>(Tag{}, (As&&) as...);
    }

    template <class SenderId, class SharedState>
    struct receiver_t {
      class __t : public stream_receiver_base {
        using Sender = stdexec::__t<SenderId>;

        SharedState& sh_state_;

       public:
        using __id = receiver_t;

        template <
          stdexec::__one_of<stdexec::set_value_t, stdexec::set_error_t, stdexec::set_stopped_t> Tag,
          class... As>
        friend void tag_invoke(Tag tag, __t&& self, As&&... as) noexcept {
          SharedState& state = self.sh_state_;

          if constexpr (stream_sender<Sender>) {
            cudaStream_t stream = state.op_state2_.get_stream();
            using tuple_t = decayed_tuple<Tag, As...>;
            state.index_ = SharedState::variant_t::template index_of<tuple_t>::value;
            copy_kernel<Tag><<<1, 1, 0, stream>>>(state.data_, (As&&) as...);
            state.status_ = STDEXEC_CHECK_CUDA_ERROR(cudaEventRecord(state.event_, stream));
          } else {
            using tuple_t = decayed_tuple<Tag, As...>;
            state.index_ = SharedState::variant_t::template index_of<tuple_t>::value;
          }

          state.notify();
        }

        friend env_t tag_invoke(stdexec::get_env_t, const __t& self) {
          return self.sh_state_.make_env();
        }

        explicit __t(SharedState& sh_state_t) noexcept
          : sh_state_(sh_state_t) {
        }
      };
    };

    struct operation_base_t {
      using notify_fn = void(operation_base_t*) noexcept;

      operation_base_t* next_{};
      notify_fn* notify_{};
    };

    template <class T>
    T* malloc_managed(cudaError_t& status) {
      T* ptr{};

      if (status == cudaSuccess) {
        if (status = STDEXEC_CHECK_CUDA_ERROR(cudaMallocManaged(&ptr, sizeof(T))); status == cudaSuccess) {
          new (ptr) T();
          return ptr;
        }
      }

      return nullptr;
    }

    inline cudaStream_t create_stream(cudaError_t& status, context_state_t context_state) {
      cudaStream_t stream{};

      if (status == cudaSuccess) {
        std::tie(stream, status) = create_stream_with_priority(context_state.priority_);
      }

      return stream;
    }

    template <class Sender>
    struct sh_state_t {
      using variant_t = variant_storage_t<Sender, env_t>;
      using inner_receiver_t = stdexec::__t<receiver_t<stdexec::__id<Sender>, sh_state_t>>;
      using task_t = continuation_task_t<inner_receiver_t, variant_t>;
      using enqueue_receiver_t =
        stdexec::__t<stream_enqueue_receiver<stdexec::__x<env_t>, stdexec::__x<variant_t>>>;
      using intermediate_receiver = //
        stdexec::__t< std::conditional_t<
          stream_sender<Sender>,
          stdexec::__id<inner_receiver_t>,
          stdexec::__id<enqueue_receiver_t>>>;
      using inner_op_state_t = stdexec::connect_result_t<Sender, intermediate_receiver>;

      context_state_t context_state_;
      cudaError_t status_{cudaSuccess};
      cudaStream_t stream_{};

      stdexec::in_place_stop_source stop_source_{};
      std::atomic<void*> head_{nullptr};
      unsigned int index_{0};
      variant_t* data_{nullptr};
      task_t* task_{nullptr};
      cudaEvent_t event_;
      inner_op_state_t op_state2_;
      ::cuda::std::atomic_flag started_{};

      explicit sh_state_t(Sender& sndr, context_state_t context_state)
        requires(stream_sender<Sender>)
        : context_state_(context_state)
        , stream_(create_stream(status_, context_state_))
        , data_(malloc_managed<variant_t>(status_))
        , op_state2_(stdexec::connect((Sender&&) sndr, inner_receiver_t{*this})) {
        if (status_ == cudaSuccess) {
          status_ = STDEXEC_CHECK_CUDA_ERROR(cudaEventCreate(&event_));
        }
      }

      explicit sh_state_t(Sender& sndr, context_state_t context_state)
        : context_state_(context_state)
        , stream_(create_stream(status_, context_state_))
        , data_(malloc_managed<variant_t>(status_))
        , task_(queue::make_host<task_t>(
                  status_,
                  context_state.pinned_resource_,
                  inner_receiver_t{*this},
                  data_,
                  stream_,
                  context_state.pinned_resource_)
                  .release())
        , op_state2_(stdexec::connect(
            (Sender&&) sndr,
            enqueue_receiver_t{make_env(), data_, task_, context_state.hub_->producer()})) {
      }

      ~sh_state_t() {
        if (!started_.test(::cuda::memory_order_relaxed)) {
          if (task_) {
            task_->free_(task_);
          }
        }

        if (data_) {
          STDEXEC_CHECK_CUDA_ERROR(cudaFree(data_));
          if constexpr (stream_sender<Sender>) {
            STDEXEC_CHECK_CUDA_ERROR(cudaEventDestroy(event_));
          }
          STDEXEC_CHECK_CUDA_ERROR(cudaStreamDestroy(stream_));
        }
      }

      env_t make_env() const {
        return make_stream_env(
          stdexec::__make_env(stdexec::__with_(stdexec::get_stop_token, stop_source_.get_token())),
          stream_);
      }

      void notify() noexcept {
        void* const completion_state = static_cast<void*>(this);
        void* old = head_.exchange(completion_state, std::memory_order_acq_rel);
        operation_base_t* op_state = static_cast<operation_base_t*>(old);

        while (op_state != nullptr) {
          operation_base_t* next = op_state->next_;
          op_state->notify_(op_state);
          op_state = next;
        }
      }
    };

    template <class SenderId, class ReceiverId>
    struct operation_t {
      class __t
        : public operation_base_t
        , public operation_state_base_t<ReceiverId> {
        using __id = operation_t;
        using Sender = stdexec::__t<SenderId>;
        using Receiver = stdexec::__t<ReceiverId>;

        struct on_stop_requested {
          stdexec::in_place_stop_source& stop_source_;

          void operator()() noexcept {
            stop_source_.request_stop();
          }
        };

        using on_stop = //
          std::optional< typename stdexec::stop_token_of_t<
            stdexec::env_of_t<Receiver>&>::template callback_type<on_stop_requested>>;

        on_stop on_stop_{};
        std::shared_ptr<sh_state_t<Sender>> shared_state_;

       public:
        __t(Receiver&& rcvr, std::shared_ptr<sh_state_t<Sender>> shared_state) //
          noexcept(std::is_nothrow_move_constructible_v<Receiver>)
          : operation_base_t{nullptr, notify}
          , operation_state_base_t<ReceiverId>(
              (Receiver&&) rcvr,
              shared_state->context_state_,
              false)
          , shared_state_(std::move(shared_state)) {
        }

        STDEXEC_IMMOVABLE(__t);

        static void notify(operation_base_t* self) noexcept {
          __t* op = static_cast<__t*>(self);
          op->on_stop_.reset();

          cudaError_t& status = op->shared_state_->status_;
          if (status == cudaSuccess) {
            if constexpr (stream_sender<Sender>) {
              status = STDEXEC_CHECK_CUDA_ERROR(
                cudaStreamWaitEvent(op->get_stream(), op->shared_state_->event_));
            }

            visit(
              [&](auto& tupl) noexcept -> void {
                ::cuda::std::apply(
                  [&](auto tag, auto&... args) noexcept -> void {
                    op->propagate_completion_signal(tag, args...);
                  },
                  tupl);
              },
              *op->shared_state_->data_,
              op->shared_state_->index_);
          } else {
            op->propagate_completion_signal(stdexec::set_error, std::move(status));
          }
        }

        friend void tag_invoke(stdexec::start_t, __t& self) noexcept {
          sh_state_t<Sender>* shared_state = self.shared_state_.get();
          std::atomic<void*>& head = shared_state->head_;
          void* const completion_state = static_cast<void*>(shared_state);
          void* old = head.load(std::memory_order_acquire);

          if (old != completion_state) {
            self.on_stop_.emplace(
              stdexec::get_stop_token(stdexec::get_env(self.receiver_)),
              on_stop_requested{shared_state->stop_source_});
          }

          do {
            if (old == completion_state) {
              self.notify(&self);
              return;
            }
            self.next_ = static_cast<operation_base_t*>(old);
          } while (!head.compare_exchange_weak(
            old, static_cast<void*>(&self), std::memory_order_release, std::memory_order_acquire));

          if (old == nullptr) {
            // the inner sender isn't running
            if (shared_state->stop_source_.stop_requested()) {
              // 1. resets head to completion state
              // 2. notifies waiting threads
              // 3. propagates "stopped" signal to `out_r'`
              shared_state->notify();
            } else {
              shared_state->started_.test_and_set(::cuda::memory_order_relaxed);
              stdexec::start(shared_state->op_state2_);
            }
          }
        }
      };
    };
  } // namespace split

  template <class SenderId>
  struct split_sender_t {
    using is_sender = void;
    using Sender = stdexec::__t<SenderId>;
    using sh_state_ = split::sh_state_t<Sender>;

    struct __t : stream_sender_base {
      using __id = split_sender_t;
      template <class Receiver>
      using operation_t =
        stdexec::__t<split::operation_t<SenderId, stdexec::__id<std::remove_cvref_t<Receiver>>>>;

      Sender sndr_;
      std::shared_ptr<sh_state_> shared_state_;

      template <stdexec::__decays_to<__t> Self, stdexec::receiver Receiver>
        requires stdexec::
          receiver_of<Receiver, stdexec::completion_signatures_of_t<Self, stdexec::empty_env>>
        friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& recvr) //
        noexcept(std::is_nothrow_constructible_v<std::decay_t<Receiver>, Receiver>)
          -> operation_t<Receiver> {
        return operation_t<Receiver>{(Receiver&&) recvr, self.shared_state_};
      }

      friend auto tag_invoke(stdexec::get_env_t, const __t& self) //
        noexcept(stdexec::__nothrow_callable<stdexec::get_env_t, const Sender&>)
          -> stdexec::__call_result_t<stdexec::get_env_t, const Sender&> {
        return stdexec::get_env(self.sndr_);
      }

      template <class... Tys>
      using set_value_t =
        stdexec::completion_signatures<stdexec::set_value_t(const std::decay_t<Tys>&...)>;

      template <class Ty>
      using set_error_t =
        stdexec::completion_signatures<stdexec::set_error_t(const std::decay_t<Ty>&)>;

      template <stdexec::__decays_to<__t> Self, class Env>
      friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
        -> stdexec::make_completion_signatures<
          Sender,
          exec::make_env_t<exec::with_t<stdexec::get_stop_token_t, stdexec::in_place_stop_token>>,
          stdexec::completion_signatures<stdexec::set_error_t(cudaError_t)>,
          set_value_t,
          set_error_t>;

      explicit __t(context_state_t context_state, Sender sndr)
        : sndr_((Sender&&) sndr)
        , shared_state_{std::make_shared<sh_state_>(sndr_, context_state)} {
      }
    };
  };
}
