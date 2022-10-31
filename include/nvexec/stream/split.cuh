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
#include "../detail/throw_on_cuda_error.cuh"

namespace nvexec::STDEXEC_STREAM_DETAIL_NS {
  namespace split {
    template <class Tag, class Variant, class... As>
      __launch_bounds__(1)
      __global__ void copy_kernel(Variant* var, As&&... as) {
        using tuple_t = decayed_tuple<Tag, As...>;
        var->template emplace<tuple_t>(Tag{}, (As&&)as...);
      }

    template <class SenderId, class SharedState>
      class receiver_t : public stream_receiver_base {
        using Sender = stdexec::__t<SenderId>;

        SharedState &sh_state_;

      public:
        template <stdexec::__one_of<stdexec::set_value_t, 
                                    stdexec::set_error_t, 
                                    stdexec::set_stopped_t> Tag, 
                  class... As _NVCXX_CAPTURE_PACK(As)>
          friend void tag_invoke(Tag tag, receiver_t&& self, As&&... as) noexcept {
            SharedState &state = self.sh_state_;

            if constexpr (stream_sender<Sender>) {
              cudaStream_t stream = state.op_state2_.stream_;
              _NVCXX_EXPAND_PACK(As, as,
                using tuple_t = decayed_tuple<Tag, As...>;
                state.index_ = SharedState::variant_t::template index_of<tuple_t>::value;
                copy_kernel<Tag><<<1, 1, 0, stream>>>(state.data_, (As&&)as...);
              );
              state.status_ = STDEXEC_DBG_ERR(cudaEventRecord(state.event_, stream));
            } else {
              _NVCXX_EXPAND_PACK(As, as,
                using tuple_t = decayed_tuple<Tag, As...>;
                state.index_ = SharedState::variant_t::template index_of<tuple_t>::value;
              );
            }

            state.notify();
          }

        friend auto tag_invoke(stdexec::get_env_t, const receiver_t& self)
          -> exec::make_env_t<exec::with_t<stdexec::get_stop_token_t, stdexec::in_place_stop_token>> {
          return exec::make_env(stdexec::__with(stdexec::get_stop_token, self.sh_state_.stop_source_.get_token()));
        }

        explicit receiver_t(SharedState &sh_state_t) noexcept
          : sh_state_(sh_state_t) {
        }
    };

    struct operation_base_t {
      using notify_fn = void(operation_base_t*) noexcept;

      operation_base_t * next_{};
      notify_fn* notify_{};
    };

    template <class T>
      T* malloc_managed(cudaError_t &status) {
        T* ptr{};

        if (status == cudaSuccess) {
          if (status = STDEXEC_DBG_ERR(cudaMallocManaged(&ptr, sizeof(T))); status == cudaSuccess) {
            new (ptr) T();
            return ptr;
          }
        }

        return nullptr;
      }

    template <class SenderId>
      struct sh_state_t {
        using Sender = stdexec::__t<SenderId>;
        using Env = exec::make_env_t<exec::with_t<stdexec::get_stop_token_t, stdexec::in_place_stop_token>>;
        using variant_t = variant_storage_t<Sender, Env>;
        using inner_receiver_t = receiver_t<SenderId, sh_state_t>;
        using task_t = continuation_task_t<inner_receiver_t, variant_t>;
        using enqueue_receiver_t = stream_enqueue_receiver<stdexec::__x<Env>, stdexec::__x<variant_t>>;
        using intermediate_receiver = 
          stdexec::__t<
            std::conditional_t<
              stream_sender<Sender>,
              stdexec::__x<inner_receiver_t>,
              stdexec::__x<enqueue_receiver_t>>>;
        using inner_op_state_t = stdexec::connect_result_t<Sender, intermediate_receiver>;

        cudaError_t status_{cudaSuccess};
        stdexec::in_place_stop_source stop_source_{};
        std::atomic<void*> head_{nullptr};
        unsigned int index_{0};
        variant_t *data_{nullptr};
        task_t *task_{nullptr}; 
        cudaEvent_t event_;
        inner_op_state_t op_state2_;
        ::cuda::std::atomic_flag started_;

        template <stream_sender S>
          explicit sh_state_t(S& sndr, queue::task_hub_t*)
            : data_(malloc_managed<variant_t>(status_))
            , op_state2_(stdexec::connect((Sender&&) sndr, inner_receiver_t{*this}))
            , started_(ATOMIC_FLAG_INIT) {
            if (status_ == cudaSuccess) {
              status_ = STDEXEC_DBG_ERR(cudaEventCreate(&event_));
            }
          }

        template <class S>
          explicit sh_state_t(S& sndr, queue::task_hub_t* hub)
            : data_(malloc_managed<variant_t>(status_))
            , task_(queue::make_host<task_t>(status_, inner_receiver_t{*this}, data_).release())
            , op_state2_(
                stdexec::connect(
                  (Sender&&)sndr,
                  enqueue_receiver_t{
                    exec::make_env(stdexec::__with(stdexec::get_stop_token, stop_source_.get_token())), 
                    data_, 
                    task_, 
                    hub->producer()})) 
            , started_(ATOMIC_FLAG_INIT) {
          }

        ~sh_state_t() {
          if (!started_.test(::cuda::memory_order_relaxed)) {
            if (task_) {
              task_->free_(task_);
            }
          }

          if (data_) {
            STDEXEC_DBG_ERR(cudaFree(data_));
            if constexpr (stream_sender<Sender>) {
              STDEXEC_DBG_ERR(cudaEventDestroy(event_));
            }
          }
        }

        void notify() noexcept {
          void* const completion_state = static_cast<void*>(this);
          void *old = head_.exchange(completion_state, std::memory_order_acq_rel);
          operation_base_t *op_state = static_cast<operation_base_t*>(old);

          while(op_state != nullptr) {
            operation_base_t *next = op_state->next_;
            op_state->notify_(op_state);
            op_state = next;
          }
        }
      };

    template <class SenderId, class ReceiverId>
      class operation_t : public operation_base_t
                        , public operation_state_base_t<ReceiverId> {
        using Sender = stdexec::__t<SenderId>;
        using Receiver = stdexec::__t<ReceiverId>;

        struct on_stop_requested {
          stdexec::in_place_stop_source& stop_source_;
          void operator()() noexcept {
            stop_source_.request_stop();
          }
        };
        using on_stop = std::optional<typename stdexec::stop_token_of_t<
            stdexec::env_of_t<Receiver> &>::template callback_type<on_stop_requested>>;

        on_stop on_stop_{};
        std::shared_ptr<sh_state_t<SenderId>> shared_state_;

      public:
        operation_t(Receiver&& rcvr,
                    std::shared_ptr<sh_state_t<SenderId>> shared_state)
            noexcept(std::is_nothrow_move_constructible_v<Receiver>)
          : operation_base_t{nullptr, notify}
          , operation_state_base_t<ReceiverId>((Receiver&&)rcvr)
          , shared_state_(move(shared_state)) {
        }
        STDEXEC_IMMOVABLE(operation_t);

        static void notify(operation_base_t* self) noexcept {
          operation_t *op = static_cast<operation_t*>(self);
          op->on_stop_.reset();

          cudaError_t& status = op->shared_state_->status_;
          if (status == cudaSuccess) {
            if constexpr (stream_sender<Sender>) {
              status = STDEXEC_DBG_ERR(cudaStreamWaitEvent(op->stream_, op->shared_state_->event_));
            }

            visit([&](auto& tupl) noexcept -> void {
              ::cuda::std::apply([&](auto tag, auto&... args) noexcept -> void {
                op->propagate_completion_signal(tag, args...);
              }, tupl);
            }, *op->shared_state_->data_, op->shared_state_->index_);
          } else {
            op->propagate_completion_signal(stdexec::set_error, std::move(status));
          }
        }

        cudaStream_t get_stream() {
          return this->allocate();
        }

        friend void tag_invoke(stdexec::start_t, operation_t& self) noexcept {
          self.stream_ = self.get_stream();

          sh_state_t<SenderId>* shared_state = self.shared_state_.get();
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
              old, static_cast<void *>(&self),
              std::memory_order_release,
              std::memory_order_acquire));

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
  } // namespace split

  template <class SenderId>
    class split_sender_t : stream_sender_base {
      using Sender = stdexec::__t<SenderId>;
      using sh_state_ = split::sh_state_t<SenderId>;
      template <class Receiver>
        using operation_t = split::operation_t<SenderId, stdexec::__x<std::remove_cvref_t<Receiver>>>;

      Sender sndr_;
      std::shared_ptr<sh_state_> shared_state_;

    public:
      template <stdexec::__decays_to<split_sender_t> Self, stdexec::receiver Receiver>
          requires stdexec::receiver_of<Receiver, stdexec::completion_signatures_of_t<Self, stdexec::__empty_env>>
        friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& recvr)
          noexcept(std::is_nothrow_constructible_v<std::decay_t<Receiver>, Receiver>)
          -> operation_t<Receiver> {
          return operation_t<Receiver>{(Receiver &&) recvr,
                                        self.shared_state_};
        }

      template <stdexec::tag_category<stdexec::forwarding_sender_query> Tag, class... As _NVCXX_CAPTURE_PACK(As)>
          requires // Always complete on GPU, so no need in (!stdexec::__is_instance_of<Tag, stdexec::get_completion_scheduler_t>) && 
            stdexec::__callable<Tag, const Sender&, As...>
        friend auto tag_invoke(Tag tag, const split_sender_t& self, As&&... as)
          noexcept(stdexec::__nothrow_callable<Tag, const Sender&, As...>)
          -> stdexec::__call_result_if_t<stdexec::tag_category<Tag, stdexec::forwarding_sender_query>, Tag, const Sender&, As...> {
          _NVCXX_EXPAND_PACK_RETURN(As, as,
            return ((Tag&&) tag)(self.sndr_, (As&&) as...);
          )
        }

      template <class... Tys>
        using set_value_t = stdexec::completion_signatures<stdexec::set_value_t(const std::decay_t<Tys>&...)>;

      template <class Ty>
        using set_error_t = stdexec::completion_signatures<stdexec::set_error_t(const std::decay_t<Ty>&)>;

      template <stdexec::__decays_to<split_sender_t> Self, class Env>
        friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env) ->
          stdexec::make_completion_signatures<
            Sender,
            exec::make_env_t<exec::with_t<stdexec::get_stop_token_t, stdexec::in_place_stop_token>>,
            stdexec::completion_signatures<stdexec::set_error_t(cudaError_t)>,
            set_value_t,
            set_error_t>;

      explicit split_sender_t(Sender sndr, queue::task_hub_t* hub)
          : sndr_((Sender&&) sndr)
          , shared_state_{std::make_shared<sh_state_>(sndr_, hub)}
      {}
    };
}

