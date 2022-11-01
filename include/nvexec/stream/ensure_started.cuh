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

#include "../detail/throw_on_cuda_error.cuh"
#include "common.cuh"

namespace nvexec::STDEXEC_STREAM_DETAIL_NS {
  namespace ensure_started {
    template <class Tag, class Variant, class... As>
      __launch_bounds__(1)
      __global__ void copy_kernel(Variant* var, As&&... as) {
        using tuple_t = decayed_tuple<Tag, As...>;
        var->template emplace<tuple_t>(Tag{}, (As&&)as...);
      }

    using env_t = 
      stdexec::__make_env_t<
        stdexec::__with_t<stdexec::get_stop_token_t, stdexec::in_place_stop_token>>;

    template <class SenderId, class SharedState>
      class receiver_t : stream_receiver_base {
        using Sender = stdexec::__t<SenderId>;

        stdexec::__intrusive_ptr<SharedState> shared_state_;

      public:
        explicit receiver_t(SharedState& shared_state) noexcept
          : shared_state_(shared_state.__intrusive_from_this()) {
        }

        template <stdexec::__one_of<stdexec::set_value_t, 
                                    stdexec::set_error_t, 
                                    stdexec::set_stopped_t> Tag, 
                  class... As _NVCXX_CAPTURE_PACK(As)>
          friend void tag_invoke(Tag tag, receiver_t&& self, As&&... as) noexcept {
            SharedState& state = *self.shared_state_;

            if constexpr (stream_sender<Sender>) {
              cudaStream_t stream = state.op_state2_.stream_;
              _NVCXX_EXPAND_PACK(As, as,
                using tuple_t = decayed_tuple<Tag, As...>;
                state.index_ = SharedState::variant_t::template index_of<tuple_t>::value;
                copy_kernel<Tag><<<1, 1, 0, stream>>>(state.data_, (As&&)as...);
              )
            } else {
              _NVCXX_EXPAND_PACK(As, as,
                using tuple_t = decayed_tuple<Tag, As...>;
                state.index_ = SharedState::variant_t::template index_of<tuple_t>::value;
              );
            }

            state.notify();
            self.shared_state_.reset();
          }

          friend env_t tag_invoke(stdexec::get_env_t, const receiver_t& self) {
            auto stok = self.shared_state_->stop_source_.get_token();
            return stdexec::__make_env(stdexec::__with(stdexec::get_stop_token, std::move(stok)));
          }
        };

    struct operation_base_t {
      using notify_fn = void(operation_base_t*) noexcept;
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
      struct sh_state_t : stdexec::__enable_intrusive_from_this<sh_state_t<SenderId>> {
        using Sender = stdexec::__t<SenderId>;
        using variant_t = variant_storage_t<Sender, env_t>;

        using inner_receiver_t = receiver_t<SenderId, sh_state_t>;
        using task_t = continuation_task_t<inner_receiver_t, variant_t>;
        using enqueue_receiver_t = stream_enqueue_receiver<stdexec::__x<env_t>, stdexec::__x<variant_t>>;
        using intermediate_receiver = 
          stdexec::__t<
            std::conditional_t<
              stream_sender<Sender>,
              stdexec::__x<inner_receiver_t>,
              stdexec::__x<enqueue_receiver_t>>>;
        using inner_op_state_t = stdexec::connect_result_t<Sender, intermediate_receiver>;

        cudaError_t status_{cudaSuccess};
        unsigned int index_{0};
        variant_t *data_{nullptr};
        task_t *task_{nullptr}; 
        stdexec::in_place_stop_source stop_source_{};

        std::atomic<void*> op_state1_;
        inner_op_state_t op_state2_;

        template <stream_sender S>
          explicit sh_state_t(S& sndr, queue::task_hub_t*)
            : data_(malloc_managed<variant_t>(status_))
            , op_state1_{nullptr}
            , op_state2_(stdexec::connect((Sender&&) sndr, inner_receiver_t{*this})) {
              stdexec::start(op_state2_);
          }

        template <class S>
          explicit sh_state_t(S& sndr, queue::task_hub_t* hub)
            : data_(malloc_managed<variant_t>(status_))
            , task_(queue::make_host<task_t>(status_, inner_receiver_t{*this}, data_).release())
            , op_state2_(
                stdexec::connect(
                  (Sender&&)sndr,
                  enqueue_receiver_t{
                    stdexec::__make_env(stdexec::__with(stdexec::get_stop_token, std::move(stop_source_.get_token()))),
                    data_, 
                    task_, 
                    hub->producer()})) {
              stdexec::start(op_state2_);
          }

        ~sh_state_t() {
          if (data_) {
            STDEXEC_DBG_ERR(cudaFree(data_));
          }
        }

        void notify() noexcept {
          void* const completion_state = static_cast<void*>(this);
          void* const old =
            op_state1_.exchange(completion_state, std::memory_order_acq_rel);
          if (old != nullptr) {
            auto* op = static_cast<operation_base_t*>(old);
            op->notify_(op);
          }
        }

        void detach() noexcept {
          stop_source_.request_stop();
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
        stdexec::__intrusive_ptr<sh_state_t<SenderId>> shared_state_;

      public:
        operation_t(Receiver rcvr,
                    stdexec::__intrusive_ptr<sh_state_t<SenderId>> shared_state)
            noexcept(std::is_nothrow_move_constructible_v<Receiver>)
          : operation_base_t{notify}
          , operation_state_base_t<ReceiverId>((Receiver&&)rcvr)
          , shared_state_(std::move(shared_state)) {
        }
        ~operation_t() {
          // Check to see if this operation was ever started. If not,
          // detach the (potentially still running) operation:
          if (nullptr == shared_state_->op_state1_.load(std::memory_order_acquire)) {
            shared_state_->detach();
          }
        }
        STDEXEC_IMMOVABLE(operation_t);

        static void notify(operation_base_t* self) noexcept {
          operation_t *op = static_cast<operation_t*>(self);
          op->on_stop_.reset();

          visit([&](auto& tupl) noexcept -> void {
            ::cuda::std::apply([&](auto tag, auto&... args) noexcept -> void {
              op->propagate_completion_signal(tag, args...);
            }, tupl);
          }, *op->shared_state_->data_, op->shared_state_->index_);
        }

        cudaStream_t get_stream() {
          cudaStream_t stream{};

          std::optional<cudaStream_t> env_stream = 
            nvexec::STDEXEC_STREAM_DETAIL_NS::get_stream(stdexec::get_env(this->receiver_));

          if (env_stream) {
            stream = *env_stream;
          } else {
            using inner_op_state_t = typename sh_state_t<SenderId>::inner_op_state_t;
            if constexpr (std::is_base_of_v<stream_op_state_base, inner_op_state_t>) {
              stream = shared_state_->op_state2_.get_stream();
            } else {
              stream = this->allocate();
            }
          }

          return stream;
        }

        friend void tag_invoke(stdexec::start_t, operation_t& self) noexcept {
          self.stream_ = self.get_stream();

          sh_state_t<SenderId>* shared_state = self.shared_state_.get();
          std::atomic<void*>& op_state1 = shared_state->op_state1_;
          void* const completion_state = static_cast<void*>(shared_state);
          void* const old = op_state1.load(std::memory_order_acquire);
          if (old == completion_state) {
            self.notify(&self);
          } else {
            // register stop callback:
            self.on_stop_.emplace(
                stdexec::get_stop_token(stdexec::get_env(self.receiver_)),
                on_stop_requested{shared_state->stop_source_});
            // Check if the stop_source has requested cancellation
            if (shared_state->stop_source_.stop_requested()) {
              // Stop has already been requested. Don't bother starting
              // the child operations.
              self.propagate_completion_signal(stdexec::set_stopped_t{});
            } else {
              // Otherwise, the inner source hasn't notified completion.
              // Set this operation as the op_state1 so it's notified.
              void* old = nullptr;
              if (!op_state1.compare_exchange_weak(
                old, &self,
                std::memory_order_release,
                std::memory_order_acquire)) {
                // We get here when the task completed during the execution
                // of this function. Complete the operation synchronously.
                STDEXEC_ASSERT(old == completion_state);
                self.notify(&self);
              }
            }
          }
        }
      };
    }

  template <class SenderId>
    class ensure_started_sender_t : stream_sender_base {
      using Sender = stdexec::__t<SenderId>;
      using sh_state_ = ensure_started::sh_state_t<SenderId>;
      template <class Receiver>
        using operation_t = ensure_started::operation_t<SenderId, stdexec::__x<std::remove_cvref_t<Receiver>>>;

      Sender sndr_;
      stdexec::__intrusive_ptr<sh_state_> shared_state_;

      template <std::same_as<ensure_started_sender_t> Self, stdexec::receiver Receiver>
          requires stdexec::receiver_of<Receiver, stdexec::completion_signatures_of_t<Self, stdexec::__empty_env>>
        friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr)
          noexcept(std::is_nothrow_constructible_v<std::decay_t<Receiver>, Receiver>)
          -> operation_t<Receiver> {
          return operation_t<Receiver>{(Receiver &&) rcvr,
                                        std::move(self).shared_state_};
        }

      template <stdexec::tag_category<stdexec::forwarding_sender_query> Tag, class... As>
          requires // Always complete on GPU, so no need in (!stdexec::__is_instance_of<Tag, stdexec::get_completion_scheduler_t>) &&
            stdexec::__callable<Tag, const Sender&, As...>
        friend auto tag_invoke(Tag tag, const ensure_started_sender_t& self, As&&... as)
          noexcept(stdexec::__nothrow_callable<Tag, const Sender&, As...>)
          -> stdexec::__call_result_if_t<stdexec::tag_category<Tag, stdexec::forwarding_sender_query>, Tag, const Sender&, As...> {
          return ((Tag&&) tag)(self.sndr_, (As&&) as...);
        }

      template <class... Tys>
        using set_value_t = 
          stdexec::completion_signatures<stdexec::set_value_t(const std::decay_t<Tys>&...)>;

      template <class Ty>
        using set_error_t = 
          stdexec::completion_signatures<stdexec::set_error_t(const std::decay_t<Ty>&)>;

      template <std::same_as<ensure_started_sender_t> Self, class Env>
        friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env) ->
          stdexec::make_completion_signatures<
            Sender,
            ensure_started::env_t,
            stdexec::completion_signatures<stdexec::set_error_t(cudaError_t),
                                                  stdexec::set_stopped_t()>, 
            set_value_t,
            set_error_t>;

     public:
      explicit ensure_started_sender_t(Sender sndr, queue::task_hub_t* hub)
        : sndr_((Sender&&) sndr)
        , shared_state_{stdexec::__make_intrusive<sh_state_>(sndr_, hub)}
      {}
      ~ensure_started_sender_t() {
        if (nullptr != shared_state_) {
          // We're detaching a potentially running operation. Request cancellation.
          shared_state_->detach(); // BUGBUG NOT TO SPEC
        }
      }
      // Move-only:
      ensure_started_sender_t(ensure_started_sender_t&&) = default;
    };
}

