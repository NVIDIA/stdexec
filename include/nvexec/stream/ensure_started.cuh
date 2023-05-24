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
  namespace __ensure_started {
    template <class Tag, class... As, class Variant>
    __launch_bounds__(1) __global__ void copy_kernel(Variant* var, As... as) {
      using tuple_t = decayed_tuple<Tag, As...>;
      var->template emplace<tuple_t>(Tag(), static_cast<As&&>(as)...);
    }

    inline auto __make_env(const in_place_stop_source& stop_source, cudaStream_t stream) noexcept {
      return make_stream_env(
        __env::__env_fn{[&](get_stop_token_t) noexcept {
          return stop_source.get_token();
        }},
        stream);
    }

    using env_t = decltype(_ensure_started::__make_env(
      __declval<const in_place_stop_source&>(),
      cudaStream_t()));

    template <class SenderId, class SharedState>
    struct receiver_t {
      class __t : stream_receiver_base {
        using Sender = stdexec::__t<SenderId>;

        __intrusive_ptr<SharedState> shared_state_;

       public:
        using __id = receiver_t;

        explicit __t(SharedState& shared_state) noexcept
          : shared_state_(shared_state.__intrusive_from_this()) {
        }

        template <class Tag, class... As>
        static void complete_(Tag, __t&& self, As&&... as) noexcept {
          SharedState& state = *self.shared_state_;

          if constexpr (stream_sender<Sender>) {
            cudaStream_t stream = state.stream_;
            using tuple_t = decayed_tuple<Tag, As...>;
            state.index_ = SharedState::variant_t::template index_of<tuple_t>::value;
            copy_kernel<Tag, As&&...><<<1, 1, 0, stream>>>(state.data_, (As&&) as...);
            state.status_ = STDEXEC_DBG_ERR(cudaEventRecord(state.event_, stream));
          } else {
            using tuple_t = decayed_tuple<Tag, As...>;
            state.index_ = SharedState::variant_t::template index_of<tuple_t>::value;
          }

          state.notify();
          self.shared_state_.reset();
        }

        template <same_as<set_value_t> Tag, class... Args>
          requires __result_constructible_from<SharedState, Tag, Args...>
        STDEXEC_DEFINE_CUSTOM(void set_value)(this __t&& self, Tag, Args&&... args) noexcept {
          complete_(Tag(), (__t&&) self, (Args&&) args...);
        }

        template <same_as<set_error_t> Tag, class Error>
          requires __result_constructible_from<SharedState, Tag, Error>
        STDEXEC_DEFINE_CUSTOM(void set_error)(this __t&& self, Tag, Error&& err) noexcept {
          complete_(Tag(), (__t&&) self, (Error&&) err);
        }

        template <same_as<set_stopped_t> Tag>
          requires __result_constructible_from<SharedState, Tag>
        STDEXEC_DEFINE_CUSTOM(void set_stopped)(this __t&& self, Tag) noexcept {
          complete_(Tag(), (__t&&) self);
        }

        STDEXEC_DEFINE_CUSTOM(env_t get_env)(this const __t& self, get_env_t) noexcept {
          return self.shared_state_->make_env();
        }
      };
    };

    struct operation_base_t {
      using notify_fn = void(operation_base_t*) noexcept;
      notify_fn* notify_{};
    };

    template <class T>
    T* malloc_managed(cudaError_t& status) {
      T* ptr{};

      if (status == cudaSuccess) {
        if (status = STDEXEC_DBG_ERR(cudaMallocManaged(&ptr, sizeof(T))); status == cudaSuccess) {
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
    struct sh_state_t : __enable_intrusive_from_this<sh_state_t<Sender>> {
      using SenderId = stdexec::__id<Sender>;
      using variant_t = variant_storage_t<Sender, env_t>;
      using inner_receiver_t = stdexec::__t<receiver_t<SenderId, sh_state_t>>;
      using task_t = continuation_task_t<inner_receiver_t, variant_t>;
      using enqueue_receiver_t =
        stdexec::__t<stream_enqueue_receiver<stdexec::__id<env_t>, variant_t>>;
      using intermediate_receiver = //
        stdexec::__t< std::conditional_t<
          stream_sender<Sender>,
          stdexec::__id<inner_receiver_t>,
          stdexec::__id<enqueue_receiver_t>>>;
      using inner_op_state_t = connect_result_t<Sender, intermediate_receiver>;

      context_state_t context_state_;
      cudaError_t status_{cudaSuccess};
      cudaStream_t stream_{};
      cudaEvent_t event_{};
      unsigned int index_{0};
      variant_t* data_{nullptr};
      task_t* task_{nullptr};
      in_place_stop_source stop_source_{};

      std::atomic<void*> op_state1_;
      inner_op_state_t op_state2_;

      env_t make_env() const noexcept {
        return _ensure_started::__make_env(stop_source_, stream_);
      }

      explicit sh_state_t(Sender& sndr, context_state_t context_state)
        requires(stream_sender<Sender>)
        : context_state_(context_state)
        , stream_(create_stream(status_, context_state_))
        , data_(malloc_managed<variant_t>(status_))
        , op_state1_{nullptr}
        , op_state2_(connect((Sender&&) sndr, inner_receiver_t{*this})) {
        if (status_ == cudaSuccess) {
          status_ = STDEXEC_DBG_ERR(cudaEventCreate(&event_));
        }

        stdexec::start(op_state2_);
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
        , op_state2_(connect(
            (Sender&&) sndr,
            enqueue_receiver_t{make_env(), data_, task_, context_state.hub_->producer()})) {
        stdexec::start(op_state2_);
      }

      ~sh_state_t() {
        if (status_ == cudaSuccess) {
          if constexpr (stream_sender<Sender>) {
            STDEXEC_DBG_ERR(cudaEventDestroy(event_));
          }
          STDEXEC_DBG_ERR(cudaStreamDestroy(stream_));
        }

        if (data_) {
          STDEXEC_DBG_ERR(cudaFree(data_));
        }
      }

      void notify() noexcept {
        void* const completion_state = static_cast<void*>(this);
        void* const old = op_state1_.exchange(completion_state, std::memory_order_acq_rel);
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
    struct operation_t {
      class __t
        : public operation_base_t
        , public operation_state_base_t<ReceiverId> {
        using Sender = stdexec::__t<SenderId>;
        using Receiver = stdexec::__t<ReceiverId>;

        struct on_stop_requested {
          in_place_stop_source& stop_source_;

          void operator()() noexcept {
            stop_source_.request_stop();
          }
        };

        using on_stop = //
          std::optional< typename stop_token_of_t< env_of_t<Receiver>&>::template callback_type<
            on_stop_requested>>;

        on_stop on_stop_{};
        __intrusive_ptr<sh_state_t<Sender>> shared_state_;

       public:
        using __id = operation_t;

        __t(Receiver rcvr, __intrusive_ptr<sh_state_t<Sender>> shared_state) //
          noexcept(std::is_nothrow_move_constructible_v<Receiver>)
          : operation_base_t{notify}
          , operation_state_base_t<ReceiverId>(
              (Receiver&&) rcvr,
              shared_state->context_state_,
              false)
          , shared_state_(std::move(shared_state)) {
        }

        ~__t() {
          // Check to see if this operation was ever started. If not,
          // detach the (potentially still running) operation:
          if (nullptr == shared_state_->op_state1_.load(std::memory_order_acquire)) {
            shared_state_->detach();
          }
        }

        STDEXEC_IMMOVABLE(__t);

        static void notify(operation_base_t* self) noexcept {
          __t* op = static_cast<__t*>(self);
          op->on_stop_.reset();

          cudaError_t& status = op->shared_state_->status_;

          if (status == cudaSuccess) {
            if constexpr (stream_sender<Sender>) {
              status = STDEXEC_DBG_ERR(
                cudaStreamWaitEvent(op->get_stream(), op->shared_state_->event_));
            }

            visit(
              [&](auto& tupl) noexcept -> void {
                ::cuda::std::apply(
                  [&]<class Tag, class... As>(Tag, As&... args) noexcept -> void {
                    op->propagate_completion_signal(Tag(), std::move(args)...);
                  },
                  tupl);
              },
              *op->shared_state_->data_,
              op->shared_state_->index_);
          } else {
            op->propagate_completion_signal(stdexec::set_error, std::move(status));
          }
        }

        STDEXEC_DEFINE_CUSTOM(void start)(this __t& self, start_t) noexcept {
          sh_state_t<Sender>* shared_state = self.shared_state_.get();
          std::atomic<void*>& op_state1 = shared_state->op_state1_;
          void* const completion_state = static_cast<void*>(shared_state);
          void* const old = op_state1.load(std::memory_order_acquire);
          if (old == completion_state) {
            self.notify(&self);
          } else {
            // register stop callback:
            self.on_stop_.emplace(
              get_stop_token(stdexec::get_env(self.receiver_)),
              on_stop_requested{shared_state->stop_source_});
            // Check if the stop_source has requested cancellation
            if (shared_state->stop_source_.stop_requested()) {
              // Stop has already been requested. Don't bother starting
              // the child operations.
              self.propagate_completion_signal(stdexec::set_stopped);
            } else {
              // Otherwise, the inner source hasn't notified completion.
              // Set this operation as the op_state1 so it's notified.
              void* old = nullptr;
              if (!op_state1.compare_exchange_weak(
                    old, &self, std::memory_order_release, std::memory_order_acquire)) {
                // We get here when the task completed during the execution
                // of this function. Complete the operation synchronously.
                STDEXEC_ASSERT(old == completion_state);
                self.notify(&self);
              }
            }
          }
        }
      };
    };
  }

  template <class SenderId>
  struct ensure_started_sender_t {
    using is_sender = void;
    using Sender = stdexec::__t<SenderId>;

    struct __t : stream_sender_base {
      using __id = ensure_started_sender_t;
      using sh_state_ = __ensure_started::sh_state_t<Sender>;
      template <class Receiver>
      using operation_t = //
        stdexec::__t< __ensure_started::operation_t<SenderId, stdexec::__id<__decay_t<Receiver>>>>;

      Sender sndr_;
      __intrusive_ptr<sh_state_> shared_state_;

      template <std::same_as<__t> Self, receiver Receiver>
        requires receiver_of<Receiver, completion_signatures_of_t<Self, empty_env>>
      friend auto tag_invoke(connect_t, Self&& self, Receiver&& rcvr) //
        noexcept(__nothrow_constructible_from<__decay_t<Receiver>, Receiver>)
          -> operation_t<Receiver> {
        return operation_t<Receiver>{(Receiver&&) rcvr, std::move(self).shared_state_};
      }

      STDEXEC_DEFINE_CUSTOM(auto get_env)(this const __t& self, get_env_t) noexcept {
        return stdexec::get_env(self.sndr_);
      }

      template <class... Tys>
      using _set_value_t = completion_signatures<set_value_t(__decay_t<Tys>&&...)>;

      template <class Ty>
      using _set_error_t = completion_signatures<set_error_t(__decay_t<Ty>&&)>;

      template <std::same_as<__t> Self, class Env>
      STDEXEC_DEFINE_CUSTOM(auto get_completion_signatures)(this Self&&, get_completion_signatures_t, Env&&)
        -> make_completion_signatures<
          Sender,
          __ensure_started::env_t,
          completion_signatures<set_error_t(cudaError_t), set_stopped_t()>,
          _set_value_t,
          _set_error_t>;

      explicit __t(context_state_t context_state, Sender sndr)
        : sndr_((Sender&&) sndr)
        , shared_state_{__make_intrusive<sh_state_>(sndr_, context_state)} {
      }

      ~__t() {
        if (nullptr != shared_state_) {
          // We're detaching a potentially running operation. Request cancellation.
          shared_state_->detach(); // BUGBUG NOT TO SPEC
        }
      }

      // Move-only:
      __t(__t&&) = default;
    };
  };
}
