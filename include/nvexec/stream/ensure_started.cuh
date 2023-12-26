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
  namespace _ensure_started {
    template <class Tag, class... As, class Variant>
    __launch_bounds__(1) __global__ void copy_kernel(Variant* var, As... as) {
      static_assert(trivially_copyable<As...>);
      using tuple_t = decayed_tuple<Tag, As...>;
      var->template emplace<tuple_t>(Tag(), static_cast<As&&>(as)...);
    }

    inline auto __make_env(
      const in_place_stop_source& stop_source,
      stream_provider_t* stream_provider) noexcept {
      return make_stream_env(
        __env::__env_fn{[&](get_stop_token_t) noexcept {
          return stop_source.get_token();
        }},
        stream_provider);
    }

    using env_t = decltype(_ensure_started::__make_env(
      __declval<const in_place_stop_source&>(),
      static_cast<stream_provider_t*>(nullptr)));

    template <class SenderId, class SharedState>
    struct receiver_t {
      class __t : public stream_receiver_base {
        using Sender = stdexec::__t<SenderId>;

        __intrusive_ptr<SharedState> shared_state_;

       public:
        using __id = receiver_t;

        explicit __t(SharedState& shared_state) noexcept
          : shared_state_(shared_state.__intrusive_from_this()) {
        }

        template < __completion_tag Tag, class... As>
        friend void tag_invoke(Tag, __t&& self, As&&... as) noexcept {
          SharedState& state = *self.shared_state_;

          if constexpr (stream_sender<Sender, env_t>) {
            cudaStream_t stream = state.stream_provider_.own_stream_.value();
            using tuple_t = decayed_tuple<Tag, As...>;
            state.index_ = SharedState::variant_t::template index_of<tuple_t>::value;
            copy_kernel<Tag, As&&...><<<1, 1, 0, stream>>>(state.data_, (As&&) as...);
            state.stream_provider_.status_ = STDEXEC_DBG_ERR(cudaEventRecord(state.event_, stream));
          } else {
            using tuple_t = decayed_tuple<Tag, As...>;
            state.index_ = SharedState::variant_t::template index_of<tuple_t>::value;
          }

          state.notify();
          self.shared_state_.reset();
        }

        friend env_t tag_invoke(get_env_t, const __t& self) noexcept {
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

    template <class Sender>
    struct sh_state_t : __enable_intrusive_from_this<sh_state_t<Sender>> {
      using SenderId = stdexec::__id<Sender>;
      using variant_t = variant_storage_t<Sender, env_t>;
      using inner_receiver_t = stdexec::__t<receiver_t<SenderId, sh_state_t>>;
      using task_t = continuation_task_t<inner_receiver_t, variant_t>;
      using enqueue_receiver_t =
        stdexec::__t<stream_enqueue_receiver<stdexec::__cvref_id<env_t>, variant_t>>;
      using intermediate_receiver = //
        stdexec::__t< std::conditional_t<
          stream_sender<Sender, env_t>,
          stdexec::__id<inner_receiver_t>,
          stdexec::__id<enqueue_receiver_t>>>;
      using inner_op_state_t = connect_result_t<Sender, intermediate_receiver>;

      context_state_t context_state_;
      stream_provider_t stream_provider_;
      cudaEvent_t event_{};
      unsigned int index_{0};
      variant_t* data_{nullptr};
      task_t* task_{nullptr};
      in_place_stop_source stop_source_{};
      host_ptr<__decay_t<env_t>> env_{};

      std::atomic<void*> op_state1_;
      inner_op_state_t op_state2_;

      env_t make_env() const noexcept {
        return _ensure_started::__make_env(
          stop_source_, &const_cast<stream_provider_t&>(stream_provider_));
      }

      explicit sh_state_t(Sender& sndr, context_state_t context_state)
        requires(stream_sender<Sender, env_t>)
        : context_state_(context_state)
        , stream_provider_(false, context_state)
        , data_(malloc_managed<variant_t>(stream_provider_.status_))
        , op_state1_{nullptr}
        , op_state2_(connect((Sender&&) sndr, inner_receiver_t{*this})) {
        if (stream_provider_.status_ == cudaSuccess) {
          stream_provider_.status_ = STDEXEC_DBG_ERR(cudaEventCreate(&event_));
        }

        start(op_state2_);
      }

      explicit sh_state_t(Sender& sndr, context_state_t context_state)
        : context_state_(context_state)
        , stream_provider_(false, context_state)
        , data_(malloc_managed<variant_t>(stream_provider_.status_))
        , task_(make_host<task_t>(
                  stream_provider_.status_,
                  context_state.pinned_resource_,
                  inner_receiver_t{*this},
                  data_,
                  stream_provider_.own_stream_.value(),
                  context_state.pinned_resource_)
                  .release())
        , env_(
            make_host(this->stream_provider_.status_, context_state_.pinned_resource_, make_env()))
        , op_state2_(connect(
            (Sender&&) sndr,
            enqueue_receiver_t{env_.get(), data_, task_, context_state.hub_->producer()})) {
        start(op_state2_);
      }

      ~sh_state_t() {
        if (stream_provider_.status_ == cudaSuccess) {
          if constexpr (stream_sender<Sender, env_t>) {
            STDEXEC_DBG_ERR(cudaEventDestroy(event_));
          }
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
          , operation_state_base_t<ReceiverId>((Receiver&&) rcvr, shared_state->context_state_)
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

          cudaError_t& status = op->shared_state_->stream_provider_.status_;

          if (status == cudaSuccess) {
            if constexpr (stream_sender<Sender, env_t>) {
              status = STDEXEC_DBG_ERR(
                cudaStreamWaitEvent(op->get_stream(), op->shared_state_->event_, 0));
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

        friend void tag_invoke(start_t, __t& self) noexcept {
          sh_state_t<Sender>* shared_state = self.shared_state_.get();
          std::atomic<void*>& op_state1 = shared_state->op_state1_;
          void* const completion_state = static_cast<void*>(shared_state);
          void* const old = op_state1.load(std::memory_order_acquire);
          if (old == completion_state) {
            self.notify(&self);
          } else {
            // register stop callback:
            self.on_stop_.emplace(
              get_stop_token(get_env(self.rcvr_)), on_stop_requested{shared_state->stop_source_});
            // Check if the stop_source has requested cancellation
            if (shared_state->stop_source_.stop_requested()) {
              // Stop has already been requested. Don't bother starting
              // the child operations.
              self.propagate_completion_signal(set_stopped_t{});
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
    using sender_concept = stdexec::sender_t;
    using Sender = stdexec::__t<SenderId>;

    struct __t : stream_sender_base {
      using __id = ensure_started_sender_t;
      using sh_state_ = _ensure_started::sh_state_t<Sender>;
      template <class Receiver>
      using operation_t = //
        stdexec::__t< _ensure_started::operation_t<SenderId, stdexec::__id<__decay_t<Receiver>>>>;

      Sender sndr_;
      __intrusive_ptr<sh_state_> shared_state_;

      template <std::same_as<__t> Self, receiver Receiver>
        requires receiver_of<Receiver, completion_signatures_of_t<Self, empty_env>>
      friend auto tag_invoke(connect_t, Self&& self, Receiver rcvr) //
        noexcept(__nothrow_constructible_from<__decay_t<Receiver>, Receiver>)
          -> operation_t<Receiver> {
        return operation_t<Receiver>{(Receiver&&) rcvr, std::move(self).shared_state_};
      }

      friend auto tag_invoke(get_env_t, const __t& self) noexcept -> env_of_t<const Sender&> {
        return get_env(self.sndr_);
      }

      template <class... Tys>
      using _set_value_t = completion_signatures<set_value_t(__decay_t<Tys>&&...)>;

      template <class Ty>
      using _set_error_t = completion_signatures<set_error_t(__decay_t<Ty>&&)>;

      template <std::same_as<__t> Self, class Env>
      friend auto tag_invoke(get_completion_signatures_t, Self&&, Env&&)
        -> __try_make_completion_signatures<
          Sender,
          _ensure_started::env_t,
          completion_signatures<set_error_t(cudaError_t), set_stopped_t()>,
          __q<_set_value_t>,
          __q<_set_error_t>> {
        return {};
      }

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

namespace stdexec::__detail {
  template <class SenderId>
  inline constexpr __mconst<
    nvexec::STDEXEC_STREAM_DETAIL_NS::ensure_started_sender_t<__name_of<__t<SenderId>>>>
    __name_of_v<nvexec::STDEXEC_STREAM_DETAIL_NS::ensure_started_sender_t<SenderId>>{};
}
