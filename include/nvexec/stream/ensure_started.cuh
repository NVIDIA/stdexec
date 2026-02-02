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

// clang-format Language: Cpp

#pragma once

#include "../../stdexec/execution.hpp"

// include these after execution.hpp
#include "../../exec/env.hpp"
#include "../detail/throw_on_cuda_error.cuh"
#include "common.cuh"

#include <atomic>
#include <optional>
#include <type_traits>
#include <utility>

#include <cuda/std/tuple>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

namespace nvexec::_strm {
  namespace _ensure_started {
    template <class Tag, class... Args, class Variant>
    STDEXEC_ATTRIBUTE(launch_bounds(1))
    __global__ void copy_kernel(Variant* var, Args... args) {
      static_assert(trivially_copyable<Args...>);
      using tuple_t = decayed_tuple_t<Tag, Args...>;
      var->template emplace<tuple_t>(Tag(), static_cast<Args&&>(args)...);
    }

    inline auto _make_env(
      const inplace_stop_source& stop_source,
      stream_provider* stream_provider) noexcept {
      return make_stream_env(
        exec::env_from{[&](get_stop_token_t) noexcept { return stop_source.get_token(); }},
        stream_provider);
    }

    using env_t = decltype(_ensure_started::_make_env(
      __declval<const inplace_stop_source&>(),
      static_cast<stream_provider*>(nullptr)));

    template <class Sender, class SharedState>
    class receiver : public stream_receiver_base {
      __intrusive_ptr<SharedState> shared_state_;

     public:
      explicit receiver(SharedState& shared_state) noexcept
        : shared_state_(shared_state.__intrusive_from_this()) {
      }

      template <class Tag, class... Args>
      void _set_result(Tag, Args&&... args) noexcept {
        using tuple_t = decayed_tuple_t<Tag, Args...>;
        using variant_t = typename SharedState::variant_t;

        if constexpr (stream_sender<Sender, env_t>) {
          cudaStream_t stream = shared_state_->stream_provider_.own_stream_.value();
          shared_state_->index_ = __mapply<__mfind_i<tuple_t>, variant_t>::value;
          copy_kernel<Tag, Args&&...>
            <<<1, 1, 0, stream>>>(shared_state_->data_, static_cast<Args&&>(args)...);
          shared_state_->stream_provider_
            .status_ = STDEXEC_LOG_CUDA_API(cudaEventRecord(shared_state_->event_, stream));
        } else {
          shared_state_->index_ = __mapply<__mfind_i<tuple_t>, variant_t>::value;
        }
      }

      template <class... Args>
      void set_value(Args&&... args) noexcept {
        _set_result(set_value_t(), static_cast<Args&&>(args)...);
        shared_state_->notify();
        shared_state_.reset();
      }

      template <class Error>
      void set_error(Error&& __err) noexcept {
        _set_result(set_error_t(), static_cast<Error&&>(__err));
        shared_state_->notify();
        shared_state_.reset();
      }

      void set_stopped() noexcept {
        _set_result(set_stopped_t());
        shared_state_->notify();
        shared_state_.reset();
      }

      [[nodiscard]]
      auto get_env() const noexcept -> env_t {
        return shared_state_->make_env();
      }
    };

    struct opstate_base {
      using notify_fn_t = void(opstate_base*) noexcept;
      notify_fn_t* notify_{};
    };

    template <class Type>
    auto malloc_managed(cudaError_t& status) -> Type* {
      Type* ptr{};

      if (status == cudaSuccess) {
        status = STDEXEC_LOG_CUDA_API(cudaMallocManaged(&ptr, sizeof(Type)));
        if (status == cudaSuccess) {
          new (ptr) Type();
          return ptr;
        }
      }

      return nullptr;
    }

    template <class Sender>
    struct sh_state : __enable_intrusive_from_this<sh_state<Sender>> {
      using variant_t = variant_storage_t<Sender, env_t>;
      using inner_receiver_t = receiver<Sender, sh_state>;
      using task_t = continuation_task<inner_receiver_t, variant_t>;
      using enqueue_receiver_t = stream_enqueue_receiver<env_t, variant_t>;
      using intermediate_receiver_t =
        std::conditional_t<stream_sender<Sender, env_t>, inner_receiver_t, enqueue_receiver_t>;
      using inner_opstate_t = connect_result_t<Sender, intermediate_receiver_t>;

      auto make_env() const noexcept -> env_t {
        return _ensure_started::_make_env(
          stop_source_, &const_cast<stream_provider&>(stream_provider_));
      }

      explicit sh_state(Sender& sndr, context ctx)
        requires(stream_sender<Sender, env_t>)
        : ctx_(ctx)
        , stream_provider_(false, ctx)
        , data_(malloc_managed<variant_t>(stream_provider_.status_))
        , opstate1_{nullptr}
        , opstate2_(connect(static_cast<Sender&&>(sndr), inner_receiver_t{*this})) {
        if (stream_provider_.status_ == cudaSuccess) {
          stream_provider_.status_ = STDEXEC_LOG_CUDA_API(
            cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
        }

        STDEXEC::start(opstate2_);
      }

      explicit sh_state(Sender& sndr, context ctx)
        : ctx_(ctx)
        , stream_provider_(false, ctx)
        , data_(malloc_managed<variant_t>(stream_provider_.status_))
        , task_(
            host_allocate<task_t>(
              stream_provider_.status_,
              ctx.pinned_resource_,
              inner_receiver_t{*this},
              data_,
              stream_provider_.own_stream_.value(),
              ctx.pinned_resource_)
              .release())
        , env_(host_allocate(this->stream_provider_.status_, ctx_.pinned_resource_, make_env()))
        , opstate2_(connect(
            static_cast<Sender&&>(sndr),
            enqueue_receiver_t{env_.get(), data_, task_, ctx.hub_->producer()})) {
        STDEXEC::start(opstate2_);
      }

      ~sh_state() {
        if (stream_provider_.status_ == cudaSuccess) {
          if constexpr (stream_sender<Sender, env_t>) {
            STDEXEC_ASSERT_CUDA_API(cudaEventDestroy(event_));
          }
        }

        if (data_) {
          STDEXEC_ASSERT_CUDA_API(cudaFree(data_));
        }
      }

      void notify() noexcept {
        void* const completion_state = static_cast<void*>(this);
        void* const old = opstate1_.exchange(completion_state, std::memory_order_acq_rel);
        if (old != nullptr) {
          auto* op = static_cast<opstate_base*>(old);
          op->notify_(op);
        }
      }

      void detach() noexcept {
        stop_source_.request_stop();
      }

      context ctx_;
      stream_provider stream_provider_;
      cudaEvent_t event_{};
      unsigned int index_{0};
      variant_t* data_{nullptr};
      task_t* task_{nullptr};
      inplace_stop_source stop_source_{};
      host_ptr_t<__decay_t<env_t>> env_{};
      std::atomic<void*> opstate1_;
      inner_opstate_t opstate2_;
    };

    template <class Sender, class Receiver>
    struct opstate
      : public opstate_base
      , public _strm::opstate_base<Receiver> {
      using on_stop_t = std::optional<
        stop_callback_for_t<stop_token_of_t<env_of_t<Receiver>>, __forward_stop_request>
      >;

      on_stop_t on_stop_{};
      __intrusive_ptr<sh_state<Sender>> shared_state_;

     public:
      opstate(Receiver rcvr, __intrusive_ptr<sh_state<Sender>> shared_state) noexcept
        : opstate_base{notify}
        , _strm::opstate_base<Receiver>(static_cast<Receiver&&>(rcvr), shared_state->ctx_)
        , shared_state_(std::move(shared_state)) {
      }

      ~opstate() {
        // Check to see if this operation was ever started. If not,
        // detach the (potentially still running) operation:
        if (nullptr == shared_state_->opstate1_.load(std::memory_order_acquire)) {
          shared_state_->detach();
        }
      }

      STDEXEC_IMMOVABLE(opstate);

      static void notify(opstate_base* self) noexcept {
        auto* op = static_cast<opstate*>(self);
        op->on_stop_.reset();

        cudaError_t& status = op->shared_state_->stream_provider_.status_;

        if (status == cudaSuccess) {
          if constexpr (stream_sender<Sender, env_t>) {
            status = STDEXEC_LOG_CUDA_API(
              cudaStreamWaitEvent(op->get_stream(), op->shared_state_->event_, 0));
          }

          nvexec::visit(
            [&](auto& tupl) noexcept -> void {
              ::cuda::std::apply(
                [&]<class Tag, class... Args>(Tag, Args&... args) noexcept -> void {
                  op->propagate_completion_signal(Tag(), std::move(args)...);
                },
                tupl);
            },
            *op->shared_state_->data_,
            op->shared_state_->index_);
        } else {
          op->propagate_completion_signal(STDEXEC::set_error, std::move(status));
        }
      }

      void start() & noexcept {
        sh_state<Sender>* shared_state = shared_state_.get();
        std::atomic<void*>& opstate1 = shared_state->opstate1_;
        void* const completion_state = static_cast<void*>(shared_state);
        void* const old = opstate1.load(std::memory_order_acquire);
        if (old == completion_state) {
          notify(this);
        } else {
          // register stop callback:
          on_stop_.emplace(
            get_stop_token(STDEXEC::get_env(this->rcvr_)),
            __forward_stop_request{shared_state->stop_source_});
          // Check if the stop_source has requested cancellation
          if (shared_state->stop_source_.stop_requested()) {
            // Stop has already been requested. Don't bother starting
            // the child operations.
            this->propagate_completion_signal(set_stopped_t{});
          } else {
            // Otherwise, the inner source hasn't notified completion.
            // Set this operation as the opstate1 so it's notified.
            void* old = nullptr;
            if (!opstate1.compare_exchange_weak(
                  old, this, std::memory_order_release, std::memory_order_acquire)) {
              // We get here when the task completed during the execution
              // of this function. Complete the operation synchronously.
              STDEXEC_ASSERT(old == completion_state);
              notify(this);
            }
          }
        }
      }
    };
  } // namespace _ensure_started

  template <class Sender>
  struct ensure_started_sender : stream_sender_base {
    using sender_concept = STDEXEC::sender_t;
    using sh_state_t = _ensure_started::sh_state<Sender>;
    template <class Receiver>
    using opstate_t = _ensure_started::opstate<Sender, Receiver>;

    explicit ensure_started_sender(context ctx, Sender sndr)
      : sndr_(static_cast<Sender&&>(sndr))
      , shared_state_{__make_intrusive<sh_state_t>(sndr_, ctx)} {
    }

    // Move-only:
    ensure_started_sender(ensure_started_sender&&) = default;

    ~ensure_started_sender() {
      if (nullptr != shared_state_) {
        // We're detaching a potentially running operation. Request cancellation.
        shared_state_->detach(); // BUGBUG NOT TO SPEC
      }
    }

    template <receiver Receiver, class Env = env<>>
      requires receiver_of<Receiver, completion_signatures_of_t<ensure_started_sender, Env>>
    auto connect(Receiver rcvr) && noexcept -> opstate_t<Receiver> {
      return opstate_t<Receiver>{static_cast<Receiver&&>(rcvr), std::move(shared_state_)};
    }

    auto get_env() const noexcept -> stream_sender_attrs<Sender> {
      return {&sndr_};
    }

    template <class... Tys>
    using _set_value_t = completion_signatures<set_value_t(__decay_t<Tys>...)>;

    template <class Ty>
    using _set_error_t = completion_signatures<set_error_t(__decay_t<Ty>)>;

    template <class Self, class... Env>
    static consteval auto get_completion_signatures() -> __try_make_completion_signatures<
      Sender,
      _ensure_started::env_t,
      completion_signatures<set_error_t(cudaError_t), set_stopped_t()>,
      __q<_set_value_t>,
      __q<_set_error_t>
    > {
      return {};
    }

    Sender sndr_;
    __intrusive_ptr<sh_state_t> shared_state_;
  };

  template <class Env>
  struct transform_sender_for<ensure_started_t, Env> {
    template <class Sender>
    using _sender_t = ensure_started_sender<__decay_t<Sender>>;

    template <stream_completing_sender<Env> Sender>
    auto operator()(__ignore, __ignore, Sender&& sndr) const -> _sender_t<Sender> {
      auto sched = get_completion_scheduler<set_value_t>(get_env(sndr), env_);
      return _sender_t<Sender>{sched.ctx_, static_cast<Sender&&>(sndr)};
    }

    const Env& env_;
  };
} // namespace nvexec::_strm

namespace STDEXEC::__detail {
  template <class Sender>
  extern __declfn_t<nvexec::_strm::ensure_started_sender<__demangle_t<Sender>>>
    __demangle_v<nvexec::_strm::ensure_started_sender<Sender>>;
} // namespace STDEXEC::__detail

STDEXEC_PRAGMA_POP()
