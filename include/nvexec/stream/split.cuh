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
#include "../detail/cuda_atomic.cuh" // IWYU pragma: keep
#include "../detail/throw_on_cuda_error.cuh"
#include "common.cuh"

#include <atomic>
#include <memory>
#include <optional>
#include <type_traits>

#include <cuda/std/tuple>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

namespace nvexec::_strm {
  namespace _split {
    inline auto _make_env(
      const inplace_stop_source& stop_source,
      stream_provider* stream_provider) noexcept {
      return make_stream_env(
        exec::env_from{[&](get_stop_token_t) noexcept { return stop_source.get_token(); }},
        stream_provider);
    }

    using env_t = decltype(_split::_make_env(
      __declval<const inplace_stop_source&>(),
      static_cast<stream_provider*>(nullptr)));

    template <class Tag, class... Args, class Variant>
    STDEXEC_ATTRIBUTE(launch_bounds(1))
    __global__ void copy_kernel(Variant* var, Args... args) {
      static_assert(trivially_copyable<Args...>);
      using tuple_t = decayed_tuple_t<Tag, Args...>;
      var->template emplace<tuple_t>(Tag(), static_cast<Args&&>(args)...);
    }

    template <class Sender, class SharedState>
    struct receiver : stream_receiver_base {
      using receiver_concept = STDEXEC::receiver_t;

      explicit receiver(SharedState& sh_state_t) noexcept
        : sh_state_(sh_state_t) {
      }

      template <class Tag, class... Args>
      void set_result(Tag, Args&&... args) noexcept {
        using tuple_t = decayed_tuple_t<Tag, Args...>;
        using variant_t = typename SharedState::variant_t;

        if constexpr (stream_sender<Sender, env_t>) {
          cudaStream_t stream = sh_state_.opstate2_.get_stream();
          sh_state_.index_ = __mapply<__mfind_i<tuple_t>, variant_t>::value;
          copy_kernel<Tag, Args&&...><<<1, 1, 0, stream>>>(sh_state_.data_, static_cast<Args&&>(args)...);
          sh_state_.stream_provider_
            .status_ = STDEXEC_LOG_CUDA_API(cudaEventRecord(sh_state_.event_, stream));
        } else {
          sh_state_.index_ = __mapply<__mfind_i<tuple_t>, variant_t>::value;
        }
      }

      template <class... Args>
      void set_value(Args&&... args) noexcept {
        set_result(set_value_t(), static_cast<Args&&>(args)...);
        sh_state_.notify();
      }

      template <class Error>
      void set_error(Error&& __err) noexcept {
        set_result(set_error_t(), static_cast<Error&&>(__err));
        sh_state_.notify();
      }

      void set_stopped() noexcept {
        set_result(set_stopped_t());
        sh_state_.notify();
      }

      [[nodiscard]]
      auto get_env() const noexcept -> env_t {
        return sh_state_.make_env();
      }

     private:
      SharedState& sh_state_;
    };

    struct opstate_base {
      using notify_fn_t = void(opstate_base*) noexcept;

      opstate_base* next_{};
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
    struct sh_state {
      using variant_t = variant_storage_t<Sender, env_t>;
      using inner_receiver_t = receiver<Sender, sh_state>;
      using task_t = continuation_task<inner_receiver_t, variant_t>;
      using enqueue_receiver_t = stream_enqueue_receiver<env_t, variant_t>;
      using intermediate_receiver_t =
        std::conditional_t<stream_sender<Sender, env_t>, inner_receiver_t, enqueue_receiver_t>;
      using inner_opstate_t = connect_result_t<Sender, intermediate_receiver_t>;

      explicit sh_state(Sender& sndr, context ctx)
        requires(stream_sender<Sender, env_t>)
        : ctx_(ctx)
        , stream_provider_(false, ctx)
        , data_(malloc_managed<variant_t>(stream_provider_.status_))
        , opstate2_(connect(static_cast<Sender&&>(sndr), inner_receiver_t{*this})) {
        if (stream_provider_.status_ == cudaSuccess) {
          stream_provider_.status_ = STDEXEC_LOG_CUDA_API(
            cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
        }
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
      }

      ~sh_state() {
        if (!started_.test(::cuda::memory_order_relaxed)) {
          if (task_) {
            task_->free_(task_);
          }
        }

        if (data_) {
          STDEXEC_ASSERT_CUDA_API(cudaFree(data_));
          if constexpr (stream_sender<Sender, env_t>) {
            STDEXEC_ASSERT_CUDA_API(cudaEventDestroy(event_));
          }
        }
      }

      auto make_env() const noexcept -> env_t {
        return _split::_make_env(stop_source_, &const_cast<stream_provider&>(stream_provider_));
      }

      void notify() noexcept {
        void* const completion_state = static_cast<void*>(this);
        void* old = head_.exchange(completion_state, std::memory_order_acq_rel);
        auto* opstate = static_cast<opstate_base*>(old);

        while (opstate != nullptr) {
          opstate_base* next = opstate->next_;
          opstate->notify_(opstate);
          opstate = next;
        }
      }

      context ctx_;
      stream_provider stream_provider_;
      inplace_stop_source stop_source_{};
      std::atomic<void*> head_{nullptr};
      unsigned int index_{0};
      variant_t* data_{nullptr};
      task_t* task_{nullptr};
      cudaEvent_t event_;
      host_ptr_t<__decay_t<env_t>> env_{};
      inner_opstate_t opstate2_;
      ::cuda::std::atomic_flag started_{};
    };

    template <class Sender, class Receiver>
    struct opstate
      : public opstate_base
      , public _strm::opstate_base<Receiver> {
      using on_stop_t = std::optional<
        stop_callback_for_t<stop_token_of_t<env_of_t<Receiver>>, __forward_stop_request>
      >;

      on_stop_t on_stop_{};
      std::shared_ptr<sh_state<Sender>> sh_state_;

     public:
      opstate(Receiver&& rcvr, std::shared_ptr<sh_state<Sender>> sh_state)
        noexcept(std::is_nothrow_move_constructible_v<Receiver>)
        : opstate_base{nullptr, notify}
        , _strm::opstate_base<Receiver>(static_cast<Receiver&&>(rcvr), sh_state->ctx_)
        , sh_state_(std::move(sh_state)) {
      }

      STDEXEC_IMMOVABLE(opstate);

      static void notify(opstate_base* self) noexcept {
        auto* op = static_cast<opstate*>(self);
        op->on_stop_.reset();

        cudaError_t& status = op->sh_state_->stream_provider_.status_;
        if (status == cudaSuccess) {
          if constexpr (stream_sender<Sender, env_t>) {
            status = STDEXEC_LOG_CUDA_API(
              cudaStreamWaitEvent(op->get_stream(), op->sh_state_->event_, 0));
          }

          nvexec::visit(
            [&](auto& tupl) noexcept -> void {
              ::cuda::std::apply(
                [&]<class Tag, class... Args>(Tag, const Args&... args) noexcept -> void {
                  op->propagate_completion_signal(Tag(), args...);
                },
                tupl);
            },
            *op->sh_state_->data_,
            op->sh_state_->index_);
        } else {
          op->propagate_completion_signal(STDEXEC::set_error, std::as_const(status));
        }
      }

      void start() & noexcept {
        sh_state<Sender>* sh_state = sh_state_.get();
        std::atomic<void*>& head = sh_state->head_;
        void* const completion_state = static_cast<void*>(sh_state);
        void* old = head.load(std::memory_order_acquire);

        if (old != completion_state) {
          on_stop_.emplace(
            get_stop_token(STDEXEC::get_env(this->rcvr_)),
            __forward_stop_request{sh_state->stop_source_});
        }

        do {
          if (old == completion_state) {
            notify(this);
            return;
          }
          next_ = static_cast<opstate_base*>(old);
        } while (!head.compare_exchange_weak(
          old, static_cast<void*>(this), std::memory_order_release, std::memory_order_acquire));

        if (old == nullptr) {
          // the inner sender isn't running
          if (sh_state->stop_source_.stop_requested()) {
            // 1. resets head to completion state
            // 2. notifies waiting threads
            // 3. propagates "stopped" signal to `out_r'`
            sh_state->notify();
          } else {
            sh_state->started_.test_and_set(::cuda::memory_order_relaxed);
            STDEXEC::start(sh_state->opstate2_);
          }
        }
      }
    };
  } // namespace _split

  template <class Sender>
  struct split_sender : stream_sender_base {
    using sender_concept = STDEXEC::sender_t;
    using sh_state_t = _split::sh_state<Sender>;

    template <class Receiver>
    using opstate_t = _split::opstate<Sender, Receiver>;

    template <class... Tys>
    using _set_value_t = completion_signatures<set_value_t(const __decay_t<Tys>&...)>;

    template <class Ty>
    using _set_error_t = completion_signatures<set_error_t(const __decay_t<Ty>&)>;

    using completion_signatures = __try_make_completion_signatures<
      Sender,
      STDEXEC::prop<get_stop_token_t, inplace_stop_token>,
      STDEXEC::completion_signatures<set_error_t(const cudaError_t&)>,
      __q<_set_value_t>,
      __q<_set_error_t>
    >;

    template <receiver_of<completion_signatures> Receiver>
    auto connect(Receiver rcvr) const & noexcept(__nothrow_move_constructible<Receiver>)
      -> opstate_t<Receiver> {
      return opstate_t<Receiver>{static_cast<Receiver&&>(rcvr), sh_state_};
    }

    auto get_env() const noexcept -> stream_sender_attrs<Sender> {
      return {&sndr_};
    }

    explicit split_sender(context ctx, Sender sndr)
      : sndr_(static_cast<Sender&&>(sndr))
      , sh_state_{std::make_shared<sh_state_t>(sndr_, ctx)} {
    }

   private:
    Sender sndr_;
    std::shared_ptr<sh_state_t> sh_state_;
  };

  template <class Env>
  struct transform_sender_for<split_t, Env> {
    template <class Sender>
    using _sender_t = split_sender<__decay_t<Sender>>;

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
  extern __declfn_t<nvexec::_strm::split_sender<__demangle_t<Sender>>>
    __demangle_v<nvexec::_strm::split_sender<Sender>>;
} // namespace STDEXEC::__detail

STDEXEC_PRAGMA_POP()
