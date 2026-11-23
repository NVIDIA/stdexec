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
#include "../../exec/completion_signatures.hpp"
#include "../../exec/ensure_started.hpp"
#include "../../exec/env.hpp"
#include "../../exec/stoppable_base.hpp"
#include "../detail/throw_on_cuda_error.cuh"
#include "common.cuh"

#include <atomic>
#include <type_traits>
#include <utility>

#include <cuda/std/tuple>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

namespace nv::execution::_strm
{
  // The sender that exec::ensure_started returns for a child that completes on a stream
  // scheduler: a facade sender (see exec/detail/shared.hpp). The stream_domain makes it from
  // the sender that ensure_started describes it with (see
  // transform_sender_for<exec::ensure_started_t> below).
  struct ensure_started_t : stream_tag_base
  {};

  namespace _ensure_started
  {
    template <class Tag, class... Args, class Variant>
    STDEXEC_ATTRIBUTE(launch_bounds(1))
    __global__ void copy_kernel(Variant* var, Args... args)
    {
      static_assert(trivially_copyable<Args...>);
      using tuple_t = decayed_tuple_t<Tag, Args...>;
      var->template emplace<tuple_t>(Tag(), static_cast<Args&&>(args)...);
    }

    inline auto
    _make_env(inplace_stop_source const & stop_source, stream_provider* stream_provider) noexcept
    {
      return make_stream_env(exec::env_from{[&](get_stop_token_t) noexcept
                                            { return stop_source.get_token(); }},
                             stream_provider);
    }

    using env_t = decltype(_ensure_started::_make_env(__declval<inplace_stop_source const &>(),
                                                      static_cast<stream_provider*>(nullptr)));

    template <class Sender, class SharedState>
    class receiver : public stream_receiver_base
    {
      __intrusive_ptr<SharedState> shared_state_;

     public:
      explicit receiver(SharedState& shared_state) noexcept
        : shared_state_(shared_state.__intrusive_from_this())
      {}

      template <class Tag, class... Args>
      void _set_result(Tag, Args&&... args) noexcept
      {
        using tuple_t   = decayed_tuple_t<Tag, Args...>;
        using variant_t = typename SharedState::variant_t;

        if constexpr (stream_sender<Sender, env_t>)
        {
          if (shared_state_->stream_provider_.status_ == cudaSuccess)
          {
            cudaStream_t stream   = shared_state_->stream_provider_.own_stream_.value();
            shared_state_->index_ = __mapply<__mfind_i<tuple_t>, variant_t>::value;
            copy_kernel<Tag, Args&&...>
              <<<1, 1, 0, stream>>>(shared_state_->data_.get(), static_cast<Args&&>(args)...);
            shared_state_->completion_copy_started_ = true;
            shared_state_->stream_provider_.status_ = STDEXEC_LOG_CUDA_API(
              cudaEventRecord(shared_state_->event_, stream));
            shared_state_->event_recorded_ = shared_state_->stream_provider_.status_ == cudaSuccess;
          }
        }
        else
        {
          shared_state_->index_ = __mapply<__mfind_i<tuple_t>, variant_t>::value;
        }
      }

      template <class... Args>
      void set_value(Args&&... args) noexcept
      {
        _set_result(set_value_t(), static_cast<Args&&>(args)...);
        shared_state_->notify();
        shared_state_.reset();
      }

      template <class Error>
      void set_error(Error&& __err) noexcept
      {
        _set_result(set_error_t(), static_cast<Error&&>(__err));
        shared_state_->notify();
        shared_state_.reset();
      }

      void set_stopped() noexcept
      {
        _set_result(set_stopped_t());
        shared_state_->notify();
        shared_state_.reset();
      }

      [[nodiscard]]
      auto get_env() const noexcept -> env_t
      {
        return shared_state_->make_env();
      }
    };

    struct opstate_base
    {
      using notify_fn_t = void(opstate_base*) noexcept;
      notify_fn_t* notify_{};
    };

    // The attributes of the shared operation, as the operation that waits for it sees them:
    // those of the child, in its environment. (The operation completes on its stream,
    // whether the shared operation completes before it starts or after.)
    template <class Sender>
    using attrs_t =
      exec::__shared::__shared_attrs<Sender, env_t, __completion_behavior::__unknown_t>;

    template <class Sender>
    struct sh_state : __enable_intrusive_from_this<sh_state<Sender>>
    {
      using variant_t          = variant_storage_t<Sender, env_t>;
      using inner_receiver_t   = receiver<Sender, sh_state>;
      using task_t             = continuation_task<inner_receiver_t, variant_t>;
      using enqueue_receiver_t = stream_enqueue_receiver<env_t, variant_t>;
      using intermediate_receiver_t =
        std::conditional_t<stream_sender<Sender, env_t>, inner_receiver_t, enqueue_receiver_t>;
      using inner_opstate_t = connect_result_t<Sender, intermediate_receiver_t>;

      auto make_env() const noexcept -> env_t
      {
        return _ensure_started::_make_env(stop_source_,
                                          &const_cast<stream_provider&>(stream_provider_));
      }

      explicit sh_state(Sender& sndr, context ctx)
        requires(stream_sender<Sender, env_t>)
        : ctx_(ctx)
        , stream_provider_(false, ctx)
        // Pinned rather than managed: the continuation task reads it on the host
        // while kernels may be running, which managed memory only allows on
        // devices with concurrent managed access.
        , data_(host_allocate<variant_t>(stream_provider_.status_, ctx.pinned_resource_))
        , attrs_(STDEXEC::get_env(sndr), make_env())
        , opstate1_{nullptr}
        , opstate2_(connect(static_cast<Sender&&>(sndr), inner_receiver_t{*this}))
      {
        if (stream_provider_.status_ == cudaSuccess)
        {
          stream_provider_.status_ = STDEXEC_LOG_CUDA_API(
            cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
          event_created_ = stream_provider_.status_ == cudaSuccess;
        }

        STDEXEC::start(opstate2_);
      }

      explicit sh_state(Sender& sndr, context ctx)
        : ctx_(ctx)
        , stream_provider_(false, ctx)
        , data_(host_allocate<variant_t>(stream_provider_.status_, ctx.pinned_resource_))
        , task_(host_allocate<task_t>(stream_provider_.status_,
                                      ctx.pinned_resource_,
                                      inner_receiver_t{*this},
                                      data_.get(),
                                      stream_provider_.own_stream_.value(),
                                      ctx.pinned_resource_)
                  .release())
        , env_(host_allocate(this->stream_provider_.status_, ctx_.pinned_resource_, make_env()))
        , attrs_(STDEXEC::get_env(sndr), make_env())
        , opstate2_(
            connect(static_cast<Sender&&>(sndr),
                    enqueue_receiver_t{env_.get(), data_.get(), task_, ctx.hub_->producer()}))
      {
        STDEXEC::start(opstate2_);
      }

      ~sh_state()
      {
        if constexpr (stream_sender<Sender, env_t>)
        {
          if (completion_copy_started_)
          {
            if (event_recorded_)
            {
              STDEXEC_ASSERT_CUDA_API(cudaEventSynchronize(event_));
            }
            else
            {
              STDEXEC_ASSERT_CUDA_API(cudaStreamSynchronize(stream_provider_.own_stream_.value()));
            }
          }

          if (event_created_)
          {
            STDEXEC_ASSERT_CUDA_API(cudaEventDestroy(event_));
          }
        }
      }

      void notify() noexcept
      {
        void* const completion_state = static_cast<void*>(this);
        void* const old = opstate1_.exchange(completion_state, std::memory_order_acq_rel);
        if (old != nullptr)
        {
          auto* op = static_cast<opstate_base*>(old);
          op->notify_(op);
        }
      }

      void detach() noexcept
      {
        stop_source_.request_stop();
      }

      context                      ctx_;
      stream_provider              stream_provider_;
      cudaEvent_t                  event_{};
      bool                         event_created_{false};
      bool                         event_recorded_{false};
      bool                         completion_copy_started_{false};
      unsigned int                 index_{0};
      host_ptr_t<variant_t>        data_{};
      task_t*                      task_{nullptr};
      inplace_stop_source          stop_source_{};
      host_ptr_t<__decay_t<env_t>> env_{};
      attrs_t<Sender>              attrs_;
      std::atomic<void*>           opstate1_;
      inner_opstate_t              opstate2_;
    };

    // The key of the shared operation, whose results the operation receives
    struct shared_op
    {};

    // The state of the operation of an ensure_started sender. It holds the reference to the
    // shared state.
    template <class Algo, class Sender, class Receiver>
    struct opstate
      : public opstate_base
      , public _strm::opstate_base<Receiver>
      , public exec::stoppable_base_for_t<stop_token_of_t<env_of_t<Receiver>>,
                                          opstate<Algo, Sender, Receiver>>
    {
      __intrusive_ptr<sh_state<Sender>> shared_state_;

      void on_stop_requested() noexcept
      {
        shared_state_->stop_source_.request_stop();
      }

      opstate(Receiver rcvr, __intrusive_ptr<sh_state<Sender>> shared_state) noexcept
        : opstate_base{notify}
        , _strm::opstate_base<Receiver>(static_cast<Receiver&&>(rcvr), shared_state->ctx_)
        , shared_state_(std::move(shared_state))
      {}

      ~opstate()
      {
        // The stopped path in __start completes without deregistering, and the
        // base outlives shared_state_, which on_stop_requested() uses.
        this->deregister_stoppable();

        // Check to see if this operation was ever started. If not,
        // detach the (potentially still running) operation:
        if (nullptr == shared_state_->opstate1_.load(std::memory_order_acquire))
        {
          shared_state_->detach();
        }
      }

      STDEXEC_IMMOVABLE(opstate);

      // The algorithm registers the stop callback in __start.
      template <class StopToken>
      void register_stop_callback(StopToken const & stok) noexcept
      {
        this->register_stoppable(stok);
      }

      // Sends the results of the shared operation, on this operation's stream, once the
      // stream has waited for them
      static void notify(opstate_base* self) noexcept
      {
        auto* op = static_cast<opstate*>(self);
        op->deregister_stoppable();

        cudaError_t& status = op->shared_state_->stream_provider_.status_;

        if (status == cudaSuccess)
        {
          if constexpr (stream_sender<Sender, env_t>)
          {
            status = STDEXEC_LOG_CUDA_API(
              cudaStreamWaitEvent(op->get_stream(), op->shared_state_->event_, 0));
          }

          nvexec::visit(
            [&](auto& tupl) noexcept -> void
            {
              ::cuda::std::apply(
                [&]<class Tag, class... Args>(Tag, Args&... args) noexcept -> void
                { (void) Algo::__complete(shared_op(), *op, Tag(), std::move(args)...); },
                tupl);
            },
            *op->shared_state_->data_,
            op->shared_state_->index_);
        }
        else
        {
          (void) Algo::__complete(shared_op(), *op, set_error_t(), std::move(status));
        }
      }
    };

    template <class Sender>
    using _decayed_t = __decay_t<Sender>;

    // The shared operation, as the operation that waits for it sees it: it completes with its
    // results, as rvalues, or with a CUDA error. It is never connected.
    template <class Sender>
    struct results_sender
    {
      using sender_concept = STDEXEC::sender_tag;

      template <class Self, class... Env>
      static consteval auto get_completion_signatures()
      {
        return exec::transform_completion_signatures(
          STDEXEC::get_completion_signatures<Sender, _ensure_started::env_t>(),
          exec::decay_arguments<set_value_t, exec::ensure_started_t>(),
          exec::decay_arguments<set_error_t, exec::ensure_started_t>(),
          {},
          completion_signatures<set_error_t(cudaError_t), set_stopped_t()>());
      }

      [[nodiscard]]
      auto get_env() const noexcept -> attrs_t<Sender>
      {
        return attrs_;
      }

      attrs_t<Sender> attrs_;
    };

    // The data of an ensure_started sender: the reference to the shared state, and the shared
    // operation's attributes. It is move-only. A sender that is destroyed without being
    // connected detaches the shared operation.
    template <class Sender>
    struct handle
    {
      using child_t = Sender;

      handle(__intrusive_ptr<sh_state<Sender>> shared_state, attrs_t<Sender> attrs) noexcept
        : shared_state_(std::move(shared_state))
        , attrs_(std::move(attrs))
      {}

      handle(handle&&) noexcept = default;

      ~handle()
      {
        if (nullptr != shared_state_)
        {
          // We're detaching a potentially running operation. Request cancellation.
          shared_state_->detach();  // BUGBUG NOT TO SPEC
        }
      }

      __intrusive_ptr<sh_state<Sender>> shared_state_;
      attrs_t<Sender>                   attrs_;
    };

    struct _AN_ENSURE_STARTED_SENDER_CAN_ONLY_BE_CONNECTED_AS_AN_RVALUE_;

    // The algorithm of an ensure_started sender: the operation receives the results of the
    // shared operation through it, on its stream (immediately, if it has completed
    // already), unless stop was requested before it starts.
    struct algo : stream_algorithm
    {
      using __dispatch_source = __facade::__via<shared_op, set_value_t, __facade::__from_start>;

      template <class Sender>
      using handle_t = __decay_t<__facade::__data_of_t<Sender>>;

      using stream_algorithm::__nested_sender;

      template <class Sender, class... Env>
      static auto __nested_sender(shared_op, Sender const & sndr, Env const &...) noexcept
        -> results_sender<typename handle_t<Sender>::child_t>
      {
        return {__facade::__get_data(sndr).attrs_};
      }

      template <class Sender, class Receiver>
      static auto
      __get_state([[maybe_unused]] Sender&& sndr, [[maybe_unused]] Receiver rcvr) noexcept
      {
        using child_t = handle_t<Sender>::child_t;
        if constexpr (std::is_lvalue_reference_v<Sender>)
        {
          return __mexception<_WHAT_(_INVALID_ARGUMENT_),
                              _WHERE_(_IN_ALGORITHM_, exec::ensure_started_t),
                              _WHY_(
                                _AN_ENSURE_STARTED_SENDER_CAN_ONLY_BE_CONNECTED_AS_AN_RVALUE_)>();
        }
        else
        {
          return opstate<algo, child_t, Receiver>(
            static_cast<Receiver&&>(rcvr),
            std::move(__facade::__get_data(static_cast<Sender&&>(sndr)).shared_state_));
        }
      }

      template <class State, class Children>
      static auto __start(State& state, Children&) noexcept
      {
        using result_t =
          __facade::__effects<__facade::__starts<shared_op>, __facade::__starts<stream_stopped>>;
        auto*               shared_state     = state.shared_state_.get();
        std::atomic<void*>& opstate1         = shared_state->opstate1_;
        void* const         completion_state = static_cast<void*>(shared_state);
        void* const         old              = opstate1.load(std::memory_order_acquire);
        if (old == completion_state)
        {
          State::notify(&state);
          return result_t(__facade::__effects<__facade::__starts<shared_op>>());
        }
        else
        {
          // register stop callback:
          state.register_stop_callback(get_stop_token(STDEXEC::get_env(state.rcvr_)));
          // Check if the stop_source has requested cancellation
          if (shared_state->stop_source_.stop_requested())
          {
            // Stop has already been requested. Don't bother starting
            // the child operations. (The operation completes on its stream, as it does
            // when it receives the shared operation's results.)
            return result_t(_strm::_report_stopped(state));
          }
          else
          {
            // Otherwise, the inner source hasn't notified completion.
            // Set this operation as the opstate1 so it's notified.
            // (A strong exchange: a weak one could fail spuriously, which would look like the
            // task having completed.)
            void* old = nullptr;
            if (!opstate1.compare_exchange_strong(old,
                                                  static_cast<opstate_base*>(&state),
                                                  std::memory_order_release,
                                                  std::memory_order_acquire))
            {
              // We get here when the task completed during the execution
              // of this function. Complete the operation synchronously.
              STDEXEC_ASSERT(old == completion_state);
              State::notify(&state);
            }
            return result_t(__facade::__effects<__facade::__starts<shared_op>>());
          }
        }
      }

      template <class State, class Tag, class... Args>
      static auto __complete(shared_op, State& state, Tag tag, Args&&... args) noexcept
      {
        return _strm::_propagate(state, tag, static_cast<Args&&>(args)...);
      }

      template <class State>
      static auto __complete(stream_stopped, State& state, set_stopped_t) noexcept
      {
        return _strm::_propagate(state, set_stopped_t());
      }
    };
  }  // namespace _ensure_started

  template <>
  struct transform_sender_for<exec::ensure_started_t>
  {
    template <class Env, class Sender>
    auto operator()(Env const & env, __ignore, __ignore, Sender&& sndr) const
    {
      using sender_t = __decay_t<Sender>;
      if constexpr (stream_completing_sender<Sender, Env>)
      {
        auto     sched = get_completion_scheduler<set_value_t>(get_env(sndr), env);
        sender_t child(static_cast<Sender&&>(sndr));
        auto     shared_state = __make_intrusive<_ensure_started::sh_state<sender_t>>(child,
                                                                                  sched.ctx_);
        auto     attrs        = shared_state->attrs_;
        return STDEXEC::__facade::__sender{
          _strm::ensure_started_t(),
          _ensure_started::handle<sender_t>{std::move(shared_state), std::move(attrs)}
        };
      }
      else
      {
        return _strm::_no_stream_scheduler_in_env<exec::ensure_started_t, Sender, Env>();
      }
    }
  };
}  // namespace nv::execution::_strm

// The stream domain's ensure_started sender is an ensure_started sender: its operation has
// started, so exec::ensure_started returns it as it is.
template <>
inline constexpr bool experimental::execution::__ensure_started::__is_started_sender_tag_v<
  nv::execution::_strm::ensure_started_t> = true;

template <>
struct STDEXEC::__facade::__algorithm<nv::execution::_strm::ensure_started_t>
  : nv::execution::_strm::_ensure_started::algo
{};

namespace nvexec = nv::execution;

STDEXEC_PRAGMA_POP()
