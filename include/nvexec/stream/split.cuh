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
#include "../../exec/env.hpp"
#include "../../exec/split.hpp"
#include "../../exec/stoppable_base.hpp"
#include "../detail/cuda_atomic.cuh"  // IWYU pragma: keep
#include "../detail/throw_on_cuda_error.cuh"
#include "common.cuh"

#include <atomic>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

#include <cuda/std/tuple>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

namespace nv::execution::_strm
{
  // The sender that exec::split returns for a child that completes on a stream scheduler: a
  // facade sender (see exec/detail/shared.hpp). The stream_domain makes it from the sender
  // that split describes it with (see transform_sender_for<exec::split_t> below).
  struct split_t : stream_tag_base
  {};

  namespace _split
  {
    inline auto
    _make_env(inplace_stop_source const & stop_source, stream_provider* stream_provider) noexcept
    {
      return make_stream_env(exec::env_from{[&](get_stop_token_t) noexcept
                                            { return stop_source.get_token(); }},
                             stream_provider);
    }

    using env_t = decltype(_split::_make_env(__declval<inplace_stop_source const &>(),
                                             static_cast<stream_provider*>(nullptr)));

    template <class Tag, class... Args, class Variant>
    STDEXEC_ATTRIBUTE(launch_bounds(1))
    __global__ void copy_kernel(Variant* var, Args... args)
    {
      static_assert(trivially_copyable<Args...>);
      using tuple_t = decayed_tuple_t<Tag, Args...>;
      var->template emplace<tuple_t>(Tag(), static_cast<Args&&>(args)...);
    }

    template <class Sender, class SharedState>
    struct receiver : stream_receiver_base
    {
      using receiver_concept = STDEXEC::receiver_tag;

      explicit receiver(SharedState& sh_state_t) noexcept
        : sh_state_(sh_state_t)
      {}

      template <class Tag, class... Args>
      void set_result(Tag, Args&&... args) noexcept
      {
        using tuple_t   = decayed_tuple_t<Tag, Args...>;
        using variant_t = typename SharedState::variant_t;

        if constexpr (stream_sender<Sender, env_t>)
        {
          // The child runs on the stream that the shared state provides it.
          cudaStream_t stream = sh_state_.stream_provider_.own_stream_.value();
          sh_state_.index_    = __mapply<__mfind_i<tuple_t>, variant_t>::value;
          copy_kernel<Tag, Args&&...>
            <<<1, 1, 0, stream>>>(sh_state_.data_.get(), static_cast<Args&&>(args)...);
          sh_state_.stream_provider_.status_ = STDEXEC_LOG_CUDA_API(
            cudaEventRecord(sh_state_.event_, stream));
        }
        else
        {
          sh_state_.index_ = __mapply<__mfind_i<tuple_t>, variant_t>::value;
        }
      }

      template <class... Args>
      void set_value(Args&&... args) noexcept
      {
        set_result(set_value_t(), static_cast<Args&&>(args)...);
        sh_state_.notify();
      }

      template <class Error>
      void set_error(Error&& __err) noexcept
      {
        set_result(set_error_t(), static_cast<Error&&>(__err));
        sh_state_.notify();
      }

      void set_stopped() noexcept
      {
        set_result(set_stopped_t());
        sh_state_.notify();
      }

      [[nodiscard]]
      auto get_env() const noexcept -> env_t
      {
        return sh_state_.make_env();
      }

     private:
      SharedState& sh_state_;
    };

    struct opstate_base
    {
      using notify_fn_t = void(opstate_base*) noexcept;

      opstate_base* next_{};
      notify_fn_t*  notify_{};
    };

    // The attributes of the shared operation, as the operations that wait for it see them:
    // those of the child, in its environment. (Each operation completes on its own stream,
    // whether the shared operation completes before it starts or after.)
    template <class Sender>
    using attrs_t =
      exec::__shared::__shared_attrs<Sender, env_t, __completion_behavior::__unknown_t>;

    template <class Sender>
    struct sh_state
    {
      using variant_t = __for_each_completion_signature_t<
        __concat_completion_signatures_t<__completion_signatures_of_t<Sender, env_t>,
                                         completion_signatures<set_stopped_t()>>,
        decayed_tuple_t,
        __munique<__q<_nullable_variant_t>>::__f>;
      using inner_receiver_t   = receiver<Sender, sh_state>;
      using task_t             = continuation_task<inner_receiver_t, variant_t>;
      using enqueue_receiver_t = stream_enqueue_receiver<env_t, variant_t>;
      using intermediate_receiver_t =
        std::conditional_t<stream_sender<Sender, env_t>, inner_receiver_t, enqueue_receiver_t>;
      using inner_opstate_t = connect_result_t<Sender, intermediate_receiver_t>;

      explicit sh_state(Sender& sndr, context ctx)
        requires(stream_sender<Sender, env_t>)
        : ctx_(ctx)
        , stream_provider_(false, ctx)
        // Pinned rather than managed: the continuation task reads it on the host
        // while kernels may be running, which managed memory only allows on
        // devices with concurrent managed access.
        , data_(host_allocate<variant_t>(stream_provider_.status_, ctx.pinned_resource_))
        , attrs_(STDEXEC::get_env(sndr), make_env())
        , opstate2_(connect(static_cast<Sender&&>(sndr), inner_receiver_t{*this}))
      {
        _initialize_stopped();
        if (stream_provider_.status_ == cudaSuccess)
        {
          stream_provider_.status_ = STDEXEC_LOG_CUDA_API(
            cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
        }
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
        _initialize_stopped();
      }

      ~sh_state()
      {
        if (!started_.test(::cuda::memory_order_relaxed))
        {
          if (task_)
          {
            task_->free_(task_);
          }
        }

        if (data_)
        {
          if constexpr (stream_sender<Sender, env_t>)
          {
            // The copy kernel may still be writing to data_, and the pinned
            // resource reuses memory without synchronizing with the device.
            STDEXEC_ASSERT_CUDA_API(cudaEventSynchronize(event_));
            STDEXEC_ASSERT_CUDA_API(cudaEventDestroy(event_));
          }
        }
      }

      auto make_env() const noexcept -> env_t
      {
        return _split::_make_env(stop_source_, &const_cast<stream_provider&>(stream_provider_));
      }

      void _initialize_stopped() noexcept
      {
        if (data_)
        {
          using tuple_t = decayed_tuple_t<set_stopped_t>;
          index_        = __mapply<__mfind_i<tuple_t>, variant_t>::value;
          data_->template emplace<tuple_t>(set_stopped_t());
        }
      }

      // Starts the shared operation, unless an operation already has. ref is the starting
      // operation's reference to the state: the state keeps a copy while the shared operation
      // runs, so that the operations that wait for it can stop waiting (see opstate) without
      // destroying it.
      void try_start(std::shared_ptr<sh_state> const & ref) noexcept
      {
        {
          std::lock_guard lock{mutex_};
          if (start_requested_)
          {
            return;
          }
          start_requested_ = true;
          running_ref_     = ref;
        }

        if (stop_source_.stop_requested())
        {
          // The shared operation was asked to stop before it started: complete with the
          // initial results, which are set_stopped.
          notify();
        }
        else
        {
          started_.test_and_set(::cuda::memory_order_relaxed);
          STDEXEC::start(opstate2_);
        }
      }

      enum class add_result
      {
        added,      // the operation receives the results when the shared operation completes
        completed,  // the shared operation has completed: the results are ready
        stopped     // the operation's stop has been requested
      };

      // Adds an operation to the ones waiting for the shared operation
      template <class StopToken>
      auto try_add_waiter(opstate_base* op, StopToken const & stok) noexcept -> add_result
      {
        std::lock_guard lock{mutex_};
        if (completed_)
        {
          return add_result::completed;
        }
        if (stok.stop_requested())
        {
          return add_result::stopped;
        }
        op->next_ = std::exchange(waiters_, op);
        return add_result::added;
      }

      // Removes an operation from the ones waiting for the shared operation, if it is one.
      // (It is not if it has not started yet, or if the shared operation has completed.)
      auto try_remove_waiter(opstate_base* op) noexcept -> bool
      {
        std::lock_guard lock{mutex_};
        for (opstate_base** link = &waiters_; *link != nullptr; link = &(*link)->next_)
        {
          if (*link == op)
          {
            *link = op->next_;
            return true;
          }
        }
        return false;
      }

      // The shared operation has completed: sends its results to the operations waiting for
      // it.
      void notify() noexcept
      {
        opstate_base*             waiters = nullptr;
        std::shared_ptr<sh_state> running_ref;
        {
          std::lock_guard lock{mutex_};
          completed_  = true;
          waiters     = std::exchange(waiters_, nullptr);
          running_ref = std::move(running_ref_);
        }

        while (waiters != nullptr)
        {
          opstate_base* next = waiters->next_;
          waiters->notify_(waiters);
          waiters = next;
        }
        // Releasing running_ref can destroy the state.
      }

      // A split sender or an operation no longer refers to the state (but still holds its
      // reference). If the shared operation is running and nothing else refers to the state,
      // nothing will receive its results, so it is asked to stop. (This is how exec::split
      // detaches.)
      void detach() noexcept
      {
        std::shared_ptr<sh_state> running_ref;
        {
          std::lock_guard lock{mutex_};
          running_ref = running_ref_;
        }
        // The caller's reference, the state's reference, and this copy of it
        if (running_ref.use_count() == 3)
        {
          stop_source_.request_stop();
        }
      }

      context                      ctx_;
      stream_provider              stream_provider_;
      inplace_stop_source          stop_source_{};
      std::mutex                   mutex_{};
      opstate_base*                waiters_{nullptr};   // guarded by mutex_
      bool                         start_requested_{};  // guarded by mutex_
      bool                         completed_{};        // guarded by mutex_
      std::shared_ptr<sh_state>    running_ref_{};      // guarded by mutex_
      unsigned int                 index_{0};
      host_ptr_t<variant_t>        data_{};
      task_t*                      task_{nullptr};
      cudaEvent_t                  event_;
      host_ptr_t<__decay_t<env_t>> env_{};
      attrs_t<Sender>              attrs_;
      inner_opstate_t              opstate2_;
      ::cuda::std::atomic_flag     started_{};
    };

    // The key of the shared operation, whose results each operation receives
    struct shared_op
    {};

    // The state of the operation of a split sender. There is one for each receiver connected
    // to the sender. It holds a reference to the shared state.
    template <class Algo, class Sender, class Receiver>
    struct opstate
      : public opstate_base
      , public _strm::opstate_base<Receiver>
      , public exec::stoppable_base_for_t<stop_token_of_t<env_of_t<Receiver>>,
                                          opstate<Algo, Sender, Receiver>>
    {
      std::shared_ptr<sh_state<Sender>> sh_state_;

      // Only this operation stops: if it is still waiting for the shared operation, it no
      // longer does, and it completes with set_stopped. (If it is not waiting, either it has
      // not been added to the ones that wait yet, and __start will see the stop request, or
      // the shared operation has completed, and its results are on their way.)
      void on_stop_requested() noexcept
      {
        if (sh_state_->try_remove_waiter(this))
        {
          std::exchange(sh_state_, {})->detach();
          (void) _strm::_report_stopped(*this);
        }
      }

      opstate(Receiver&& rcvr, std::shared_ptr<sh_state<Sender>> sh_state)
        noexcept(std::is_nothrow_move_constructible_v<Receiver>)
        : opstate_base{nullptr, notify}
        , _strm::opstate_base<Receiver>(static_cast<Receiver&&>(rcvr), sh_state->ctx_)
        , sh_state_(std::move(sh_state))
      {}

      STDEXEC_IMMOVABLE(opstate);

      ~opstate()
      {
        if (sh_state_)
        {
          sh_state_->detach();
        }
      }

      // The algorithm registers the stop callback in __start.
      template <class StopToken>
      void register_stop_callback(StopToken const & stok) noexcept
      {
        this->register_stoppable(stok);
      }

      void deregister_stop_callback() noexcept
      {
        this->deregister_stoppable();
      }

      // Sends the results of the shared operation, on this operation's stream, once the
      // stream has waited for them
      static void notify(opstate_base* self) noexcept
      {
        auto* op = static_cast<opstate*>(self);
        op->deregister_stoppable();

        cudaError_t& status = op->sh_state_->stream_provider_.status_;
        if (status == cudaSuccess)
        {
          if constexpr (stream_sender<Sender, env_t>)
          {
            status = STDEXEC_LOG_CUDA_API(
              cudaStreamWaitEvent(op->get_stream(), op->sh_state_->event_, 0));
          }

          nvexec::visit(
            [&](auto& tupl) noexcept -> void
            {
              ::cuda::std::apply([&]<class Tag, class... Args>(Tag, Args const &... args) noexcept
                                   -> void
                                 { (void) Algo::__complete(shared_op(), *op, Tag(), args...); },
                                 tupl);
            },
            *op->sh_state_->data_,
            op->sh_state_->index_);
        }
        else
        {
          (void) Algo::__complete(shared_op(), *op, set_error_t(), std::as_const(status));
        }
      }
    };

    template <class Ty>
    using _decayed_cref_t = __decay_t<Ty> const &;
  }  // namespace _split

  namespace _split
  {
    // The shared operation, as the operations that wait for it see it: it completes with
    // its results, as const lvalue references, or with a CUDA error. It is never connected.
    template <class Sender>
    struct results_sender
    {
      using sender_concept = STDEXEC::sender_tag;

      template <class Self, class... Env>
      static consteval auto get_completion_signatures()
      {
        using env_t = STDEXEC::prop<get_stop_token_t, inplace_stop_token>;
        return exec::transform_completion_signatures(
          STDEXEC::get_completion_signatures<Sender, env_t>(),
          exec::transform_arguments<STDEXEC::set_value_t, _split::_decayed_cref_t, exec::split_t>(),
          exec::transform_arguments<STDEXEC::set_error_t, _split::_decayed_cref_t, exec::split_t>(),
          exec::keep_completion<STDEXEC::set_stopped_t>(),
          STDEXEC::completion_signatures<set_error_t(cudaError_t const &), set_stopped_t()>());
      }

      [[nodiscard]]
      auto get_env() const noexcept -> attrs_t<Sender>
      {
        return attrs_;
      }

      attrs_t<Sender> attrs_;
    };

    // The data of a split sender: a reference to the shared state, and the shared operation's
    // attributes. It is copyable. A split sender that no longer refers to the shared state
    // (because it is destroyed, or assigned over) detaches from it (see sh_state::detach).
    template <class Sender>
    struct handle
    {
      using child_t = Sender;

      handle(std::shared_ptr<sh_state<Sender>> sh_state, attrs_t<Sender> attrs) noexcept
        : sh_state_(std::move(sh_state))
        , attrs_(std::move(attrs))
      {}

      handle(handle const &)    = default;
      handle(handle&&) noexcept = default;

      auto operator=(handle&& other) noexcept -> handle&
      {
        if (this != &other)
        {
          detach();
          sh_state_ = std::move(other.sh_state_);
          attrs_    = std::move(other.attrs_);
        }
        return *this;
      }

      auto operator=(handle const & other) -> handle&
      {
        if (this != &other)
        {
          handle copy(other);
          *this = std::move(copy);
        }
        return *this;
      }

      ~handle()
      {
        detach();
      }

      void detach() noexcept
      {
        if (sh_state_)
        {
          sh_state_->detach();
        }
      }

      std::shared_ptr<sh_state<Sender>> sh_state_;
      attrs_t<Sender>                   attrs_;
    };

    // The algorithm of a split sender: when the operation starts, it starts the shared
    // operation if no other operation has, and its results reach it through the shared
    // operation, on its stream (immediately, if it has completed already).
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
      static auto __get_state(Sender&& sndr, Receiver rcvr) noexcept
        -> opstate<algo, typename handle_t<Sender>::child_t, Receiver>
      {
        return opstate<algo, typename handle_t<Sender>::child_t, Receiver>(
          static_cast<Receiver&&>(rcvr),
          __facade::__get_data(static_cast<Sender&&>(sndr)).sh_state_);
      }

      // The operation starts the shared operation, unless another has, and waits for its
      // results. If its receiver asks it to stop first, it stops waiting (see
      // opstate::on_stop_requested), and the shared operation is stopped only if nothing
      // else waits for it (see sh_state::detach).
      template <class State, class Children>
      static auto __start(State& state, Children&) noexcept
        -> __facade::__effects<__facade::__starts<shared_op>, __facade::__starts<stream_stopped>>
      {
        auto const stok = get_stop_token(STDEXEC::get_env(state.rcvr_));
        state.register_stop_callback(stok);

        if (!stok.stop_requested())
        {
          auto& sh_state = *state.sh_state_;
          sh_state.try_start(state.sh_state_);

          using add_result = __decay_t<decltype(sh_state)>::add_result;
          switch (sh_state.try_add_waiter(&state, stok))
          {
          case add_result::added:
            return {};  // the results come when the shared operation completes
          case add_result::completed:
            State::notify(&state);
            return {};
          case add_result::stopped:
            break;
          }
        }

        // The operation's stop has been requested, before it waits for the shared operation.
        state.deregister_stop_callback();
        std::exchange(state.sh_state_, {})->detach();
        return _strm::_report_stopped(state);
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
  }  // namespace _split

  template <>
  struct transform_sender_for<exec::split_t>
  {
    template <class Env, class Sender>
    auto operator()(Env const & env, __ignore, __ignore, Sender&& sndr) const
    {
      using sender_t = __decay_t<Sender>;
      if constexpr (stream_completing_sender<Sender, Env>)
      {
        auto     sched = get_completion_scheduler<set_value_t>(get_env(sndr), env);
        sender_t child(static_cast<Sender&&>(sndr));
        auto     sh_state = std::make_shared<_split::sh_state<sender_t>>(child, sched.ctx_);
        auto     attrs    = sh_state->attrs_;
        return STDEXEC::__facade::__sender{
          _strm::split_t(),
          _split::handle<sender_t>{std::move(sh_state), std::move(attrs)}
        };
      }
      else
      {
        return _strm::_no_stream_scheduler_in_env<exec::split_t, Sender, Env>();
      }
    }
  };
}  // namespace nv::execution::_strm

template <>
struct STDEXEC::__facade::__algorithm<nv::execution::_strm::split_t>
  : nv::execution::_strm::_split::algo
{};

namespace nvexec = nv::execution;

STDEXEC_PRAGMA_POP()
