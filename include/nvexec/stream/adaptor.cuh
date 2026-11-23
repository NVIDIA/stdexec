/*
 * Copyright (c) 2026 NVIDIA Corporation
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

#include <cstddef>
#include <type_traits>
#include <utility>

#include <cuda/std/utility>

#include "common.cuh"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

// The pieces that the stream algorithms with one child share. (They are facade algorithms;
// see stream_algorithm in common.cuh.)
namespace nv::execution::_strm
{
  // The state of a stream algorithm with one child: the operation state of the stream
  // (opstate_base: the receiver, the stream, and temporary storage), and the algorithm's data
  template <class Data, class Receiver>
  struct adaptor_state : _strm::opstate_base<Receiver>
  {
    template <class CvData>
    explicit adaptor_state(CvData&& data, Receiver rcvr, context ctx)
      : _strm::opstate_base<Receiver>(static_cast<Receiver&&>(rcvr), ctx)
      , data_(static_cast<CvData&&>(data))
    {}

    ~adaptor_state()
    {
      if (this->temp_storage_)
      {
        this->ctx_.managed_resource_->deallocate(this->temp_storage_, temp_size_);
      }
    }

    STDEXEC_IMMOVABLE(adaptor_state);

    Data        data_;
    std::size_t temp_size_{0};
  };

  // Allocates the state's temporary storage, in managed memory. Returns null if that fails.
  template <class State>
  auto _allocate_temp_storage(State& state, std::size_t size) noexcept -> void*
  {
    void* memory = nullptr;
    STDEXEC_TRY
    {
      memory = state.ctx_.managed_resource_->allocate(size);
    }
    STDEXEC_CATCH_ALL {}
    if (memory != nullptr)
    {
      state.temp_storage_ = memory;
      state.temp_size_    = size;
    }
    return memory;
  }

  // The base of the stream algorithms with one child, which runs on the algorithm's stream.
  // The operation runs on the stream of the scheduler on which the child completes. A
  // CUDA error from starting it, and the completions of the child that the algorithm does
  // not handle, are passed on. Derived is the algorithm.
  template <class Derived>
  struct stream_adaptor : stream_algorithm
  {
    using stream_algorithm::__env_for;
    using stream_algorithm::__get_env;

    // The child shares the stream.
    template <class Sender, class Env>
    static auto __env_for(STDEXEC::__facade::__child<0>, Sender const &, Env&& env) noexcept
    {
      return _strm::make_stream_env(static_cast<Env&&>(env),
                                    static_cast<stream_provider*>(nullptr));
    }

    template <class State>
    static auto __get_env(STDEXEC::__facade::__child<0>, State const & state) noexcept
    {
      return state.make_env();
    }

    template <class Sender, class Receiver>
    static auto __get_state(Sender&& sndr, Receiver rcvr)
      -> adaptor_state<__decay_t<STDEXEC::__facade::__data_of_t<Sender>>, Receiver>
    {
      using state_t = adaptor_state<__decay_t<STDEXEC::__facade::__data_of_t<Sender>>, Receiver>;
      auto sched    = STDEXEC::get_completion_scheduler<set_value_t>(
        STDEXEC::__facade::__nested_attrs_of<Derived>(STDEXEC::__facade::__child<0>(),
                                                      sndr,
                                                      STDEXEC::get_env(rcvr)),
        Derived::__env_for(STDEXEC::__facade::__child<0>(), sndr, STDEXEC::get_env(rcvr)));
      context ctx = sched.ctx_;
      return state_t(STDEXEC::__facade::__get_data(static_cast<Sender&&>(sndr)),
                     static_cast<Receiver&&>(rcvr),
                     ctx);
    }

    template <class State, class Children>
    static auto __start(State& state, Children& children) noexcept
    {
      cudaError_t status = state.stream_provider_.status_;
      return STDEXEC::__facade::__choose(
        status == cudaSuccess,
        [&]() noexcept { return children.__start_all(); },
        [&]() noexcept { return _strm::_report_error(state, status); });
    }

    // The child's completions that the algorithm does not handle, and CUDA errors (see
    // _report_error), are passed on.
    template <class Key, class State, class Tag, class... Args>
    static auto __complete(Key, State& state, Tag tag, Args&&... args) noexcept
    {
      return _strm::_propagate(state, tag, static_cast<Args&&>(args)...);
    }
  };

  template <class... Args, class Fun>
  STDEXEC_ATTRIBUTE(launch_bounds(1))
  __global__ void _invoke_kernel(Fun fn, Args... args)
  {
    static_assert(trivially_copyable<Fun, Args...>);
    ::cuda::std::move(fn)(static_cast<Args&&>(args)...);
  }

  template <class ResultT, class... Args, class Fun>
  STDEXEC_ATTRIBUTE(launch_bounds(1))
  __global__
    void _invoke_kernel_with_result(Fun fn, _stored_result_t<ResultT>* result, Args... args)
  {
    static_assert(trivially_copyable<Fun, Args...>);
    _store_result<ResultT>(result, ::cuda::std::move(fn), static_cast<Args&&>(args)...);
  }

  // Calls the function with the arguments in a kernel on the state's stream, and sends its
  // result as the operation's value (or a CUDA error)
  template <class State, class Fun, class... Args>
  auto _invoke_on_stream(State& state, Fun& fun, Args&&... args) noexcept
  {
    using result_t      = __invoke_result_t<Fun, Args...>;
    cudaStream_t stream = state.get_stream();

    if constexpr (std::is_same_v<void, result_t>)
    {
      _strm::_invoke_kernel<Args&&...>
        <<<1, 1, 0, stream>>>(std::move(fun), static_cast<Args&&>(args)...);
      cudaError_t status = STDEXEC_LOG_CUDA_API(cudaPeekAtLastError());
      return STDEXEC::__facade::__choose(
        status == cudaSuccess,
        [&]() noexcept { return _strm::_propagate(state, set_value_t()); },
        [&]() noexcept { return _strm::_report_error(state, status); });
    }
    else
    {
      using stored_t = _stored_result_t<result_t>;
      void* storage  = _strm::_allocate_temp_storage(state, sizeof(stored_t));

      auto launch = [&]() noexcept
      {
        auto* d_result = static_cast<stored_t*>(storage);
        _strm::_invoke_kernel_with_result<result_t, Args&&...>
          <<<1, 1, 0, stream>>>(std::move(fun), d_result, static_cast<Args&&>(args)...);
        state.defer_temp_storage_destruction(d_result);

        cudaError_t status = STDEXEC_LOG_CUDA_API(cudaPeekAtLastError());
        if (status == cudaSuccess)
        {
          status = _wait_for_result<result_t>(stream);
        }
        return STDEXEC::__facade::__choose(
          status == cudaSuccess,
          [&]() noexcept
          { return _strm::_propagate(state, set_value_t(), _load_result<result_t>(d_result)); },
          [&]() noexcept { return _strm::_report_error(state, status); });
      };

      return STDEXEC::__facade::__choose(
        storage != nullptr,
        launch,
        [&]() noexcept { return _strm::_report_error(state, cudaErrorMemoryAllocation); });
    }
  }
}  // namespace nv::execution::_strm

STDEXEC_PRAGMA_POP()
