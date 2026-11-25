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

#include <concepts>
#include <memory>
#include <utility>

#include "adaptor.cuh"
#include "common.cuh"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

#if defined(STDEXEC_CLANGD_INVOKED) && STDEXEC_CLANG() && STDEXEC_CUDA_COMPILATION()
// clangd doesn't understand CUDA's new/delete operators
__host__ auto operator new[](std::size_t) -> void*;
#endif

namespace nv::execution::_strm
{
  // bulk as a facade algorithm. The stream_domain customizes bulk(sndr, pol, shape, fn) by
  // re-tagging it with this tag (see transform_sender_for<bulk_t> below).
  struct bulk_t : stream_tag_base
  {};

  namespace _bulk
  {
    template <int BlockThreads, class... Args, std::integral Shape, class Fun>
    STDEXEC_ATTRIBUTE(launch_bounds(BlockThreads))
    __global__ void _bulk_kernel(Shape shape, Fun fn, Args... args)
    {
      static_assert(trivially_copyable<Shape, Fun, Args...>);
      Shape const tid = static_cast<Shape>(threadIdx.x + blockIdx.x * blockDim.x);

      if (tid < shape)
      {
        ::cuda::std::move(fn)(Shape(tid), static_cast<Args&&>(args)...);
      }
    }

    template <int BlockThreads, class... Args, std::integral Shape, class Fun>
    STDEXEC_ATTRIBUTE(launch_bounds(BlockThreads))
    __global__ void _multi_bulk_kernel(Shape begin, Shape end, Fun fn, Args... args)
    {
      static_assert(trivially_copyable<Shape, Fun, Args...>);
      Shape const i = begin + static_cast<Shape>(threadIdx.x + blockIdx.x * blockDim.x);

      if (i < end)
      {
        ::cuda::std::move(fn)(i, ::cuda::std::forward<Args>(args)...);
      }
    }

    // The state of a bulk operation whose child completes on a multi-GPU stream scheduler:
    // it also has a stream on each device.
    template <class Data, class Receiver>
    struct multi_gpu_state : adaptor_state<Data, Receiver>
    {
      template <class CvData>
      explicit multi_gpu_state(CvData&& data, Receiver rcvr, context ctx, int num_devices)
        : adaptor_state<Data, Receiver>(static_cast<CvData&&>(data),
                                        static_cast<Receiver&&>(rcvr),
                                        ctx)
        , num_devices_(num_devices)
        , streams_(new cudaStream_t[num_devices_])
        , ready_to_complete_(new cudaEvent_t[num_devices_])
      {
        // TODO Manage errors
        cudaGetDevice(&current_device_);
        cudaEventCreateWithFlags(&ready_to_launch_, cudaEventDisableTiming);
        for (int dev = 0; dev < num_devices_; dev++)
        {
          cudaSetDevice(dev);
          cudaStreamCreate(streams_.get() + dev);
          cudaEventCreateWithFlags(ready_to_complete_.get() + dev, cudaEventDisableTiming);
        }
        cudaSetDevice(current_device_);
      }

      ~multi_gpu_state()
      {
        // TODO Manage errors
        for (int dev = 0; dev < num_devices_; dev++)
        {
          cudaSetDevice(dev);
          cudaStreamDestroy(streams_[dev]);
          cudaEventDestroy(ready_to_complete_[dev]);
        }
        cudaSetDevice(current_device_);
        cudaEventDestroy(ready_to_launch_);
      }

      STDEXEC_IMMOVABLE(multi_gpu_state);

      int                             num_devices_{};
      int                             current_device_{};
      std::unique_ptr<cudaStream_t[]> streams_;
      std::unique_ptr<cudaEvent_t[]>  ready_to_complete_;
      cudaEvent_t                     ready_to_launch_;
    };

    template <class Shape>
    auto
    _even_share(Shape n, std::size_t rank, std::size_t size) noexcept -> std::pair<Shape, Shape>
    {
      auto const avg_per_thread = n / size;
      auto const n_big_share    = avg_per_thread + 1;
      auto const big_shares     = n % size;
      auto const is_big_share   = rank < big_shares;
      auto const begin          = is_big_share
                                  ? n_big_share * rank
                                  : n_big_share * big_shares + (rank - big_shares) * avg_per_thread;
      auto const end            = begin + (is_big_share ? n_big_share : avg_per_thread);

      return std::make_pair(begin, end);
    }

    // Launches the kernel on the state's stream
    template <class State, class... Args>
    void _launch(State& state, Args&... args) noexcept
    {
      auto const shape = state.data_.__shape_;
      if (shape)
      {
        cudaStream_t  stream        = state.get_stream();
        constexpr int block_threads = 256;
        int const     grid_blocks   = (static_cast<int>(shape) + block_threads - 1) / block_threads;
        _bulk_kernel<block_threads, Args&...>
          <<<grid_blocks, block_threads, 0, stream>>>(shape,
                                                      std::move(state.data_.__fun_),
                                                      args...);
      }
    }

    // Launches the kernel on each device, sharing the shape among them
    template <class Data, class Receiver, class... Args>
    void _launch(multi_gpu_state<Data, Receiver>& state, Args&... args) noexcept
    {
      // TODO Manage errors
      auto const   shape           = state.data_.__shape_;
      auto&        fun             = state.data_.__fun_;
      cudaStream_t baseline_stream = state.get_stream();
      cudaEventRecord(state.ready_to_launch_, baseline_stream);

      if (shape)
      {
        constexpr int block_threads = 256;
        for (int dev = 0; dev < state.num_devices_; dev++)
        {
          if (state.current_device_ != dev)
          {
            cudaStream_t stream  = state.streams_[dev];
            auto [begin, end]    = _bulk::_even_share(shape, dev, state.num_devices_);
            auto const count     = static_cast<int>(end - begin);
            int const  grid_size = (count + block_threads - 1) / block_threads;

            if (begin < end)
            {
              cudaSetDevice(dev);
              cudaStreamWaitEvent(stream, state.ready_to_launch_, 0);
              _multi_bulk_kernel<block_threads, Args&...>
                <<<grid_size, block_threads, 0, stream>>>(begin, end, fun, args...);
              cudaEventRecord(state.ready_to_complete_[dev], state.streams_[dev]);
            }
          }
        }

        {
          int const dev = state.current_device_;
          cudaSetDevice(dev);
          auto [begin, end]    = _bulk::_even_share(shape, dev, state.num_devices_);
          auto const count     = static_cast<int>(end - begin);
          int const  grid_size = (count + block_threads - 1) / block_threads;

          if (begin < end)
          {
            _multi_bulk_kernel<block_threads, Args&...>
              <<<grid_size, block_threads, 0, baseline_stream>>>(begin, end, fun, args...);
          }
        }

        for (int dev = 0; dev < state.num_devices_; dev++)
        {
          if (dev != state.current_device_)
          {
            cudaStreamWaitEvent(baseline_stream, state.ready_to_complete_[dev], 0);
          }
        }
      }
    }

    // bulk(sndr, pol, shape, fn): when the child completes with values, fn is called with
    // each index in [0, shape) and the values, in parallel on the stream, and then the values
    // are sent. If the child completes on a multi-GPU stream scheduler, the indices are
    // shared among the devices. The child's other completions are passed on.
    struct algo : stream_adaptor<algo>
    {
      // The scheduler on which the child completes
      template <class Sender, class Receiver>
      using _sched_t = decltype(STDEXEC::get_completion_scheduler<set_value_t>(
        STDEXEC::__facade::__nested_attrs_of<algo>(STDEXEC::__facade::__child<0>(),
                                                   __declval<__decay_t<Sender> const &>(),
                                                   __declval<env_of_t<Receiver> const &>()),
        algo::__env_for(STDEXEC::__facade::__child<0>(),
                        __declval<__decay_t<Sender> const &>(),
                        __declval<env_of_t<Receiver>>())));

      template <class Sender, class Receiver>
      static constexpr bool _multi_gpu = requires {
        __minvoke_or_q<_sched_t, stream_scheduler, Sender, Receiver>::num_devices_;
      } || requires(__minvoke_or_q<_sched_t, stream_scheduler, Sender, Receiver> sched) {
        sched.num_devices_;
      };

      template <class Sender, class Receiver>
      using _state_t =
        __if_c<_multi_gpu<Sender, Receiver>,
               multi_gpu_state<__decay_t<STDEXEC::__facade::__data_of_t<Sender>>, Receiver>,
               adaptor_state<__decay_t<STDEXEC::__facade::__data_of_t<Sender>>, Receiver>>;

      template <class Sender, class Receiver>
      static auto __get_state(Sender&& sndr, Receiver rcvr) -> _state_t<Sender, Receiver>
      {
        auto sched = STDEXEC::get_completion_scheduler<set_value_t>(
          STDEXEC::__facade::__nested_attrs_of<algo>(STDEXEC::__facade::__child<0>(),
                                                     sndr,
                                                     STDEXEC::get_env(rcvr)),
          algo::__env_for(STDEXEC::__facade::__child<0>(), sndr, STDEXEC::get_env(rcvr)));
        context ctx = sched.ctx_;
        if constexpr (_multi_gpu<Sender, Receiver>)
        {
          return _state_t<Sender, Receiver>(STDEXEC::__facade::__get_data(
                                              static_cast<Sender&&>(sndr)),
                                            static_cast<Receiver&&>(rcvr),
                                            ctx,
                                            sched.num_devices_);
        }
        else
        {
          return _state_t<Sender, Receiver>(STDEXEC::__facade::__get_data(
                                              static_cast<Sender&&>(sndr)),
                                            static_cast<Receiver&&>(rcvr),
                                            ctx);
        }
      }

      template <class State, class... Args>
      static auto
      __complete(STDEXEC::__facade::__child<0>, State& state, set_value_t, Args&&... args) noexcept
      {
        using data_t  = decltype(state.data_);
        using shape_t = decltype(data_t::__shape_);
        using fun_t   = decltype(data_t::__fun_);
        if constexpr (!__callable<fun_t, shape_t, Args&...>)
        {
          return STDEXEC::__facade::__type_error(
            __mexception<_WHAT_(_FUNCTION_IS_NOT_CALLABLE_WITH_THE_GIVEN_ARGUMENTS_),
                         _WHERE_(_IN_ALGORITHM_, STDEXEC::bulk_t),
                         _WITH_FUNCTION_(fun_t),
                         _WITH_ARGUMENTS_(shape_t, Args & ...)>());
        }
        else
        {
          _bulk::_launch(state, args...);
          cudaError_t status = STDEXEC_LOG_CUDA_API(cudaPeekAtLastError());
          return STDEXEC::__facade::__choose(
            status == cudaSuccess,
            [&]() noexcept
            { return _strm::_propagate(state, set_value_t(), static_cast<Args&&>(args)...); },
            [&]() noexcept { return _strm::_report_error(state, status); });
        }
      }

      using stream_adaptor<algo>::__complete;
    };
  }  // namespace _bulk

  template <>
  struct transform_sender_for<STDEXEC::bulk_t>
  {
    template <class Env, class Data, class Sender>
    auto operator()(Env const &, __ignore, Data&& data, Sender&& sndr) const
    {
      if constexpr (stream_completing_sender<Sender, Env>)
      {
        return STDEXEC::__facade::__sender{_strm::bulk_t(),
                                           static_cast<Data&&>(data),
                                           static_cast<Sender&&>(sndr)};
      }
      else
      {
        return _strm::_no_stream_scheduler_in_env<STDEXEC::bulk_t, Sender, Env>();
      }
    }
  };
}  // namespace nv::execution::_strm

template <>
struct STDEXEC::__facade::__algorithm<nv::execution::_strm::bulk_t>
  : nv::execution::_strm::_bulk::algo
{};

namespace nvexec = nv::execution;

STDEXEC_PRAGMA_POP()
