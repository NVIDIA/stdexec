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

#include "../../exec/completion_signatures.hpp"

#include <concepts>
#include <memory>
#include <utility>

#include "common.cuh"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

#if defined(STDEXEC_CLANGD_INVOKED) && STDEXEC_CLANG() && STDEXEC_CUDA_COMPILATION()
// clangd doesn't understand CUDA's new/delete operators
__host__ auto operator new[](std::size_t) -> void*;
#endif

namespace nv::execution::_strm
{
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

    template <class Receiver, std::integral Shape, class Fun>
    struct receiver : public stream_receiver_base
    {
     private:
      using env_t = _strm::opstate_base<Receiver>::env_t;

      Shape                          shape_;
      Fun                            fun_;
      _strm::opstate_base<Receiver>& opstate_;

     public:
      explicit receiver(Shape shape, Fun fun, _strm::opstate_base<Receiver>& opstate)
        : shape_(shape)
        , fun_(static_cast<Fun&&>(fun))
        , opstate_(opstate)
      {}

      template <class... Args>
        requires __callable<Fun, Shape, Args&...>
      void set_value(Args&&... args) noexcept
      {
        _strm::opstate_base<Receiver>& opstate = opstate_;

        if (shape_)
        {
          cudaStream_t  stream        = opstate.get_stream();
          constexpr int block_threads = 256;
          int const grid_blocks = (static_cast<int>(shape_) + block_threads - 1) / block_threads;
          _bulk_kernel<block_threads, Args&...>
            <<<grid_blocks, block_threads, 0, stream>>>(shape_, std::move(fun_), args...);
        }

        if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaPeekAtLastError()); status == cudaSuccess)
        {
          opstate.propagate_completion_signal(STDEXEC::set_value, static_cast<Args&&>(args)...);
        }
        else
        {
          opstate.propagate_completion_signal(STDEXEC::set_error, std::move(status));
        }
      }

      template <class Error>
      void set_error(Error&& err) noexcept
      {
        opstate_.propagate_completion_signal(set_error_t(), static_cast<Error&&>(err));
      }

      void set_stopped() noexcept
      {
        opstate_.propagate_completion_signal(set_stopped_t());
      }

      [[nodiscard]]
      auto get_env() const noexcept -> env_t
      {
        return opstate_.make_env();
      }
    };

    template <class Fun, std::integral Shape>
    inline constexpr auto value_cmplsigs_transform()
    {
      return []<class... Args>()
      {
        if constexpr (__callable<Fun, Shape, Args&...>)
        {
          return completion_signatures<set_value_t(Args...), set_error_t(cudaError_t)>();
        }
        else
        {
          return exec::throw_compile_time_error<
            _WHAT_(_FUNCTION_IS_NOT_CALLABLE_WITH_THE_GIVEN_ARGUMENTS_),
            _WHERE_(_IN_ALGORITHM_, bulk_t),
            _WITH_FUNCTION_(Fun),
            _WITH_ARGUMENTS_(Shape, Args & ...)>();
        }
      };
    }
  }  // namespace _bulk

  template <class Sender, std::integral Shape, class Fun>
  struct bulk_sender : stream_sender_base
  {
    template <class Receiver>
    using receiver_t = _bulk::receiver<Receiver, Shape, Fun>;

    template <__decays_to<bulk_sender> Self, receiver Receiver>
    STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver rcvr)
      -> stream_opstate_t<__copy_cvref_t<Self, Sender>, receiver_t<Receiver>, Receiver>
    {
      return stream_opstate<__copy_cvref_t<Self, Sender>>(
        static_cast<Self&&>(self).sndr_,
        static_cast<Receiver&&>(rcvr),
        [&](_strm::opstate_base<Receiver>& stream_provider) -> receiver_t<Receiver>
        {
          return receiver_t<Receiver>(self.shape_, static_cast<Self&&>(self).fun_, stream_provider);
        });
    }
    STDEXEC_EXPLICIT_THIS_END(connect)

    template <__decays_to<bulk_sender> Self, class... Env>
    static consteval auto get_completion_signatures()
    {
      return exec::transform_completion_signatures(
        STDEXEC::get_completion_signatures<__copy_cvref_t<Self, Sender>, Env...>(),
        _bulk::value_cmplsigs_transform<Fun, Shape>());
    }

    auto get_env() const noexcept -> stream_sender_attrs<Sender>
    {
      return {&sndr_};
    }

    Sender sndr_;
    Shape  shape_;
    Fun    fun_;
  };

  namespace multi_gpu_bulk
  {
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

    template <class CvSender, class Receiver, class Shape, class Fun>
    struct opstate;

    template <class CvSender, class Receiver, std::integral Shape, class Fun>
    struct receiver : public stream_receiver_base
    {
      explicit receiver(Shape shape, Fun fun, opstate<CvSender, Receiver, Shape, Fun>& opstate)
        : shape_(shape)
        , fun_(static_cast<Fun&&>(fun))
        , opstate_(opstate)
      {}

      template <class... Args>
        requires __callable<Fun, Shape, Args&...>
      void set_value(Args&&... args) noexcept
      {
        // TODO Manage errors
        // TODO Usual logic when there's only a single GPU
        cudaStream_t baseline_stream = opstate_.get_stream();
        cudaEventRecord(opstate_.ready_to_launch_, baseline_stream);

        if (shape_)
        {
          constexpr int block_threads = 256;
          for (int dev = 0; dev < opstate_.num_devices_; dev++)
          {
            if (opstate_.current_device_ != dev)
            {
              cudaStream_t stream   = opstate_.streams_[dev];
              auto [begin, end]     = even_share(shape_, dev, opstate_.num_devices_);
              auto      shape       = static_cast<int>(end - begin);
              int const grid_blocks = (shape + block_threads - 1) / block_threads;

              if (begin < end)
              {
                cudaSetDevice(dev);
                cudaStreamWaitEvent(stream, opstate_.ready_to_launch_, 0);
                _multi_bulk_kernel<block_threads, Args&...>
                  <<<grid_blocks, block_threads, 0, stream>>>(begin, end, fun_, args...);
                cudaEventRecord(opstate_.ready_to_complete_[dev], opstate_.streams_[dev]);
              }
            }
          }

          {
            int const dev = opstate_.current_device_;
            cudaSetDevice(dev);
            auto [begin, end]     = even_share(shape_, dev, opstate_.num_devices_);
            auto      shape       = static_cast<int>(end - begin);
            int const grid_blocks = (shape + block_threads - 1) / block_threads;

            if (begin < end)
            {
              _multi_bulk_kernel<block_threads, Args&...>
                <<<grid_blocks, block_threads, 0, baseline_stream>>>(begin, end, fun_, args...);
            }
          }

          for (int dev = 0; dev < opstate_.num_devices_; dev++)
          {
            if (dev != opstate_.current_device_)
            {
              cudaStreamWaitEvent(baseline_stream, opstate_.ready_to_complete_[dev], 0);
            }
          }
        }

        if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaPeekAtLastError()); status == cudaSuccess)
        {
          opstate_.propagate_completion_signal(STDEXEC::set_value, static_cast<Args&&>(args)...);
        }
        else
        {
          opstate_.propagate_completion_signal(STDEXEC::set_error, std::move(status));
        }
      }

      template <class Error>
      void set_error(Error&& __err) noexcept
      {
        opstate_.propagate_completion_signal(set_error_t(), static_cast<Error&&>(__err));
      }

      void set_stopped() noexcept
      {
        opstate_.propagate_completion_signal(set_stopped_t());
      }

      [[nodiscard]]
      auto get_env() const noexcept -> env_of_t<Receiver>
      {
        return STDEXEC::get_env(opstate_.rcvr_);
      }

     private:
      static auto
      even_share(Shape n, std::size_t rank, std::size_t size) noexcept -> std::pair<Shape, Shape>
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

      Shape                                    shape_;
      Fun                                      fun_;
      opstate<CvSender, Receiver, Shape, Fun>& opstate_;
    };

    template <class Sender, class Receiver, class Shape, class Fun>
    using opstate_base_t = _strm::opstate<Sender, receiver<Sender, Receiver, Shape, Fun>, Receiver>;

    template <class CvSender, class Receiver, class Shape, class Fun>
    struct opstate : opstate_base_t<CvSender, Receiver, Shape, Fun>
    {
      explicit opstate(int        num_devices,
                       CvSender&& sndr,
                       Receiver   rcvr,
                       Shape      shape,
                       Fun        fun,
                       context    ctx)
        : opstate_base_t<CvSender, Receiver, Shape, Fun>(
            static_cast<CvSender&&>(sndr),
            static_cast<Receiver&&>(rcvr),
            [&](_strm::opstate_base<Receiver>&) -> receiver<CvSender, Receiver, Shape, Fun>
            { return receiver<CvSender, Receiver, Shape, Fun>(shape, std::move(fun), *this); },
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

      ~opstate()
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

      STDEXEC_IMMOVABLE(opstate);

      int                             num_devices_{};
      int                             current_device_{};
      std::unique_ptr<cudaStream_t[]> streams_;
      std::unique_ptr<cudaEvent_t[]>  ready_to_complete_;
      cudaEvent_t                     ready_to_launch_;
    };
  }  // namespace multi_gpu_bulk

  template <class Sender, class Shape, class Fun>
  struct multi_gpu_bulk_sender : stream_sender_base
  {
    static_assert(std::integral<Shape>);
    using sender_concept = STDEXEC::sender_tag;

    template <__decays_to<multi_gpu_bulk_sender> Self, receiver Receiver>
    STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver&& rcvr)
      -> multi_gpu_bulk::opstate<__copy_cvref_t<Self, Sender>, Receiver, Shape, Fun>
    {
      auto    sch = STDEXEC::get_completion_scheduler<set_value_t>(STDEXEC::get_env(self.sndr_),
                                                                STDEXEC::get_env(rcvr));
      context ctx = sch.ctx_;
      return multi_gpu_bulk::opstate<__copy_cvref_t<Self, Sender>, Receiver, Shape, Fun>(
        self.num_devices_,
        static_cast<Self&&>(self).sndr_,
        static_cast<Receiver&&>(rcvr),
        self.shape_,
        static_cast<Self&&>(self).fun_,
        ctx);
    }
    STDEXEC_EXPLICIT_THIS_END(connect)

    template <__decays_to<multi_gpu_bulk_sender> Self, class... Env>
    static consteval auto get_completion_signatures()
    {
      return exec::transform_completion_signatures(
        STDEXEC::get_completion_signatures<__copy_cvref_t<Self, Sender>, Env...>(),
        _bulk::value_cmplsigs_transform<Fun, Shape>());
    }

    auto get_env() const noexcept -> stream_sender_attrs<Sender>
    {
      return {&sndr_};
    }

    int    num_devices_;
    Sender sndr_;
    Shape  shape_;
    Fun    fun_;
  };

  template <>
  struct transform_sender_for<STDEXEC::bulk_t>
  {
    template <class Env, class Data, class Sender>
    auto operator()(Env const & env, __ignore, Data data, Sender&& sndr) const
    {
      if constexpr (stream_completing_sender<Sender, Env>)
      {
        auto [policy, shape, fun] = static_cast<Data&&>(data);
        using shape_t             = decltype(shape);
        using fun_t               = decltype(fun);
        auto sched                = get_completion_scheduler<set_value_t>(get_env(sndr), env);
        if constexpr (__std::same_as<decltype(sched), stream_scheduler>)
        {
          // Use the bulk sender for a single GPU
          using _sender_t = bulk_sender<__decay_t<Sender>, shape_t, fun_t>;
          return _sender_t{{}, static_cast<Sender&&>(sndr), shape, static_cast<fun_t&&>(fun)};
        }
        else
        {
          // Use the bulk sender for a multiple GPUs
          using _sender_t = multi_gpu_bulk_sender<__decay_t<Sender>, shape_t, fun_t>;
          return _sender_t{{},
                           sched.num_devices_,
                           static_cast<Sender&&>(sndr),
                           shape,
                           static_cast<fun_t&&>(fun)};
        }
      }
      else
      {
        return _strm::_no_stream_scheduler_in_env<STDEXEC::bulk_t, Sender, Env>();
      }
    }
  };
}  // namespace nv::execution::_strm

namespace nvexec = nv::execution;

STDEXEC_PRAGMA_POP()
