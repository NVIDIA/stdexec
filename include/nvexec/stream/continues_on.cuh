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

#include "../detail/variant.cuh"

#include "common.cuh"

#include <cuda/std/tuple>

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

namespace nv::execution::_strm
{
  namespace _trnsfr
  {
    template <class Tag, class Storage, class... Args>
    STDEXEC_ATTRIBUTE(launch_bounds(1))
    __global__ void _continues_on_kernel(Storage* storage, Args... args)
    {
      ::new (storage) Storage();
      storage->template emplace<decayed_tuple_t<Tag, Args...>>(Tag(), static_cast<Args&&>(args)...);
    }

    template <class CvSender, class Receiver>
    struct receiver : stream_receiver_base
    {
      using env_t     = terminal_stream_env_t<env_of_t<Receiver>>;
      using storage_t = variant_storage_t<CvSender, env_t>;

      static constexpr std::size_t memory_allocation_size() noexcept
      {
        return sizeof(storage_t);
      }

      _strm::opstate_base<Receiver>& opstate_;
      // Set when the work before the transition runs on another scheduler.
      stream_provider* src_provider_{nullptr};

      template <class Tag, class... Args>
      void complete(Tag, Args&&... args) noexcept
      {
        using tuple_t = decayed_tuple_t<Tag, Args...>;

        // Args an optimization, if there are no values to persist to temporary
        // storage, skip it and simply propagate the completion signal.
        if constexpr (sizeof...(Args) == 0)
        {
          opstate_.propagate_completion_signal(Tag());
        }
        else
        {
          // If there are values in the completion channel, we have to construct
          // the temporary storage. If the values are trivially copyable, we launch
          // a _continues_on_kernel and construct the temporary storage on the device to avoid managed
          // memory movements. Otherwise, we construct the temporary storage on the host
          // and prefetch it to the device.
          auto*          storage             = static_cast<storage_t*>(opstate_.temp_storage_);
          constexpr bool construct_on_device = trivially_copyable<__decay_t<Args>...>;

          if constexpr (!construct_on_device)
          {
            ::new (storage) storage_t();
            storage->template emplace<tuple_t>(Tag(), static_cast<Args&&>(args)...);
          }

          auto complete_error = [storage, &opstate = opstate_](cudaError_t status) noexcept
          {
            if constexpr (!construct_on_device)
            {
              storage->~storage_t();
            }
            opstate.propagate_completion_signal(STDEXEC::set_error, std::move(status));
          };

          int dev_id{};
          if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaGetDevice(&dev_id));
              status != cudaSuccess)
          {
            complete_error(std::move(status));
            return;
          }

          int concurrent_managed_access{};
          if (cudaError_t status = STDEXEC_LOG_CUDA_API(
                cudaDeviceGetAttribute(&concurrent_managed_access,
                                       cudaDevAttrConcurrentManagedAccess,
                                       dev_id));
              status != cudaSuccess)
          {
            complete_error(std::move(status));
            return;
          }

          cudaStream_t stream = opstate_.get_stream();

          if (concurrent_managed_access)
          {
            if (cudaError_t status = STDEXEC_LOG_CUDA_API(
                  cudaMemPrefetchAsync(storage, sizeof(storage_t), dev_id, stream));
                status != cudaSuccess)
            {
              complete_error(std::move(status));
              return;
            }
          }

          if constexpr (construct_on_device)
          {
            _continues_on_kernel<Tag, storage_t, __decay_t<Args>...>
              <<<1, 1, 0, stream>>>(storage, args...);

            if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaPeekAtLastError());
                status != cudaSuccess)
            {
              complete_error(std::move(status));
              return;
            }
          }

          opstate_.defer_temp_storage_destruction(storage);

          unsigned int index = __mapply<__mfind_i<tuple_t>, storage_t>::value;

          nvexec::visit(
            [&](auto& tpl) noexcept
            {
              ::cuda::std::apply(
                [&]<class Tag2, class... Bs>(Tag2, Bs&... tas) noexcept
                { opstate_.propagate_completion_signal(Tag2(), std::move(tas)...); },
                tpl);
            },
            *storage,
            index);
        }
      }

      template <class... Args>
      void set_value(Args&&... args) noexcept
      {
        complete(set_value_t(), static_cast<Args&&>(args)...);
      }

      template <class Error>
      void set_error(Error&& __err) noexcept
      {
        complete(set_error_t(), static_cast<Error&&>(__err));
      }

      void set_stopped() noexcept
      {
        complete(set_stopped_t());
      }

      [[nodiscard]]
      auto get_env() const noexcept -> env_t
      {
        return make_terminal_stream_env(STDEXEC::get_env(opstate_.rcvr_),
                                        src_provider_ ? src_provider_
                                                      : opstate_.get_stream_provider());
      }
    };

    template <class CvSourceSender, class Receiver>
    struct opstate
    {
      using operation_state_concept = STDEXEC::operation_state_tag;
      using receiver_t              = receiver<CvSourceSender, Receiver>;
      using inner_opstate_t         = stream_opstate_t<CvSourceSender, receiver_t, Receiver>;

      opstate(CvSourceSender&& sndr, Receiver rcvr, context dst_ctx, std::optional<context> src_ctx)
        : src_provider_(src_ctx ? std::make_unique<stream_provider>(*src_ctx) : nullptr)
        , inner_(_strm::stream_opstate(
            static_cast<CvSourceSender&&>(sndr),
            static_cast<Receiver&&>(rcvr),
            [this](_strm::opstate_base<Receiver>& op) -> receiver_t
            { return receiver_t{{}, op, src_provider_.get()}; },
            dst_ctx))
      {}

      STDEXEC_IMMOVABLE(opstate);

      void start() & noexcept
      {
        if (src_provider_ && src_provider_->status_ != cudaSuccess)
        {
          inner_.propagate_completion_signal(STDEXEC::set_error,
                                             cudaError_t(src_provider_->status_));
          return;
        }
        STDEXEC::start(inner_);
      }

     private:
      std::unique_ptr<stream_provider> src_provider_;
      inner_opstate_t                  inner_;
    };

    template <class Sender>
    struct source_sender : stream_sender_base
    {
      explicit source_sender(Sender sndr)
        : sndr_(static_cast<Sender&&>(sndr))
      {}

      template <__decay_copyable Self, STDEXEC::receiver Receiver>
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver rcvr)
        -> connect_result_t<__copy_cvref_t<Self, Sender>, Receiver>
      {
        return STDEXEC::connect(static_cast<Self&&>(self).sndr_, static_cast<Receiver&&>(rcvr));
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      template <__decay_copyable Self, class... Env>
      static consteval auto get_completion_signatures()
      {
        return exec::concat_completion_signatures(
          STDEXEC::get_completion_signatures<__copy_cvref_t<Self, Sender>, Env...>(),
          STDEXEC::completion_signatures<set_error_t(cudaError_t)>());
      }

      [[nodiscard]]
      auto get_env() const noexcept -> env_of_t<Sender const &>
      {
        // TODO - this code is not exercised by any test
        return STDEXEC::get_env(sndr_);
      }

     private:
      Sender sndr_;
    };
  }  // namespace _trnsfr

  template <class Scheduler, class Sender>
  struct continues_on_sender : stream_sender_base
  {
    using source_sender_t = _trnsfr::source_sender<Sender>;

    explicit continues_on_sender(Scheduler sched, Sender sndr)
      : sched_(sched)
      , sndr_{static_cast<Sender&&>(sndr)}
    {}

    template <class Self, class Receiver>
    using opstate_t = _trnsfr::opstate<__copy_cvref_t<Self, source_sender_t>, Receiver>;

    template <__decays_to<continues_on_sender> Self, receiver Receiver>
    STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver rcvr)  //
      -> opstate_t<Self, Receiver>
    {
      // Work before the transition keeps running on its own scheduler's stream.
      std::optional<context> src_ctx;
      if constexpr (stream_sender<Sender, env_of_t<Receiver>>
                    && stream_completing_sender<Sender, env_of_t<Receiver>>)
      {
        auto src_sched = get_completion_scheduler<set_value_t>(STDEXEC::get_env(self.sndr_),
                                                               STDEXEC::get_env(rcvr));
        if (src_sched != self.sched_)
        {
          src_ctx = src_sched.ctx_;
        }
      }

      return opstate_t<Self, Receiver>{static_cast<Self&&>(self).sndr_,
                                       static_cast<Receiver&&>(rcvr),
                                       self.sched_.ctx_,
                                       src_ctx};
    }
    STDEXEC_EXPLICIT_THIS_END(connect)

    template <__decays_to<continues_on_sender> Self, class... Env>
    static consteval auto get_completion_signatures()
    {
      return exec::transform_completion_signatures(
        STDEXEC::get_completion_signatures<__copy_cvref_t<Self, Sender>, Env...>(),
        exec::decay_arguments<set_value_t, continues_on_t>(),
        exec::decay_arguments<set_error_t, continues_on_t>(),
        {},
        completion_signatures<set_error_t(cudaError_t)>());
    }

    [[nodiscard]]
    auto get_env() const noexcept -> __sched_attrs<Scheduler>
    {
      return {sched_};
    }

   private:
    Scheduler       sched_;
    source_sender_t sndr_;
  };

  template <>
  struct transform_sender_for<STDEXEC::continues_on_t>
  {
    template <class Env, class Sched, class Sender>
    auto operator()(Env const &, __ignore, Sched sched, Sender&& sndr) const
    {
      static_assert(gpu_stream_scheduler<Sched, Env>);
      using __sender_t = continues_on_sender<Sched, __decay_t<Sender>>;
      return __sender_t{sched, static_cast<Sender&&>(sndr)};
    }
  };
}  // namespace nv::execution::_strm

namespace nvexec = nv::execution;

STDEXEC_PRAGMA_POP()
