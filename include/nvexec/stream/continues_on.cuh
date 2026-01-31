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

#include "../detail/variant.cuh"
#include "common.cuh"

#include <cuda/std/tuple>

#include <cstddef>
#include <utility>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

namespace nvexec::_strm {
  namespace _trnsfr {
    template <class Tag, class Storage, class... Args>
    STDEXEC_ATTRIBUTE(launch_bounds(1))
    __global__ void _continues_on_kernel(Storage* storage, Args... args) {
      ::new (storage) Storage();
      storage->template emplace<decayed_tuple_t<Tag, Args...>>(Tag(), static_cast<Args&&>(args)...);
    }

    template <class CvSender, class Receiver>
    struct receiver : stream_receiver_base {
      using env_t = _strm::opstate_base<Receiver>::env_t;
      using storage_t = variant_storage_t<CvSender, env_t>;

      static constexpr std::size_t memory_allocation_size() noexcept {
        return sizeof(storage_t);
      }

      _strm::opstate_base<Receiver>& opstate_;

      template <class Tag, class... Args>
      void complete(Tag, Args&&... args) noexcept {
        using tuple_t = decayed_tuple_t<Tag, Args...>;

        // Args an optimization, if there are no values to persist to temporary
        // storage, skip it and simply propagate the completion signal.
        if constexpr (sizeof...(Args) == 0) {
          opstate_.propagate_completion_signal(Tag());
        } else {
          // If there are values in the completion channel, we have to construct
          // the temporary storage. If the values are trivially copyable, we launch
          // a _continues_on_kernel and construct the temporary storage on the device to avoid managed
          // memory movements. Otherwise, we construct the temporary storage on the host
          // and prefetch it to the device.
          auto* storage = static_cast<storage_t*>(opstate_.temp_storage_);
          constexpr bool construct_on_device = trivially_copyable<__decay_t<Args>...>;

          if constexpr (!construct_on_device) {
            ::new (storage) storage_t();
            storage->template emplace<tuple_t>(Tag(), static_cast<Args&&>(args)...);
          }

          int dev_id{};
          if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaGetDevice(&dev_id));
              status != cudaSuccess) {
            opstate_.propagate_completion_signal(STDEXEC::set_error, std::move(status));
            return;
          }

          int concurrent_managed_access{};
          if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaDeviceGetAttribute(
                &concurrent_managed_access, cudaDevAttrConcurrentManagedAccess, dev_id));
              status != cudaSuccess) {
            opstate_.propagate_completion_signal(STDEXEC::set_error, std::move(status));
            return;
          }

          cudaStream_t stream = opstate_.get_stream();

          if (concurrent_managed_access) {
            if (cudaError_t status = STDEXEC_LOG_CUDA_API(
                  cudaMemPrefetchAsync(storage, sizeof(storage_t), dev_id, stream));
                status != cudaSuccess) {
              opstate_.propagate_completion_signal(STDEXEC::set_error, std::move(status));
              return;
            }
          }

          if constexpr (construct_on_device) {
            _continues_on_kernel<Tag, storage_t, __decay_t<Args>...>
              <<<1, 1, 0, stream>>>(storage, args...);

            if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaPeekAtLastError());
                status != cudaSuccess) {
              opstate_.propagate_completion_signal(STDEXEC::set_error, std::move(status));
              return;
            }
          }

          opstate_.defer_temp_storage_destruction(storage);

          unsigned int index = __mapply<__mfind_i<tuple_t>, storage_t>::value;

          nvexec::visit(
            [&](auto& tpl) noexcept {
              ::cuda::std::apply(
                [&]<class Tag2, class... Bs>(Tag2, Bs&... tas) noexcept {
                  opstate_.propagate_completion_signal(Tag2(), std::move(tas)...);
                },
                tpl);
            },
            *storage,
            index);
        }
      }

      template <class... Args>
      void set_value(Args&&... args) noexcept {
        complete(set_value_t(), static_cast<Args&&>(args)...);
      }

      template <class Error>
      void set_error(Error&& __err) noexcept {
        complete(set_error_t(), static_cast<Error&&>(__err));
      }

      void set_stopped() noexcept {
        complete(set_stopped_t());
      }

      [[nodiscard]]
      auto get_env() const noexcept -> env_t {
        return opstate_.make_env();
      }
    };

    template <class Sender>
    struct source_sender : stream_sender_base {
      using schedule_from_sender_t = __result_of<schedule_from, Sender>;

      explicit source_sender(Sender sndr)
        : sndr_(schedule_from(static_cast<Sender&&>(sndr))) {
      }

      template <__decay_copyable Self, STDEXEC::receiver Receiver>
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver rcvr)
        -> connect_result_t<__copy_cvref_t<Self, schedule_from_sender_t>, Receiver> {
        return STDEXEC::connect(static_cast<Self&&>(self).sndr_, static_cast<Receiver&&>(rcvr));
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      template <__decay_copyable Self, class... Env>
      static consteval auto get_completion_signatures()
        -> __completion_signatures_of_t<__copy_cvref_t<Self, Sender>, Env...> {
        return {};
      }

      [[nodiscard]]
      auto get_env() const noexcept -> env_of_t<const Sender&> {
        // TODO - this code is not exercised by any test
        return STDEXEC::get_env(sndr_);
      }

     private:
      __result_of<schedule_from, Sender> sndr_;
    };

    template <class... Ty>
    using value_completions_t = completion_signatures<set_value_t(__decay_t<Ty>...)>;

    template <class Ty>
    using error_completions_t = completion_signatures<set_error_t(__decay_t<Ty>)>;
  } // namespace _trnsfr

  template <class Scheduler, class Sender>
  struct continues_on_sender : stream_sender_base {
    using source_sender_t = _trnsfr::source_sender<Sender>;

    template <class Self, class Receiver>
    using receiver_t = _trnsfr::receiver<__copy_cvref_t<Self, Sender>, Receiver>;

    explicit continues_on_sender(Scheduler sched, Sender sndr)
      : sched_(sched)
      , sndr_{static_cast<Sender&&>(sndr)} {
    }

    template <__decays_to<continues_on_sender> Self, receiver Receiver>
      requires sender_to<__copy_cvref_t<Self, source_sender_t>, Receiver>
    STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver rcvr) //
      -> stream_opstate_t<
        __copy_cvref_t<Self, source_sender_t>,
        receiver_t<Self, Receiver>,
        Receiver
      > {
      auto receiver_factory =
        [&](_strm::opstate_base<Receiver>& stream_provider) -> receiver_t<Self, Receiver> {
        return receiver_t<Self, Receiver>{{}, stream_provider};
      };

      return _strm::stream_opstate(
        static_cast<Self&&>(self).sndr_,
        static_cast<Receiver&&>(rcvr),
        receiver_factory,
        self.sched_.ctx_);
    }
    STDEXEC_EXPLICIT_THIS_END(connect)

    template <__decays_to<continues_on_sender> Self, class... Env>
    static consteval auto get_completion_signatures() -> transform_completion_signatures<
      __completion_signatures_of_t<__copy_cvref_t<Self, Sender>, Env...>,
      completion_signatures<set_error_t(cudaError_t)>,
      _trnsfr::value_completions_t,
      _trnsfr::error_completions_t
    > {
      return {};
    }

    [[nodiscard]]
    auto get_env() const noexcept -> __sched_attrs<Scheduler> {
      return {sched_};
    }

   private:
    Scheduler sched_;
    source_sender_t sndr_;
  };

  template <class Env>
  struct transform_sender_for<STDEXEC::continues_on_t, Env> {
    template <class Sched, class Sender>
    auto operator()(__ignore, Sched sched, Sender&& sndr) const {
      static_assert(gpu_stream_scheduler<Sched, Env>);
      using __sender_t = continues_on_sender<Sched, __decay_t<Sender>>;
      return __sender_t{sched, static_cast<Sender&&>(sndr)};
    }

    const Env& env_;
  };
} // namespace nvexec::_strm

// Decode the sender name for diagnostics:
namespace STDEXEC::__detail {
  template <class Scheduler, class Sender>
  extern __declfn_t<nvexec::_strm::continues_on_sender<Scheduler, __demangle_t<Sender>>>
    __demangle_v<nvexec::_strm::continues_on_sender<Scheduler, Sender>>;
} // namespace STDEXEC::__detail

STDEXEC_PRAGMA_POP()
