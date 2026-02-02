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

#include "common.cuh"

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <cuda/std/utility>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

namespace nvexec::_strm {
  namespace _upon_error {
    template <class... Args, class Fun>
    STDEXEC_ATTRIBUTE(launch_bounds(1))
    __global__ void _upon_error_kernel(Fun fn, Args... args) {
      static_assert(trivially_copyable<Fun, Args...>);
      ::cuda::std::move(fn)(static_cast<Args&&>(args)...);
    }

    template <class... Args, class Fun, class ResultT>
    STDEXEC_ATTRIBUTE(launch_bounds(1))
    __global__ void _upon_error_kernel_with_result(Fun fn, ResultT* result, Args... args) {
      static_assert(trivially_copyable<Fun, Args...>);
      new (result) ResultT(::cuda::std::move(fn)(static_cast<Args&&>(args)...));
    }

    template <std::size_t MemoryAllocationSize, class Receiver, class Fun>
    struct receiver : stream_receiver_base {
      using receiver_concept = STDEXEC::receiver_t;
      using env_t = _strm::opstate_base<Receiver>::env_t;

      static constexpr std::size_t memory_allocation_size() noexcept {
        return MemoryAllocationSize;
      }

      explicit receiver(Fun fun, _strm::opstate_base<Receiver>& opstate)
        : f_(static_cast<Fun&&>(fun))
        , opstate_(opstate) {
      }

      template <class... Args>
      void set_value(Args&&... args) noexcept {
        opstate_.propagate_completion_signal(set_value_t(), static_cast<Args&&>(args)...);
      }

      template <class Error>
        requires std::invocable<Fun, Error>
      void set_error(Error&& error) noexcept {
        using result_t = std::invoke_result_t<Fun, Error>;
        constexpr bool does_not_return_a_value = std::is_same_v<void, result_t>;
        cudaStream_t stream = opstate_.get_stream();

        if constexpr (does_not_return_a_value) {
          _upon_error_kernel<Error&&>
            <<<1, 1, 0, stream>>>(std::move(f_), static_cast<Error&&>(error));
          if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaPeekAtLastError());
              status == cudaSuccess) {
            opstate_.propagate_completion_signal(STDEXEC::set_value);
          } else {
            opstate_.propagate_completion_signal(STDEXEC::set_error, std::move(status));
          }
        } else {
          using decayed_result_t = __decay_t<result_t>;
          auto* d_result = static_cast<decayed_result_t*>(opstate_.temp_storage_);
          _upon_error_kernel_with_result<Error&&>
            <<<1, 1, 0, stream>>>(std::move(f_), d_result, static_cast<Error&&>(error));
          if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaPeekAtLastError());
              status == cudaSuccess) {
            opstate_.defer_temp_storage_destruction(d_result);
            opstate_.propagate_completion_signal(STDEXEC::set_value, std::move(*d_result));
          } else {
            opstate_.propagate_completion_signal(STDEXEC::set_error, std::move(status));
          }
        }
      }

      void set_stopped() noexcept {
        opstate_.propagate_completion_signal(set_stopped_t());
      }

      [[nodiscard]]
      auto get_env() const noexcept -> env_t {
        return opstate_.make_env();
      }

      Fun f_;
      _strm::opstate_base<Receiver>& opstate_;
    };
  } // namespace _upon_error

  template <class Sender, class Fun>
  struct upon_error_sender : stream_sender_base {
    template <class Receiver>
      requires sender_in<Sender, env_of_t<Receiver>>
    struct max_result_size
      : STDEXEC::__gather_completions_of_t<
          set_error_t,
          Sender,
          env_of_t<Receiver>,
          __mbind_front<result_size_for, Fun>,
          maxsize
        > { };

    template <class Receiver>
    using receiver_t = _upon_error::receiver<max_result_size<Receiver>::value, Receiver, Fun>;

    template <class Error>
    using _set_error_t = __set_value_from_t<Fun, Error>;

    template <class Self, class... Env>
    using completion_signatures = transform_completion_signatures<
      __completion_signatures_of_t<__copy_cvref_t<Self, Sender>, Env...>,
      completion_signatures<set_error_t(cudaError_t)>,
      __cmplsigs::__default_set_value,
      _set_error_t
    >;

    explicit upon_error_sender(Sender sndr, Fun fun)
      noexcept(__nothrow_move_constructible<Sender, Fun>)
      : sndr_(static_cast<Sender&&>(sndr))
      , fun_(static_cast<Fun&&>(fun)) {
    }

    template <__decays_to<upon_error_sender> Self, receiver Receiver>
      requires receiver_of<Receiver, completion_signatures<Self, env_of_t<Receiver>>>
    STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver rcvr)
      -> stream_opstate_t<__copy_cvref_t<Self, Sender>, receiver_t<Receiver>, Receiver> {
      return stream_opstate<__copy_cvref_t<Self, Sender>>(
        static_cast<Self&&>(self).sndr_,
        static_cast<Receiver&&>(rcvr),
        [&](_strm::opstate_base<Receiver>& stream_provider) -> receiver_t<Receiver> {
          return receiver_t<Receiver>(self.fun_, stream_provider);
        });
    }
    STDEXEC_EXPLICIT_THIS_END(connect)

    template <__decays_to<upon_error_sender> Self, class... Env>
    static consteval auto get_completion_signatures() -> completion_signatures<Self, Env...> {
      return {};
    }

    auto get_env() const noexcept -> stream_sender_attrs<Sender> {
      return {&sndr_};
    }

   private:
    Sender sndr_;
    Fun fun_;
  };

  template <class Env>
  struct transform_sender_for<STDEXEC::upon_error_t, Env> {
    template <class Fun, stream_completing_sender<Env> Sender>
    auto operator()(__ignore, Fun fun, Sender&& sndr) const {
      using _sender_t = upon_error_sender<__decay_t<Sender>, Fun>;
      return _sender_t{static_cast<Sender&&>(sndr), static_cast<Fun&&>(fun)};
    }

    const Env& env_;
  };
} // namespace nvexec::_strm

namespace STDEXEC::__detail {
  template <class Sender, class Fun>
  extern __declfn_t<nvexec::_strm::upon_error_sender<__demangle_t<Sender>, Fun>>
    __demangle_v<nvexec::_strm::upon_error_sender<Sender, Fun>>;
} // namespace STDEXEC::__detail

STDEXEC_PRAGMA_POP()
