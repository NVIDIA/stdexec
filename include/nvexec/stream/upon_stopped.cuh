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
#include <cstddef>
#include <exception>
#include <type_traits>

#include <cuda/std/utility>

#include "common.cuh"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

namespace nvexec::_strm {
  namespace _upon_stopped {
    template <class Fun>
    STDEXEC_ATTRIBUTE(launch_bounds(1))
    __global__ void _upon_stopped_kernel(Fun fn) {
      static_assert(trivially_copyable<Fun>);
      ::cuda::std::move(fn)();
    }

    template <class Fun, class ResultT>
    STDEXEC_ATTRIBUTE(launch_bounds(1))
    __global__ void _upon_stopped_kernel_with_result(Fun fn, ResultT* result) {
      static_assert(trivially_copyable<Fun>);
      new (result) ResultT(::cuda::std::move(fn)());
    }

    template <class Receiver, class Fun>
    struct receiver : public stream_receiver_base {
      using receiver_concept = STDEXEC::receiver_t;
      using _result_t = std::invoke_result_t<Fun>;
      using _env_t = _strm::opstate_base<Receiver>::env_t;

      static constexpr std::size_t memory_allocation_size() noexcept {
        return _sizeof_v<_result_t>;
      }

      explicit receiver(Fun fun, _strm::opstate_base<Receiver>& opstate)
        : fun_(static_cast<Fun&&>(fun))
        , opstate_(opstate) {
      }

      template <class... Args>
      void set_value(Args&&... args) noexcept {
        opstate_.propagate_completion_signal(set_value_t(), static_cast<Args&&>(args)...);
      }

      template <class Error>
      void set_error(Error __err) noexcept {
        opstate_.propagate_completion_signal(set_error_t(), static_cast<Error&&>(__err));
      }

      void set_stopped() noexcept {
        constexpr bool does_not_return_a_value = std::is_same_v<void, _result_t>;
        cudaStream_t stream = opstate_.get_stream();

        if constexpr (does_not_return_a_value) {
          _upon_stopped_kernel<<<1, 1, 0, stream>>>(std::move(fun_));
          if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaPeekAtLastError());
              status == cudaSuccess) {
            opstate_.propagate_completion_signal(STDEXEC::set_value);
          } else {
            opstate_.propagate_completion_signal(STDEXEC::set_error, std::move(status));
          }
        } else {
          using decayed_result_t = __decay_t<_result_t>;
          auto* d_result = static_cast<decayed_result_t*>(opstate_.temp_storage_);
          _upon_stopped_kernel_with_result<<<1, 1, 0, stream>>>(std::move(fun_), d_result);
          if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaPeekAtLastError());
              status == cudaSuccess) {
            opstate_.defer_temp_storage_destruction(d_result);
            opstate_.propagate_completion_signal(STDEXEC::set_value, *d_result);
          } else {
            opstate_.propagate_completion_signal(STDEXEC::set_error, std::move(status));
          }
        }
      }

      [[nodiscard]]
      auto get_env() const noexcept -> _env_t {
        return opstate_.make_env();
      }

     private:
      Fun fun_;
      _strm::opstate_base<Receiver>& opstate_;
    };
  } // namespace _upon_stopped

  template <class Sender, class Fun>
  struct upon_stopped_sender : stream_sender_base {
    using sender_concept = STDEXEC::sender_t;
    using _set_error_t = completion_signatures<set_error_t(std::exception_ptr)>;

    template <class Receiver>
    using receiver_t = _upon_stopped::receiver<Receiver, Fun>;

    template <class Self, class... Env>
    using completion_signatures = transform_completion_signatures<
      __completion_signatures_of_t<__copy_cvref_t<Self, Sender>, Env...>,
      __with_error_invoke_t<
        __mbind_front_q<__callable_error_t, upon_stopped_t>,
        set_stopped_t,
        Fun,
        __copy_cvref_t<Self, Sender>,
        Env...
      >,
      __cmplsigs::__default_set_value,
      __cmplsigs::__default_set_error,
      __set_value_from_t<Fun>
    >;

    explicit upon_stopped_sender(Sender sndr, Fun fun)
      noexcept(__nothrow_move_constructible<Sender, Fun>)
      : sndr_(static_cast<Sender&&>(sndr))
      , fun_(static_cast<Fun&&>(fun)) {
    }

    template <__decays_to<upon_stopped_sender> Self, STDEXEC::receiver Receiver>
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

    template <__decays_to<upon_stopped_sender> Self, class... Env>
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
  struct transform_sender_for<STDEXEC::upon_stopped_t, Env> {
    template <class Fun, stream_completing_sender<Env> CvSender>
    auto operator()(__ignore, Fun fun, CvSender&& sndr) const {
      using _sender_t = upon_stopped_sender<__decay_t<CvSender>, Fun>;
      return _sender_t{static_cast<CvSender&&>(sndr), static_cast<Fun&&>(fun)};
    }

    const Env& env_;
  };
} // namespace nvexec::_strm

namespace STDEXEC::__detail {
  template <class Sender, class Fun>
  extern __declfn_t<nvexec::_strm::upon_stopped_sender<__demangle_t<Sender>, Fun>>
    __demangle_v<nvexec::_strm::upon_stopped_sender<Sender, Fun>>;
} // namespace STDEXEC::__detail

STDEXEC_PRAGMA_POP()
