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
#include <cstddef>
#include <type_traits>
#include <utility>

#include <cuda/std/utility>

#include "common.cuh"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

namespace nvexec::_strm {

  namespace _then {
    template <class... Args, class Fun>
    STDEXEC_ATTRIBUTE(launch_bounds(1))
    __global__ void _then_kernel(Fun fn, Args... args) {
      static_assert(trivially_copyable<Fun, Args...>);
      ::cuda::std::move(fn)(static_cast<Args&&>(args)...);
    }

    template <class... Args, class Fun, class ResultT>
    STDEXEC_ATTRIBUTE(launch_bounds(1))
    __global__ void _then_kernel_with_result(Fun fn, ResultT* result, Args... args) {
      static_assert(trivially_copyable<Fun, Args...>);
      new (result) ResultT(::cuda::std::move(fn)(static_cast<Args&&>(args)...));
    }

    template <std::size_t MemoryAllocationSize, class Receiver, class Fun>
    struct receiver : public stream_receiver_base {
      using receiver_concept = STDEXEC::receiver_t;
      static constexpr std::size_t memory_allocation_size() noexcept {
        return MemoryAllocationSize;
      }

      explicit receiver(Fun fun, _strm::opstate_base<Receiver>& opstate)
        : f_(static_cast<Fun&&>(fun))
        , opstate_(opstate) {
      }

      template <class... Args>
        requires std::invocable<Fun, __decay_t<Args>...>
      void set_value(Args&&... args) noexcept {
        using result_t = std::invoke_result_t<Fun, __decay_t<Args>...>;
        constexpr bool does_not_return_a_value = std::is_same_v<void, result_t>;
        _strm::opstate_base<Receiver>& opstate = opstate_;
        cudaStream_t stream = opstate.get_stream();

        if constexpr (does_not_return_a_value) {
          _then_kernel<Args&&...><<<1, 1, 0, stream>>>(std::move(f_), static_cast<Args&&>(args)...);

          if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaPeekAtLastError());
              status == cudaSuccess) {
            opstate.propagate_completion_signal(STDEXEC::set_value);
          } else {
            opstate.propagate_completion_signal(STDEXEC::set_error, std::move(status));
          }
        } else {
          using decayed_result_t = __decay_t<result_t>;
          auto* d_result = static_cast<decayed_result_t*>(opstate.temp_storage_);
          _then_kernel_with_result<Args&&...>
            <<<1, 1, 0, stream>>>(std::move(f_), d_result, static_cast<Args&&>(args)...);
          opstate.defer_temp_storage_destruction(d_result);

          if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaPeekAtLastError());
              status == cudaSuccess) {
            opstate.propagate_completion_signal(STDEXEC::set_value, std::move(*d_result));
          } else {
            opstate.propagate_completion_signal(STDEXEC::set_error, std::move(status));
          }
        }
      }

      template <class Error>
      void set_error(Error&& err) noexcept {
        opstate_.propagate_completion_signal(set_error_t(), static_cast<Error&&>(err));
      }

      void set_stopped() noexcept {
        opstate_.propagate_completion_signal(set_stopped_t());
      }

      auto get_env() const noexcept -> _strm::opstate_base<Receiver>::env_t {
        return opstate_.make_env();
      }

     private:
      Fun f_;
      _strm::opstate_base<Receiver>& opstate_;
    };
  } // namespace _then

  template <class Sender, class Fun>
  struct then_sender : stream_sender_base {
    template <class Receiver>
      requires sender_in<Sender, env_of_t<Receiver>>
    struct max_result_size
      : __gather_completions_of_t<
          set_value_t,
          Sender,
          env_of_t<Receiver>,
          __mbind_front<result_size_for, Fun>,
          maxsize
        > { };

    template <class Receiver>
    using receiver_t = _then::receiver<max_result_size<Receiver>::value, Receiver, Fun>;

    template <class Error>
    using _set_error_t = completion_signatures<set_error_t(Error)>;

    template <class Self, class... Env>
    using __error_completions_t = __minvoke_q<
      __concat_completion_signatures_t,
      __with_error_invoke_t<
        __mbind_front_q<__callable_error_t, then_t>,
        set_value_t,
        Fun,
        __copy_cvref_t<Self, Sender>,
        Env...
      >,
      completion_signatures<set_error_t(cudaError_t)>
    >;

    template <class... Args>
    using _set_value_t = __set_value_from_t<Fun, Args...>;

    template <class Self, class... Env>
    using _completions_t = transform_completion_signatures<
      __completion_signatures_of_t<__copy_cvref_t<Self, Sender>, Env...>,
      __error_completions_t<Self, Env...>,
      _set_value_t,
      _set_error_t
    >;

    explicit then_sender(Sender sndr, Fun fun) noexcept(__nothrow_move_constructible<Sender, Fun>)
      : sndr_(static_cast<Sender&&>(sndr))
      , fun_(static_cast<Fun&&>(fun)) {
    }

    template <__decays_to<then_sender> Self, STDEXEC::receiver Receiver>
      requires receiver_of<Receiver, _completions_t<Self, env_of_t<Receiver>>>
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

    template <__decays_to<then_sender> Self, class... Env>
    static consteval auto get_completion_signatures() -> _completions_t<Self, Env...> {
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
  struct transform_sender_for<STDEXEC::then_t, Env> {
    template <class Fn, stream_completing_sender<Env> CvSender>
    auto operator()(__ignore, Fn fun, CvSender&& sndr) const {
      using _sender_t = then_sender<__decay_t<CvSender>, Fn>;
      return _sender_t{static_cast<CvSender&&>(sndr), static_cast<Fn&&>(fun)};
    }

    const Env& env_;
  };
} // namespace nvexec::_strm

namespace STDEXEC::__detail {
  template <class Sender, class Fun>
  extern __declfn_t<nvexec::_strm::then_sender<__demangle_t<Sender>, Fun>>
    __demangle_v<nvexec::_strm::then_sender<Sender, Fun>>;
} // namespace STDEXEC::__detail

STDEXEC_PRAGMA_POP()
