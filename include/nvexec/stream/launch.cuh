/*
 * Copyright (c) 2022-2024 NVIDIA Corporation
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

#include <cuda/std/utility>

#include "common.cuh"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

namespace nvexec {
  namespace _strm {
    struct launch_t;

    struct launch_params {
      std::size_t grid_size = 1;
      std::size_t block_size = 1;
      std::size_t shared_memory = 0;
    };

    namespace _launch {
      template <class... Args, class Fun>
      __global__ void _launch_kernel(Fun fn, cudaStream_t stream, Args... args) {
        static_assert(trivially_copyable<Fun, Args...>);
        ::cuda::std::move(fn)(stream, args...);
      }

      template <class Receiver, class Fun>
      class receiver : public stream_receiver_base {
        _strm::opstate_base<Receiver>& opstate_;
        Fun fun_;
        launch_params params_;

       public:
        explicit receiver(_strm::opstate_base<Receiver>& opstate, Fun fun, launch_params params)
          : opstate_(opstate)
          , fun_(static_cast<Fun&&>(fun))
          , params_(params) {
        }

        template <class... Args>
          requires std::invocable<Fun, cudaStream_t, Args&...>
        void set_value(Args&&... args) noexcept {
          cudaStream_t stream = opstate_.get_stream();
          launch_params p = params_;
          _launch_kernel<Args&...><<<p.grid_size, p.block_size, p.shared_memory, stream>>>(
            std::move(fun_), stream, args...);

          if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaPeekAtLastError());
              status == cudaSuccess) {
            opstate_.propagate_completion_signal(STDEXEC::set_value, static_cast<Args&&>(args)...);
          } else {
            opstate_.propagate_completion_signal(STDEXEC::set_error, std::move(status));
          }
        }

        template <class Error>
        void set_error(Error&& err) noexcept {
          opstate_.propagate_completion_signal(set_error_t(), static_cast<Error&&>(err));
        }

        void set_stopped() noexcept {
          opstate_.propagate_completion_signal(set_stopped_t());
        }

        [[nodiscard]]
        auto get_env() const noexcept -> _strm::opstate_base<Receiver>::env_t {
          return opstate_.make_env();
        }
      };

      template <class Fun, class... Args>
      using launch_error_t = __callable_error_t<launch_t, Fun, cudaStream_t, Args&...>;

      template <class Fun, class... Args>
      using _set_value_t = __minvoke<
        __if_c<
          __callable<Fun, cudaStream_t, Args&...>,
          __mcompose<__q<completion_signatures>, __qf<set_value_t>>,
          __mbind_front_q<launch_error_t, Fun>
        >,
        Args...
      >;

      template <class Fun, class CvSender, class... Env>
      using completions_t = transform_completion_signatures<
        __completion_signatures_of_t<CvSender, Env...>,
        completion_signatures<set_error_t(std::exception_ptr)>,
        __mbind_front_q<_set_value_t, Fun>::template __f
      >;
    } // namespace _launch

    template <class Sender, class Fun>
    struct launch_sender : stream_sender_base {
      template <class Self, class... Env>
      using completions_t = _launch::completions_t<Fun, __copy_cvref_t<Self, Sender>, Env...>;

      template <class Receiver>
      using receiver_t = _launch::receiver<Receiver, Fun>;

      template <__decays_to<launch_sender> Self, STDEXEC::receiver Receiver>
        requires receiver_of<Receiver, completions_t<Self, env_of_t<Receiver>>>
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver rcvr)
        -> stream_opstate_t<__copy_cvref_t<Self, Sender>, receiver_t<Receiver>, Receiver> {
        return stream_opstate<__copy_cvref_t<Self, Sender>>(
          static_cast<Self&&>(self).sndr_,
          static_cast<Receiver&&>(rcvr),
          [&](_strm::opstate_base<Receiver>& stream_provider) -> receiver_t<Receiver> {
            return receiver_t<Receiver>(stream_provider, self.fun_, self.params_);
          });
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      template <__decays_to<launch_sender> Self, class... Env>
      static consteval auto get_completion_signatures() -> completions_t<Self, Env...> {
        return {};
      }

      auto get_env() const noexcept -> stream_sender_attrs<Sender> {
        return {&sndr_};
      }

      Sender sndr_;
      Fun fun_;
      launch_params params_;
    };

    struct launch_t {
      template <class CvSender, class Fun>
      using sender_t = launch_sender<__decay_t<CvSender>, Fun>;

      template <sender CvSender, __movable_value Fun>
      constexpr auto operator()(CvSender&& sndr, Fun&& fun) const -> sender_t<CvSender, Fun> {
        return {{}, static_cast<CvSender&&>(sndr), static_cast<Fun&&>(fun), {}};
      }

      template <sender CvSender, __movable_value Fun>
      constexpr auto operator()(CvSender&& sndr, launch_params params, Fun&& fun) const //
        -> sender_t<CvSender, Fun> {
        return {{}, static_cast<CvSender&&>(sndr), static_cast<Fun&&>(fun), params};
      }

      template <__movable_value Fun>
      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()(Fun&& fun) const {
        return STDEXEC::__closure(*this, static_cast<Fun&&>(fun));
      }

      template <__movable_value Fun>
      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()(launch_params params, Fun&& fun) const {
        return STDEXEC::__closure(*this, params, static_cast<Fun&&>(fun));
      }
    };
  } // namespace _strm

  inline constexpr _strm::launch_t launch{};

} // namespace nvexec

namespace STDEXEC::__detail {
  template <class Sender, class Fun>
  extern __declfn_t<nvexec::_strm::launch_sender<__demangle_t<Sender>, Fun>>
    __demangle_v<nvexec::_strm::launch_sender<Sender, Fun>>;
} // namespace STDEXEC::__detail

STDEXEC_PRAGMA_POP()
