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

#include "adaptor.cuh"
#include "common.cuh"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

namespace nv::execution::_strm
{
  // upon_stopped as a facade algorithm. The stream_domain customizes upon_stopped(sndr, fn) by
  // re-tagging it with this tag (see transform_sender_for<upon_stopped_t> below).
  struct upon_stopped_t : stream_tag_base
  {};

  namespace _upon_stopped
  {
    // upon_stopped(sndr, fn): when the child completes with stopped, fn is called in a kernel on
    // the stream, and its result is the operation's value. The child's other completions are
    // passed on.
    struct algo : stream_adaptor<algo>
    {
      template <class State>
      static auto __complete(STDEXEC::__facade::__child<0>, State& state, set_stopped_t) noexcept
      {
        using fn_t = decltype(state.data_);
        if constexpr (!__invocable<fn_t>)
        {
          return STDEXEC::__facade::__type_error(
            STDEXEC::__callable_error_t<STDEXEC::upon_stopped_t, fn_t>());
        }
        else
        {
          return _strm::_invoke_on_stream(state, state.data_);
        }
      }

      using stream_adaptor<algo>::__complete;
    };
  }  // namespace _upon_stopped

  template <>
  struct transform_sender_for<STDEXEC::upon_stopped_t>
  {
    template <class Env, class Fun, class CvSender>
    auto operator()(Env const &, __ignore, Fun&& fun, CvSender&& sndr) const
    {
      if constexpr (stream_completing_sender<CvSender, Env>)
      {
        return STDEXEC::__facade::__sender{_strm::upon_stopped_t(),
                                           static_cast<Fun&&>(fun),
                                           static_cast<CvSender&&>(sndr)};
      }
      else
      {
        return _strm::_no_stream_scheduler_in_env<STDEXEC::upon_stopped_t, CvSender, Env>();
      }
    }
  };
}  // namespace nv::execution::_strm

template <>
struct STDEXEC::__facade::__algorithm<nv::execution::_strm::upon_stopped_t>
  : nv::execution::_strm::_upon_stopped::algo
{};

namespace nvexec = nv::execution;

STDEXEC_PRAGMA_POP()
