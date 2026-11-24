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
  // then as a facade algorithm. The stream_domain customizes then(sndr, fn) by re-tagging
  // it with this tag (see transform_sender_for<then_t> below).
  struct then_t : stream_tag_base
  {};

  namespace _then
  {
    // then(sndr, fn): when the child completes with values, fn is called with them in a
    // kernel on the stream, and its result is the operation's value. The child's other
    // completions are passed on.
    struct algo : stream_adaptor<algo>
    {
      template <class State, class... Args>
      static auto
      __complete(STDEXEC::__facade::__child<0>, State& state, set_value_t, Args&&... args) noexcept
      {
        using fn_t = decltype(state.data_);
        if constexpr (!__invocable<fn_t, Args...>)
        {
          return STDEXEC::__facade::__type_error(
            STDEXEC::__callable_error_t<STDEXEC::then_t, fn_t, Args...>());
        }
        else
        {
          return _strm::_invoke_on_stream(state, state.data_, static_cast<Args&&>(args)...);
        }
      }

      using stream_adaptor<algo>::__complete;
    };
  }  // namespace _then

  template <>
  struct transform_sender_for<STDEXEC::then_t>
  {
    template <class Env, class Fn, class CvSender>
    auto operator()(Env const &, __ignore, Fn&& fun, CvSender&& sndr) const
    {
      if constexpr (stream_completing_sender<CvSender, Env>)
      {
        return STDEXEC::__facade::__sender{_strm::then_t(),
                                           static_cast<Fn&&>(fun),
                                           static_cast<CvSender&&>(sndr)};
      }
      else
      {
        return _strm::_no_stream_scheduler_in_env<STDEXEC::then_t, CvSender, Env>();
      }
    }
  };
}  // namespace nv::execution::_strm

template <>
struct STDEXEC::__facade::__algorithm<nv::execution::_strm::then_t>
  : nv::execution::_strm::_then::algo
{};

namespace nvexec = nv::execution;

STDEXEC_PRAGMA_POP()
