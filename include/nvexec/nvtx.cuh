/*
 * Copyright (c) 2023 NVIDIA Corporation
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

#include <nvtx3/nvToolsExt.h>

#include "../stdexec/execution.hpp"
#include <string>
#include <utility>

#include "stream/adaptor.cuh"
#include "stream/common.cuh"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace nv::execution
{

  namespace _strm::nvtx
  {

    enum class kind
    {
      push,
      pop
    };

    // The tag of an nvtx sender, which is a facade sender
    template <kind Kind>
    struct nvtx_t : stream_tag_base
    {};

    // nvtx::push(sndr, name) and nvtx::pop(sndr): when the child completes, an NVTX range
    // named name is pushed, or the innermost range is popped, and then the completion is
    // passed on.
    template <kind Kind>
    struct algo : stream_adaptor<algo<Kind>>
    {
      template <class State, class Tag, class... Args>
      static auto
      __complete(STDEXEC::__facade::__child<0>, State& state, Tag tag, Args&&... args) noexcept
      {
        if constexpr (Kind == kind::push)
        {
          nvtxRangePushA(state.data_.c_str());
        }
        else
        {
          nvtxRangePop();
        }

        return _strm::_propagate(state, tag, static_cast<Args&&>(args)...);
      }

      using stream_adaptor<algo<Kind>>::__complete;
    };

    struct push_t
    {
      template <STDEXEC::sender Sender>
      auto operator()(Sender&& sndr, std::string&& name) const
      {
        return STDEXEC::__facade::__sender{nvtx_t<kind::push>(),
                                           std::move(name),
                                           static_cast<Sender&&>(sndr)};
      }

      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(std::string name) const
      {
        return STDEXEC::__closure(*this, std::move(name));
      }
    };

    struct pop_t
    {
      template <STDEXEC::sender Sender>
      auto operator()(Sender&& sndr) const
      {
        return STDEXEC::__facade::__sender{nvtx_t<kind::pop>(),
                                           std::string(),
                                           static_cast<Sender&&>(sndr)};
      }

      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()() const noexcept
      {
        return STDEXEC::__closure(*this);
      }
    };

    inline constexpr push_t push{};
    inline constexpr pop_t  pop{};

    struct scoped_t
    {
      template <STDEXEC::sender Sender, STDEXEC::__sender_adaptor_closure Closure>
      auto operator()(Sender&& __sndr, std::string&& name, Closure closure) const noexcept
      {
        return static_cast<Sender&&>(__sndr) | push(std::move(name)) | closure | pop();
      }

      template <STDEXEC::__sender_adaptor_closure Closure>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(std::string name, Closure closure) const
      {
        return STDEXEC::__closure(*this, std::move(name), static_cast<Closure&&>(closure));
      }
    };

    inline constexpr scoped_t scoped{};

  }  // namespace _strm::nvtx

  namespace nvtx
  {
    using _strm::nvtx::push;
    using _strm::nvtx::pop;
    using _strm::nvtx::scoped;
  }  // namespace nvtx

}  // namespace nv::execution

template <nv::execution::_strm::nvtx::kind Kind>
struct STDEXEC::__facade::__algorithm<nv::execution::_strm::nvtx::nvtx_t<Kind>>
  : nv::execution::_strm::nvtx::algo<Kind>
{};

namespace nvexec = nv::execution;

STDEXEC_PRAGMA_POP()
