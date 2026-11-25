/*
 * Copyright (c) 2021-2024 NVIDIA Corporation
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
#pragma once

#include "__config.hpp"

#if STDEXEC_USE_MODULES() && !defined(STDEXEC_IN_MODULE_PURVIEW)

import stdexec;

#else

#  include "__execution_fwd.hpp"

#  include "../functional.hpp"
#  include "__completion_signatures.hpp"
#  include "__concepts.hpp"
#  include "__diagnostics.hpp"
#  include "__partition_completion_signatures.hpp"
#  include "__receivers.hpp"
#  include "__sender_facade.hpp"

#  if !STDEXEC_USE_MODULES()
#    include <exception>
#  endif

#  include "__prologue.hpp"

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  //////////////////////////////////////////////////////////////////////////////
  // Helpers for [exec.then]
  namespace __upon
  {
    // then(sndr, fn), upon_error(sndr, fn), and upon_stopped(sndr, fn): the child's _SetTag
    // completions invoke the function, and its result is sent as a value. Other completions
    // are passed through. _AlgoTag names the algorithm in diagnostics.
    template <class _AlgoTag, class _SetTag>
    struct __algo : __facade::__defaults
    {
      template <class _State, class _Tag, class... _Args>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __complete(__facade::__child<0>,
                                       _State& __state,
                                       _Tag    __tag,
                                       _Args&&... __args) noexcept
      {
        if constexpr (__same_as<_Tag, _SetTag>)
        {
          return __facade::__set_value_from<_AlgoTag>(__state.__rcvr_,
                                                      static_cast<_State&&>(__state).__data_,
                                                      static_cast<_Args&&>(__args)...);
        }
        else
        {
          return __facade::__send(__state.__rcvr_, __tag, static_cast<_Args&&>(__args)...);
        }
      }
    };
  }  // namespace __upon
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
