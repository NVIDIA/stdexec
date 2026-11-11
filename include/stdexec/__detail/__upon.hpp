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
#  include "__meta.hpp"
#  include "__partition_completion_signatures.hpp"
#  include "__receivers.hpp"

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
    template <class _SetTag>
    static constexpr auto __complete =
      []<class _Tag, class _State, class... _Args>(__ignore,
                                                   _State& __state,
                                                   _Tag,
                                                   _Args&&... __args) noexcept -> void
    {
      if constexpr (__same_as<_Tag, _SetTag>)
      {
        STDEXEC::__set_value_from(static_cast<_State&&>(__state).__rcvr_,
                                  static_cast<_State&&>(__state).__data_,
                                  static_cast<_Args&&>(__args)...);
      }
      else
      {
        _Tag()(static_cast<_State&&>(__state).__rcvr_, static_cast<_Args&&>(__args)...);
      }
    };

    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Fun, class AlgoTag>
    struct __transform_completion
    {
      template <class... _Args>
      constexpr auto operator()() const
      {
        if constexpr (__nothrow_invocable<_Fun, _Args...>)
        {
          return completion_signatures<__single_value_sig_t<__invoke_result_t<_Fun, _Args...>>>();
        }
        else if constexpr (__invocable<_Fun, _Args...>)
        {
          return completion_signatures<__single_value_sig_t<__invoke_result_t<_Fun, _Args...>>,
                                       set_error_t(std::exception_ptr)>();
        }
        else
        {
          return STDEXEC::__throw_compile_time_error(__callable_error_t<AlgoTag, _Fun, _Args...>());
        }
      }
    };
  }  // namespace __upon
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
