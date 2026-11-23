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

// include these after __execution_fwd.hpp
#  include "__completion_signatures_of.hpp"
#  include "__queries.hpp"
#  include "__sender_facade.hpp"
#  include "__sender_introspection.hpp"

#  include "__prologue.hpp"

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  //////////////////////////////////////////////////////////////////////////////
  // [exec.schedule.from]
  struct schedule_from_t
  {
    template <sender _Sender>
    constexpr auto operator()(_Sender&& __sndr) const
    {
      return __facade::__sender{schedule_from_t(), __(), static_cast<_Sender&&>(__sndr)};
    }
  };

  inline constexpr schedule_from_t schedule_from{};

  namespace __schedule_from
  {
    // schedule_from(sndr): completes as sndr does. It exists to be customized by the domain
    // in which sndr completes, which can then move the completions elsewhere (as a CUDA
    // stream scheduler moves them off the device).
    struct __algo : __facade::__defaults
    {
      // The child sees the forwarded environment of the receiver
      template <class _Sender, class _Env>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __env_for(__facade::__child<0>, _Sender const &, _Env&& __env) noexcept
        -> __fwd_env_t<_Env>
      {
        return __fwd_env(static_cast<_Env&&>(__env));
      }

      template <class _State>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __get_env(__facade::__child<0>, _State const & __state) noexcept
        -> decltype(auto)
      {
        return __fwd_env(STDEXEC::get_env(__state.__rcvr_));
      }
    };
  }  // namespace __schedule_from

  template <>
  struct __facade::__algorithm<schedule_from_t> : __schedule_from::__algo
  {};
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
