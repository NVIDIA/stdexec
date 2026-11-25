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
#  include "__concepts.hpp"
#  include "__continues_on.hpp"
#  include "__just.hpp"
#  include "__schedulers.hpp"
#  include "__sender_facade.hpp"

#  include "__prologue.hpp"

// Hide transfer_just from Doxygen since it's deprecated and we don't want to
// document it:
#  if !defined(STDEXEC_DOXYGEN_INVOKED)

STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  //////////////////////////////////////////////////////////////////////////////
  struct __transfer_just_t
  {
    // The sender is continues_on(just(vals...), sch) with this tag, so that a domain can
    // customize it: it has the parts of that sender (the scheduler, and just's sender as
    // continues_on keeps it), and continues_on's algorithm.
    template <scheduler _Scheduler, __movable_value... _Values>
    constexpr auto
    operator()(_Scheduler&& __sched, _Values&&... __vals) const -> __well_formed_sender auto
    {
      return __facade::__retagged<__transfer_just_t>(
        continues_on(just(static_cast<_Values&&>(__vals)...), static_cast<_Scheduler&&>(__sched)));
    }

    // transform __transfer_just(sch, vals...) into continues_on(just(vals...), sch): the
    // same parts with continues_on's tag, so that the customizations of continues_on apply
    template <class _Sender>
    static constexpr auto transform_sender(set_value_t, _Sender&& __sndr, __ignore)
    {
      if constexpr (__decay_copyable<_Sender>)
      {
        return __facade::__retagged<continues_on_t>(static_cast<_Sender&&>(__sndr));
      }
      else
      {
        return __not_a_sender<_SENDER_TYPE_IS_NOT_DECAY_COPYABLE_, _WITH_SENDER_<_Sender>>();
      }
    }
  };

  inline constexpr __transfer_just_t __transfer_just{};

  template <>
  struct __facade::__algorithm<__transfer_just_t> : __trnsfr::__algo
  {};
}  // namespace STDEXEC

#  endif  // STDEXEC_DOXYGEN_INVOKED

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
