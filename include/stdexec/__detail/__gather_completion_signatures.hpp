/*
 * Copyright (c) 2026 NVIDIA Corporation
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
#  include "__meta.hpp"
#  include "__partition_completion_signatures.hpp"

#  include "__prologue.hpp"

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  //////////////////////////////////////////////////////////////////////////////
  // __gather_completion_signatures_t
  namespace __detail
  {
    template <class _WantedTag>
    struct __gather_sigs_fn;

    template <>
    struct __gather_sigs_fn<set_value_t>
    {
      template <class _Sigs, class _Tuple, class _Variant>
      using __f = __value_types_t<_Sigs, _Tuple, _Variant>;
    };

    template <>
    struct __gather_sigs_fn<set_error_t>
    {
      template <class _Sigs, class _Tuple, class _Variant>
      using __f = __error_types_t<_Sigs, _Variant, _Tuple>;
    };

    template <>
    struct __gather_sigs_fn<set_stopped_t>
    {
      template <class _Sigs, class _Tuple, class _Variant>
      using __f = __stopped_types_t<_Sigs, _Variant, __minvoke<_Tuple>>;
    };
  }  // namespace __detail

  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class _WantedTag, class _Sigs, class _Tuple, class _Variant>
  using __gather_completion_signatures_t =
    __mcall<__detail::__gather_sigs_fn<_WantedTag>, _Sigs, _Tuple, _Variant>;
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
