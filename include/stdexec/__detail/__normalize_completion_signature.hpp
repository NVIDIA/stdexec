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

#  include "__meta.hpp"
#  include "__type_traits.hpp"

#  include "__prologue.hpp"

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  namespace __cmplsigs
  {
    // The following code is used to normalize completion signatures.
    // "Normalization" means that that rvalue-references are stripped from the
    // types in the completion signatures. For example, the completion signature
    // `set_value_t(int &&)` would be normalized to `set_value_t(int)`, but
    // `set_value_t(int)` and `set_value_t(int &)` would remain unchanged.
    template <class... _Sigs>
    constexpr auto __repack(_Sigs*...) -> completion_signatures<_Sigs...>;

    template <class _Tag, class... _Args>
    constexpr auto __normalize_fn(_Args&&...) -> _Tag (*)(_Args...);

    template <class _Tag, class... _Args>
    constexpr auto __normalize(_Tag (*)(_Args...))
      -> decltype(__cmplsigs::__normalize_fn<_Tag>(__declval<_Args>()...));

    template <class... _Sigs>
    constexpr auto __normalize(completion_signatures<_Sigs...>*)
      -> decltype(__cmplsigs::__repack(__cmplsigs::__normalize(__midentity<_Sigs*>())...));

    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _CompletionsOrSig>
    using __normalize_t = decltype(__cmplsigs::__normalize(__midentity<_CompletionsOrSig*>()));
  }  // namespace __cmplsigs

  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class... _SigPtrs>
  using __completion_signature_ptrs_t = decltype(__cmplsigs::__repack(_SigPtrs()...));
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
