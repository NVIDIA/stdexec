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

#  include "__prologue.hpp"

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  STDEXEC_PRAGMA_PUSH()
  STDEXEC_PRAGMA_IGNORE_EDG(expr_has_no_effect)
  STDEXEC_PRAGMA_IGNORE_GNU("-Wunused-value")

  //////////////////////////////////////////////////////////////////////////////
  // concat_completion_signatures
  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class... _Sigs>
  using __concat_completion_signatures_t =
    __mcall<__mconcat<__munique<__qq<completion_signatures>>>, _Sigs...>;

  namespace __detail
  {
    STDEXEC_MODULE_EXPORT_AUTHORING
    struct __concat_completion_signatures_fn
    {
      template <STDEXEC::__valid_completion_signatures... Sigs>
      [[nodiscard]]
      consteval auto operator()(Sigs...) const noexcept
      {
        return __concat_completion_signatures_t<Sigs...>();
      }

      template <class... Errors>
      [[nodiscard]]
      consteval auto operator()(Errors...) const noexcept
      {
        // NB: this uses an overloaded comma operator on the _ERROR_ type to
        // find an error in a pack of types.
        using __error_t = decltype(+(Errors(), ...));
        static_assert(__merror<__error_t>);
        return STDEXEC::__throw_compile_time_error(__error_t());
      }
    };
  }  // namespace __detail

  inline constexpr auto __concat_completion_signatures =
    __detail::__concat_completion_signatures_fn{};

  STDEXEC_PRAGMA_POP()
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
