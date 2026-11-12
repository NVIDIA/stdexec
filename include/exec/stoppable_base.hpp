/*
 * Copyright (c) 2026 Lauri Ojell-Vasama
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

#include "../stdexec/stop_token.hpp"

namespace experimental::execution
{

#if STDEXEC_NAMESPACE_IS_WITHIN_STD()
#  define STDEXEC_STOP_TOKEN_NAMESPACE std
#else
#  define STDEXEC_STOP_TOKEN_NAMESPACE STDEXEC
#endif

  template <class _Token, class _Stoppable, class _Tag = void>
  using stoppable_base_for_t =
    STDEXEC_STOP_TOKEN_NAMESPACE::__stoppable_base_for_t<_Token, _Stoppable, _Tag>;

  template <class _Stoppable, class _Tag = void>
  using inplace_stoppable_base =
    STDEXEC_STOP_TOKEN_NAMESPACE::__inplace_stoppable_base<_Stoppable, _Tag>;

#undef STDEXEC_STOP_TOKEN_NAMESPACE

}  // namespace experimental::execution

namespace exec = experimental::execution;
