/*
 * Copyright (c) 2021-2026 NVIDIA Corporation
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
#  include "__tuple.hpp"  // IWYU pragma: keep for __is_tuple and __tuple_size_v

#  if !STDEXEC_USE_MODULES()
#    include <exception>  // IWYU pragma: keep for std::terminate
#  endif

#  include "__prologue.hpp"

namespace STDEXEC
{
  namespace __detail
  {
    template <class... _Ts>
    auto __std_tuple_sizer(std::tuple<_Ts...> const &) -> __msize_t<sizeof...(_Ts)>;
  }  // namespace __detail

  template <class _Tuple>
  concept __is_std_tuple = requires(_Tuple &&__arg) { __detail::__std_tuple_sizer(__arg); };

  template <class _Ty>
  inline constexpr int __structured_binding_size_v = -1;

#  if STDEXEC_HAS_BUILTIN(__builtin_structured_binding_size)
  template <class _Ty>
    requires(__builtin_structured_binding_size(_Ty) >= 0U)
  inline constexpr int __structured_binding_size_v<_Ty> = __builtin_structured_binding_size(_Ty);
#  else
  template <__is_tuple _Ty>
  inline constexpr int __structured_binding_size_v<_Ty> = __tuple_size_v<_Ty>;

  template <__is_std_tuple _Ty>
  inline constexpr int __structured_binding_size_v<_Ty> = decltype(__detail::__std_tuple_sizer(
    __declval<_Ty>()))::value;

  // For types that are *not* tuples, __structured_binding_size_v must be
  // specialized explicitly.
#  endif
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
