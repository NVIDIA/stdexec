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
#pragma once

#include "__config.hpp"
#include "__type_traits.hpp"

#include <type_traits>

namespace stdexec {
  namespace __detail {
    template <class _Cpcvref>
    inline constexpr auto __forward_like = []<class _Uy>(_Uy&& __uy) noexcept -> auto&& {
      return static_cast<typename _Cpcvref::template __f<std::remove_reference_t<_Uy>>>(__uy);
    };
  }

  template <class _Ty>
  inline constexpr auto const & __forward_like = __detail::__forward_like<__copy_cvref_fn<_Ty&&>>;
}
