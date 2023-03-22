/*
 * Copyright (c) 2022 NVIDIA Corporation
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

#include "../stdexec/__detail/__scope.hpp"

namespace exec {

  template <class _Fn, class... _Ts>
    requires stdexec::__nothrow_callable<_Fn, _Ts...>
  struct scope_guard {
    stdexec::__scope_guard<_Fn, _Ts...> __guard_;

    void dismiss() noexcept {
      __guard_.__dismiss();
    }
  };
  template <class _Fn, class... _Ts>
  scope_guard(_Fn, _Ts...) -> scope_guard<_Fn, _Ts...>;

} // namespace exec
