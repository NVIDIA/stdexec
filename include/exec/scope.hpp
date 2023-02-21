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

#include "../stdexec/__detail/__meta.hpp"

namespace exec {
  template <stdexec::__nothrow_callable _Fn>
  struct scope_guard {
    [[no_unique_address]] _Fn __fn_;
    [[no_unique_address]] stdexec::__immovable __hidden_{};
    bool __dismissed_{false};

    ~scope_guard() {
      if (!__dismissed_)
        ((_Fn&&) __fn_)();
    }

    void dismiss() noexcept {
      __dismissed_ = true;
    }
  };

  template <stdexec::__nothrow_callable _Fn>
  scope_guard(_Fn) -> scope_guard<_Fn>;
} // namespace exec
