/*
 * Copyright (c) 2023 Maikel Nadolski
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

#include "../../stdexec/concepts.hpp"

#include <cstddef>
#include <memory>
#include <type_traits>

namespace exec {

  template <class _Ty>
  class __manual_lifetime {
   public:
    __manual_lifetime() noexcept {
    }

    ~__manual_lifetime() {
    }

    __manual_lifetime(const __manual_lifetime&) = delete;
    auto operator=(const __manual_lifetime&) -> __manual_lifetime& = delete;

    __manual_lifetime(__manual_lifetime&&) = delete;
    auto operator=(__manual_lifetime&&) -> __manual_lifetime& = delete;

    template <class... _Args>
    auto __construct(_Args&&... __args) noexcept(
      stdexec::__nothrow_constructible_from<_Ty, _Args...>) -> _Ty& {
      return *::new (static_cast<void*>(std::addressof(__value_)))
        _Ty(static_cast<_Args&&>(__args)...);
    }

    template <class _Func>
    auto __construct_with(_Func&& func) -> _Ty& {
      return *::new (static_cast<void*>(std::addressof(__value_)))
        _Ty((static_cast<_Func&&>(func))());
    }

    void __destroy() noexcept {
      __value_.~_Ty();
    }

    auto __get() & noexcept -> _Ty& {
      return __value_;
    }

    auto __get() && noexcept -> _Ty&& {
      return static_cast<_Ty&&>(__value_);
    }

    auto __get() const & noexcept -> const _Ty& {
      return __value_;
    }

    auto __get() const && noexcept -> const _Ty&& {
      return static_cast<const _Ty&&>(__value_);
    }

   private:
    union {
      _Ty __value_;
    };
  };
} // namespace exec
