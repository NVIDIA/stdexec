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
    __manual_lifetime& operator=(const __manual_lifetime&) = delete;

    __manual_lifetime(__manual_lifetime&&) = delete;
    __manual_lifetime& operator=(__manual_lifetime&&) = delete;

    template <class... _Args>
    _Ty& __construct(_Args&&... __args) noexcept(
      stdexec::__nothrow_constructible_from<_Ty, _Args...>) {
      return *::new (static_cast<void*>(std::addressof(__value_))) _Ty((_Args&&) __args...);
    }

    template <class _Func>
    _Ty& __construct_with(_Func&& func) {
      return *::new (static_cast<void*>(std::addressof(__value_))) _Ty(((_Func&&) func)());
    }

    void __destroy() noexcept {
      __value_.~_Ty();
    }

    _Ty& __get() & noexcept {
      return __value_;
    }

    _Ty&& __get() && noexcept {
      return (_Ty&&) __value_;
    }

    const _Ty& __get() const & noexcept {
      return __value_;
    }

    const _Ty&& __get() const && noexcept {
      return (const _Ty&&) __value_;
    }

   private:
    union {
      _Ty __value_;
    };
  };
}
