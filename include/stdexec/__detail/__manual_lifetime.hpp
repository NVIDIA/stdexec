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

#include "__concepts.hpp"
#include "__utility.hpp"

#include <memory>
#include <new>
#include <type_traits>

namespace STDEXEC
{
  //! Holds storage for a `_Ty`, but allows clients to `__construct(...)`, `__destry()`,
  //! and `__get()` the `_Ty` without regard for usual lifetime rules.
  template <class _Ty>
  class __manual_lifetime
  {
   public:
    //! Constructor does nothing: It's on you to call `__construct(...)` or `__construct_from(...)`
    //! if you want the `_Ty`'s lifetime to begin.
    constexpr __manual_lifetime() noexcept = default;

    //! Destructor does nothing: It's on you to call `__destroy()` if you mean to.
    constexpr ~__manual_lifetime() = default;

    STDEXEC_IMMOVABLE(__manual_lifetime);

    //! Construct the `_Ty` in place.
    //! There are no safeties guarding against the case that there's already one there.
    template <class... _Args>
    constexpr auto __construct(_Args&&... __args)
      noexcept(STDEXEC::__nothrow_constructible_from<_Ty, _Args...>) -> _Ty&
    {
      // Use placement new instead of std::construct_at to support aggregate initialization with
      // brace elision.
      return *std::launder(::new (static_cast<void*>(__buffer_))
                             _Ty{static_cast<_Args&&>(__args)...});
    }

    //! Construct the `_Ty` in place from the result of calling `func`.
    //! There are no safeties guarding against the case that there's already one there.
    template <class _Func, class... _Args>
    constexpr auto __construct_from(_Func&& func, _Args&&... __args) -> _Ty&
    {
      // Use placement new instead of std::construct_at in case the function returns an immovable
      // type.
      return *std::launder(::new (static_cast<void*>(__buffer_))
                             _Ty{static_cast<_Func&&>(func)(static_cast<_Args&&>(__args)...)});
    }

    //! End the lifetime of the contained `_Ty`.
    //! \pre The lifetime has started.
    constexpr void __destroy() noexcept
    {
      std::destroy_at(&__get());
    }

    //! Get access to the `_Ty`.
    //! \pre The lifetime has started.
    constexpr auto __get() & noexcept -> _Ty&
    {
      return *reinterpret_cast<_Ty*>(__buffer_);
    }

    //! Get access to the `_Ty`.
    //! \pre The lifetime has started.
    constexpr auto __get() && noexcept -> _Ty&&
    {
      return static_cast<_Ty&&>(*reinterpret_cast<_Ty*>(__buffer_));
    }

    //! Get access to the `_Ty`.
    //! \pre The lifetime has started.
    constexpr auto __get() const & noexcept -> _Ty const &
    {
      return *reinterpret_cast<_Ty const *>(__buffer_);
    }

    constexpr auto __get() const && noexcept -> _Ty const && = delete;

    constexpr auto operator->() noexcept -> _Ty*
    {
      return reinterpret_cast<_Ty*>(__buffer_);
    }

    constexpr auto operator->() const noexcept -> _Ty const *
    {
      return reinterpret_cast<_Ty const *>(__buffer_);
    }

   private:
    alignas(_Ty) unsigned char __buffer_[sizeof(_Ty)]{};
  };

  template <class _Reference>
    requires std::is_reference_v<_Reference>
  class __manual_lifetime<_Reference>
  {
   public:
    constexpr __manual_lifetime() noexcept = default;
    STDEXEC_IMMOVABLE(__manual_lifetime);

    constexpr auto __construct(_Reference __ref) noexcept -> _Reference
    {
      __ptr_ = std::addressof(__ref);
      return static_cast<_Reference>(*__ptr_);
    }

    template <class _Func, class... _Args>
    constexpr auto __construct_from(_Func&& func, _Args&&... __args)
      noexcept(__nothrow_callable<_Func, _Args...>) -> _Reference
    {
      decltype(auto) __result = static_cast<_Func&&>(func)(static_cast<_Args&&>(__args)...);
      static_assert(std::is_reference_v<decltype(__result)>, "Result must be a reference");
      __ptr_ = std::addressof(__result);
      return static_cast<_Reference>(*__ptr_);
    }

    constexpr void __destroy() noexcept {}

    constexpr auto __get() const noexcept -> _Reference
    {
      return static_cast<_Reference>(*__ptr_);
    }

    constexpr auto operator->() const noexcept -> std::add_pointer_t<_Reference>
    {
      return __ptr_;
    }

   private:
    std::add_pointer_t<_Reference> __ptr_{nullptr};
  };

  template <>
  class __manual_lifetime<void>
  {
   public:
    constexpr __manual_lifetime() noexcept = default;
    STDEXEC_IMMOVABLE(__manual_lifetime);

    template <class... _Args>
    constexpr void __construct(_Args&&...) noexcept
    {}

    template <class _Func, class... _Args>
    constexpr void __construct_from(_Func&& __func, _Args&&... __args) noexcept
    {
      (void) static_cast<_Func&&>(__func)(static_cast<_Args&&>(__args)...);
    }

    constexpr void __destroy() noexcept {}

    constexpr auto __get() const noexcept -> void {}

    constexpr auto operator->() const noexcept -> void*
    {
      return nullptr;
    }
  };
}  // namespace STDEXEC
