/*
 * Copyright (c) 2021-2024 NVIDIA Corporation
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

#include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "__concepts.hpp"
#include "__scope.hpp"

#include <exception>
#include <memory>
#include <new> // IWYU pragma: keep for ::new
#include <utility>

namespace STDEXEC {
  namespace __opt {
    struct __bad_optional_access : std::exception {
      [[nodiscard]]
      auto what() const noexcept -> const char* override {
        return "STDEXEC::__optional: bad access";
      }
    };

    inline constexpr auto __mk_has_value_guard(bool& __has_value) noexcept {
      __has_value = true;
      return __scope_guard{[&]() noexcept { __has_value = false; }};
    }

    inline constexpr struct __nullopt_t {
    } __nullopt{};

    // A simplified version of std::optional for better compile times
    template <class _Tp>
    struct __optional {
      static_assert(__std::destructible<_Tp>);

      union {
        _Tp __value_;
      };

      bool __has_value_ = false;

      constexpr __optional() noexcept {
      }

      constexpr __optional(__nullopt_t) noexcept {
      }

      constexpr __optional(__optional&&) = delete; // immovable for simplicity's sake

      template <__not_decays_to<__optional> _Up>
        requires __std::constructible_from<_Tp, _Up>
      constexpr __optional(_Up&& __val) noexcept(__nothrow_constructible_from<_Tp, _Up>) {
        emplace(static_cast<_Up&&>(__val));
      }

      template <class... _Us>
        requires __std::constructible_from<_Tp, _Us...>
      constexpr __optional(std::in_place_t, _Us&&... __us)
        noexcept(__nothrow_constructible_from<_Tp, _Us...>) {
        emplace(static_cast<_Us&&>(__us)...);
      }

      constexpr ~__optional() {
        if (__has_value_) {
          std::destroy_at(std::addressof(__value_));
        }
      }

      // The following emplace function must take great care to avoid use-after-free bugs.
      // If the object being constructed calls `start` on a newly created operation state
      // (as does the object returned from `submit`), and if `start` completes inline, it
      // could cause the destruction of the outer operation state that owns *this. The
      // function below uses the following pattern to avoid this:
      // 1. Set __has_value_ to true.
      // 2. Create a scope guard that will reset __has_value_ to false if the constructor
      //    throws.
      // 3. Construct the new object in the storage, which may cause the invalidation of
      //    *this. The emplace function must not access any members of *this after this point.
      // 4. Dismiss the scope guard, which will leave __has_value_ set to true.
      // 5. Return a reference to the new object -- which may be invalid! Calling code
      //    must be aware of the danger.
      template <class... _Us>
        requires __std::constructible_from<_Tp, _Us...>
      constexpr auto
        emplace(_Us&&... __us) noexcept(__nothrow_constructible_from<_Tp, _Us...>) -> _Tp& {
        reset();
        auto __sg = __mk_has_value_guard(__has_value_);
        auto* __p = std::construct_at(std::addressof(__value_), static_cast<_Us&&>(__us)...);
        __sg.__dismiss();
        return *std::launder(__p);
      }

      template <class _Fn, class... _Args>
        requires __std::same_as<_Tp, __call_result_t<_Fn, _Args...>>
      auto __emplace_from(_Fn&& __f, _Args&&... __args) noexcept(__nothrow_callable<_Fn, _Args...>)
        -> _Tp& {
        reset();
        auto __sg = __mk_has_value_guard(__has_value_);
        auto* __p = ::new (static_cast<void*>(std::addressof(__value_)))
          _Tp(static_cast<_Fn&&>(__f)(static_cast<_Args&&>(__args)...));
        __sg.__dismiss();
        return *std::launder(__p);
      }

      constexpr auto value() & -> _Tp& {
        if (!__has_value_) {
          STDEXEC_THROW(__bad_optional_access());
        }
        return __value_;
      }

      constexpr auto value() const & -> const _Tp& {
        if (!__has_value_) {
          STDEXEC_THROW(__bad_optional_access());
        }
        return __value_;
      }

      constexpr auto value() && -> _Tp&& {
        if (!__has_value_) {
          STDEXEC_THROW(__bad_optional_access());
        }
        return static_cast<_Tp&&>(static_cast<_Tp&>(__value_));
      }

      constexpr auto operator*() & noexcept -> _Tp& {
        STDEXEC_ASSERT(__has_value_);
        return __value_;
      }

      constexpr auto operator*() const & noexcept -> const _Tp& {
        STDEXEC_ASSERT(__has_value_);
        return __value_;
      }

      constexpr auto operator*() && noexcept -> _Tp&& {
        STDEXEC_ASSERT(__has_value_);
        return static_cast<_Tp&&>(static_cast<_Tp&>(__value_));
      }

      constexpr auto operator->() & noexcept -> std::add_pointer_t<_Tp> {
        STDEXEC_ASSERT(__has_value_);
        return std::addressof(static_cast<_Tp&>(__value_));
      }

      constexpr auto operator->() const & noexcept -> std::add_pointer_t<const _Tp> {
        STDEXEC_ASSERT(__has_value_);
        return std::addressof(static_cast<const _Tp&>(__value_));
      }

      [[nodiscard]]
      constexpr auto has_value() const noexcept -> bool {
        return __has_value_;
      }

      constexpr void reset() noexcept {
        if (__has_value_) {
          std::destroy_at(std::addressof(__value_));
          __has_value_ = false;
        }
      }
    };

    // __optional<T&>
    template <class _Tp>
    struct __optional<_Tp&> {
      _Tp* __value_ = nullptr;

      __optional() noexcept = default;

      __optional(__nullopt_t) noexcept {
      }

      template <__not_decays_to<__optional> _Up>
        requires __std::constructible_from<_Tp&, _Up>
      constexpr __optional(_Up&& __val) noexcept(__nothrow_constructible_from<_Tp&, _Up>) {
        emplace(static_cast<_Up&&>(__val));
      }

      template <__not_decays_to<__optional> _Up>
        requires __std::constructible_from<_Tp&, _Up>
      constexpr __optional(std::in_place_t, _Up&& __val)
        noexcept(__nothrow_constructible_from<_Tp&, _Up>) {
        emplace(static_cast<_Up&&>(__val));
      }

      template <class _Up>
        requires __std::constructible_from<_Tp&, _Up>
      constexpr auto emplace(_Up&& __us) noexcept(__nothrow_constructible_from<_Tp&, _Up>) -> _Tp& {
        __value_ = std::addressof(static_cast<_Up&&>(__us));
        return *__value_;
      }

      template <class _Fn, class... _Args>
        requires __std::same_as<_Tp&, __call_result_t<_Fn, _Args...>>
      auto __emplace_from(_Fn&& __f, _Args&&... __args) noexcept(__nothrow_callable<_Fn, _Args...>)
        -> _Tp& {
        __value_ = std::addressof(static_cast<_Fn&&>(__f)(static_cast<_Args&&>(__args)...));
        return *__value_;
      }

      constexpr auto value() const -> _Tp& {
        if (__value_ == nullptr) {
          STDEXEC_THROW(__bad_optional_access());
        }
        return *__value_;
      }

      constexpr auto operator*() const -> _Tp& {
        STDEXEC_ASSERT(__value_ != nullptr);
        return *__value_;
      }

      constexpr auto operator->() const -> _Tp* {
        STDEXEC_ASSERT(__value_ != nullptr);
        return __value_;
      }

      [[nodiscard]]
      constexpr auto has_value() const noexcept -> bool {
        return __value_ != nullptr;
      }

      constexpr void reset() noexcept {
        __value_ = nullptr;
      }
    };
  } // namespace __opt

  using __opt::__optional;
  using __opt::__bad_optional_access;
  using __opt::__nullopt;
} // namespace STDEXEC
