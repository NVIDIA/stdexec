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

#include <new> // IWYU pragma: keep for ::new
#include <exception>
#include <memory>
#include <utility>

namespace stdexec {
  namespace __opt {
    struct __bad_optional_access : std::exception {
      [[nodiscard]]
      auto what() const noexcept -> const char* override {
        return "stdexec::__optional: bad access";
      }
    };

    inline auto __mk_has_value_guard(bool& __has_value) noexcept {
      __has_value = true;
      return __scope_guard{[&]() noexcept {
        __has_value = false;
      }};
    }

    inline constexpr struct __nullopt_t {
    } __nullopt{};

    // A simplified version of std::optional for better compile times
    template <class _Tp>
    struct __optional {
      static_assert(destructible<_Tp>);

      union {
        _Tp __value_;
      };

      bool __has_value_ = false;

      __optional() noexcept {
      }

      __optional(__nullopt_t) noexcept {
      }

      __optional(__optional&&) = delete; // immovable for simplicity's sake

      template <__not_decays_to<__optional> _Up>
        requires constructible_from<_Tp, _Up>
      __optional(_Up&& __v) noexcept(__nothrow_constructible_from<_Tp, _Up>) {
        emplace(static_cast<_Up&&>(__v));
      }

      template <class... _Us>
        requires constructible_from<_Tp, _Us...>
      __optional(std::in_place_t, _Us&&... __us) noexcept(
        __nothrow_constructible_from<_Tp, _Us...>) {
        emplace(static_cast<_Us&&>(__us)...);
      }

      ~__optional() {
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
        requires constructible_from<_Tp, _Us...>
      auto emplace(_Us&&... __us) noexcept(__nothrow_constructible_from<_Tp, _Us...>) -> _Tp& {
        reset();
        auto __sg = __mk_has_value_guard(__has_value_);
        auto* __p = ::new (static_cast<void*>(std::addressof(__value_)))
          _Tp{static_cast<_Us&&>(__us)...};
        __sg.__dismiss();
        return *std::launder(__p);
      }

      auto value() & -> _Tp& {
        if (!__has_value_) {
          throw __bad_optional_access();
        }
        return __value_;
      }

      auto value() const & -> const _Tp& {
        if (!__has_value_) {
          throw __bad_optional_access();
        }
        return __value_;
      }

      auto value() && -> _Tp&& {
        if (!__has_value_) {
          throw __bad_optional_access();
        }
        return static_cast<_Tp&&>(__value_);
      }

      auto operator*() & noexcept -> _Tp& {
        STDEXEC_ASSERT(__has_value_);
        return __value_;
      }

      auto operator*() const & noexcept -> const _Tp& {
        STDEXEC_ASSERT(__has_value_);
        return __value_;
      }

      auto operator*() && noexcept -> _Tp&& {
        STDEXEC_ASSERT(__has_value_);
        return static_cast<_Tp&&>(__value_);
      }

      auto operator->() & noexcept -> _Tp* {
        STDEXEC_ASSERT(__has_value_);
        return &__value_;
      }

      auto operator->() const & noexcept -> const _Tp* {
        STDEXEC_ASSERT(__has_value_);
        return &__value_;
      }

      [[nodiscard]]
      auto has_value() const noexcept -> bool {
        return __has_value_;
      }

      void reset() noexcept {
        if (__has_value_) {
          std::destroy_at(std::addressof(__value_));
          __has_value_ = false;
        }
      }
    };
  } // namespace __opt

  using __opt::__optional;
  using __opt::__bad_optional_access;
  using __opt::__nullopt;
} // namespace stdexec
