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

#include "__execution_fwd.hpp" // IWYU pragma: keep

// include these after __execution_fwd.hpp
#include "__concepts.hpp"

#include <new> // IWYU pragma: keep for ::new
#include <exception>
#include <memory>
#include <utility>

namespace stdexec {
  namespace __opt {
    struct __bad_optional_access : std::exception {
      const char* what() const noexcept override {
        return "stdexec::__optional: bad access";
      }
    };

    inline constexpr struct __nullopt_t {
    } __nullopt{};

    // A simplified version of std::optional for better compile times
    template <class _Tp>
    struct __optional {
      static_assert(destructible<_Tp>);

      union {
        _Tp __value;
      };

      bool __has_value = false;

      __optional() noexcept {
      }

      __optional(__nullopt_t) noexcept {
      }

      __optional(__optional&&) = delete; // immovable for simplicity's sake

      template <__not_decays_to<__optional> _Up>
        requires constructible_from<_Tp, _Up>
      __optional(_Up&& __v)
        : __value(static_cast<_Up&&>(__v))
        , __has_value(true) {
      }

      template <class... _Us>
        requires constructible_from<_Tp, _Us...>
      __optional(std::in_place_t, _Us&&... __us)
        : __value(static_cast<_Us&&>(__us)...)
        , __has_value(true) {
      }

      ~__optional() {
        if (__has_value) {
          std::destroy_at(std::addressof(__value));
        }
      }

      template <class... _Us>
        requires constructible_from<_Tp, _Us...>
      _Tp& emplace(_Us&&... __us) noexcept(__nothrow_constructible_from<_Tp, _Us...>) {
        reset(); // sets __has_value to false in case the next line throws
        ::new (&__value) _Tp{static_cast<_Us&&>(__us)...};
        __has_value = true;
        return __value;
      }

      _Tp& value() & {
        if (!__has_value) {
          throw __bad_optional_access();
        }
        return __value;
      }

      const _Tp& value() const & {
        if (!__has_value) {
          throw __bad_optional_access();
        }
        return __value;
      }

      _Tp&& value() && {
        if (!__has_value) {
          throw __bad_optional_access();
        }
        return static_cast<_Tp&&>(__value);
      }

      _Tp& operator*() & noexcept {
        STDEXEC_ASSERT(__has_value);
        return __value;
      }

      const _Tp& operator*() const & noexcept {
        STDEXEC_ASSERT(__has_value);
        return __value;
      }

      _Tp&& operator*() && noexcept {
        STDEXEC_ASSERT(__has_value);
        return static_cast<_Tp&&>(__value);
      }

      _Tp* operator->() & noexcept {
        STDEXEC_ASSERT(__has_value);
        return &__value;
      }

      const _Tp* operator->() const & noexcept {
        STDEXEC_ASSERT(__has_value);
        return &__value;
      }

      bool has_value() const noexcept {
        return __has_value;
      }

      void reset() noexcept {
        if (__has_value) {
          std::destroy_at(std::addressof(__value));
          __has_value = false;
        }
      }
    };
  } // namespace __opt

  using __opt::__optional;
  using __opt::__bad_optional_access;
  using __opt::__nullopt;
} // namespace stdexec
