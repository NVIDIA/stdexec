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
#include "__concepts.hpp"
#include "__type_traits.hpp"

#include <initializer_list>
#include <type_traits>

namespace stdexec {
  constexpr std::size_t __npos = ~0UL;

  template <class...>
  struct __undefined;

  struct __ { };

  struct __ignore {
    __ignore() = default;

    STDEXEC_ATTRIBUTE(always_inline) constexpr __ignore(auto&&...) noexcept {
    }
  };

#if STDEXEC_MSVC()
  // MSVCBUG https://developercommunity.visualstudio.com/t/Incorrect-function-template-argument-sub/10437827

  template <std::size_t>
  struct __ignore_t {
    __ignore_t() = default;

    constexpr __ignore_t(auto&&...) noexcept {
    }
  };
#else
  template <std::size_t>
  using __ignore_t = __ignore;
#endif

  struct __none_such { };

  namespace {
    struct __anon { };
  } // namespace

  struct __immovable {
    __immovable() = default;
   private:
    STDEXEC_IMMOVABLE(__immovable);
  };

  struct __move_only {
    __move_only() = default;

    __move_only(__move_only&&) noexcept = default;
    auto operator=(__move_only&&) noexcept -> __move_only& = default;

    __move_only(const __move_only&) = delete;
    auto operator=(const __move_only&) -> __move_only& = delete;
  };

  template <class... _Fns>
  struct __overload : _Fns... {
    using _Fns::operator()...;
  };

  template <class... _Fns>
  __overload(_Fns...) -> __overload<_Fns...>;

  inline constexpr auto __umax(std::initializer_list<std::size_t> __il) noexcept -> std::size_t {
    std::size_t __m = 0;
    for (std::size_t __i: __il) {
      if (__i > __m) {
        __m = __i;
      }
    }
    return __m;
  }

  inline constexpr auto
    __pos_of(const bool* const __first, const bool* const __last) noexcept -> std::size_t {
    for (const bool* __where = __first; __where != __last; ++__where) {
      if (*__where) {
        return static_cast<std::size_t>(__where - __first);
      }
    }
    return __npos;
  }

  template <class _Ty, class... _Ts>
  inline constexpr auto __index_of() noexcept -> std::size_t {
    constexpr bool __same[] = {STDEXEC_IS_SAME(_Ty, _Ts)..., false};
    return __pos_of(__same, __same + sizeof...(_Ts));
  }

  namespace __detail {
    template <class _Cpcvref>
    struct __forward_like_fn {
      template <class _Uy>
      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()(_Uy&& __uy) const noexcept -> auto&& {
        return static_cast<typename _Cpcvref::template __f<std::remove_reference_t<_Uy>>>(__uy);
      }
    };
  } // namespace __detail

  template <class _Ty>
  inline constexpr __detail::__forward_like_fn<__copy_cvref_fn<_Ty&&>> __forward_like{};

  STDEXEC_PRAGMA_PUSH()
  STDEXEC_PRAGMA_IGNORE_GNU("-Wold-style-cast")

  // A derived-to-base cast that works even when the base is not accessible from derived.
  template <class _Tp, class _Up>
  STDEXEC_ATTRIBUTE(host, device)
  auto __c_upcast(_Up&& u) noexcept -> __copy_cvref_t<_Up&&, _Tp>
    requires __decays_to<_Tp, _Tp>
  {
    static_assert(STDEXEC_IS_BASE_OF(_Tp, __decay_t<_Up>));
    return (__copy_cvref_t<_Up&&, _Tp>) static_cast<_Up&&>(u);
  }

  // A base-to-derived cast that works even when the base is not accessible from derived.
  template <class _Tp, class _Up>
  STDEXEC_ATTRIBUTE(host, device)
  auto __c_downcast(_Up&& u) noexcept -> __copy_cvref_t<_Up&&, _Tp>
    requires __decays_to<_Tp, _Tp>
  {
    static_assert(STDEXEC_IS_BASE_OF(__decay_t<_Up>, _Tp));
    return (__copy_cvref_t<_Up&&, _Tp>) static_cast<_Up&&>(u);
  }

  STDEXEC_PRAGMA_POP()

  template <class _Ty>
  auto __decay_copy(_Ty) noexcept -> _Ty;

  template <class _Ty>
  struct __indestructible {
    template <class... _Us>
    constexpr __indestructible(_Us&&... __us) noexcept(__nothrow_constructible_from<_Ty, _Us...>)
      : __value(static_cast<_Us&&>(__us)...) {
    }

    constexpr ~__indestructible() {
    }

    auto get() noexcept -> _Ty& {
      return __value;
    }

    auto get() const noexcept -> const _Ty& {
      return __value;
    }

    union {
      _Ty __value;
    };
  };
} // namespace stdexec

#if defined(__cpp_auto_cast) && (__cpp_auto_cast >= 2021'10L)
#  define STDEXEC_DECAY_COPY(...) auto(__VA_ARGS__)
#else
#  define STDEXEC_DECAY_COPY(...) (true ? (__VA_ARGS__) : stdexec::__decay_copy(__VA_ARGS__))
#endif
