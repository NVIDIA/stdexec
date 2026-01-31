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

#include "__concepts.hpp"
#include "__config.hpp"
#include "__meta.hpp"

#include <initializer_list>

namespace STDEXEC {
  constexpr std::size_t __npos = ~0UL;

  template <class...>
  struct __undefined;

  using __empty = struct __ { };

  struct __ignore {
    constexpr __ignore() = default;

    template <class... _Ts>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr __ignore(_Ts&&...) noexcept {
    }
  };

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

  // Helper to combine multiple function objects into one overload set
  template <class... _Fns>
  struct __overload : _Fns... {
    using _Fns::operator()...;
  };

  template <class... _Fns>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __overload(_Fns...) -> __overload<_Fns...>;

#if STDEXEC_EDG()
  // nvc++ doesn't cache the results of alias template specializations.
  // To avoid repeated computation of the same function return type,
  // cache the result ourselves in a class template specialization.
  template <class _Fun, class... _As>
  using __call_result_i = decltype(__declval<_Fun>()(__declval<_As>()...));
  template <class _Fun, class... _As>
  using __call_result_t = __mmemoize_q<__call_result_i, _Fun, _As...>;
#else
  template <class _Fun, class... _As>
  using __call_result_t = decltype(__declval<_Fun>()(__declval<_As>()...));
#endif

  template <class _Fun, class _Default, class... _As>
  using __call_result_or_t =
    __mcall<__mtry_catch_q<__call_result_t, __mconst<_Default>>, _Fun, _As...>;

// BUGBUG TODO file this bug with nvc++
#if STDEXEC_EDG()
  template <const auto &_Fun, class... _As>
  using __result_of = __call_result_t<decltype(_Fun), _As...>;
#else
  template <const auto &_Fun, class... _As>
  using __result_of = decltype(_Fun(__declval<_As>()...));
#endif

  template <const auto &_Fun, class... _As>
  inline constexpr bool __noexcept_of = noexcept(_Fun(__declval<_As>()...));

  // For emplacing non-movable types into optionals:
  template <__nothrow_move_constructible _Fn>
  struct __emplace_from {
    _Fn __fn_;
    using __t = __call_result_t<_Fn>;

    constexpr operator __t() && noexcept(__nothrow_callable<_Fn>) {
      return static_cast<_Fn &&>(__fn_)();
    }

    constexpr auto operator()() && noexcept(__nothrow_callable<_Fn>) -> __t {
      return static_cast<_Fn &&>(__fn_)();
    }
  };

  template <class _Fn>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __emplace_from(_Fn) -> __emplace_from<_Fn>;

  // Helper to make a type ill-formed if it is one of the given types
  template <class _Ty, class... _Us>
    requires __none_of<_Ty, _Us...>
  using __unless_one_of_t = _Ty;

  // Helper to select overloads by priority:
  template <int _Iy>
  struct __priority : __priority<_Iy - 1> { };

  template <>
  struct __priority<0> { };

  inline constexpr auto __umin(std::initializer_list<std::size_t> __il) noexcept -> std::size_t {
    std::size_t __m = 0;
    for (std::size_t __i: __il) {
      if (__i < __m) {
        __m = __i;
      }
    }
    return __m;
  }

  inline constexpr auto __umax(std::initializer_list<std::size_t> __il) noexcept -> std::size_t {
    std::size_t __m = 0;
    for (std::size_t __i: __il) {
      if (__m < __i) {
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

  template <class _Ty, class _Uy>
  STDEXEC_ATTRIBUTE(nodiscard, always_inline)
  constexpr auto __forward_like(_Uy&& __uy) noexcept -> auto&& {
    return static_cast<__copy_cvref_t<_Ty&&, STDEXEC_REMOVE_REFERENCE(_Uy)>>(__uy);
  }

  STDEXEC_PRAGMA_PUSH()
  STDEXEC_PRAGMA_IGNORE_GNU("-Wold-style-cast")

  // A derived-to-base cast that works even when the base is not accessible from derived.
  template <class _Tp, class _Up>
  STDEXEC_ATTRIBUTE(host, device)
  constexpr auto __c_upcast(_Up&& u) noexcept -> __copy_cvref_t<_Up&&, _Tp>
    requires __decays_to<_Tp, _Tp>
  {
    static_assert(STDEXEC_IS_BASE_OF(_Tp, __decay_t<_Up>));
    return (__copy_cvref_t<_Up&&, _Tp>) static_cast<_Up&&>(u);
  }

  // A base-to-derived cast that works even when the base is not accessible from derived.
  template <class _Tp, class _Up>
  STDEXEC_ATTRIBUTE(host, device)
  constexpr auto __c_downcast(_Up&& u) noexcept -> __copy_cvref_t<_Up&&, _Tp>
    requires __decays_to<_Tp, _Tp>
  {
    static_assert(STDEXEC_IS_BASE_OF(__decay_t<_Up>, _Tp));
    return (__copy_cvref_t<_Up&&, _Tp>) static_cast<_Up&&>(u);
  }

  STDEXEC_PRAGMA_POP()

  template <class _Ty>
  struct __indestructible {
    template <class... _Us>
    constexpr __indestructible(_Us&&... __us) noexcept(__nothrow_constructible_from<_Ty, _Us...>)
      : __value(static_cast<_Us&&>(__us)...) {
    }

    constexpr ~__indestructible() {
    }

    constexpr auto get() noexcept -> _Ty& {
      return __value;
    }

    constexpr auto get() const noexcept -> const _Ty& {
      return __value;
    }

    union {
      _Ty __value;
    };
  };

  template <class _Ty>
  constexpr auto __decay_copy(_Ty) noexcept -> _Ty;

#if defined(__cpp_auto_cast) && (__cpp_auto_cast >= 2021'10L)
#  define STDEXEC_DECAY_COPY(...) auto(__VA_ARGS__)
#else
#  define STDEXEC_DECAY_COPY(...) (true ? (__VA_ARGS__) : STDEXEC::__decay_copy(__VA_ARGS__))
#endif
} // namespace STDEXEC
