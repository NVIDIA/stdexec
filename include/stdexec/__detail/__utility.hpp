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
#include "__type_traits.hpp"

#include <type_traits>

namespace stdexec {
  template <class...>
  struct __undefined;

  struct __ { };

  struct __ignore {
    __ignore() = default;

    STDEXEC_ATTRIBUTE((always_inline))
    constexpr __ignore(auto&&...) noexcept {
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

  namespace __detail {
    template <class _Cpcvref>
    inline constexpr auto __forward_like = []<class _Uy>(_Uy&& __uy) noexcept -> auto&& {
      return static_cast<typename _Cpcvref::template __f<std::remove_reference_t<_Uy>>>(__uy);
    };
  } // namespace __detail

  template <class _Ty>
  inline constexpr auto const & __forward_like = __detail::__forward_like<__copy_cvref_fn<_Ty&&>>;

  STDEXEC_PRAGMA_PUSH()
  STDEXEC_PRAGMA_IGNORE_GNU("-Wold-style-cast")

  // A derived-to-base cast that works even when the base is not accessible from derived.
  template <class _Tp, class _Up>
  STDEXEC_ATTRIBUTE((host, device))
  auto
    __c_upcast(_Up&& u) noexcept -> __copy_cvref_t<_Up&&, _Tp>
    requires __decays_to<_Tp, _Tp>
  {
    static_assert(STDEXEC_IS_BASE_OF(_Tp, __decay_t<_Up>));
    return (__copy_cvref_t<_Up&&, _Tp>) static_cast<_Up&&>(u);
  }

  // A base-to-derived cast that works even when the base is not accessible from derived.
  template <class _Tp, class _Up>
  STDEXEC_ATTRIBUTE((host, device))
  auto
    __c_downcast(_Up&& u) noexcept -> __copy_cvref_t<_Up&&, _Tp>
    requires __decays_to<_Tp, _Tp>
  {
    static_assert(STDEXEC_IS_BASE_OF(__decay_t<_Up>, _Tp));
    return (__copy_cvref_t<_Up&&, _Tp>) static_cast<_Up&&>(u);
  }

  STDEXEC_PRAGMA_POP()

  template <class _Ty>
  _Ty __decay_copy(_Ty) noexcept;
} // namespace stdexec

#if defined(__cpp_auto_cast) && (__cpp_auto_cast >= 202110UL)
#  define STDEXEC_DECAY_COPY(...) auto(__VA_ARGS__)
#else
#  define STDEXEC_DECAY_COPY(...) (true ? (__VA_ARGS__) : stdexec::__decay_copy(__VA_ARGS__))
#endif
