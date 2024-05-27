/*
 * Copyright (c) 2024 NVIDIA Corporation
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

#include "__meta.hpp"
#include "__type_traits.hpp"
#include "__utility.hpp"

#include <cstddef>
#include <new>
#include <type_traits>

/********************************************************************************/
/* NB: The variant type implemented here default-constructs into the valueless  */
/* state. This is different from std::variant which default-constructs into the */
/* first alternative. This is done to simplify the implementation and to avoid  */
/* the need for a default constructor for each alternative type.                */
/********************************************************************************/

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace stdexec {
  inline constexpr std::size_t __variant_npos = ~0UL;

  struct __monostate { };

  namespace __var {
    template <class _Idx, class... _Ts>
    class __variant;

    template <>
    class __variant<std::index_sequence<>> {
     public:
      template <class _Fn, class... _Us>
      STDEXEC_ATTRIBUTE((host, device))
      void
        visit(_Fn &&, _Us &&...) const noexcept {
      }

      STDEXEC_ATTRIBUTE((host, device))
      static constexpr bool
        index() noexcept {
        return __variant_npos;
      }

      STDEXEC_ATTRIBUTE((host, device))
      static constexpr bool
        is_valueless() noexcept {
        return true;
      }
    };

    template <std::size_t... _Idx, class... _Ts>
    class __variant<std::index_sequence<_Idx...>, _Ts...> {
      static constexpr std::size_t __max_size = stdexec::__umax({sizeof(_Ts)...});
      static_assert(__max_size != 0);
      std::size_t __index_{__variant_npos};
      alignas(_Ts...) unsigned char __storage_[__max_size];

      STDEXEC_ATTRIBUTE((host, device))
      void
        __destroy() noexcept {
        auto __index = std::exchange(__index_, __variant_npos);
        if (__variant_npos != __index) {
#if STDEXEC_NVHPC()
          // Unknown nvc++ name lookup bug
          ((_Idx == __index ? reinterpret_cast<const __at<_Idx> *>(__storage_)->_Ts::~_Ts()
                            : void(0)),
           ...);
#else
          // casting the destructor expression to void is necessary for MSVC in
          // /permissive- mode.
          ((_Idx == __index ? void(reinterpret_cast<const __at<_Idx> *>(__storage_)->~_Ts())
                            : void(0)),
           ...);
#endif
        }
      }

      template <std::size_t _Ny>
      using __at = __m_at_c<_Ny, _Ts...>;

     public:
      // immovable:
      __variant(__variant &&) = delete;

      STDEXEC_ATTRIBUTE((host, device))
      __variant() noexcept {
      }

      STDEXEC_ATTRIBUTE((host, device))
      ~__variant() {
        __destroy();
      }

      STDEXEC_ATTRIBUTE((host, device, always_inline))
      void *
        __get_ptr() noexcept {
        return __storage_;
      }

      STDEXEC_ATTRIBUTE((host, device, always_inline))
      std::size_t
        index() const noexcept {
        return __index_;
      }

      STDEXEC_ATTRIBUTE((host, device, always_inline))
      bool
        is_valueless() const noexcept {
        return __index_ == __variant_npos;
      }

      template <class _Ty, class... _As>
      STDEXEC_ATTRIBUTE((host, device))
      _Ty &
        emplace(_As &&...__as) //
        noexcept(__nothrow_constructible_from<_Ty, _As...>) {
        constexpr std::size_t __new_index = stdexec::__index_of<_Ty, _Ts...>();
        static_assert(__new_index != __variant_npos, "Type not in variant");

        __destroy();
        ::new (__storage_) _Ty{static_cast<_As &&>(__as)...};
        __index_ = __new_index;
        return *reinterpret_cast<_Ty *>(__storage_);
      }

      template <std::size_t _Ny, class... _As>
      STDEXEC_ATTRIBUTE((host, device))
      __at<_Ny> &
        emplace(_As &&...__as) //
        noexcept(__nothrow_constructible_from<__at<_Ny>, _As...>) {
        static_assert(_Ny < sizeof...(_Ts), "variant index is too large");

        __destroy();
        ::new (__storage_) __at<_Ny>{static_cast<_As &&>(__as)...};
        __index_ = _Ny;
        return *reinterpret_cast<__at<_Ny> *>(__storage_);
      }

      template <class _Fn, class... _As>
      STDEXEC_ATTRIBUTE((host, device))
      auto
        emplace_from(_Fn &&__fn, _As &&...__as) //
        noexcept(__nothrow_callable<_Fn, _As...>) -> __call_result_t<_Fn, _As...> & {
        using __result_t = __call_result_t<_Fn, _As...>;
        constexpr std::size_t __new_index = stdexec::__index_of<__result_t, _Ts...>();
        static_assert(__new_index != __variant_npos, "Type not in variant");

        __destroy();
        ::new (__storage_) __result_t(static_cast<_Fn &&>(__fn)(static_cast<_As &&>(__as)...));
        __index_ = __new_index;
        return *reinterpret_cast<__result_t *>(__storage_);
      }

      template <class _Fn, class _Self, class... _As>
      STDEXEC_ATTRIBUTE((host, device))
      static void
        visit(_Fn &&__fn, _Self &&__self, _As &&...__as) //
        noexcept((__nothrow_callable<_Fn, _As..., __copy_cvref_t<_Self, _Ts>> && ...)) {
        STDEXEC_ASSERT(__self.__index_ != __variant_npos);
        auto __index = __self.__index_; // make it local so we don't access it after it's deleted.
        ((_Idx == __index ? static_cast<_Fn &&>(__fn)(
            static_cast<_As &&>(__as)..., static_cast<_Self &&>(__self).template get<_Idx>())
                          : void()),
         ...);
      }

      template <std::size_t _Ny>
      STDEXEC_ATTRIBUTE((host, device))
      decltype(auto)
        get() && noexcept {
        STDEXEC_ASSERT(_Ny == __index_);
        return static_cast<__at<_Ny> &&>(*reinterpret_cast<__at<_Ny> *>(__storage_));
      }

      template <std::size_t _Ny>
      STDEXEC_ATTRIBUTE((host, device))
      decltype(auto)
        get() & noexcept {
        STDEXEC_ASSERT(_Ny == __index_);
        return *reinterpret_cast<__at<_Ny> *>(__storage_);
      }

      template <std::size_t _Ny>
      STDEXEC_ATTRIBUTE((host, device))
      decltype(auto)
        get() const & noexcept {
        STDEXEC_ASSERT(_Ny == __index_);
        return *reinterpret_cast<const __at<_Ny> *>(__storage_);
      }
    };
  } // namespace __var

  using __var::__variant;

  template <class... _Ts>
  using __variant_for = __variant<std::make_index_sequence<sizeof...(_Ts)>, _Ts...>;

  template <class... Ts>
  using __uniqued_variant_for = __mcall<__munique<__qq<__variant_for>>, __decay_t<Ts>...>;
} // namespace stdexec

STDEXEC_PRAGMA_POP()
