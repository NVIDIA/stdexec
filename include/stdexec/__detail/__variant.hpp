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
#include "__scope.hpp"

#include <cstddef>
#include <new>
#include <utility>

/********************************************************************************/
/* NB: The variant type implemented here default-constructs into the valueless  */
/* state. This is different from std::variant which default-constructs into the */
/* first alternative. This is done to simplify the implementation and to avoid  */
/* the need for a default constructor for each alternative type.                */
/********************************************************************************/

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace stdexec {
#if STDEXEC_NVHPC()
  enum __variant_npos_t : std::size_t {
    __variant_npos = ~0UL
  };
#else
  STDEXEC_GLOBAL_CONSTANT std::size_t __variant_npos = ~0UL;
#endif

  struct __monostate { };

  namespace __var {
    template <class _Ty>
    STDEXEC_ATTRIBUTE(host, device, always_inline)
    void __destroy_at(_Ty *ptr) noexcept {
      ptr->~_Ty();
    }

    STDEXEC_ATTRIBUTE(host, device)
    inline auto __mk_index_guard(std::size_t &__index, std::size_t __new) noexcept {
      __index = __new;
      return __scope_guard{[&__index]() noexcept { __index = __variant_npos; }};
    }

    template <auto _Idx, class... _Ts>
    class __variant;

    template <>
    class __variant<__indices<>{}> {
     public:
      template <class _Fn, class... _Us>
      STDEXEC_ATTRIBUTE(host, device)
      void visit(_Fn &&, _Us &&...) const noexcept {
        STDEXEC_ASSERT(false);
      }

      STDEXEC_ATTRIBUTE(host, device) static constexpr auto index() noexcept -> std::size_t {
        return __variant_npos;
      }

      STDEXEC_ATTRIBUTE(host, device) static constexpr auto is_valueless() noexcept -> bool {
        return true;
      }
    };

    template <std::size_t... _Is, __indices<_Is...> _Idx, class... _Ts>
    class __variant<_Idx, _Ts...> {
      static constexpr std::size_t __max_size = stdexec::__umax({sizeof(_Ts)...});
      static_assert(__max_size != 0);
      std::size_t __index_{__variant_npos};
      alignas(_Ts...) unsigned char __storage_[__max_size];

      STDEXEC_ATTRIBUTE(host, device) void __destroy() noexcept {
        auto __index = std::exchange(__index_, __variant_npos);
        if (__variant_npos != __index) {
          ((_Is == __index ? __var::__destroy_at(static_cast<_Ts *>(__get_ptr())) : void(0)), ...);
        }
      }

      template <std::size_t _Ny>
      using __at = __m_at_c<_Ny, _Ts...>;

     public:
      // immovable:
      __variant(__variant &&) = delete;

      STDEXEC_ATTRIBUTE(host, device) __variant() noexcept = default;

      STDEXEC_ATTRIBUTE(host, device) ~__variant() {
        __destroy();
      }

      [[nodiscard]]
      STDEXEC_ATTRIBUTE(host, device, always_inline) auto __get_ptr() noexcept -> void * {
        return __storage_;
      }

      [[nodiscard]]
      STDEXEC_ATTRIBUTE(host, device, always_inline) auto index() const noexcept -> std::size_t {
        return __index_;
      }

      [[nodiscard]]
      STDEXEC_ATTRIBUTE(host, device, always_inline) auto is_valueless() const noexcept -> bool {
        return __index_ == __variant_npos;
      }

      // The following emplace functions must take great care to avoid use-after-free bugs.
      // If the object being constructed calls `start` on a newly created operation state
      // (as does the object returned from `submit`), and if `start` completes inline, it
      // could cause the destruction of the outer operation state that owns *this. The
      // function below uses the following pattern to avoid this:
      // 1. Store the new index in __index_.
      // 2. Create a scope guard that will reset __index_ to __variant_npos if the
      //    constructor throws.
      // 3. Construct the new object in the storage, which may cause the invalidation of
      //    *this. The emplace function must not access any members of *this after this point.
      // 4. Dismiss the scope guard, which will leave __index_ set to the new index.
      // 5. Return a reference to the new object -- which may be invalid! Calling code
      //    must be aware of the danger.
      template <class _Ty, class... _As>
      STDEXEC_ATTRIBUTE(host, device)
      auto emplace(_As &&...__as) noexcept(__nothrow_constructible_from<_Ty, _As...>) -> _Ty & {
        constexpr std::size_t __new_index = stdexec::__index_of<_Ty, _Ts...>();
        static_assert(__new_index != __variant_npos, "Type not in variant");

        __destroy();
        auto __sg = __mk_index_guard(__index_, __new_index);
        auto *__p = ::new (__storage_) _Ty{static_cast<_As &&>(__as)...};
        __sg.__dismiss();
        return *std::launder(__p);
      }

      template <std::size_t _Ny, class... _As>
      STDEXEC_ATTRIBUTE(host, device)
      auto emplace(_As &&...__as) noexcept(__nothrow_constructible_from<__at<_Ny>, _As...>)
        -> __at<_Ny> & {
        static_assert(_Ny < sizeof...(_Ts), "variant index is too large");

        __destroy();
        auto __sg = __mk_index_guard(__index_, _Ny);
        auto *__p = ::new (__storage_) __at<_Ny>{static_cast<_As &&>(__as)...};
        __sg.__dismiss();
        return *std::launder(__p);
      }

      template <std::size_t _Ny, class _Fn, class... _As>
      STDEXEC_ATTRIBUTE(host, device)
      auto emplace_from_at(_Fn &&__fn, _As &&...__as) noexcept(__nothrow_callable<_Fn, _As...>)
        -> __at<_Ny> & {
        static_assert(
          __same_as<__call_result_t<_Fn, _As...>, __at<_Ny>>,
          "callable does not return the correct type");

        __destroy();
        auto __sg = __mk_index_guard(__index_, _Ny);
        auto *__p = ::new (__storage_)
          __at<_Ny>(static_cast<_Fn &&>(__fn)(static_cast<_As &&>(__as)...));
        __sg.__dismiss();
        return *std::launder(__p);
      }

      template <class _Fn, class... _As>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      auto emplace_from(_Fn &&__fn, _As &&...__as) noexcept(__nothrow_callable<_Fn, _As...>)
        -> __call_result_t<_Fn, _As...> & {
        using __result_t = __call_result_t<_Fn, _As...>;
        constexpr std::size_t __new_index = stdexec::__index_of<__result_t, _Ts...>();
        static_assert(__new_index != __variant_npos, "Type not in variant");
        return emplace_from_at<__new_index>(
          static_cast<_Fn &&>(__fn), static_cast<_As &&>(__as)...);
      }

      template <class _Fn, class _Self, class... _As>
      STDEXEC_ATTRIBUTE(host, device)
      static void visit(_Fn &&__fn, _Self &&__self, _As &&...__as)
        noexcept((__nothrow_callable<_Fn, _As..., __copy_cvref_t<_Self, _Ts>> && ...)) {
        STDEXEC_ASSERT(__self.__index_ != __variant_npos);
        auto __index = __self.__index_; // make it local so we don't access it after it's deleted.
        ((_Is == __index
            ? static_cast<_Fn &&>(__fn)(
                static_cast<_As &&>(__as)..., static_cast<_Self &&>(__self).template get<_Is>())
            : void()),
         ...);
      }

      template <std::size_t _Ny>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      auto get() && noexcept -> decltype(auto) {
        STDEXEC_ASSERT(_Ny == __index_);
        return static_cast<__at<_Ny> &&>(*reinterpret_cast<__at<_Ny> *>(__storage_));
      }

      template <std::size_t _Ny>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      auto get() & noexcept -> decltype(auto) {
        STDEXEC_ASSERT(_Ny == __index_);
        return *reinterpret_cast<__at<_Ny> *>(__storage_);
      }

      template <std::size_t _Ny>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      auto get() const & noexcept -> decltype(auto) {
        STDEXEC_ASSERT(_Ny == __index_);
        return *reinterpret_cast<const __at<_Ny> *>(__storage_);
      }
    };
  } // namespace __var

  using __var::__variant;

  template <class... _Ts>
  using __variant_for = __variant<__indices_for<_Ts...>{}, _Ts...>;

  template <class... Ts>
  using __uniqued_variant_for = __mcall<__munique<__qq<__variant_for>>, __decay_t<Ts>...>;

  // So we can use __variant as a typelist
  template <auto _Idx, class... _Ts>
  struct __muncurry_<__variant<_Idx, _Ts...>> {
    template <class _Fn>
    using __f = __minvoke<_Fn, _Ts...>;
  };
} // namespace stdexec

STDEXEC_PRAGMA_POP()
