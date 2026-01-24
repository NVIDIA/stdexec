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

#include "__concepts.hpp"
#include "__receivers.hpp"
#include "__type_traits.hpp"
#include "__utility.hpp"

namespace STDEXEC {
  namespace __adaptors {
    namespace __no {
      struct __nope { };

      struct __receiver : __nope {
        using receiver_concept = receiver_t;

        constexpr void set_error(std::exception_ptr) noexcept;
        constexpr void set_stopped() noexcept;
        [[nodiscard]]
        constexpr auto get_env() const noexcept -> env<>;
      };
    } // namespace __no

    using __not_a_receiver = __no::__receiver;

    template <class _Base>
    struct __adaptor_base {
      template <class _T1>
        requires __std::constructible_from<_Base, _T1>
      explicit __adaptor_base(_T1&& __base)
        : __base_(static_cast<_T1&&>(__base)) {
      }

     private:
      STDEXEC_ATTRIBUTE(no_unique_address) _Base __base_;

     protected:
      STDEXEC_ATTRIBUTE(host, device, always_inline) auto base() & noexcept -> _Base& {
        return __base_;
      }

      STDEXEC_ATTRIBUTE(host, device, always_inline)
      auto base() const & noexcept -> const _Base& {
        return __base_;
      }

      STDEXEC_ATTRIBUTE(host, device, always_inline) auto base() && noexcept -> _Base&& {
        return static_cast<_Base&&>(__base_);
      }
    };

    template <__std::derived_from<__no::__nope> _Base>
    struct __adaptor_base<_Base> { };

// BUGBUG Not to spec: on gcc and nvc++, member functions in derived classes
// don't shadow type aliases of the same name in base classes. :-O
// On mingw gcc, 'bool(type::existing_member_function)' evaluates to true,
// but 'int(type::existing_member_function)' is an error (as desired).
#define STDEXEC_DISPATCH_MEMBER(_TAG)                                                              \
  template <class _Self, class... _Ts>                                                             \
  STDEXEC_ATTRIBUTE(host, device, always_inline)                                                   \
  static auto __call_##_TAG(_Self&& __self, _Ts&&... __ts) noexcept                                \
    -> decltype((static_cast<_Self&&>(__self))._TAG(static_cast<_Ts&&>(__ts)...)) {                \
    static_assert(noexcept((static_cast<_Self&&>(__self))._TAG(static_cast<_Ts&&>(__ts)...)));     \
    return static_cast<_Self&&>(__self)._TAG(static_cast<_Ts&&>(__ts)...);                         \
  } /**/
#define STDEXEC_CALL_MEMBER(_TAG, ...) __call_##_TAG(__VA_ARGS__)

#if STDEXEC_CLANG()
// Only clang gets this right.
#  define STDEXEC_MISSING_MEMBER(_Dp, _TAG) requires { typename _Dp::_TAG; }
#  define STDEXEC_DEFINE_MEMBER(_TAG)       STDEXEC_DISPATCH_MEMBER(_TAG) using _TAG = void
#else
#  define STDEXEC_MISSING_MEMBER(_Dp, _TAG) (__missing_##_TAG<_Dp>())
#  define STDEXEC_DEFINE_MEMBER(_TAG)                                                              \
    template <class _Dp>                                                                           \
    static constexpr bool __missing_##_TAG() noexcept {                                            \
      return requires { requires bool(int(_Dp::_TAG)); };                                          \
    }                                                                                              \
    STDEXEC_DISPATCH_MEMBER(_TAG)                                                                  \
    static constexpr int _TAG = 1 /**/
#endif

    template <__class _Derived, class _Base = __not_a_receiver>
    struct receiver_adaptor
      : __adaptor_base<_Base>
      , receiver_t {

      static constexpr bool __has_base = !__std::derived_from<_Base, __no::__nope>;

      template <class _Self>
      using __base_from_derived_t = decltype(__declval<_Self>().base());

      using __get_base_fn =
        __if_c<__has_base, __mbind_back_q<__copy_cvref_t, _Base>, __q<__base_from_derived_t>>;

      template <class _Self>
      using __base_t = __minvoke<__get_base_fn, _Self&&>;

      template <class _Self>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr static auto __get_base(_Self&& __self) noexcept -> __base_t<_Self> {
        if constexpr (__has_base) {
          return __c_upcast<receiver_adaptor>(static_cast<_Self&&>(__self)).base();
        } else {
          return static_cast<_Self&&>(__self).base();
        }
      }

     public:
      using receiver_concept = receiver_t;

      constexpr receiver_adaptor() = default;
      using __adaptor_base<_Base>::__adaptor_base;

      template <class... _As, class _Self = _Derived>
        requires __callable<set_value_t, __base_t<_Self>, _As...>
      STDEXEC_ATTRIBUTE(host, device)
      void set_value(_As&&... __as) && noexcept {
        return STDEXEC::set_value(
          __get_base(static_cast<_Self&&>(*this)), static_cast<_As&&>(__as)...);
      }

      template <class _Error, class _Self = _Derived>
        requires __callable<set_error_t, __base_t<_Self>, _Error>
      STDEXEC_ATTRIBUTE(host, device)
      void set_error(_Error&& __err) && noexcept {
        return STDEXEC::set_error(
          __get_base(static_cast<_Self&&>(*this)), static_cast<_Error&&>(__err));
      }

      template <class _Self = _Derived>
        requires __callable<set_stopped_t, __base_t<_Self>>
      STDEXEC_ATTRIBUTE(host, device)
      void set_stopped() && noexcept {
        return STDEXEC::set_stopped(__get_base(static_cast<_Self&&>(*this)));
      }

      template <class _Self = _Derived>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr auto get_env() const noexcept -> env_of_t<__base_t<const _Self&>> {
        return STDEXEC::get_env(__get_base(static_cast<const _Self&>(*this)));
      }
    };
  } // namespace __adaptors

  template <__class _Derived, receiver _Base = __adaptors::__not_a_receiver>
  using receiver_adaptor = __adaptors::receiver_adaptor<_Derived, _Base>;
} // namespace STDEXEC
