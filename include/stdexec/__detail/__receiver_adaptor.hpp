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
#include "__cpo.hpp"
#include "__receivers.hpp"
#include "__tag_invoke.hpp"
#include "__type_traits.hpp"
#include "__utility.hpp"

namespace stdexec {
  namespace __adaptors {
    namespace __no {
      struct __nope { };

      struct __receiver : __nope {
        using receiver_concept = receiver_t;
      };

      template <same_as<set_error_t> _Tag>
      void tag_invoke(_Tag, __receiver, std::exception_ptr) noexcept;
      template <same_as<set_stopped_t> _Tag>
      void tag_invoke(_Tag, __receiver) noexcept;
      auto tag_invoke(get_env_t, __receiver) noexcept -> empty_env;
    } // namespace __no

    using __not_a_receiver = __no::__receiver;

    template <class _Base>
    struct __adaptor_base {
      template <class _T1>
        requires constructible_from<_Base, _T1>
      explicit __adaptor_base(_T1&& __base)
        : __base_(static_cast<_T1&&>(__base)) {
      }

     private:
      STDEXEC_ATTRIBUTE((no_unique_address))
      _Base __base_;

     protected:
      STDEXEC_ATTRIBUTE((host, device, always_inline))
      _Base&
        base() & noexcept {
        return __base_;
      }

      STDEXEC_ATTRIBUTE((host, device, always_inline))
      const _Base&
        base() const & noexcept {
        return __base_;
      }

      STDEXEC_ATTRIBUTE((host, device, always_inline))
      _Base&&
        base() && noexcept {
        return static_cast<_Base&&>(__base_);
      }
    };

    template <derived_from<__no::__nope> _Base>
    struct __adaptor_base<_Base> { };

// BUGBUG Not to spec: on gcc and nvc++, member functions in derived classes
// don't shadow type aliases of the same name in base classes. :-O
// On mingw gcc, 'bool(type::existing_member_function)' evaluates to true,
// but 'int(type::existing_member_function)' is an error (as desired).
#define STDEXEC_DISPATCH_MEMBER(_TAG)                                                              \
  template <class _Self, class... _Ts>                                                             \
  STDEXEC_ATTRIBUTE((host, device, always_inline))                                                 \
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
      friend _Derived;
      STDEXEC_DEFINE_MEMBER(set_value);
      STDEXEC_DEFINE_MEMBER(set_error);
      STDEXEC_DEFINE_MEMBER(set_stopped);
      STDEXEC_DEFINE_MEMBER(get_env);

      static constexpr bool __has_base = !derived_from<_Base, __no::__nope>;

      template <class _Dp>
      using __base_from_derived_t = decltype(__declval<_Dp>().base());

      using __get_base_t =
        __if_c<__has_base, __mbind_back_q<__copy_cvref_t, _Base>, __q<__base_from_derived_t>>;

      template <class _Dp>
      using __base_t = __minvoke<__get_base_t, _Dp&&>;

      template <class _Dp>
      STDEXEC_ATTRIBUTE((host, device))
      static auto
        __get_base(_Dp&& __self) noexcept -> __base_t<_Dp> {
        if constexpr (__has_base) {
          return __c_upcast<receiver_adaptor>(static_cast<_Dp&&>(__self)).base();
        } else {
          return static_cast<_Dp&&>(__self).base();
        }
      }

      template <__same_as<set_value_t> _SetValue, class... _As>
      STDEXEC_ATTRIBUTE((host, device, always_inline))
      friend auto
        tag_invoke(_SetValue, _Derived&& __self, _As&&... __as) noexcept //
        -> __msecond<                                                    //
          __if_c<__same_as<set_value_t, _SetValue>>,
          decltype(STDEXEC_CALL_MEMBER(
            set_value,
            static_cast<_Derived&&>(__self),
            static_cast<_As&&>(__as)...))> {
        static_assert(noexcept(STDEXEC_CALL_MEMBER(
          set_value, static_cast<_Derived&&>(__self), static_cast<_As&&>(__as)...)));
        STDEXEC_CALL_MEMBER(set_value, static_cast<_Derived&&>(__self), static_cast<_As&&>(__as)...);
      }

      template <__same_as<set_value_t> _SetValue, class _Dp = _Derived, class... _As>
        requires STDEXEC_MISSING_MEMBER(_Dp, set_value) && tag_invocable<_SetValue, __base_t<_Dp>, _As...>
      STDEXEC_ATTRIBUTE((host, device, always_inline))
      friend void
        tag_invoke(_SetValue, _Derived&& __self, _As&&... __as) noexcept {
        stdexec::set_value(__get_base(static_cast<_Dp&&>(__self)), static_cast<_As&&>(__as)...);
      }

      template <__same_as<set_error_t> _SetError, class _Error>
      STDEXEC_ATTRIBUTE((host, device, always_inline))
      friend auto
        tag_invoke(_SetError, _Derived&& __self, _Error&& __err) noexcept //
        -> __msecond<                                                     //
          __if_c<__same_as<set_error_t, _SetError>>,
          decltype(STDEXEC_CALL_MEMBER(
            set_error,
            static_cast<_Derived&&>(__self),
            static_cast<_Error&&>(__err)))> {
        static_assert(noexcept(STDEXEC_CALL_MEMBER(
          set_error, static_cast<_Derived&&>(__self), static_cast<_Error&&>(__err))));
        STDEXEC_CALL_MEMBER(
          set_error, static_cast<_Derived&&>(__self), static_cast<_Error&&>(__err));
      }

      template <__same_as<set_error_t> _SetError, class _Error, class _Dp = _Derived>
        requires STDEXEC_MISSING_MEMBER(_Dp, set_error) && tag_invocable<_SetError, __base_t<_Dp>, _Error>
      STDEXEC_ATTRIBUTE((host, device, always_inline))
      friend void
        tag_invoke(_SetError, _Derived&& __self, _Error&& __err) noexcept {
        stdexec::set_error(
          __get_base(static_cast<_Derived&&>(__self)), static_cast<_Error&&>(__err));
      }

      template <__same_as<set_stopped_t> _SetStopped, class _Dp = _Derived>
      STDEXEC_ATTRIBUTE((host, device, always_inline))
      friend auto
        tag_invoke(_SetStopped, _Derived&& __self) noexcept //
        -> __msecond<                                       //
          __if_c<__same_as<set_stopped_t, _SetStopped>>,
          decltype(STDEXEC_CALL_MEMBER(set_stopped, static_cast<_Dp&&>(__self)))> {
        static_assert(noexcept(STDEXEC_CALL_MEMBER(set_stopped, static_cast<_Derived&&>(__self))));
        STDEXEC_CALL_MEMBER(set_stopped, static_cast<_Derived&&>(__self));
      }

      template <__same_as<set_stopped_t> _SetStopped, class _Dp = _Derived>
        requires STDEXEC_MISSING_MEMBER(_Dp, set_stopped) && tag_invocable<_SetStopped, __base_t<_Dp>>
      STDEXEC_ATTRIBUTE((host, device, always_inline))
      friend void
        tag_invoke(_SetStopped, _Derived&& __self) noexcept {
        stdexec::set_stopped(__get_base(static_cast<_Derived&&>(__self)));
      }

      // Pass through the get_env receiver query
      template <__same_as<get_env_t> _GetEnv, class _Dp = _Derived>
      STDEXEC_ATTRIBUTE((host, device, always_inline))
      friend auto
        tag_invoke(_GetEnv, const _Derived& __self) noexcept
        -> decltype(STDEXEC_CALL_MEMBER(get_env, static_cast<const _Dp&>(__self))) {
        static_assert(noexcept(STDEXEC_CALL_MEMBER(get_env, __self)));
        return STDEXEC_CALL_MEMBER(get_env, __self);
      }

      template <__same_as<get_env_t> _GetEnv, class _Dp = _Derived>
        requires STDEXEC_MISSING_MEMBER(_Dp, get_env)
          STDEXEC_ATTRIBUTE((host, device, always_inline))
      friend auto
        tag_invoke(_GetEnv, const _Derived& __self) noexcept -> env_of_t<__base_t<const _Dp&>> {
        return stdexec::get_env(__get_base(__self));
      }

     public:
      receiver_adaptor() = default;
      using __adaptor_base<_Base>::__adaptor_base;

      using receiver_concept = receiver_t;
    };
  } // namespace __adaptors

  template <__class _Derived, receiver _Base = __adaptors::__not_a_receiver>
  using receiver_adaptor = __adaptors::receiver_adaptor<_Derived, _Base>;
} // namespace stdexec
