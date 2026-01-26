/*
 * Copyright (c) 2022 NVIDIA Corporation
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

#include "__atomic.hpp"
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#if STDEXEC_TSAN()
#  include <sanitizer/tsan_interface.h>
#endif

namespace STDEXEC {
  namespace __ptr {
    template <std::size_t _ReservedBits>
    struct __count_and_bits {
      static constexpr std::size_t __ref_count_increment = 1ul << _ReservedBits;

      enum struct __bits : std::size_t {
      };

      friend constexpr auto __count(__bits __b) noexcept -> std::size_t {
        return static_cast<std::size_t>(__b) / __ref_count_increment;
      }

      template <std::size_t _Bit>
      friend constexpr auto __bit(__bits __b) noexcept -> bool {
        static_assert(_Bit < _ReservedBits, "Bit index out of range");
        return (static_cast<std::size_t>(__b) & (1ul << _Bit)) != 0;
      }
    };

    template <std::size_t _ReservedBits>
    using __bits_t = __count_and_bits<_ReservedBits>::__bits;

    template <class _Ty, std::size_t _ReservedBits>
    struct __make_intrusive_t;

    template <class _Ty, std::size_t _ReservedBits = 0ul>
    class __intrusive_ptr;

    template <class _Ty, std::size_t _ReservedBits = 0ul>
    struct __enable_intrusive_from_this {
      constexpr auto __intrusive_from_this() noexcept -> __intrusive_ptr<_Ty, _ReservedBits>;
      constexpr auto
        __intrusive_from_this() const noexcept -> __intrusive_ptr<const _Ty, _ReservedBits>;
     private:
      using __bits_t = __count_and_bits<_ReservedBits>::__bits;
      friend _Ty;
      constexpr auto __inc_ref() noexcept -> __bits_t;
      constexpr auto __dec_ref() noexcept -> __bits_t;

      template <std::size_t _Bit>
      [[nodiscard]]
      constexpr auto __is_set() const noexcept -> bool;
      template <std::size_t _Bit>
      constexpr auto __set_bit() noexcept -> __bits_t;
      template <std::size_t _Bit>
      constexpr auto __clear_bit() noexcept -> __bits_t;
    };

    STDEXEC_PRAGMA_PUSH()
    STDEXEC_PRAGMA_IGNORE_GNU("-Wtsan")

    template <class _Ty, std::size_t _ReservedBits>
    struct __control_block {
      using __bits_t = __count_and_bits<_ReservedBits>::__bits;
      static constexpr std::size_t __ref_count_increment = 1ul << _ReservedBits;

      union {
        _Ty __value;
      };
      __std::atomic<std::size_t> __ref_count_;

      template <class... _Us>
      constexpr explicit __control_block(_Us&&... __us)
        noexcept(__nothrow_constructible_from<_Ty, _Us...>)
        : __ref_count_(__ref_count_increment) {
        // Construct the value *after* the initialization of the atomic in case the constructor of
        // _Ty calls __intrusive_from_this() (which increments the ref count):
        std::construct_at(std::addressof(__value), static_cast<_Us&&>(__us)...);
      }

      constexpr ~__control_block() {
        __value.~_Ty();
      }

      constexpr auto __inc_ref_() noexcept -> __bits_t {
        auto __old = __ref_count_.fetch_add(__ref_count_increment, __std::memory_order_relaxed);
        return static_cast<__bits_t>(__old);
      }

      constexpr auto __dec_ref_() noexcept -> __bits_t {
        auto __old = __ref_count_.fetch_sub(__ref_count_increment, __std::memory_order_acq_rel);
        if (__count(static_cast<__bits_t>(__old)) == 1) {
          delete this;
        }
        return static_cast<__bits_t>(__old);
      }

      // Returns true if the bit was set, false if it was already set.
      template <std::size_t _Bit>
      [[nodiscard]]
      constexpr auto __is_set_() const noexcept -> bool {
        auto __old = __ref_count_.load(__std::memory_order_relaxed);
        return __bit<_Bit>(static_cast<__bits_t>(__old));
      }

      template <std::size_t _Bit>
      constexpr auto __set_bit_() noexcept -> __bits_t {
        static_assert(_Bit < _ReservedBits, "Bit index out of range");
        constexpr std::size_t __mask = 1ul << _Bit;
        auto __old = __ref_count_.fetch_or(__mask, __std::memory_order_acq_rel);
        return static_cast<__bits_t>(__old);
      }

      // Returns true if the bit was cleared, false if it was already cleared.
      template <std::size_t _Bit>
      constexpr auto __clear_bit_() noexcept -> __bits_t {
        static_assert(_Bit < _ReservedBits, "Bit index out of range");
        constexpr std::size_t __mask = 1ul << _Bit;
        auto __old = __ref_count_.fetch_and(~__mask, __std::memory_order_acq_rel);
        return static_cast<__bits_t>(__old);
      }
    };

    STDEXEC_PRAGMA_POP()

    template <class _Ty, std::size_t _ReservedBits /* = 0ul */>
    class __intrusive_ptr {
      using _UncvTy = std::remove_cv_t<_Ty>;
      using __enable_intrusive_t = __enable_intrusive_from_this<_UncvTy, _ReservedBits>;
      friend _Ty;
      friend struct __make_intrusive_t<_Ty, _ReservedBits>;
      friend struct __enable_intrusive_from_this<_UncvTy, _ReservedBits>;

      __control_block<_UncvTy, _ReservedBits>* __data_{nullptr};

      constexpr explicit __intrusive_ptr(__control_block<_UncvTy, _ReservedBits>* __data) noexcept
        : __data_(__data) {
      }

      constexpr void __inc_ref_() noexcept {
        if (__data_) {
          __data_->__inc_ref_();
        }
      }

      constexpr void __dec_ref_() noexcept {
        if (__data_) {
          __data_->__dec_ref_();
        }
      }

      // For use when types want to take over manual control of the reference count.
      // Very unsafe, but useful for implementing custom reference counting.
      [[nodiscard]]
      constexpr auto __release_() noexcept -> __enable_intrusive_t* {
        auto* __data = std::exchange(__data_, nullptr);
        return __data ? &__c_upcast<__enable_intrusive_t>(__data->__value) : nullptr;
      }

     public:
      using element_type = _Ty;

      constexpr __intrusive_ptr() = default;

      constexpr __intrusive_ptr(__intrusive_ptr&& __that) noexcept
        : __data_(std::exchange(__that.__data_, nullptr)) {
      }

      constexpr __intrusive_ptr(const __intrusive_ptr& __that) noexcept
        : __data_(__that.__data_) {
        __inc_ref_();
      }

      constexpr __intrusive_ptr(__enable_intrusive_from_this<_Ty, _ReservedBits>* __that) noexcept
        : __intrusive_ptr(__that ? __that->__intrusive_from_this() : __intrusive_ptr()) {
      }

      constexpr auto operator=(__intrusive_ptr&& __that) noexcept -> __intrusive_ptr& {
        [[maybe_unused]]
        __intrusive_ptr __old{std::exchange(__data_, std::exchange(__that.__data_, nullptr))};
        return *this;
      }

      constexpr auto operator=(const __intrusive_ptr& __that) noexcept -> __intrusive_ptr& {
        return operator=(__intrusive_ptr(__that));
      }

      constexpr auto operator=(__enable_intrusive_from_this<_Ty, _ReservedBits>* __that) noexcept
        -> __intrusive_ptr& {
        return operator=(__that ? __that->__intrusive_from_this() : __intrusive_ptr());
      }

      constexpr ~__intrusive_ptr() {
        __dec_ref_();
      }

      constexpr void reset() noexcept {
        operator=({});
      }

      constexpr void swap(__intrusive_ptr& __that) noexcept {
        std::swap(__data_, __that.__data_);
      }

      constexpr auto get() const noexcept -> _Ty* {
        return &__data_->__value;
      }

      constexpr auto operator->() const noexcept -> _Ty* {
        return &__data_->__value;
      }

      constexpr auto operator*() const noexcept -> _Ty& {
        return __data_->__value;
      }

      constexpr explicit operator bool() const noexcept {
        return __data_ != nullptr;
      }

      constexpr auto operator!() const noexcept -> bool {
        return __data_ == nullptr;
      }

      constexpr auto operator==(const __intrusive_ptr&) const -> bool = default;

      constexpr auto operator==(std::nullptr_t) const noexcept -> bool {
        return __data_ == nullptr;
      }
    };

    template <class _Ty, std::size_t _ReservedBits>
    constexpr auto
      __enable_intrusive_from_this<_Ty, _ReservedBits>::__intrusive_from_this() noexcept
      -> __intrusive_ptr<_Ty, _ReservedBits> {
      auto* __data = reinterpret_cast<__control_block<_Ty, _ReservedBits>*>(
        &__c_downcast<_Ty>(*this));
      __data->__inc_ref_();
      return __intrusive_ptr<_Ty, _ReservedBits>{__data};
    }

    template <class _Ty, std::size_t _ReservedBits>
    constexpr auto
      __enable_intrusive_from_this<_Ty, _ReservedBits>::__intrusive_from_this() const noexcept
      -> __intrusive_ptr<const _Ty, _ReservedBits> {
      auto* __data = reinterpret_cast<__control_block<_Ty, _ReservedBits>*>(
        &__c_downcast<_Ty>(*this));
      __data->__inc_ref_();
      return __intrusive_ptr<const _Ty, _ReservedBits>{__data};
    }

    template <class _Ty, std::size_t _ReservedBits>
    constexpr auto __enable_intrusive_from_this<_Ty, _ReservedBits>::__inc_ref() noexcept
      -> __ptr::__bits_t<_ReservedBits> {
      auto* __data = reinterpret_cast<__control_block<_Ty, _ReservedBits>*>(
        &__c_downcast<_Ty>(*this));
      return __data->__inc_ref_();
    }

    template <class _Ty, std::size_t _ReservedBits>
    constexpr auto __enable_intrusive_from_this<_Ty, _ReservedBits>::__dec_ref() noexcept
      -> __ptr::__bits_t<_ReservedBits> {

      auto* __data = reinterpret_cast<__control_block<_Ty, _ReservedBits>*>(
        &__c_downcast<_Ty>(*this));
      return __data->__dec_ref_();
    }

    template <class _Ty, std::size_t _ReservedBits>
    template <std::size_t _Bit>
    constexpr auto
      __enable_intrusive_from_this<_Ty, _ReservedBits>::__is_set() const noexcept -> bool {
      auto* __data = reinterpret_cast<const __control_block<_Ty, _ReservedBits>*>(
        &__c_downcast<_Ty>(*this));
      return __data->template __is_set_<_Bit>();
    }

    template <class _Ty, std::size_t _ReservedBits>
    template <std::size_t _Bit>
    constexpr auto __enable_intrusive_from_this<_Ty, _ReservedBits>::__set_bit() noexcept
      -> __ptr::__bits_t<_ReservedBits> {
      auto* __data = reinterpret_cast<__control_block<_Ty, _ReservedBits>*>(
        &__c_downcast<_Ty>(*this));
      return __data->template __set_bit_<_Bit>();
    }

    template <class _Ty, std::size_t _ReservedBits>
    template <std::size_t _Bit>
    constexpr auto __enable_intrusive_from_this<_Ty, _ReservedBits>::__clear_bit() noexcept
      -> __ptr::__bits_t<_ReservedBits> {
      auto* __data = reinterpret_cast<__control_block<_Ty, _ReservedBits>*>(
        &__c_downcast<_Ty>(*this));
      return __data->template __clear_bit_<_Bit>();
    }

    template <class _Ty, std::size_t _ReservedBits>
    struct __make_intrusive_t {
      template <class... _Us>
        requires __std::constructible_from<_Ty, _Us...>
      auto operator()(_Us&&... __us) const -> __intrusive_ptr<_Ty, _ReservedBits> {
        using _UncvTy = std::remove_cv_t<_Ty>;
        return __intrusive_ptr<_Ty, _ReservedBits>{
          ::new __control_block<_UncvTy, _ReservedBits>{static_cast<_Us&&>(__us)...}};
      }
    };
  } // namespace __ptr

  using __ptr::__intrusive_ptr;
  using __ptr::__enable_intrusive_from_this;
  template <class _Ty, std::size_t _ReservedBits = 0ul>
  inline constexpr __ptr::__make_intrusive_t<_Ty, _ReservedBits> __make_intrusive{};

} // namespace STDEXEC
