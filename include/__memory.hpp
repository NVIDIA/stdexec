/*
 * Copyright (c) NVIDIA
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

#include <__utility.hpp>
#include <concepts.hpp>

#include <atomic>
#include <memory>
#include <new>

namespace _P2300 {
  namespace __ptr {
    template <class _Ty>
      struct __make_intrusive_ptr_t;

    template <class _Ty>
      struct __enable_intrusive_from_this;

    template <class _Ty>
      struct __control_block {
        alignas(_Ty) unsigned char __value_[sizeof(_Ty)];
        std::atomic<unsigned long> __refcount_;

        template <class... _Us>
          explicit __control_block(_Us&&... __us)
            noexcept(noexcept(_Ty{__declval<_Us>()...}))
            : __refcount_(1u) {
            // Construct the value *after* the initialization of the
            // atomic in case the constructor of _Ty calls
            // __intrusive_from_this() (which increments the atomic):
            ::new ((void*) __value_) _Ty{(_Us&&) __us...};
          }
        ~__control_block() {
          __value().~_Ty();
        }

        _Ty& __value() const noexcept {
          return *(_Ty*) __value_;
        }
      };

    template <class _Ty>
      class __intrusive_ptr {
        using _UncvTy = std::remove_cv_t<_Ty>;
        friend struct __make_intrusive_ptr_t<_Ty>;
        friend struct __enable_intrusive_from_this<_UncvTy>;

        __control_block<_UncvTy>* __data_{nullptr};

        explicit __intrusive_ptr(__control_block<_UncvTy>* __data) noexcept
          : __data_(__data) {}

        void __addref_() noexcept {
          if (__data_) {
            __data_->__refcount_.fetch_add(1, std::memory_order_relaxed);
          }
        }

        void __release_() noexcept {
          if (__data_ && (1u == __data_->__refcount_.fetch_sub(1, std::memory_order_release))) {
            std::atomic_thread_fence(std::memory_order_acquire);
            delete __data_;
          }
        }

       public:
        __intrusive_ptr() = default;
        bool operator==(const __intrusive_ptr&) const = default;

        __intrusive_ptr(__intrusive_ptr&& __that) noexcept
          : __data_(std::exchange(__that.__data_, nullptr)) {}

        __intrusive_ptr(const __intrusive_ptr&& __that) noexcept
          : __data_(__that.__data_) {
          __addref_();
        }

        __intrusive_ptr& operator=(__intrusive_ptr&& __that) noexcept {
          __release_();
          __data_ = std::exchange(__that.__data_, nullptr);
        }

        __intrusive_ptr& operator=(const __intrusive_ptr&& __that) noexcept {
          __release_();
          __data_ = __that.__data_;
          __addref_();
        }

        ~__intrusive_ptr() {
          __release_();
        }

        void reset() noexcept {
          __release_();
          __data_ = nullptr;
        }

        _Ty* get() const noexcept {
          return &__data_->__value();
        }

        _Ty* operator->() const noexcept {
          return &__data_->__value();
        }

        _Ty& operator*() const noexcept {
          return __data_->__value();
        }
      };

    template <class _Ty>
      struct __enable_intrusive_from_this {
        __intrusive_ptr<_Ty> __intrusive_from_this() noexcept {
          static_assert(0 == offsetof(__control_block<_Ty>, __value_));
          _Ty* __this = static_cast<_Ty*>(this);
          __intrusive_ptr<_Ty> __p{(__control_block<_Ty>*) __this};
          __p.__addref_();
          return __p;
        }

        __intrusive_ptr<const _Ty> __intrusive_from_this() const noexcept {
          static_assert(0 == offsetof(__control_block<_Ty>, __value_));
          const _Ty* __this = static_cast<const _Ty*>(this);
          __intrusive_ptr<const _Ty> __p{(__control_block<_Ty>*) __this};
          __p.__addref_();
          return __p;
        }
      };

    template <class _Ty>
      struct __make_intrusive_ptr_t {
        template <class... _Us>
            requires constructible_from<_Ty, _Us...>
          __intrusive_ptr<_Ty> operator()(_Us&&... __us) const {
            using _UncvTy = std::remove_cv_t<_Ty>;
            return __intrusive_ptr<_Ty>{::new __control_block<_UncvTy>{(_Us&&) __us...}};
          }
      };
  } // namespace __ptr

  using __ptr::__intrusive_ptr;
  using __ptr::__enable_intrusive_from_this;
  template <class _Ty>
    inline constexpr __ptr::__make_intrusive_ptr_t<_Ty> __make_intrusive_ptr {};

} // namespace _P2300
