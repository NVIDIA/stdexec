/*
 * Copyright (c) 2021-2022 Facebook, Inc. and its affiliates
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

#include "__detail/__stop_token.hpp" // IWYU pragma: export
#include "__detail/__utility.hpp"

#include <version>
#include <cstdint>
#include <utility>
#include <atomic>
#include <thread>

#if __has_include(<stop_token>) && __cpp_lib_jthread >= 2019'11L
#  include <stop_token> // IWYU pragma: export
#endif

namespace stdexec {
  namespace __stok {
    struct __inplace_stop_callback_base {
      void __execute() noexcept {
        this->__execute_(this);
      }

     protected:
      using __execute_fn_t = void(__inplace_stop_callback_base*) noexcept;

      explicit __inplace_stop_callback_base(
        const inplace_stop_source* __source,
        __execute_fn_t* __execute) noexcept
        : __source_(__source)
        , __execute_(__execute) {
      }

      void __register_callback_() noexcept;

      friend inplace_stop_source;

      const inplace_stop_source* __source_;
      __execute_fn_t* __execute_;
      __inplace_stop_callback_base* __next_ = nullptr;
      __inplace_stop_callback_base** __prev_ptr_ = nullptr;
      bool* __removed_during_callback_ = nullptr;
      std::atomic<bool> __callback_completed_{false};
    };

    struct __spin_wait {
      __spin_wait() noexcept = default;

      void __wait() noexcept {
        if (__count_++ < __yield_threshold_) {
          // TODO: _mm_pause();
        } else {
          if (__count_ == 0)
            __count_ = __yield_threshold_;
          std::this_thread::yield();
        }
      }

     private:
      static constexpr uint32_t __yield_threshold_ = 20;
      uint32_t __count_ = 0;
    };
  } // namespace __stok

  // [stoptoken.never], class never_stop_token
  struct never_stop_token {
   private:
    struct __callback_type {
      explicit __callback_type(never_stop_token, __ignore) noexcept {
      }
    };
   public:
    template <class>
    using callback_type = __callback_type;

    static constexpr auto stop_requested() noexcept -> bool {
      return false;
    }

    static constexpr auto stop_possible() noexcept -> bool {
      return false;
    }

    auto operator==(const never_stop_token&) const noexcept -> bool = default;
  };

  template <class _Callback>
  class inplace_stop_callback;

  // [stopsource.inplace], class inplace_stop_source
  class inplace_stop_source {
   public:
    inplace_stop_source() noexcept = default;
    ~inplace_stop_source();
    inplace_stop_source(inplace_stop_source&&) = delete;

    auto get_token() const noexcept -> inplace_stop_token;

    auto request_stop() noexcept -> bool;

    auto stop_requested() const noexcept -> bool {
      return (__state_.load(std::memory_order_acquire) & __stop_requested_flag_) != 0;
    }

   private:
    friend inplace_stop_token;
    friend __stok::__inplace_stop_callback_base;
    template <class>
    friend class inplace_stop_callback;

    auto __lock_() const noexcept -> uint8_t;
    void __unlock_(uint8_t) const noexcept;

    auto __try_lock_unless_stop_requested_(bool) const noexcept -> bool;

    auto __try_add_callback_(__stok::__inplace_stop_callback_base*) const noexcept -> bool;

    void __remove_callback_(__stok::__inplace_stop_callback_base*) const noexcept;

    static constexpr uint8_t __stop_requested_flag_ = 1;
    static constexpr uint8_t __locked_flag_ = 2;

    mutable std::atomic<uint8_t> __state_{0};
    mutable __stok::__inplace_stop_callback_base* __callbacks_ = nullptr;
    std::thread::id __notifying_thread_;
  };

  // [stoptoken.inplace], class inplace_stop_token
  class inplace_stop_token {
   public:
    template <class _Fun>
    using callback_type = inplace_stop_callback<_Fun>;

    inplace_stop_token() noexcept
      : __source_(nullptr) {
    }

    inplace_stop_token(const inplace_stop_token& __other) noexcept = default;

    inplace_stop_token(inplace_stop_token&& __other) noexcept
      : __source_(std::exchange(__other.__source_, {})) {
    }

    auto operator=(const inplace_stop_token& __other) noexcept -> inplace_stop_token& = default;

    auto operator=(inplace_stop_token&& __other) noexcept -> inplace_stop_token& {
      __source_ = std::exchange(__other.__source_, nullptr);
      return *this;
    }

    [[nodiscard]]
    auto stop_requested() const noexcept -> bool {
      return __source_ != nullptr && __source_->stop_requested();
    }

    [[nodiscard]]
    auto stop_possible() const noexcept -> bool {
      return __source_ != nullptr;
    }

    void swap(inplace_stop_token& __other) noexcept {
      std::swap(__source_, __other.__source_);
    }

    auto operator==(const inplace_stop_token&) const noexcept -> bool = default;

   private:
    friend inplace_stop_source;
    template <class>
    friend class inplace_stop_callback;

    explicit inplace_stop_token(const inplace_stop_source* __source) noexcept
      : __source_(__source) {
    }

    const inplace_stop_source* __source_;
  };

  inline auto inplace_stop_source::get_token() const noexcept -> inplace_stop_token {
    return inplace_stop_token{this};
  }

  // [stopcallback.inplace], class template inplace_stop_callback
  template <class _Fun>
  class inplace_stop_callback : __stok::__inplace_stop_callback_base {
   public:
    template <class _Fun2>
      requires constructible_from<_Fun, _Fun2>
    explicit inplace_stop_callback(inplace_stop_token __token, _Fun2&& __fun)
      noexcept(__nothrow_constructible_from<_Fun, _Fun2>)
      : __stok::__inplace_stop_callback_base(
          __token.__source_,
          &inplace_stop_callback::__execute_impl_)
      , __fun_(static_cast<_Fun2&&>(__fun)) {
      __register_callback_();
    }

    ~inplace_stop_callback() {
      if (__source_ != nullptr)
        __source_->__remove_callback_(this);
    }

   private:
    static void __execute_impl_(__stok::__inplace_stop_callback_base* cb) noexcept {
      std::move(static_cast<inplace_stop_callback*>(cb)->__fun_)();
    }

    STDEXEC_ATTRIBUTE(no_unique_address) _Fun __fun_;
  };

  namespace __stok {
    inline void __inplace_stop_callback_base::__register_callback_() noexcept {
      if (__source_ != nullptr) {
        if (!__source_->__try_add_callback_(this)) {
          __source_ = nullptr;
          // Callback not registered because stop_requested() was true.
          // Execute inline here.
          __execute();
        }
      }
    }
  } // namespace __stok

  inline inplace_stop_source::~inplace_stop_source() {
    STDEXEC_ASSERT((__state_.load(std::memory_order_relaxed) & __locked_flag_) == 0);
    STDEXEC_ASSERT(__callbacks_ == nullptr);
  }

  inline auto inplace_stop_source::request_stop() noexcept -> bool {
    if (!__try_lock_unless_stop_requested_(true))
      return true;

    __notifying_thread_ = std::this_thread::get_id();

    // We are responsible for executing callbacks.
    while (__callbacks_ != nullptr) {
      auto* __callbk = __callbacks_;
      __callbk->__prev_ptr_ = nullptr;
      __callbacks_ = __callbk->__next_;
      if (__callbacks_ != nullptr)
        __callbacks_->__prev_ptr_ = &__callbacks_;

      __state_.store(__stop_requested_flag_, std::memory_order_release);

      bool __removed_during_callback = false;
      __callbk->__removed_during_callback_ = &__removed_during_callback;

      __callbk->__execute();

      if (!__removed_during_callback) {
        __callbk->__removed_during_callback_ = nullptr;
        __callbk->__callback_completed_.store(true, std::memory_order_release);
      }

      __lock_();
    }

    __state_.store(__stop_requested_flag_, std::memory_order_release);
    return false;
  }

  inline auto inplace_stop_source::__lock_() const noexcept -> uint8_t {
    __stok::__spin_wait __spin;
    auto __old_state = __state_.load(std::memory_order_relaxed);
    do {
      while ((__old_state & __locked_flag_) != 0) {
        __spin.__wait();
        __old_state = __state_.load(std::memory_order_relaxed);
      }
    } while (!__state_.compare_exchange_weak(
      __old_state,
      __old_state | __locked_flag_,
      std::memory_order_acquire,
      std::memory_order_relaxed));

    return __old_state;
  }

  inline void inplace_stop_source::__unlock_(uint8_t __old_state) const noexcept {
    (void) __state_.store(__old_state, std::memory_order_release);
  }

  inline auto
    inplace_stop_source::__try_lock_unless_stop_requested_(bool __set_stop_requested) const noexcept
    -> bool {
    __stok::__spin_wait __spin;
    auto __old_state = __state_.load(std::memory_order_relaxed);
    do {
      while (true) {
        if ((__old_state & __stop_requested_flag_) != 0) {
          // Stop already requested.
          return false;
        } else if (__old_state == 0) {
          break;
        } else {
          __spin.__wait();
          __old_state = __state_.load(std::memory_order_relaxed);
        }
      }
    } while (!__state_.compare_exchange_weak(
      __old_state,
      __set_stop_requested ? (__locked_flag_ | __stop_requested_flag_) : __locked_flag_,
      std::memory_order_acq_rel,
      std::memory_order_relaxed));

    // Lock acquired successfully
    return true;
  }

  inline auto inplace_stop_source::__try_add_callback_(
    __stok::__inplace_stop_callback_base* __callbk) const noexcept -> bool {
    if (!__try_lock_unless_stop_requested_(false)) {
      return false;
    }

    __callbk->__next_ = __callbacks_;
    __callbk->__prev_ptr_ = &__callbacks_;
    if (__callbacks_ != nullptr) {
      __callbacks_->__prev_ptr_ = &__callbk->__next_;
    }
    __callbacks_ = __callbk;

    __unlock_(0);

    return true;
  }

  inline void inplace_stop_source::__remove_callback_(
    __stok::__inplace_stop_callback_base* __callbk) const noexcept {
    auto __old_state = __lock_();

    if (__callbk->__prev_ptr_ != nullptr) {
      // Callback has not been executed yet.
      // Remove from the list.
      *__callbk->__prev_ptr_ = __callbk->__next_;
      if (__callbk->__next_ != nullptr) {
        __callbk->__next_->__prev_ptr_ = __callbk->__prev_ptr_;
      }
      __unlock_(__old_state);
    } else {
      auto __notifying_thread = __notifying_thread_;
      __unlock_(__old_state);

      // Callback has either already been executed or is
      // currently executing on another thread.
      if (std::this_thread::get_id() == __notifying_thread) {
        if (__callbk->__removed_during_callback_ != nullptr) {
          *__callbk->__removed_during_callback_ = true;
        }
      } else {
        // Concurrently executing on another thread.
        // Wait until the other thread finishes executing the callback.
        __stok::__spin_wait __spin;
        while (!__callbk->__callback_completed_.load(std::memory_order_acquire)) {
          __spin.__wait();
        }
      }
    }
  }

  using in_place_stop_token
    [[deprecated("in_place_stop_token has been renamed inplace_stop_token")]] = inplace_stop_token;

  using in_place_stop_source [[deprecated(
    "in_place_stop_token has been renamed inplace_stop_source")]] = inplace_stop_source;

  template <class _Fun>
  using in_place_stop_callback
    [[deprecated("in_place_stop_callback has been renamed inplace_stop_callback")]] =
      inplace_stop_callback<_Fun>;
} // namespace stdexec
