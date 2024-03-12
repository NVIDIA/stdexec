/*
 * Copyright (c) 2021-2022 Facebook, Inc. and its affiliates
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#include "concepts.hpp"
#include "__detail/__config.hpp"

#include <version>
#include <cstdint>
#include <utility>
#include <type_traits>
#include <atomic>
#include <thread>
#include <concepts>

#if __has_include(<stop_token>) && __cpp_lib_jthread >= 201911
#  include <stop_token>
#endif


namespace stdexec {
  // [stoptoken.inplace], class in_place_stop_token
  class in_place_stop_token;

  // [stopsource.inplace], class in_place_stop_source
  class in_place_stop_source;

  // [stopcallback.inplace], class template in_place_stop_callback
  template <class _Callback>
  class in_place_stop_callback;

  namespace __stok {
    struct __in_place_stop_callback_base {
      void __execute() noexcept {
        this->__execute_(this);
      }

     protected:
      using __execute_fn_t = void(__in_place_stop_callback_base*) noexcept;

      explicit __in_place_stop_callback_base( //
        const in_place_stop_source* __source, //
        __execute_fn_t* __execute) noexcept
        : __source_(__source)
        , __execute_(__execute) {
      }

      void __register_callback_() noexcept;

      friend in_place_stop_source;

      const in_place_stop_source* __source_;
      __execute_fn_t* __execute_;
      __in_place_stop_callback_base* __next_ = nullptr;
      __in_place_stop_callback_base** __prev_ptr_ = nullptr;
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

    template <template <class> class>
    struct __check_type_alias_exists;
  } // namespace __stok

  // [stoptoken.never], class never_stop_token
  struct never_stop_token {
   private:
    struct __callback_type {
      explicit __callback_type(never_stop_token, auto&&) noexcept {
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
  class in_place_stop_callback;

  // [stopsource.inplace], class in_place_stop_source
  class in_place_stop_source {
   public:
    in_place_stop_source() noexcept = default;
    ~in_place_stop_source();
    in_place_stop_source(in_place_stop_source&&) = delete;

    auto get_token() const noexcept -> in_place_stop_token;

    auto request_stop() noexcept -> bool;

    auto stop_requested() const noexcept -> bool {
      return (__state_.load(std::memory_order_acquire) & __stop_requested_flag_) != 0;
    }

   private:
    friend in_place_stop_token;
    friend __stok::__in_place_stop_callback_base;
    template <class>
    friend class in_place_stop_callback;

    auto __lock_() const noexcept -> uint8_t;
    void __unlock_(uint8_t) const noexcept;

    auto __try_lock_unless_stop_requested_(bool) const noexcept -> bool;

    auto __try_add_callback_(__stok::__in_place_stop_callback_base*) const noexcept -> bool;

    void __remove_callback_(__stok::__in_place_stop_callback_base*) const noexcept;

    static constexpr uint8_t __stop_requested_flag_ = 1;
    static constexpr uint8_t __locked_flag_ = 2;

    mutable std::atomic<uint8_t> __state_{0};
    mutable __stok::__in_place_stop_callback_base* __callbacks_ = nullptr;
    std::thread::id __notifying_thread_;
  };

  // [stoptoken.inplace], class in_place_stop_token
  class in_place_stop_token {
   public:
    template <class _Fun>
    using callback_type = in_place_stop_callback<_Fun>;

    in_place_stop_token() noexcept
      : __source_(nullptr) {
    }

    in_place_stop_token(const in_place_stop_token& __other) noexcept = default;

    in_place_stop_token(in_place_stop_token&& __other) noexcept
      : __source_(std::exchange(__other.__source_, {})) {
    }

    auto operator=(const in_place_stop_token& __other) noexcept -> in_place_stop_token& = default;

    auto operator=(in_place_stop_token&& __other) noexcept -> in_place_stop_token& {
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

    void swap(in_place_stop_token& __other) noexcept {
      std::swap(__source_, __other.__source_);
    }

    auto operator==(const in_place_stop_token&) const noexcept -> bool = default;

   private:
    friend in_place_stop_source;
    template <class>
    friend class in_place_stop_callback;

    explicit in_place_stop_token(const in_place_stop_source* __source) noexcept
      : __source_(__source) {
    }

    const in_place_stop_source* __source_;
  };

  inline auto in_place_stop_source::get_token() const noexcept -> in_place_stop_token {
    return in_place_stop_token{this};
  }

  // [stopcallback.inplace], class template in_place_stop_callback
  template <class _Fun>
  class in_place_stop_callback : __stok::__in_place_stop_callback_base {
   public:
    template <class _Fun2>
      requires constructible_from<_Fun, _Fun2>
    explicit in_place_stop_callback(
      in_place_stop_token __token,
      _Fun2&& __fun) //
      noexcept(__nothrow_constructible_from<_Fun, _Fun2>)
      : __stok::__in_place_stop_callback_base(
        __token.__source_,
        &in_place_stop_callback::__execute_impl_)
      , __fun_(static_cast<_Fun2&&>(__fun)) {
      __register_callback_();
    }

    ~in_place_stop_callback() {
      if (__source_ != nullptr)
        __source_->__remove_callback_(this);
    }

   private:
    static void __execute_impl_(__stok::__in_place_stop_callback_base* cb) noexcept {
      std::move(static_cast<in_place_stop_callback*>(cb)->__fun_)();
    }

    STDEXEC_ATTRIBUTE((no_unique_address))
    _Fun __fun_;
  };

  namespace __stok {
    inline void __in_place_stop_callback_base::__register_callback_() noexcept {
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

  inline in_place_stop_source::~in_place_stop_source() {
    STDEXEC_ASSERT((__state_.load(std::memory_order_relaxed) & __locked_flag_) == 0);
    STDEXEC_ASSERT(__callbacks_ == nullptr);
  }

  inline auto in_place_stop_source::request_stop() noexcept -> bool {
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

  inline auto in_place_stop_source::__lock_() const noexcept -> uint8_t {
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

  inline void in_place_stop_source::__unlock_(uint8_t __old_state) const noexcept {
    (void) __state_.store(__old_state, std::memory_order_release);
  }

  inline auto in_place_stop_source::__try_lock_unless_stop_requested_(
    bool __set_stop_requested) const noexcept -> bool {
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

  inline auto in_place_stop_source::__try_add_callback_(
    __stok::__in_place_stop_callback_base* __callbk) const noexcept -> bool {
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

  inline void in_place_stop_source::__remove_callback_(
    __stok::__in_place_stop_callback_base* __callbk) const noexcept {
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

  template <class _Token>
  concept stoppable_token =
    copy_constructible<_Token> &&                   //
    move_constructible<_Token> &&                   //
    std::is_nothrow_copy_constructible_v<_Token> && //
    std::is_nothrow_move_constructible_v<_Token> && //
    equality_comparable<_Token> &&                  //
    requires(const _Token& __token) {
      { __token.stop_requested() } noexcept -> __boolean_testable_;
      { __token.stop_possible() } noexcept -> __boolean_testable_;
    // workaround ICE in appleclang 13.1
#if !defined(__clang__)
      typename __stok::__check_type_alias_exists<_Token::template callback_type>;
#endif
    };

  template <class _Token, typename _Callback, typename _Initializer = _Callback>
  concept stoppable_token_for =
    stoppable_token<_Token> && __callable<_Callback> &&                 //
    requires { typename _Token::template callback_type<_Callback>; } && //
    constructible_from<_Callback, _Initializer> &&                      //
    constructible_from<                                                 //
      typename _Token::template callback_type<_Callback>,
      _Token,
      _Initializer>
    &&                  //
    constructible_from< //
      typename _Token::template callback_type<_Callback>,
      _Token&,
      _Initializer>
    &&                  //
    constructible_from< //
      typename _Token::template callback_type<_Callback>,
      const _Token,
      _Initializer>
    && //
    constructible_from<
      typename _Token::template callback_type<_Callback>,
      const _Token&,
      _Initializer>;

  template <class _Token>
  concept unstoppable_token =  //
    stoppable_token<_Token> && //
    requires {
      { _Token::stop_possible() } -> __boolean_testable_;
    } && //
    (!_Token::stop_possible());
} // namespace stdexec
