/*
 * Copyright (c) Facebook
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

#include <version>
#include <cassert>
#include <cstdint>
#include <utility>
#include <type_traits>
#include <atomic>
#include <thread>

#if __has_include(<stop_token>) && __cpp_lib_jthread >= 201911
#include <stop_token>
#endif

namespace std {
  // [stoptoken.inplace], class in_place_stop_token
  class in_place_stop_token;

  // [stopsource.inplace], class in_place_stop_source
  class in_place_stop_source;

  // [stopcallback.inplace], class template in_place_stop_callback
  template <class Callback>
    class in_place_stop_callback;

  namespace detail {
    struct in_place_stop_callback_base {
      void execute() noexcept {
        this->execute_(this);
      }

    protected:
      using execute_fn = void(in_place_stop_callback_base*) noexcept;
      explicit in_place_stop_callback_base(in_place_stop_source* source, execute_fn* execute) noexcept
        : source_(source), execute_(execute) {}

      void register_callback() noexcept;

      friend in_place_stop_source;

      in_place_stop_source* source_;
      execute_fn* execute_;
      in_place_stop_callback_base* next_ = nullptr;
      in_place_stop_callback_base** prev_ptr_ = nullptr;
      bool* removed_during_callback_ = nullptr;
      std::atomic<bool> callback_completed_{false};
    };

    struct spin_wait {
      spin_wait() noexcept = default;

      void wait() noexcept {
        if (count_++ < yield_threshold) {
          // TODO: _mm_pause();
        } else {
          if (count_ == 0)
            count_ = yield_threshold;
          std::this_thread::yield();
        }
      }

    private:
      static constexpr uint32_t yield_threshold = 20;
      uint32_t count_ = 0;
    };

    template<template <class> class>
      struct __check_type_alias_exists;
  }

  // [stoptoken.never], class never_stop_token
  struct never_stop_token {
    template <class F>
      struct callback_type {
        explicit callback_type(never_stop_token, F&&) noexcept
        {}
      };
    static constexpr bool stop_requested() noexcept {
      return false;
    }
    static constexpr bool stop_possible() noexcept {
      return false;
    }
  };

  template <class Callback>
    class in_place_stop_callback;

  // [stopsource.inplace], class in_place_stop_source
  class in_place_stop_source {
  public:
    in_place_stop_source() noexcept = default;
    ~in_place_stop_source();
    in_place_stop_source(in_place_stop_source&&) = delete;

    in_place_stop_token get_token() noexcept;

    bool request_stop() noexcept;
    bool stop_requested() const noexcept {
      return (state_.load(memory_order_acquire) & stop_requested_flag) != 0;
    }

  private:
    friend in_place_stop_token;
    friend detail::in_place_stop_callback_base;
    template <class F>
      friend class in_place_stop_callback;

    uint8_t lock() noexcept;
    void unlock(uint8_t) noexcept;

    bool try_lock_unless_stop_requested(bool) noexcept;

    bool try_add_callback(detail::in_place_stop_callback_base*) noexcept;

    void remove_callback(detail::in_place_stop_callback_base*) noexcept;

    static constexpr uint8_t stop_requested_flag = 1;
    static constexpr uint8_t locked_flag = 2;

    std::atomic<uint8_t> state_{0};
    detail::in_place_stop_callback_base* callbacks_ = nullptr;
    std::thread::id notifying_thread_;
  };

  // [stoptoken.inplace], class in_place_stop_token
  class in_place_stop_token {
  public:
    template <class F>
      using callback_type = in_place_stop_callback<F>;

    in_place_stop_token() noexcept : source_(nullptr) {}

    in_place_stop_token(const in_place_stop_token& other) noexcept = default;

    in_place_stop_token(in_place_stop_token&& other) noexcept
      : source_(std::exchange(other.source_, {})) {}

    in_place_stop_token& operator=(const in_place_stop_token& other) noexcept = default;

    in_place_stop_token& operator=(in_place_stop_token&& other) noexcept {
      source_ = std::exchange(other.source_, nullptr);
      return *this;
    }

    bool stop_requested() const noexcept {
      return source_ != nullptr && source_->stop_requested();
    }

    bool stop_possible() const noexcept {
      return source_ != nullptr;
    }

    void swap(in_place_stop_token& other) noexcept {
      std::swap(source_, other.source_);
    }

    bool operator==(const in_place_stop_token&) const noexcept = default;

  private:
    friend in_place_stop_source;
    template <class F>
      friend class in_place_stop_callback;

    explicit in_place_stop_token(in_place_stop_source* source) noexcept
      : source_(source) {}

    in_place_stop_source* source_;
  };

  inline in_place_stop_token in_place_stop_source::get_token() noexcept {
    return in_place_stop_token{this};
  }

  // [stopcallback.inplace], class template in_place_stop_callback
  template <class F>
    class in_place_stop_callback : detail::in_place_stop_callback_base {
    public:
      template <class T>
        requires constructible_from<F, T>
      explicit in_place_stop_callback(in_place_stop_token token, T&& func)
          noexcept(std::is_nothrow_constructible_v<F, T>)
        : detail::in_place_stop_callback_base(token.source_, &in_place_stop_callback::execute_impl)
        , func_((T&&) func) {
        register_callback();
      }

      ~in_place_stop_callback() {
        if (source_ != nullptr)
          source_->remove_callback(this);
      }

    private:
      static void execute_impl(detail::in_place_stop_callback_base* cb) noexcept {
        std::move(static_cast<in_place_stop_callback*>(cb)->func_)();
      }

      [[no_unique_address]] F func_;
    };

  namespace detail {
    inline void in_place_stop_callback_base::register_callback() noexcept {
      if (source_ != nullptr) {
        if (!source_->try_add_callback(this)) {
          source_ = nullptr;
          // Callback not registered because stop_requested() was true.
          // Execute inline here.
          execute();
        }
      }
    }
  }

  inline in_place_stop_source::~in_place_stop_source() {
    assert((state_.load(memory_order_relaxed) & locked_flag) == 0);
    assert(callbacks_ == nullptr);
  }

  inline bool in_place_stop_source::request_stop() noexcept {
    if (!try_lock_unless_stop_requested(true))
      return true;

    notifying_thread_ = this_thread::get_id();

    // We are responsible for executing callbacks.
    while (callbacks_ != nullptr) {
      auto* callback = callbacks_;
      callback->prev_ptr_ = nullptr;
      callbacks_ = callback->next_;
      if (callbacks_ != nullptr)
        callbacks_->prev_ptr_ = &callbacks_;

      state_.store(stop_requested_flag, memory_order_release);

      bool removed_during_callback = false;
      callback->removed_during_callback_ = &removed_during_callback;

      callback->execute();

      if (!removed_during_callback) {
        callback->removed_during_callback_ = nullptr;
        callback->callback_completed_.store(true, memory_order_release);
      }

      lock();
    }

    state_.store(stop_requested_flag, memory_order_release);
    return false;
  }

  uint8_t in_place_stop_source::lock() noexcept {
    detail::spin_wait spin;
    auto old_state = state_.load(memory_order_relaxed);
    do {
      while ((old_state & locked_flag) != 0) {
        spin.wait();
        old_state = state_.load(memory_order_relaxed);
      }
    } while (!state_.compare_exchange_weak(
        old_state,
        old_state | locked_flag,
        memory_order_acquire,
        memory_order_relaxed));

    return old_state;
  }

  void in_place_stop_source::unlock(uint8_t old_state) noexcept {
    (void)state_.store(old_state, memory_order_release);
  }

  bool in_place_stop_source::try_lock_unless_stop_requested(
      bool set_stop_requested) noexcept {
    detail::spin_wait spin;
    auto old_state = state_.load(memory_order_relaxed);
    do {
      while (true) {
        if ((old_state & stop_requested_flag) != 0) {
          // Stop already requested.
          return false;
        } else if (old_state == 0) {
          break;
        } else {
          spin.wait();
          old_state = state_.load(memory_order_relaxed);
        }
      }
    } while (!state_.compare_exchange_weak(
        old_state,
        set_stop_requested ? (locked_flag | stop_requested_flag) : locked_flag,
        memory_order_acq_rel,
        memory_order_relaxed));

    // Lock acquired successfully
    return true;
  }

  bool in_place_stop_source::try_add_callback(
      detail::in_place_stop_callback_base* callback) noexcept {
    if (!try_lock_unless_stop_requested(false)) {
      return false;
    }

    callback->next_ = callbacks_;
    callback->prev_ptr_ = &callbacks_;
    if (callbacks_ != nullptr) {
      callbacks_->prev_ptr_ = &callback->next_;
    }
    callbacks_ = callback;

    unlock(0);

    return true;
  }

  void in_place_stop_source::remove_callback(
      detail::in_place_stop_callback_base* callback) noexcept {
    auto old_state = lock();

    if (callback->prev_ptr_ != nullptr) {
      // Callback has not been executed yet.
      // Remove from the list.
      *callback->prev_ptr_ = callback->next_;
      if (callback->next_ != nullptr) {
        callback->next_->prev_ptr_ = callback->prev_ptr_;
      }
      unlock(old_state);
    } else {
      auto notifying_thread = notifying_thread_;
      unlock(old_state);

      // Callback has either already been executed or is
      // currently executing on another thread.
      if (std::this_thread::get_id() == notifying_thread) {
        if (callback->removed_during_callback_ != nullptr) {
          *callback->removed_during_callback_ = true;
        }
      } else {
        // Concurrently executing on another thread.
        // Wait until the other thread finishes executing the callback.
        detail::spin_wait spin;
        while (!callback->callback_completed_.load(memory_order_acquire)) {
          spin.wait();
        }
      }
    }
  }

  template <class T>
    concept stoppable_token =
      copy_constructible<T> &&
      move_constructible<T> &&
      is_nothrow_copy_constructible_v<T> &&
      is_nothrow_move_constructible_v<T> &&
      equality_comparable<T> &&
      requires (const T& token) {
        { token.stop_requested() } noexcept -> __boolean_testable;
        { token.stop_possible() } noexcept -> __boolean_testable;
        typename detail::__check_type_alias_exists<T::template callback_type>;
      };

  template <class T, typename CB, typename Initializer = CB>
    concept stoppable_token_for =
      stoppable_token<T> &&
      invocable<CB> &&
      requires {
        typename T::template callback_type<CB>;
      } &&
      constructible_from<CB, Initializer> &&
      constructible_from<typename T::template callback_type<CB>, T, Initializer> &&
      constructible_from<typename T::template callback_type<CB>, T&, Initializer> &&
      constructible_from<typename T::template callback_type<CB>, const T, Initializer> &&
      constructible_from<typename T::template callback_type<CB>, const T&, Initializer>;

  template <class T>
    concept unstoppable_token =
      stoppable_token<T> &&
      requires {
        { T::stop_possible() } -> __boolean_testable;
      } &&
      (!T::stop_possible());
}
