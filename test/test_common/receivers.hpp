/*
 * Copyright (c) 2022 Lucian Radu Teodorescu
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

#include <catch2/catch.hpp>
#include <test_common/type_helpers.hpp>
#include <stdexec/execution.hpp>

#include <atomic>
#include <exception>
#include <tuple>
#include <typeinfo>

namespace ex = stdexec;

namespace {

  namespace empty_recv {
    struct recv0 {
      using receiver_concept = stdexec::receiver_t;

      void set_value() noexcept {
      }

      void set_stopped() noexcept {
      }

      void set_error(std::exception_ptr) noexcept {
      }
    };

    struct recv_int {
      using receiver_concept = stdexec::receiver_t;

      void set_value(int) noexcept {
      }

      void set_stopped() noexcept {
      }

      void set_error(std::exception_ptr) noexcept {
      }
    };

    struct recv0_ec {
      using receiver_concept = stdexec::receiver_t;

      void set_value() noexcept {
      }

      void set_stopped() noexcept {
      }

      void set_error(std::error_code) noexcept {
      }

      void set_error(std::exception_ptr) noexcept {
      }
    };

    struct recv_int_ec {
      using receiver_concept = stdexec::receiver_t;

      void set_value(int) noexcept {
      }

      void set_stopped() noexcept {
      }

      void set_error(std::error_code) noexcept {
      }

      void set_error(std::exception_ptr) noexcept {
      }
    };

  } // namespace empty_recv

  template <class _Env = empty_env>
  class base_expect_receiver {
    std::atomic<bool> called_{false};
    _Env env_{};

   public:
    using receiver_concept = stdexec::receiver_t;
    base_expect_receiver() = default;

    ~base_expect_receiver() {
      CHECK(called_.load());
    }

    explicit base_expect_receiver(_Env env)
      : env_(std::move(env)) {
    }

    base_expect_receiver(base_expect_receiver&& other)
      : called_(other.called_.exchange(true))
      , env_(std::move(other.env_)) {
    }

    base_expect_receiver& operator=(base_expect_receiver&& other) = delete;

    void set_called() {
      called_.store(true);
    }

    _Env get_env() const noexcept {
      return env_;
    }
  };

  template <class _Env = empty_env>
  struct expect_void_receiver : base_expect_receiver<_Env> {
    expect_void_receiver() = default;

    explicit expect_void_receiver(_Env env)
      : base_expect_receiver<_Env>(std::move(env)) {
    }

    void set_value() noexcept {
      this->set_called();
    }

    template <typename... Ts>
    void set_value(Ts...) noexcept {
      FAIL_CHECK("set_value called on expect_void_receiver with some non-void value");
    }

    void set_stopped() noexcept {
      FAIL_CHECK("set_stopped called on expect_void_receiver");
    }

    void set_error(std::exception_ptr) noexcept {
      FAIL_CHECK("set_error called on expect_void_receiver");
    }
  };

  struct expect_void_receiver_ex {
    using receiver_concept = stdexec::receiver_t;

    expect_void_receiver_ex(bool& executed)
      : executed_(&executed) {
    }

    template <class... Ty>
    void set_value(const Ty&...) noexcept {
      *executed_ = true;
    }

    void set_stopped() noexcept {
      FAIL_CHECK("set_stopped called on expect_void_receiver_ex");
    }

    void set_error(std::exception_ptr) noexcept {
      FAIL_CHECK("set_error called on expect_void_receiver_ex");
    }

   private:
    bool* executed_;
  };

  struct env_tag { };

  template <class Env = empty_env, typename... Ts>
  struct expect_value_receiver : base_expect_receiver<Env> {
    explicit(sizeof...(Ts) != 1) expect_value_receiver(Ts... vals)
      : values_(std::move(vals)...) {
    }

    expect_value_receiver(env_tag, Env env, Ts... vals)
      : base_expect_receiver<Env>(std::move(env))
      , values_(std::move(vals)...) {
    }

    void set_value(const Ts&... vals) noexcept {
      CHECK(values_ == std::tie(vals...));
      this->set_called();
    }

    template <typename... Us>
    void set_value(const Us&...) noexcept {
      FAIL_CHECK("set_value called with wrong value types on expect_value_receiver");
    }

    void set_stopped() noexcept {
      FAIL_CHECK("set_stopped called on expect_value_receiver");
    }

    template <typename E>
    void set_error(E) noexcept {
      FAIL_CHECK("set_error called on expect_value_receiver");
    }

   private:
    std::tuple<Ts...> values_;
  };

  template <typename T, class Env = empty_env>
  class expect_value_receiver_ex {
    T* dest_;
    Env env_{};

   public:
    using receiver_concept = stdexec::receiver_t;

    explicit expect_value_receiver_ex(T& dest)
      : dest_(&dest) {
    }

    expect_value_receiver_ex(Env env, T& dest)
      : dest_(&dest)
      , env_(std::move(env)) {
    }

    void set_value(T val) noexcept {
      *dest_ = val;
    }

    template <typename... Ts>
    void set_value(Ts...) noexcept {
      FAIL_CHECK("set_value called with wrong value types on expect_value_receiver_ex");
    }

    void set_stopped() noexcept {
      FAIL_CHECK("set_stopped called on expect_value_receiver_ex");
    }

    void set_error(std::exception_ptr) noexcept {
      FAIL_CHECK("set_error called on expect_value_receiver_ex");
    }

    Env get_env() const noexcept {
      return env_;
    }
  };

  template <class Env = empty_env>
  struct expect_stopped_receiver : base_expect_receiver<Env> {
    expect_stopped_receiver() = default;

    expect_stopped_receiver(Env env)
      : base_expect_receiver<Env>(std::move(env)) {
    }

    template <typename... Ts>
    void set_value(Ts...) noexcept {
      FAIL_CHECK("set_value called on expect_stopped_receiver");
    }

    void set_stopped() noexcept {
      this->set_called();
    }

    void set_error(std::exception_ptr) noexcept {
      FAIL_CHECK("set_error called on expect_stopped_receiver");
    }
  };

  template <class Env = empty_env>
  struct expect_stopped_receiver_ex {
    using receiver_concept = stdexec::receiver_t;

    explicit expect_stopped_receiver_ex(bool& executed)
      : executed_(&executed) {
    }

    expect_stopped_receiver_ex(Env env, bool& executed)
      : executed_(&executed)
      , env_(std::move(env)) {
    }

    template <typename... Ts>
    void set_value(Ts...) noexcept {
      FAIL_CHECK("set_value called on expect_stopped_receiver_ex");
    }

    void set_stopped() noexcept {
      *executed_ = true;
    }

    void set_error(std::exception_ptr) //
      noexcept {
      FAIL_CHECK("set_error called on expect_stopped_receiver_ex");
    }

    Env get_env() const noexcept {
      return env_;
    }

   private:
    bool* executed_;
    Env env_;
  };

  inline std::pair<const std::type_info&, std::string> to_comparable(std::exception_ptr eptr) {
    try {
      std::rethrow_exception(eptr);
    } catch (const std::exception& e) {
      return {typeid(e), e.what()};
    } catch (...) {
      return {typeid(void), "<unknown>"};
    }
  }

  template <typename T>
  inline T to_comparable(T value) {
    return value;
  }

  template <class T = std::exception_ptr, class Env = empty_env>
  struct expect_error_receiver : base_expect_receiver<Env> {
    expect_error_receiver() = default;

    explicit expect_error_receiver(T error)
      : error_(std::move(error)) {
    }

    expect_error_receiver(Env env, T error)
      : base_expect_receiver<Env>{std::move(env)}
      , error_(std::move(error)) {
    }

    // these do not move error_ and cannot be defaulted
    expect_error_receiver(expect_error_receiver&& other)
      : base_expect_receiver<Env>(std::move(other))
      , error_() {
    }

    expect_error_receiver& operator=(expect_error_receiver&& other) noexcept {
      base_expect_receiver<Env>::operator=(std::move(other));
      error_.reset();
      return *this;
    }

    template <typename... Ts>
    void set_value(Ts...) noexcept {
      FAIL_CHECK("set_value called on expect_error_receiver");
    }

    void set_stopped() noexcept {
      FAIL_CHECK("set_stopped called on expect_error_receiver");
    }

    void set_error(T err) noexcept {
      this->set_called();
      if (error_) {
        CHECK(to_comparable(err) == to_comparable(*error_));
      }
    }

    template <typename E>
    void set_error(E) noexcept {
      FAIL_CHECK("set_error called on expect_error_receiver with wrong error type");
    }

   private:
    std::optional<T> error_;
  };

  template <class T, class Env = empty_env>
  struct expect_error_receiver_ex {
    using receiver_concept = stdexec::receiver_t;

    explicit expect_error_receiver_ex(T& value)
      : value_(&value) {
    }

    expect_error_receiver_ex(Env env, T& value)
      : value_(&value)
      , env_(std::move(env)) {
    }

    template <typename... Ts>
    void set_value(Ts...) noexcept {
      FAIL_CHECK("set_value called on expect_error_receiver_ex");
    }

    void set_stopped() noexcept {
      FAIL_CHECK("set_stopped called on expect_error_receiver_ex");
    }

    template <class Err>
    void set_error(Err) noexcept {
      FAIL_CHECK("set_error called on expect_error_receiver_ex with the wrong error type");
    }

    void set_error(T value) noexcept {
      *value_ = std::move(value);
    }

    Env get_env() const noexcept {
      return env_;
    }

   private:
    T* value_;
    Env env_{};
  };

  struct logging_receiver {
    using receiver_concept = stdexec::receiver_t;

    logging_receiver(int& state)
      : state_(&state) {
    }

    template <typename... Args>
    void set_value(Args...) noexcept {
      *state_ = 0;
    }

    void set_stopped() noexcept {
      *state_ = 1;
    }

    template <typename E>
    void set_error(E) noexcept {
      *state_ = 2;
    }

   private:
    int* state_;
  };

  enum class typecat {
    undefined,
    value,
    ref,
    cref,
    rvalref,
  };

  template <typename T>
  struct typecat_receiver {
    using receiver_concept = stdexec::receiver_t;
    T* value_;
    typecat* cat_;

    // void set_value(T v) noexcept {
    //     *value_ = v;
    //     *cat_ = typecat::value;
    // }
    void set_value(T& v) noexcept {
      *value_ = v;
      *cat_ = typecat::ref;
    }

    void set_value(const T& v) noexcept {
      *value_ = v;
      *cat_ = typecat::cref;
    }

    void set_value(T&& v) noexcept {
      *value_ = v;
      *cat_ = typecat::rvalref;
    }

    void set_stopped() noexcept {
      FAIL_CHECK("set_stopped called");
    }

    void set_error(std::exception_ptr) noexcept {
      FAIL_CHECK("set_error called");
    }
  };

  template <typename F>
  struct fun_receiver {
    using receiver_concept = stdexec::receiver_t;
    F f_;

    template <typename... Ts>
    void set_value(Ts... vals) noexcept {
      try {
        std::move(f_)(static_cast<Ts&&>(vals)...);
      } catch (...) {
        ex::set_error(std::move(*this), std::current_exception());
      }
    }

    void set_stopped() noexcept {
      FAIL("Done called");
    }

    void set_error(std::exception_ptr eptr) noexcept {
      try {
        if (eptr)
          std::rethrow_exception(eptr);
        FAIL("Empty exception thrown");
      } catch (const std::exception& e) {
        FAIL("Exception thrown: " << e.what());
      }
    }
  };

  template <typename F>
  fun_receiver<F> make_fun_receiver(F f) {
    return fun_receiver<F>{std::forward<F>(f)};
  }

  template <ex::sender S, typename... Ts>
  inline void wait_for_value(S&& snd, Ts&&... val) {
    // Ensure that the given sender type has only one variant for set_value calls
    // If not, sync_wait will not work
    static_assert(
      stdexec::__single_value_variant_sender<S, ex::__sync_wait::__env>,
      "Sender passed to sync_wait needs to have one variant for sending set_value");

    std::optional<std::tuple<Ts...>> res = stdexec::sync_wait(static_cast<S&&>(snd));
    CHECK(res.has_value());
    std::tuple<Ts...> expected(static_cast<Ts&&>(val)...);
    if constexpr (std::tuple_size_v<std::tuple<Ts...>> == 1)
      CHECK(std::get<0>(res.value()) == std::get<0>(expected));
    else
      CHECK(res.value() == expected);
  }
} // namespace
