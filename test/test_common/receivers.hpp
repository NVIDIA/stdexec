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
#include <tuple>

namespace ex = stdexec;

namespace empty_recv {

  using ex::get_env_t;
  using ex::set_error_t;
  using ex::set_stopped_t;
  using ex::set_value_t;

  struct recv0 {
    using is_receiver = void;

    STDEXEC_DEFINE_CUSTOM(void set_value)(this recv0&&, set_value_t) noexcept {
    }

    STDEXEC_DEFINE_CUSTOM(void set_stopped)(this recv0&&, set_stopped_t) noexcept {
    }

    STDEXEC_DEFINE_CUSTOM(void set_error)(this recv0&&, set_error_t, std::exception_ptr) noexcept {
    }

    STDEXEC_DEFINE_CUSTOM(empty_env get_env)(this const recv0&, get_env_t) noexcept {
      return {};
    }
  };

  struct recv_int {
    using is_receiver = void;

    STDEXEC_DEFINE_CUSTOM(void set_value)(this recv_int&&, set_value_t, int) noexcept {
    }

    STDEXEC_DEFINE_CUSTOM(void set_stopped)(this recv_int&&, set_stopped_t) noexcept {
    }

    STDEXEC_DEFINE_CUSTOM(void set_error)(
      this recv_int&&,
      set_error_t,
      std::exception_ptr) noexcept {
    }

    STDEXEC_DEFINE_CUSTOM(empty_env get_env)(this const recv_int&, get_env_t) noexcept {
      return {};
    }
  };

  struct recv0_ec {
    using is_receiver = void;

    STDEXEC_DEFINE_CUSTOM(void set_value)(this recv0_ec&&, set_value_t) noexcept {
    }

    STDEXEC_DEFINE_CUSTOM(void set_stopped)(this recv0_ec&&, set_stopped_t) noexcept {
    }

    STDEXEC_DEFINE_CUSTOM(void set_error)(this recv0_ec&&, set_error_t, std::error_code) noexcept {
    }

    STDEXEC_DEFINE_CUSTOM(void set_error)(
      this recv0_ec&&,
      set_error_t,
      std::exception_ptr) noexcept {
    }

    STDEXEC_DEFINE_CUSTOM(empty_env get_env)(this const recv0_ec&, get_env_t) noexcept {
      return {};
    }
  };

  struct recv_int_ec {
    using is_receiver = void;

    STDEXEC_DEFINE_CUSTOM(void set_value)(this recv_int_ec&&, set_value_t, int) noexcept {
    }

    STDEXEC_DEFINE_CUSTOM(void set_stopped)(this recv_int_ec&&, set_stopped_t) noexcept {
    }

    STDEXEC_DEFINE_CUSTOM(void set_error)(
      this recv_int_ec&&,
      set_error_t,
      std::error_code) noexcept {
    }

    STDEXEC_DEFINE_CUSTOM(void set_error)(
      this recv_int_ec&&,
      set_error_t,
      std::exception_ptr) noexcept {
    }

    STDEXEC_DEFINE_CUSTOM(empty_env get_env)(this const recv_int_ec&, get_env_t) noexcept {
      return {};
    }
  };

} // namespace empty_recv

template <class _Env = empty_env>
class base_expect_receiver {
  std::atomic<bool> called_{false};
  _Env env_{};

 public:
  template <ex::derived_from<base_expect_receiver> Self>
  STDEXEC_DEFINE_CUSTOM(_Env get_env)(this const Self& self, ex::get_env_t) noexcept {
    return self.env_;
  }

 public:
  using is_receiver = void;
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

  base_expect_receiver& operator=(base_expect_receiver&& other) {
    called_.store(other.called_.exchange(true));
    env_ = std::move(other.env_);
    return *this;
  }

  void set_called() {
    called_.store(true);
  }
};

template <class _Env = empty_env>
struct expect_void_receiver : base_expect_receiver<_Env> {
  expect_void_receiver() = default;

  explicit expect_void_receiver(_Env env)
    : base_expect_receiver<_Env>(std::move(env)) {
  }

  STDEXEC_DEFINE_CUSTOM(void set_value)(
    this expect_void_receiver&& self,
    ex::set_value_t) noexcept {
    self.set_called();
  }

  template <typename... Ts>
  STDEXEC_DEFINE_CUSTOM(void set_value)(
    this expect_void_receiver&&,
    ex::set_value_t,
    Ts...) noexcept {
    FAIL_CHECK("set_value called on expect_void_receiver with some non-void value");
  }

  STDEXEC_DEFINE_CUSTOM(void set_stopped)(this expect_void_receiver&&, ex::set_stopped_t) noexcept {
    FAIL_CHECK("set_stopped called on expect_void_receiver");
  }

  STDEXEC_DEFINE_CUSTOM(void set_error)(
    this expect_void_receiver&&,
    ex::set_error_t,
    std::exception_ptr) noexcept {
    FAIL_CHECK("set_error called on expect_void_receiver");
  }
};

struct expect_void_receiver_ex {
  using is_receiver = void;

  expect_void_receiver_ex(bool& executed)
    : executed_(&executed) {
  }

 private:
  STDEXEC_CPO_ACCESS(ex::set_value_t);
  STDEXEC_CPO_ACCESS(ex::set_error_t);
  STDEXEC_CPO_ACCESS(ex::set_stopped_t);
  STDEXEC_CPO_ACCESS(ex::get_env_t);

  template <class... Ty>
  STDEXEC_DEFINE_CUSTOM(void set_value)(
    this expect_void_receiver_ex&& self,
    ex::set_value_t,
    const Ty&...) noexcept {
    *self.executed_ = true;
  }

  STDEXEC_DEFINE_CUSTOM(void set_stopped)(
    this expect_void_receiver_ex&&,
    ex::set_stopped_t) noexcept {
    FAIL_CHECK("set_stopped called on expect_void_receiver_ex");
  }

  STDEXEC_DEFINE_CUSTOM(void set_error)(
    this expect_void_receiver_ex&&,
    ex::set_error_t,
    std::exception_ptr) noexcept {
    FAIL_CHECK("set_error called on expect_void_receiver_ex");
  }

  STDEXEC_DEFINE_CUSTOM(empty_env get_env)(
    this const expect_void_receiver_ex&,
    ex::get_env_t) noexcept {
    return {};
  }

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

  STDEXEC_DEFINE_CUSTOM(void set_value)(
    this expect_value_receiver&& self,
    ex::set_value_t,
    const Ts&... vals) //
    noexcept {
    CHECK(self.values_ == std::tie(vals...));
    self.set_called();
  }

  template <typename... Us>
  STDEXEC_DEFINE_CUSTOM(void set_value)(
    this expect_value_receiver&&,
    ex::set_value_t,
    const Us&...) noexcept {
    FAIL_CHECK("set_value called with wrong value types on expect_value_receiver");
  }

  STDEXEC_DEFINE_CUSTOM(void set_stopped)(
    this expect_value_receiver&& self,
    ex::set_stopped_t) noexcept {
    FAIL_CHECK("set_stopped called on expect_value_receiver");
  }

  template <typename E>
  STDEXEC_DEFINE_CUSTOM(void set_error)(this expect_value_receiver&&, ex::set_error_t, E) noexcept {
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
  using is_receiver = void;

  explicit expect_value_receiver_ex(T& dest)
    : dest_(&dest) {
  }

  expect_value_receiver_ex(Env env, T& dest)
    : dest_(&dest)
    , env_(std::move(env)) {
  }

  STDEXEC_DEFINE_CUSTOM(void set_value)(
    this expect_value_receiver_ex self,
    ex::set_value_t,
    T val) noexcept {
    *self.dest_ = val;
  }

  template <typename... Ts>
  STDEXEC_DEFINE_CUSTOM(void set_value)(
    this expect_value_receiver_ex,
    ex::set_value_t,
    Ts...) noexcept {
    FAIL_CHECK("set_value called with wrong value types on expect_value_receiver_ex");
  }

  STDEXEC_DEFINE_CUSTOM(void set_stopped)(
    this expect_value_receiver_ex,
    ex::set_stopped_t) noexcept {
    FAIL_CHECK("set_stopped called on expect_value_receiver_ex");
  }

  STDEXEC_DEFINE_CUSTOM(void set_error)(
    this expect_value_receiver_ex,
    ex::set_error_t,
    std::exception_ptr) noexcept {
    FAIL_CHECK("set_error called on expect_value_receiver_ex");
  }

  STDEXEC_DEFINE_CUSTOM(Env get_env)(
    this const expect_value_receiver_ex& self,
    ex::get_env_t) noexcept {
    return self.env_;
  }
};

template <class Env = empty_env>
struct expect_stopped_receiver : base_expect_receiver<Env> {
  expect_stopped_receiver() = default;

  expect_stopped_receiver(Env env)
    : base_expect_receiver<Env>(std::move(env)) {
  }

  template <typename... Ts>
  STDEXEC_DEFINE_CUSTOM(void set_value)(
    this expect_stopped_receiver&&,
    ex::set_value_t,
    Ts...) noexcept {
    FAIL_CHECK("set_value called on expect_stopped_receiver");
  }

  STDEXEC_DEFINE_CUSTOM(void set_error)(
    this expect_stopped_receiver&&,
    ex::set_error_t,
    std::exception_ptr) noexcept {
    FAIL_CHECK("set_error called on expect_stopped_receiver");
  }

  STDEXEC_DEFINE_CUSTOM(void set_stopped)(
    this expect_stopped_receiver&& self,
    ex::set_stopped_t) noexcept {
    self.set_called();
  }
};

template <class Env = empty_env>
struct expect_stopped_receiver_ex {
  using is_receiver = void;

  explicit expect_stopped_receiver_ex(bool& executed)
    : executed_(&executed) {
  }

  expect_stopped_receiver_ex(Env env, bool& executed)
    : executed_(&executed)
    , env_(std::move(env)) {
  }

 private:
  STDEXEC_CPO_ACCESS(ex::set_value_t);
  STDEXEC_CPO_ACCESS(ex::set_error_t);
  STDEXEC_CPO_ACCESS(ex::set_stopped_t);
  STDEXEC_CPO_ACCESS(ex::get_env_t);

  template <typename... Ts>
  STDEXEC_DEFINE_CUSTOM(void set_value)(
    this expect_stopped_receiver_ex&&,
    ex::set_value_t,
    Ts...) noexcept {
    FAIL_CHECK("set_value called on expect_stopped_receiver_ex");
  }

  STDEXEC_DEFINE_CUSTOM(void set_stopped)(
    this expect_stopped_receiver_ex&& self,
    ex::set_stopped_t) noexcept {
    *self.executed_ = true;
  }

  STDEXEC_DEFINE_CUSTOM(void set_error)(
    this expect_stopped_receiver_ex&&,
    ex::set_error_t,
    std::exception_ptr) //
    noexcept {
    FAIL_CHECK("set_error called on expect_stopped_receiver_ex");
  }

  STDEXEC_DEFINE_CUSTOM(Env get_env)(
    this const expect_stopped_receiver_ex& self,
    ex::get_env_t) noexcept {
    return self.env_;
  }

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

 private:
  STDEXEC_CPO_ACCESS(ex::set_value_t);
  STDEXEC_CPO_ACCESS(ex::set_error_t);
  STDEXEC_CPO_ACCESS(ex::set_stopped_t);

  template <typename... Ts>
  STDEXEC_DEFINE_CUSTOM(void set_value)(
    this expect_error_receiver&&,
    ex::set_value_t,
    Ts...) noexcept {
    FAIL_CHECK("set_value called on expect_error_receiver");
  }

  STDEXEC_DEFINE_CUSTOM(void set_stopped)(
    this expect_error_receiver&& self,
    ex::set_stopped_t) noexcept {
    FAIL_CHECK("set_stopped called on expect_error_receiver");
  }

  STDEXEC_DEFINE_CUSTOM(void set_error)(
    this expect_error_receiver&& self,
    ex::set_error_t,
    T err) noexcept {
    self.set_called();
    if (self.error_) {
      CHECK(to_comparable(err) == to_comparable(*self.error_));
    }
  }

  template <typename E>
  STDEXEC_DEFINE_CUSTOM(void set_error)(
    this expect_error_receiver&& self,
    ex::set_error_t,
    E) noexcept {
    FAIL_CHECK("set_error called on expect_error_receiver with wrong error type");
  }

  std::optional<T> error_;
};

template <class T, class Env = empty_env>
struct expect_error_receiver_ex {
  using is_receiver = void;

  explicit expect_error_receiver_ex(T& value)
    : value_(&value) {
  }

  expect_error_receiver_ex(Env env, T& value)
    : value_(&value)
    , env_(std::move(env)) {
  }

 private:
  STDEXEC_CPO_ACCESS(ex::set_value_t);
  STDEXEC_CPO_ACCESS(ex::set_error_t);
  STDEXEC_CPO_ACCESS(ex::set_stopped_t);
  STDEXEC_CPO_ACCESS(ex::get_env_t);

  template <typename... Ts>
  STDEXEC_DEFINE_CUSTOM(void set_value)(
    this expect_error_receiver_ex&&,
    ex::set_value_t,
    Ts...) noexcept {
    FAIL_CHECK("set_value called on expect_error_receiver_ex");
  }

  STDEXEC_DEFINE_CUSTOM(void set_stopped)(
    this expect_error_receiver_ex&&,
    ex::set_stopped_t) noexcept {
    FAIL_CHECK("set_stopped called on expect_error_receiver_ex");
  }

  template <class Err>
  STDEXEC_DEFINE_CUSTOM(void set_error)(
    this expect_error_receiver_ex&&,
    ex::set_error_t,
    Err) noexcept {
    FAIL_CHECK("set_error called on expect_error_receiver_ex with the wrong error type");
  }

  STDEXEC_DEFINE_CUSTOM(void set_error)(
    this expect_error_receiver_ex&& self,
    ex::set_error_t,
    T value) noexcept {
    *self.value_ = std::move(value);
  }

  STDEXEC_DEFINE_CUSTOM(Env get_env)(
    this const expect_error_receiver_ex& self,
    ex::get_env_t) noexcept {
    return self.env_;
  }

  T* value_;
  Env env_{};
};

struct logging_receiver {
  using is_receiver = void;

  logging_receiver(int& state)
    : state_(&state) {
  }

 private:
  STDEXEC_CPO_ACCESS(ex::set_value_t);
  STDEXEC_CPO_ACCESS(ex::set_error_t);
  STDEXEC_CPO_ACCESS(ex::set_stopped_t);
  STDEXEC_CPO_ACCESS(ex::get_env_t);

  template <typename... Args>
  STDEXEC_DEFINE_CUSTOM(void set_value)(
    this logging_receiver&& self,
    ex::set_value_t,
    Args...) noexcept {
    *self.state_ = 0;
  }

  STDEXEC_DEFINE_CUSTOM(void set_stopped)(
    this logging_receiver&& self,
    ex::set_stopped_t) noexcept {
    *self.state_ = 1;
  }

  template <typename E>
  STDEXEC_DEFINE_CUSTOM(void set_error)(this logging_receiver&& self, ex::set_error_t, E) noexcept {
    *self.state_ = 2;
  }

  STDEXEC_DEFINE_CUSTOM(empty_env get_env)(this const logging_receiver&, ex::get_env_t) noexcept {
    return {};
  }

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
  using is_receiver = void;
  T* value_;
  typecat* cat_;

  // STDEXEC_DEFINE_CUSTOM(void set_value)(this typecat_receiver self, ex::set_value_t, T v) noexcept {
  //     *self.value_ = v;
  //     *self.cat_ = typecat::value;
  // }
  STDEXEC_DEFINE_CUSTOM(void set_value)(
    this typecat_receiver self,
    ex::set_value_t,
    T& v) noexcept {
    *self.value_ = v;
    *self.cat_ = typecat::ref;
  }

  STDEXEC_DEFINE_CUSTOM(void set_value)(
    this typecat_receiver self,
    ex::set_value_t,
    const T& v) noexcept {
    *self.value_ = v;
    *self.cat_ = typecat::cref;
  }

  STDEXEC_DEFINE_CUSTOM(void set_value)(
    this typecat_receiver self,
    ex::set_value_t,
    T&& v) noexcept {
    *self.value_ = v;
    *self.cat_ = typecat::rvalref;
  }

  STDEXEC_DEFINE_CUSTOM(void set_stopped)(this typecat_receiver self, ex::set_stopped_t) noexcept {
    FAIL_CHECK("set_stopped called");
  }

  STDEXEC_DEFINE_CUSTOM(void set_error)(
    this typecat_receiver self,
    ex::set_error_t,
    std::exception_ptr) noexcept {
    FAIL_CHECK("set_error called");
  }

  STDEXEC_DEFINE_CUSTOM(empty_env get_env)(this const typecat_receiver&, ex::get_env_t) noexcept {
    return {};
  }
};

template <typename F>
struct fun_receiver {
  using is_receiver = void;
  F f_;

  template <typename... Ts>
  STDEXEC_DEFINE_CUSTOM(void set_value)(
    this fun_receiver&& self,
    ex::set_value_t,
    Ts... vals) noexcept {
    try {
      std::move(self.f_)((Ts&&) vals...);
    } catch (...) {
      ex::set_error(std::move(self), std::current_exception());
    }
  }

  STDEXEC_DEFINE_CUSTOM(void set_stopped)(this fun_receiver, ex::set_stopped_t) noexcept {
    FAIL("Done called");
  }

  STDEXEC_DEFINE_CUSTOM(void set_error)(
    this fun_receiver,
    ex::set_error_t,
    std::exception_ptr eptr) noexcept {
    try {
      if (eptr)
        std::rethrow_exception(eptr);
      FAIL("Empty exception thrown");
    } catch (const std::exception& e) {
      FAIL("Exception thrown: " << e.what());
    }
  }

  STDEXEC_DEFINE_CUSTOM(empty_env get_env)(this const fun_receiver&, ex::get_env_t) noexcept {
    return {};
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

  std::optional<std::tuple<Ts...>> res = stdexec::sync_wait((S&&) snd);
  CHECK(res.has_value());
  std::tuple<Ts...> expected((Ts&&) val...);
  if constexpr (std::tuple_size_v<std::tuple<Ts...>> == 1)
    CHECK(std::get<0>(res.value()) == std::get<0>(expected));
  else
    CHECK(res.value() == expected);
}
