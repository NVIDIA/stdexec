/*
 * Copyright (c) Lucian Radu Teodorescu
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
#include <execution.hpp>

namespace ex = std::execution;

namespace empty_recv {

using ex::set_error_t;
using ex::set_stopped_t;
using ex::set_value_t;

struct recv0 {
  friend void tag_invoke(set_value_t, recv0&&) {}
  friend void tag_invoke(set_stopped_t, recv0&&) noexcept {}
  friend void tag_invoke(set_error_t, recv0&&, std::exception_ptr) noexcept {}
};
struct recv_int {
  friend void tag_invoke(set_value_t, recv_int&&, int) {}
  friend void tag_invoke(set_stopped_t, recv_int&&) noexcept {}
  friend void tag_invoke(set_error_t, recv_int&&, std::exception_ptr) noexcept {}
};

struct recv0_ec {
  friend void tag_invoke(set_value_t, recv0_ec&&) {}
  friend void tag_invoke(set_stopped_t, recv0_ec&&) noexcept {}
  friend void tag_invoke(set_error_t, recv0_ec&&, std::error_code) noexcept {}
  friend void tag_invoke(set_error_t, recv0_ec&&, std::exception_ptr) noexcept {}
};
struct recv_int_ec {
  friend void tag_invoke(set_value_t, recv_int_ec&&, int) {}
  friend void tag_invoke(set_stopped_t, recv_int_ec&&) noexcept {}
  friend void tag_invoke(set_error_t, recv_int_ec&&, std::error_code) noexcept {}
  friend void tag_invoke(set_error_t, recv_int_ec&&, std::exception_ptr) noexcept {}
};

} // namespace empty_recv

class expect_void_receiver {
  bool called_{false};

  public:
  expect_void_receiver() = default;
  ~expect_void_receiver() { CHECK(called_); }

  expect_void_receiver(expect_void_receiver&& other)
      : called_(other.called_) {
    other.called_ = true;
  }
  expect_void_receiver& operator=(expect_void_receiver&& other) {
    called_ = other.called_;
    other.called_ = true;
    return *this;
  }

  friend void tag_invoke(ex::set_value_t, expect_void_receiver&& self) noexcept {
    self.called_ = true;
  }
  template <typename... Ts>
  friend void tag_invoke(ex::set_value_t, expect_void_receiver&&, Ts...) noexcept {
    FAIL_CHECK("set_value called on expect_void_receiver with some non-void value");
  }
  friend void tag_invoke(ex::set_stopped_t, expect_void_receiver&&) noexcept {
    FAIL_CHECK("set_stopped called on expect_void_receiver");
  }
  friend void tag_invoke(ex::set_error_t, expect_void_receiver&&, std::exception_ptr) noexcept {
    FAIL_CHECK("set_error called on expect_void_receiver");
  }
};

struct expect_void_receiver_ex {
  bool* executed_;

  template <typename... Ts>
  friend void tag_invoke(ex::set_value_t, expect_void_receiver_ex&& self, Ts...) noexcept {
    *self.executed_ = true;
  }
  friend void tag_invoke(ex::set_stopped_t, expect_void_receiver_ex&&) noexcept {
    FAIL_CHECK("set_stopped called on expect_void_receiver_ex");
  }
  friend void tag_invoke(ex::set_error_t, expect_void_receiver_ex&&, std::exception_ptr) noexcept {
    FAIL_CHECK("set_error called on expect_void_receiver_ex");
  }
};

template <typename T>
class expect_value_receiver {
  bool called_{false};
  T value_;

  public:
  explicit expect_value_receiver(T val)
      : value_(std::forward<T>(val)) {}
  ~expect_value_receiver() { CHECK(called_); }

  expect_value_receiver(expect_value_receiver&& other)
      : called_(other.called_)
      , value_(std::move(other.value_)) {
    other.called_ = true;
  }
  expect_value_receiver& operator=(expect_value_receiver&& other) {
    value_ = std::move(other.value_);
    called_ = other.called_;
    other.called_ = true;
    return *this;
  }

  template <typename... Ts>
  friend void tag_invoke(ex::set_value_t, expect_value_receiver&& self, const T& val) noexcept {
    CHECK(val == self.value_);
    self.called_ = true;
  }
  template <typename... Ts>
  friend void tag_invoke(ex::set_value_t, expect_value_receiver&&, Ts...) noexcept {
    FAIL_CHECK("set_value called with wrong value types on expect_value_receiver");
  }
  friend void tag_invoke(ex::set_stopped_t, expect_value_receiver&& self) noexcept {
    FAIL_CHECK("set_stopped called on expect_value_receiver");
  }
  friend void tag_invoke(ex::set_error_t, expect_value_receiver&&, std::exception_ptr) noexcept {
    FAIL_CHECK("set_error called on expect_value_receiver");
  }
};

template <typename T>
class expect_value_receiver_ex {
  T* dest_;

  public:
  explicit expect_value_receiver_ex(T* dest)
      : dest_(dest) {}

  friend void tag_invoke(ex::set_value_t, expect_value_receiver_ex self, T val) noexcept {
    *self.dest_ = val;
  }
  template <typename... Ts>
  friend void tag_invoke(ex::set_value_t, expect_value_receiver_ex, Ts...) noexcept {
    FAIL_CHECK("set_value called with wrong value types on expect_value_receiver_ex");
  }
  friend void tag_invoke(ex::set_stopped_t, expect_value_receiver_ex) noexcept {
    FAIL_CHECK("set_stopped called on expect_value_receiver_ex");
  }
  friend void tag_invoke(ex::set_error_t, expect_value_receiver_ex, std::exception_ptr) noexcept {
    FAIL_CHECK("set_error called on expect_value_receiver");
  }
};

class expect_stopped_receiver {
  bool called_{false};

  public:
  expect_stopped_receiver() = default;
  ~expect_stopped_receiver() { CHECK(called_); }

  expect_stopped_receiver(expect_stopped_receiver&& other)
      : called_(other.called_) {
    other.called_ = true;
  }
  expect_stopped_receiver& operator=(expect_stopped_receiver&& other) {
    called_ = other.called_;
    other.called_ = true;
    return *this;
  }

  template <typename... Ts>
  friend void tag_invoke(ex::set_value_t, expect_stopped_receiver&&, Ts...) noexcept {
    FAIL_CHECK("set_value called on expect_stopped_receiver");
  }
  friend void tag_invoke(ex::set_stopped_t, expect_stopped_receiver&& self) noexcept {
    self.called_ = true;
  }
  friend void tag_invoke(ex::set_error_t, expect_stopped_receiver&&, std::exception_ptr) noexcept {
    FAIL_CHECK("set_error called on expect_stopped_receiver");
  }
};

struct expect_stopped_receiver_ex {
  bool* executed_;

  template <typename... Ts>
  friend void tag_invoke(ex::set_value_t, expect_stopped_receiver_ex&&, Ts...) noexcept {
    FAIL_CHECK("set_value called on expect_stopped_receiver_ex");
  }
  friend void tag_invoke(ex::set_stopped_t, expect_stopped_receiver_ex&& self) noexcept {
    *self.executed_ = true;
  }
  friend void tag_invoke(
      ex::set_error_t, expect_stopped_receiver_ex&&, std::exception_ptr) noexcept {
    FAIL_CHECK("set_error called on expect_stopped_receiver_ex");
  }
};

class expect_error_receiver {
  bool called_{false};

  public:
  expect_error_receiver() = default;
  ~expect_error_receiver() { CHECK(called_); }

  expect_error_receiver(expect_error_receiver&& other)
      : called_(other.called_) {
    other.called_ = true;
  }
  expect_error_receiver& operator=(expect_error_receiver&& other) {
    called_ = other.called_;
    other.called_ = true;
    return *this;
  }

  template <typename... Ts>
  friend void tag_invoke(ex::set_value_t, expect_error_receiver&&, Ts...) noexcept {
    FAIL_CHECK("set_value called on expect_error_receiver");
  }
  friend void tag_invoke(ex::set_stopped_t, expect_error_receiver&& self) noexcept {
    FAIL_CHECK("set_stopped called on expect_error_receiver");
  }
  friend void tag_invoke(
      ex::set_error_t, expect_error_receiver&& self, std::exception_ptr) noexcept {
    self.called_ = true;
  }
  template <typename E>
  friend void tag_invoke(ex::set_error_t, expect_error_receiver&& self, E) noexcept {
    self.called_ = true;
  }
};

struct expect_error_receiver_ex {
  bool* executed_;

  template <typename... Ts>
  friend void tag_invoke(ex::set_value_t, expect_error_receiver_ex&&, Ts...) noexcept {
    FAIL_CHECK("set_value called on expect_error_receiver_ex");
  }
  friend void tag_invoke(ex::set_stopped_t, expect_error_receiver_ex&&) noexcept {
    FAIL_CHECK("set_stopped called on expect_error_receiver_ex");
  }
  friend void tag_invoke(
      ex::set_error_t, expect_error_receiver_ex&& self, std::exception_ptr) noexcept {
    *self.executed_ = true;
  }
};

struct logging_receiver {
  int* state_;
  bool should_throw_{false};
  friend void tag_invoke(ex::set_value_t, logging_receiver&& self) {
    if (self.should_throw_)
      throw std::logic_error("test");
    *self.state_ = 0;
  }
  friend void tag_invoke(ex::set_stopped_t, logging_receiver&& self) noexcept { *self.state_ = 1; }
  friend void tag_invoke(ex::set_error_t, logging_receiver&& self, std::exception_ptr) noexcept {
    *self.state_ = 2;
  }
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
  T* value_;
  typecat* cat_;

  // friend void tag_invoke(ex::set_value_t, typecat_receiver self, T v) {
  //     *self.value_ = v;
  //     *self.cat_ = typecat::value;
  // }
  friend void tag_invoke(ex::set_value_t, typecat_receiver self, T& v) {
    *self.value_ = v;
    *self.cat_ = typecat::ref;
  }
  friend void tag_invoke(ex::set_value_t, typecat_receiver self, const T& v) {
    *self.value_ = v;
    *self.cat_ = typecat::cref;
  }
  friend void tag_invoke(ex::set_value_t, typecat_receiver self, T&& v) {
    *self.value_ = v;
    *self.cat_ = typecat::rvalref;
  }
  friend void tag_invoke(ex::set_stopped_t, typecat_receiver self) noexcept {
    FAIL_CHECK("set_stopped called");
  }
  friend void tag_invoke(ex::set_error_t, typecat_receiver self, std::exception_ptr) noexcept {
    FAIL_CHECK("set_error called");
  }
};

template <typename F>
struct fun_receiver {
  F f_;

  template <typename... Ts>
  friend void tag_invoke(ex::set_value_t, fun_receiver&& self, Ts... vals) {
    std::move(self.f_)((Ts &&) vals...);
  }
  template <typename... Ts>
  friend void tag_invoke(ex::set_value_t, const fun_receiver& self, Ts... vals) {
    self.f_((Ts &&) vals...);
  }

  friend void tag_invoke(ex::set_stopped_t, fun_receiver) noexcept { FAIL("Done called"); }
  friend void tag_invoke(ex::set_error_t, fun_receiver, std::exception_ptr eptr) noexcept {
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
  static_assert(ex::__single_value_variant_sender<S>,
      "Sender passed to sync_wait needs to have one variant for sending set_value");

  std::optional<std::tuple<Ts...>> res = std::this_thread::sync_wait((S &&) snd);
  CHECK(res.has_value());
  std::tuple<Ts...> expected((Ts &&) val...);
  if constexpr (std::tuple_size_v<std::tuple<Ts...>> == 1)
    CHECK(std::get<0>(res.value()) == std::get<0>(expected));
  else
    CHECK(res.value() == expected);
}