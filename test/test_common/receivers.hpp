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
#include <execution.hpp>
#include <tuple>

namespace ex = std::execution;

namespace empty_recv {

using ex::set_error_t;
using ex::set_stopped_t;
using ex::set_value_t;
using ex::get_env_t;

struct recv0 {
  friend void tag_invoke(set_value_t, recv0&&) noexcept {}
  friend void tag_invoke(set_stopped_t, recv0&&) noexcept {}
  friend void tag_invoke(set_error_t, recv0&&, std::exception_ptr) noexcept {}
  friend empty_env tag_invoke(get_env_t, const recv0&) noexcept { return {}; }
};
struct recv_int {
  friend void tag_invoke(set_value_t, recv_int&&, int) noexcept {}
  friend void tag_invoke(set_stopped_t, recv_int&&) noexcept {}
  friend void tag_invoke(set_error_t, recv_int&&, std::exception_ptr) noexcept {}
  friend empty_env tag_invoke(get_env_t, const recv_int&) noexcept { return {}; }
};

struct recv0_ec {
  friend void tag_invoke(set_value_t, recv0_ec&&) noexcept {}
  friend void tag_invoke(set_stopped_t, recv0_ec&&) noexcept {}
  friend void tag_invoke(set_error_t, recv0_ec&&, std::error_code) noexcept {}
  friend void tag_invoke(set_error_t, recv0_ec&&, std::exception_ptr) noexcept {}
  friend empty_env tag_invoke(get_env_t, const recv0_ec&) noexcept { return {}; }
};
struct recv_int_ec {
  friend void tag_invoke(set_value_t, recv_int_ec&&, int) noexcept {}
  friend void tag_invoke(set_stopped_t, recv_int_ec&&) noexcept {}
  friend void tag_invoke(set_error_t, recv_int_ec&&, std::error_code) noexcept {}
  friend void tag_invoke(set_error_t, recv_int_ec&&, std::exception_ptr) noexcept {}
  friend empty_env tag_invoke(get_env_t, const recv_int_ec&) noexcept { return {}; }
};

} // namespace empty_recv

class base_expect_receiver {
  std::atomic<bool> called_{false};

  public:
  ~base_expect_receiver() { CHECK(called_.load()); }
  base_expect_receiver() : called_(false) {}
  base_expect_receiver(base_expect_receiver&& other)
      : called_(other.called_.exchange(true)) {
  }
  base_expect_receiver& operator=(base_expect_receiver&& other) {
    called_.store(other.called_.exchange(true));
    return *this;
  }
  void set_called() {
    called_.store(true);
  }
};

template<class _Env = empty_env>
class expect_void_receiver : base_expect_receiver {
  _Env env_;

  public:
  expect_void_receiver(_Env env = _Env{}) : env_(std::move(env)) {}

  friend void tag_invoke(ex::set_value_t, expect_void_receiver&& self) noexcept {
    self.set_called();
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
  friend _Env tag_invoke(ex::get_env_t, const expect_void_receiver& self) noexcept {
    return self.env_;
  }
};

struct expect_void_receiver_ex {
  expect_void_receiver_ex(bool& executed)
    : executed_(&executed)
  {}

private:
  bool* executed_;

  friend void tag_invoke(ex::set_value_t, expect_void_receiver_ex&& self, const auto&...) noexcept {
    *self.executed_ = true;
  }
  friend void tag_invoke(ex::set_stopped_t, expect_void_receiver_ex&&) noexcept {
    FAIL_CHECK("set_stopped called on expect_void_receiver_ex");
  }
  friend void tag_invoke(ex::set_error_t, expect_void_receiver_ex&&, std::exception_ptr) noexcept {
    FAIL_CHECK("set_error called on expect_void_receiver_ex");
  }
  friend empty_env tag_invoke(ex::get_env_t, const expect_void_receiver_ex&) noexcept {
    return {};
  }
};

template <bool RuntimeCheck /*= true*/, typename... Ts>
class basic_expect_value_receiver  : base_expect_receiver {
  std::tuple<Ts...> values_;

public:
  explicit basic_expect_value_receiver(Ts... vals)
      : values_(std::move(vals)...) {}

  friend void tag_invoke(ex::set_value_t, basic_expect_value_receiver&& self, const Ts&... vals) noexcept {
    CHECK(self.values_ == std::tie(vals...));
    self.set_called();
  }
  template<typename... Us>
  friend void tag_invoke(ex::set_value_t, basic_expect_value_receiver&&, const Us&...) noexcept
      requires RuntimeCheck {
    FAIL_CHECK("set_value called with wrong value types on basic_expect_value_receiver");
  }
  friend void tag_invoke(ex::set_stopped_t, basic_expect_value_receiver&& self) noexcept {
    FAIL_CHECK("set_stopped called on basic_expect_value_receiver");
  }
  template <typename E>
  friend void tag_invoke(ex::set_error_t, basic_expect_value_receiver&&, E) noexcept {
    FAIL_CHECK("set_error called on basic_expect_value_receiver");
  }
  friend empty_env tag_invoke(ex::get_env_t, const basic_expect_value_receiver&) noexcept {
    return {};
  }
};

template <typename... Ts>
struct expect_value_receiver
  : public basic_expect_value_receiver<true, Ts...> {
  using basic_expect_value_receiver<true, Ts...>::basic_expect_value_receiver;
};

template <typename... Ts>
expect_value_receiver(Ts...) -> expect_value_receiver<Ts...>;

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
    FAIL_CHECK("set_error called on expect_value_receiver_ex");
  }
  friend empty_env tag_invoke(ex::get_env_t, const expect_value_receiver_ex&) noexcept {
    return {};
  }
};

class expect_stopped_receiver : base_expect_receiver {
  public:
  expect_stopped_receiver() = default;

  template <typename... Ts>
  friend void tag_invoke(ex::set_value_t, expect_stopped_receiver&&, Ts...) noexcept {
    FAIL_CHECK("set_value called on expect_stopped_receiver");
  }
  friend void tag_invoke(ex::set_stopped_t, expect_stopped_receiver&& self) noexcept {
    self.set_called();
  }
  friend void tag_invoke(ex::set_error_t, expect_stopped_receiver&&, std::exception_ptr) noexcept {
    FAIL_CHECK("set_error called on expect_stopped_receiver");
  }
  friend empty_env tag_invoke(ex::get_env_t, const expect_stopped_receiver&) noexcept {
    return {};
  }
};

struct expect_stopped_receiver_ex {
  explicit expect_stopped_receiver_ex(bool& executed)
    : executed_(&executed)
  {}

private:
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
  friend empty_env tag_invoke(ex::get_env_t, const expect_stopped_receiver_ex&) noexcept {
    return {};
  }
  bool* executed_;
};

inline std::pair<const std::type_info&, std::string> to_comparable(std::exception_ptr eptr) try {
  std::rethrow_exception(eptr);
} catch(const std::exception& e) {
  return {typeid(e), e.what()};
} catch(...) {
  return {typeid(void), "<unknown>"};
}

template <typename T>
inline T to_comparable(T value) {
  return value;
}

template <class T = std::exception_ptr>
struct expect_error_receiver : base_expect_receiver {
  expect_error_receiver() = default;
  expect_error_receiver(T error)
    : error_(std::move(error))
  {}

  // these do not move error_ and cannot be defaulted
  expect_error_receiver(expect_error_receiver&& other)
      : base_expect_receiver(std::move(other))
      , error_()
  {}
  expect_error_receiver& operator=(expect_error_receiver&& other) noexcept {
    base_expect_receiver::operator=(std::move(other));
    error_.reset();
    return *this;
  }
private:
  std::optional<T> error_;

  template <typename... Ts>
  friend void tag_invoke(ex::set_value_t, expect_error_receiver&&, Ts...) noexcept {
    FAIL_CHECK("set_value called on expect_error_receiver");
  }
  friend void tag_invoke(ex::set_stopped_t, expect_error_receiver&& self) noexcept {
    FAIL_CHECK("set_stopped called on expect_error_receiver");
  }
  friend void tag_invoke(ex::set_error_t, expect_error_receiver&& self, T err) noexcept {
    self.set_called();
    if (self.error_) {
      CHECK(to_comparable(err) == to_comparable(*self.error_));
    }
  }
  template <typename E>
  friend void tag_invoke(ex::set_error_t, expect_error_receiver&& self, E) noexcept {
    FAIL_CHECK("set_error called on expect_error_receiver with wrong error type");
  }
  friend empty_env tag_invoke(ex::get_env_t, const expect_error_receiver&) noexcept {
    return {};
  }
};

struct expect_error_receiver_ex {
  expect_error_receiver_ex(bool& executed)
    : executed_(&executed)
  {}
private:
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
  friend empty_env tag_invoke(ex::get_env_t, const expect_error_receiver_ex&) noexcept {
    return {};
  }
};

struct logging_receiver {
  logging_receiver(int& state)
    : state_(&state)
  {}
private:
  int* state_;
  template<typename... Args>
  friend void tag_invoke(ex::set_value_t, logging_receiver&& self, Args...) noexcept {
    *self.state_ = 0;
  }
  friend void tag_invoke(ex::set_stopped_t, logging_receiver&& self) noexcept {
    *self.state_ = 1;
  }
  template <typename E>
  friend void tag_invoke(ex::set_error_t, logging_receiver&& self, E) noexcept {
    *self.state_ = 2;
  }
  friend empty_env tag_invoke(ex::get_env_t, const logging_receiver&) noexcept {
    return {};
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

  // friend void tag_invoke(ex::set_value_t, typecat_receiver self, T v) noexcept {
  //     *self.value_ = v;
  //     *self.cat_ = typecat::value;
  // }
  friend void tag_invoke(ex::set_value_t, typecat_receiver self, T& v) noexcept {
    *self.value_ = v;
    *self.cat_ = typecat::ref;
  }
  friend void tag_invoke(ex::set_value_t, typecat_receiver self, const T& v) noexcept {
    *self.value_ = v;
    *self.cat_ = typecat::cref;
  }
  friend void tag_invoke(ex::set_value_t, typecat_receiver self, T&& v) noexcept {
    *self.value_ = v;
    *self.cat_ = typecat::rvalref;
  }
  friend void tag_invoke(ex::set_stopped_t, typecat_receiver self) noexcept {
    FAIL_CHECK("set_stopped called");
  }
  friend void tag_invoke(ex::set_error_t, typecat_receiver self, std::exception_ptr) noexcept {
    FAIL_CHECK("set_error called");
  }
  friend empty_env tag_invoke(ex::get_env_t, const typecat_receiver&) noexcept {
    return {};
  }
};

template <typename F>
struct fun_receiver {
  F f_;

  template <typename... Ts>
  friend void tag_invoke(ex::set_value_t, fun_receiver&& self, Ts... vals) noexcept try {
    std::move(self.f_)((Ts &&) vals...);
  } catch(...) {
    ex::set_error(std::move(self), std::current_exception());
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
  friend empty_env tag_invoke(ex::get_env_t, const fun_receiver&) noexcept {
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
  static_assert(ex::__single_value_variant_sender<S>,
      "Sender passed to sync_wait needs to have one variant for sending set_value");

  std::optional<std::tuple<Ts...>> res = _P2300::this_thread::sync_wait((S &&) snd);
  CHECK(res.has_value());
  std::tuple<Ts...> expected((Ts &&) val...);
  if constexpr (std::tuple_size_v<std::tuple<Ts...>> == 1)
    CHECK(std::get<0>(res.value()) == std::get<0>(expected));
  else
    CHECK(res.value() == expected);
}