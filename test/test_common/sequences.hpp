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
#include <test_common/type_helpers.hpp>
#include <stdexec/execution.hpp>
#include <exec/sequence.hpp>

namespace ex = stdexec;

template<class _Env = empty_env>
class expect_empty_sequence_receiver {
  _Env env_;
  bool called_{false};

  public:
  expect_empty_sequence_receiver(_Env env = _Env{}) : env_(env), called_(false) {}
  ~expect_empty_sequence_receiver() { CHECK(called_); }

  expect_empty_sequence_receiver(expect_empty_sequence_receiver&& other)
      : env_(other.env_)
      , called_(std::exchange(other.called_, true)) {
  }
  expect_empty_sequence_receiver& operator=(expect_empty_sequence_receiver&& other) {
    env_ = other.env_;
    called_ = std::exchange(other.called_, true);
    return *this;
  }


  template <typename ValueSender>
  friend void tag_invoke(_P0TBD::execution::set_next_t, expect_empty_sequence_receiver&, ValueSender&&) noexcept {
    FAIL_CHECK("set_next called on expect_empty_sequence_receiver");
  }
  friend void tag_invoke(ex::set_value_t, expect_empty_sequence_receiver&& self) noexcept {
    self.called_ = true;
  }
  template <typename... Ts>
  friend void tag_invoke(ex::set_value_t, expect_empty_sequence_receiver&&, Ts...) noexcept {
    FAIL_CHECK("set_value called on expect_empty_sequence_receiver with some non-void value");
  }
  friend void tag_invoke(ex::set_stopped_t, expect_empty_sequence_receiver&&) noexcept {
    FAIL_CHECK("set_stopped called on expect_empty_sequence_receiver");
  }
  friend void tag_invoke(ex::set_error_t, expect_empty_sequence_receiver&&, std::exception_ptr) noexcept {
    FAIL_CHECK("set_error called on expect_empty_sequence_receiver");
  }
  friend _Env tag_invoke(ex::get_env_t, const expect_empty_sequence_receiver& self) noexcept {
    return self.env_;
  }
};

struct expect_empty_sequence_receiver_ex {
  bool* executed_;

  template <typename ValueSender>
  friend void tag_invoke(_P0TBD::execution::set_next_t, expect_empty_sequence_receiver_ex& self, ValueSender&&) noexcept {
    FAIL_CHECK("set_next called on expect_empty_sequence_receiver_ex");
  }
  friend void tag_invoke(ex::set_value_t, expect_empty_sequence_receiver_ex&& self, const auto&...) noexcept {
    *self.executed_ = true;
  }
  friend void tag_invoke(ex::set_stopped_t, expect_empty_sequence_receiver_ex&&) noexcept {
    FAIL_CHECK("set_stopped called on expect_empty_sequence_receiver_ex");
  }
  friend void tag_invoke(ex::set_error_t, expect_empty_sequence_receiver_ex&&, std::exception_ptr) noexcept {
    FAIL_CHECK("set_error called on expect_empty_sequence_receiver_ex");
  }
  friend empty_env tag_invoke(ex::get_env_t, const expect_empty_sequence_receiver_ex&) noexcept {
    return {};
  }
};

template <typename ValidFn, class _Env = empty_env, bool RuntimeCheck = true>
class expect_sequence_receiver {
  std::int_least32_t called_{0};
  std::atomic_bool ended_{false};
  ValidFn valid_;
  _Env env_;

  public:
  explicit expect_sequence_receiver(ValidFn valid, _Env env = {})
      : valid_(std::move(valid))
      , env_(std::move(env)) {}
  ~expect_sequence_receiver() { CHECK(ended_.load() != false); }

  expect_sequence_receiver(expect_sequence_receiver&& other)
      : called_(std::exchange(other.called_, -1))
      , ended_(other.ended_.load())
      , valid_(std::move(other.valid_))
      , env_(std::move(other.env_))
  {other.ended_ = true;}
  expect_sequence_receiver& operator=(expect_sequence_receiver&& other) {
    called_ = std::exchange(other.called_, -1);
    ended_ = other.ended_.load();
    other.ended_ = true;
    valid_ = std::move(other.valid_);
    env_ = std::move(other.env_);
    return *this;
  }

  template<class ValueSender>
  friend auto tag_invoke(_P0TBD::execution::set_next_t, expect_sequence_receiver& self, ValueSender&& vs) noexcept {
    return ex::then(std::forward<ValueSender>(vs),
      [&self](auto&&... tn) noexcept requires std::is_invocable_v<ValidFn, decltype(tn)...> {
        CHECK(self.valid_(std::forward<decltype(tn)>(tn)...));
        ++self.called_;
      });
  }
  friend void tag_invoke(ex::set_value_t, expect_sequence_receiver&& self) noexcept {
    // end of sequence
    self.ended_ = true;
  }
  friend void tag_invoke(ex::set_value_t, expect_sequence_receiver&&, const auto&...) noexcept
      requires RuntimeCheck {
    FAIL_CHECK("set_value called with wrong value types on expect_sequence_receiver");
  }
  friend void tag_invoke(ex::set_stopped_t, expect_sequence_receiver&& ) noexcept {
    FAIL_CHECK("set_stopped called on expect_sequence_receiver");
  }
  template <typename E>
  friend void tag_invoke(ex::set_error_t, expect_sequence_receiver&&, E) noexcept {
    FAIL_CHECK("set_error called on expect_sequence_receiver");
  }
  friend _Env tag_invoke(ex::get_env_t, const expect_sequence_receiver& __self) noexcept {
    return __self.env_;
  }
};

template <typename ValidFn, class _Env = empty_env, bool RuntimeCheck = true>
expect_sequence_receiver<ValidFn, _Env, RuntimeCheck> make_expect_sequence_receiver(ValidFn&& valid, _Env&& env = {}) {
  return expect_sequence_receiver<ValidFn, _Env, RuntimeCheck>{std::forward<ValidFn>(valid), std::forward<_Env>(env)};
}
