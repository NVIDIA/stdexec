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

#include <cassert>
#include <variant>
#include <utility>
#include <exception>

#include <coroutine.hpp>
#include <execution.hpp>

template <class T>
struct task {
  struct promise_type;
  struct final_awaitable {
    bool await_ready() const noexcept {
      return false;
    }
    auto await_suspend(coro::coroutine_handle<promise_type> h) const noexcept {
      return h.promise().parent_;
    }
    void await_resume() const noexcept {
    }
  };
  struct _promise_base {
    void return_value(T value) noexcept {
      data_.template emplace<1>(std::move(value));
    }
    std::variant<std::monostate, T, std::exception_ptr> data_{};
  };
  struct promise_type : _promise_base {
    task get_return_object() noexcept {
      return task(coro::coroutine_handle<promise_type>::from_promise(*this));
    }
    coro::suspend_always initial_suspend() noexcept {
      return {};
    }
    final_awaitable final_suspend() noexcept {
      return {};
    }
    void unhandled_exception() noexcept {
      this->data_.template emplace<2>(std::current_exception());
    }
    coro::coroutine_handle<> parent_{};
  };

  task(task&& that) noexcept
    : coro_(std::exchange(that.coro_, {}))
  {}

  ~task() {
    if (coro_)
      coro_.destroy();
  }

  struct task_awaitable {
    task& t;
    bool await_ready() const noexcept {
      return false;
    }
    auto await_suspend(coro::coroutine_handle<> parent) noexcept {
      t.coro_.promise().parent_ = parent;
      return t.coro_;
    }
    T await_resume() const {
      if (t.coro_.promise().data_.index() == 2)
        std::rethrow_exception(std::get<2>(t.coro_.promise().data_));
      if constexpr (!std::is_void_v<T>)
        return std::get<T>(t.coro_.promise().data_);
    }
  };

  friend task_awaitable operator co_await(task&& t) noexcept {
    return task_awaitable{t};
  }

private:
  explicit task(coro::coroutine_handle<promise_type> coro) noexcept
    : coro_(coro)
  {}
  coro::coroutine_handle<promise_type> coro_;
};

template<>
struct task<void>::_promise_base {
  struct _void {};
  void return_void() noexcept {
    data_.template emplace<1>(_void{});
  }
  std::variant<std::monostate, _void, std::exception_ptr> data_{};
};
