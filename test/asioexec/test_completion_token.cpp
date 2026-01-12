/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *                         Copyright (c) 2025 Robert Leahy. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <asioexec/completion_token.hpp>

#include <asioexec/asio_config.hpp>
#include <barrier>
#include <catch2/catch.hpp>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <stdexec/execution.hpp>
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>
#include <thread>
#include <type_traits>
#include <utility>

using namespace STDEXEC;
using namespace asioexec;

namespace {

  static_assert(noexcept(detail::completion_token::convert<const int&>(std::declval<int&>())));
  static_assert(noexcept(detail::completion_token::convert<const int>(std::declval<int>())));
  static_assert(
    noexcept(detail::completion_token::convert<std::string>(std::declval<std::string>())));
  static_assert(
    !noexcept(detail::completion_token::convert<std::string>(std::declval<const std::string&>())));
  static_assert(
    !noexcept(detail::completion_token::convert<std::string>(std::declval<const char*>())));

  //  connect_shared and start_shared ensure the operation state's lifetime ends
  //  within the completion signal handling of the receiver thereby ensuring any
  //  use of the operation state by the operation after it's sent a completion
  //  signal is caught be AddressSanitizer

  template <typename Receiver>
  class connect_shared_receiver {
    Receiver r_;
    std::shared_ptr<void>& ptr_;

    template <typename Tag, typename... Args>
    void complete_(const Tag& tag, Args&&... args) noexcept {
      CHECK(ptr_);
      CHECK(ptr_.use_count() == 1);
      tag(std::move(r_), std::forward<Args>(args)...);
      ptr_.reset();
    }
   public:
    using receiver_concept = receiver_t;

    template <typename T>
      requires std::constructible_from<Receiver, T>
    constexpr explicit connect_shared_receiver(T&& t, std::shared_ptr<void>& ptr) noexcept
      : r_(std::forward<T>(t))
      , ptr_(ptr) {
    }

    constexpr void set_stopped() && noexcept
      requires ::STDEXEC::receiver_of<
        Receiver,
        ::STDEXEC::completion_signatures<::STDEXEC::set_stopped_t()>
      >
    {
      complete_(::STDEXEC::set_stopped);
    }

    template <typename T>
      requires ::STDEXEC::receiver_of<
        Receiver,
        ::STDEXEC::completion_signatures<::STDEXEC::set_error_t(T)>
      >
    constexpr void set_error(T&& t) && noexcept {
      complete_(::STDEXEC::set_error, std::forward<T>(t));
    }

    template <typename... Args>
      requires ::STDEXEC::receiver_of<
        Receiver,
        ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(Args...)>
      >
    constexpr void set_value(Args&&... args) && noexcept {
      complete_(::STDEXEC::set_value, std::forward<Args>(args)...);
    }

    constexpr decltype(auto) get_env() const noexcept {
      return ::STDEXEC::get_env(r_);
    }
  };

  template <typename Sender, typename Receiver>
  class connect_shared_operation_state {
    using receiver_ = connect_shared_receiver<std::remove_cvref_t<Receiver>>;
    std::shared_ptr<void> self_;
    ::STDEXEC::connect_result_t<Sender, receiver_> op_;
   public:
    constexpr explicit connect_shared_operation_state(Sender&& s, Receiver&& r)
      : op_(
          ::STDEXEC::connect(
            std::forward<Sender>(s),
            receiver_(std::forward<Receiver>(r), self_))) {
    }

    void start(std::shared_ptr<connect_shared_operation_state>&& ptr) & noexcept {
      CHECK(ptr.get() == this);
      CHECK(ptr.use_count() == 1);
      self_ = std::move(ptr);
      ::STDEXEC::start(op_);
    }
  };

  template <typename Sender, typename Receiver>
  auto connect_shared(Sender&& sender, Receiver&& receiver) {
    return std::make_shared<connect_shared_operation_state<Sender, Receiver>>(
      std::forward<Sender>(sender), std::forward<Receiver>(receiver));
  }

  template <typename Sender, typename Receiver>
  void
    start_shared(std::shared_ptr<connect_shared_operation_state<Sender, Receiver>>&& ptr) noexcept {
    REQUIRE(ptr);
    auto&& state = *ptr;
    state.start(std::move(ptr));
  }

  TEST_CASE(
    "Asio-based asynchronous operation ends with error when canceled",
    "[asioexec][completion_token]") {
    inplace_stop_source source;

    const struct {
      auto query(const get_stop_token_t&) const noexcept {
        return source_.get_token();
      }

      inplace_stop_source& source_;
    } e{source};

    error_code err;
    asio_impl::io_context ctx;
    asio_impl::system_timer t(ctx);
    t.expires_after(std::chrono::years(1));
    auto sender = t.async_wait(completion_token);
    static_assert(set_equivalent<
                  completion_signatures_of_t<decltype(sender), env<>>,
                  completion_signatures<
                    set_stopped_t(),
                    set_error_t(std::exception_ptr),
                    set_value_t(error_code)
                  >
    >);
    CHECK(!ctx.poll());
    CHECK(ctx.stopped());
    ctx.restart();
    auto op = connect_shared(std::move(sender), expect_value_receiver_ex(e, err));
    CHECK(!ctx.poll());
    CHECK(ctx.stopped());
    ctx.restart();
    source.request_stop();
    start_shared(std::move(op));
    CHECK(!err);
    CHECK(ctx.poll());
    CHECK(ctx.stopped());
    CHECK(err == asio_impl::error::operation_aborted);
  }

  TEST_CASE("Posting may be redirected to a sender", "[asioexec][completion_token]") {
    asio_impl::io_context ctx;
    asio_impl::system_timer t(ctx);
    t.expires_after(std::chrono::years(1));
    auto sender = asio_impl::post(ctx.get_executor(), completion_token);
    CHECK(!ctx.poll());
    CHECK(ctx.stopped());
    ctx.restart();
    auto op = connect_shared(std::move(sender), expect_value_receiver<>{});
    CHECK(!ctx.poll());
    CHECK(ctx.stopped());
    ctx.restart();
    start_shared(std::move(op));
    CHECK(ctx.poll());
    CHECK(ctx.stopped());
  }

  template <typename CompletionToken>
  decltype(auto) async_throw_from_initiation(CompletionToken&& token) {
    using signature_type = void();
    return asio_impl::async_initiate<CompletionToken, signature_type>(
      [](const auto&) { throw std::logic_error("Test"); }, token);
  }

  TEST_CASE(
    "When initiating the asynchronous operation throws this causes the "
    "sender to send a std::exception_ptr",
    "[asioexec][completion_token]") {
    std::exception_ptr ex;
    auto op =
      connect_shared(async_throw_from_initiation(completion_token), expect_error_receiver_ex(ex));
    CHECK(!ex);
    start_shared(std::move(op));
    CHECK(ex);
  }

  template <typename Executor, typename CompletionToken>
  decltype(auto) async_throw_from_completion(const Executor& ex, CompletionToken&& token) {
    using signature_type = void();
    return asio_impl::async_initiate<CompletionToken, signature_type>(
      [ex](auto h) {
        const auto assoc = asio_impl::get_associated_executor(h, ex);
        asio_impl::post(ex, asio_impl::bind_executor(assoc, [h = std::move(h)]() {
                          throw std::logic_error("Test");
                        }));
      },
      token);
  }

  TEST_CASE(
    "When a completion handler invoked as part of a composed "
    "asynchronous operation throws that exception is captured and sent as a "
    "std::exception_ptr",
    "[asioexec][completion_token]") {
    std::exception_ptr ex;
    asio_impl::io_context ctx;
    auto sender = async_throw_from_completion(ctx.get_executor(), completion_token);
    CHECK(!ctx.poll());
    CHECK(ctx.stopped());
    ctx.restart();
    auto op = connect_shared(std::move(sender), expect_error_receiver_ex<std::exception_ptr>(ex));
    CHECK(!ctx.poll());
    CHECK(ctx.stopped());
    ctx.restart();
    CHECK(!ex);
    start_shared(std::move(op));
    CHECK(ctx.poll());
    CHECK(ctx.stopped());
    CHECK(ex);
  }

  TEST_CASE(
    "When an operation is abandoned this is reported via a stopped "
    "signal",
    "[asioexec][completion_token]") {
    bool stopped = false;
    expect_stopped_receiver_ex r(stopped);
    using sender_type = decltype(std::declval<asio_impl::system_timer&>()
                                   .async_wait(completion_token));
    using operation_state_type = connect_result_t<sender_type, decltype(r)>;
    std::optional<operation_state_type> op;
    {
      asio_impl::io_context ctx;
      asio_impl::system_timer t(ctx);
      t.expires_after(std::chrono::years(1));

      struct {
        operator operation_state_type() {
          return ::STDEXEC::connect(std::move(s_), std::move(r_));
        }

        sender_type s_;
        decltype(r) r_;
      } elide{t.async_wait(completion_token), std::move(r)};

      op.emplace(std::move(elide));
      start(*op);
      CHECK(!stopped);
    }
    CHECK(stopped);
  }

  template <typename Executor, typename CompletionToken>
  decltype(auto) async_indirect_completion_handler(
    const Executor& ex,
    std::shared_ptr<void>& ptr,
    CompletionToken&& token) {
    using signature_type = void();
    return asio_impl::async_initiate<CompletionToken, signature_type>(
      [&ptr](auto h, const auto& ex) {
        auto local = std::make_shared<decltype(h)>(std::move(h));
        const auto assoc = asio_impl::get_associated_executor(*local, ex);
        ptr = local;
        asio_impl::post(ex, asio_impl::bind_executor(assoc, [local]() mutable {
                          std::invoke(std::move(*local));
                          local.reset();
                        }));
      },
      token,
      ex);
  }

  TEST_CASE(
    "If the completion handler outlives completion of the operation "
    "the receiver contract is still satisfied eagerly",
    "[asioexec][completion_token]") {
    bool invoked = false;
    std::shared_ptr<void> ptr;
    asio_impl::io_context ctx;
    auto sender = async_indirect_completion_handler(ctx.get_executor(), ptr, completion_token);
    CHECK(!ctx.poll());
    CHECK(ctx.stopped());
    ctx.restart();
    CHECK(!ptr);
    auto op = connect_shared(std::move(sender), expect_void_receiver_ex(invoked));
    CHECK(!ctx.poll());
    CHECK(ctx.stopped());
    ctx.restart();
    CHECK(!ptr);
    start_shared(std::move(op));
    CHECK(ptr);
    CHECK(!invoked);
    CHECK(ctx.poll());
    CHECK(ctx.stopped());
    CHECK(invoked);
    CHECK(ptr.use_count() == 1U);
  }

  template <typename Executor, typename CompletionToken>
  decltype(auto) async_indirect_completion_handler_throw_from_completion(
    const Executor& ex,
    std::shared_ptr<void>& ptr,
    CompletionToken&& token) {
    //  This function is seemingly fragile?
    //
    //  In release mode slightly different arrangements produce bizarre outcomes:
    //
    //  /root/stdexec/test/asioexec/test_completion_token.cpp:380: FAILED:
    //    CHECK( !ex )
    //  with expansion:
    //    false
    //
    //  /root/stdexec/test/asioexec/test_completion_token.cpp:382: FAILED:
    //    CHECK( ptr.use_count() == 1U )
    //  with expansion:
    //    -1564306560 == 1
    using signature_type = void();
    return asio_impl::async_initiate<CompletionToken, signature_type>(
      [&ptr](auto h, const auto& ex) {
        auto local = std::make_shared<decltype(h)>(std::move(h));
        const auto assoc = asio_impl::get_associated_executor(*local, ex);
        ptr = local;
        asio_impl::post(assoc, [&ptr, local = std::move(local)]() mutable {
          CHECK(local.use_count() == 2);
          CHECK(ptr.use_count() == 2);
          auto cpy = std::move(local);
          cpy.reset();
          CHECK(ptr.use_count() == 1);
          throw std::logic_error("Test");
        });
      },
      token,
      ex);
  }

  TEST_CASE(
    "If the completion handler outlives completion of the operation "
    "satisfaction of the receiver contract is deferred until the end of the "
    "completion handler's lifetime (this is necessary in situations where "
    "asynchronous control flow bifurcates and one of the child operations ends "
    "via exception)",
    "[asioexec][completion_token]") {
    std::exception_ptr ex;
    std::shared_ptr<void> ptr;
    asio_impl::io_context ctx;
    auto sender = async_indirect_completion_handler_throw_from_completion(
      ctx.get_executor(), ptr, completion_token);
    CHECK(!ctx.poll());
    CHECK(ctx.stopped());
    ctx.restart();
    CHECK(!ptr);
    auto op = connect_shared(std::move(sender), expect_error_receiver_ex(ex));
    CHECK(!ctx.poll());
    CHECK(ctx.stopped());
    ctx.restart();
    CHECK(!ptr);
    start_shared(std::move(op));
    CHECK(ptr);
    CHECK(ctx.poll());
    CHECK(ctx.stopped());
    CHECK(!ex);
    CHECK(ptr.use_count() == 1U);
    ptr.reset();
    CHECK(ex);
  }

  template <typename CompletionToken>
  decltype(auto) async_multishot(CompletionToken&& token) {
    using signature_type = void();
    return asio_impl::async_initiate<CompletionToken, signature_type>(
      [](auto&& h) { std::forward<decltype(h)>(h)(); }, token);
  }

  TEST_CASE("When appropriate the yielded sender is multi-shot", "[asioexec][completion_token]") {
    const auto sender = async_multishot(completion_token);
    auto a = connect_shared(sender, expect_void_receiver{});
    auto b = connect_shared(sender, expect_void_receiver{});
    start_shared(std::move(a));
    start_shared(std::move(b));
  }

  template <typename CompletionToken>
  decltype(auto) async_single_shot(CompletionToken&& token) {
    using signature_type = void();
    return asio_impl::async_initiate<CompletionToken, signature_type>(
      [ptr = std::make_unique<int>(5)](auto&& h) { std::forward<decltype(h)>(h)(); }, token);
  }

  TEST_CASE("When appropriate the yielded sender is single shot", "[asioexec][completion_token]") {
    auto sender = async_single_shot(completion_token);
    static_assert(!::STDEXEC::sender_to<const decltype(sender)&, expect_void_receiver<>>);
    start_shared(connect_shared(std::move(sender), expect_void_receiver{}));
  }

  TEST_CASE(
    "When an operation is abandoned by the initiating function "
    "set_stopped is sent immediately",
    "[asioexec][completion_token]") {
    const auto initiating_function = [&](auto&& token) {
      return asio_impl::async_initiate<decltype(token), void()>([](auto&&) noexcept { }, token);
    };
    start_shared(connect_shared(initiating_function(completion_token), expect_stopped_receiver{}));
  }

  class thread {
    asio_impl::io_context ctx_;
    asio_impl::executor_work_guard<asio_impl::io_context::executor_type> g_;
    std::thread t_;
   public:
    thread()
      : g_(ctx_.get_executor())
      , t_([&]() noexcept {
        STDEXEC_TRY {
          ctx_.run();
        }
        STDEXEC_CATCH_ALL {
          FAIL("Exception thrown in background thread");
        }
      }) {
    }

    ~thread() noexcept {
      g_.reset();
      if (t_.joinable()) {
        t_.join();
      }
    }

    void join() noexcept {
      g_.reset();
      t_.join();
    }

    asio_impl::io_context& context() noexcept {
      return ctx_;
    }
  };

  template <typename CompletionHandler>
  struct ping_pong {
    explicit ping_pong(asio_impl::io_context& a, asio_impl::io_context& b, CompletionHandler h)
      : a_(a)
      , a_g_(a.get_executor())
      , b_(b)
      , b_g_(b.get_executor())
      , h_(std::move(h)) {
    }

    void operator()() && {
      if (i_ == 10000) {
        a_g_.reset();
        b_g_.reset();
        std::invoke(std::move(h_));
        return;
      }
      const auto ex = [&]() noexcept {
        if (i_ % 2) {
          return a_.get_executor();
        }
        return b_.get_executor();
      }();
      ++i_;
      asio_impl::post(ex, std::move(*this));
    }

    asio_impl::io_context& a_;
    asio_impl::executor_work_guard<asio_impl::io_context::executor_type> a_g_;
    asio_impl::io_context& b_;
    asio_impl::executor_work_guard<asio_impl::io_context::executor_type> b_g_;
    CompletionHandler h_;
    std::size_t i_{0};
  };

  TEST_CASE(
    "Intermediate completion handlers being passed between threads has "
    "no effect on the transformation of an initiating function into a sender",
    "[asioexec][completion_token]") {
    thread a;
    thread b;
    const auto initiating_function = [&](auto&& token) {
      return asio_impl::async_initiate<decltype(token), void()>(
        [&a, &b](auto&& h) {
          ping_pong<std::remove_cvref_t<decltype(h)>>{
            a.context(), b.context(), std::forward<decltype(h)>(h)}();
        },
        token);
    };
    start_shared(connect_shared(initiating_function(completion_token), expect_void_receiver{}));
    a.join();
    b.join();
  }

  TEST_CASE(
    "When the initiating function posts and then throws, and the "
    "posted operation simply abandons the completion handler, the operation "
    "completes after the post with the thrown error",
    "[asioexec][completion_token]") {
    std::exception_ptr ex;
    asio_impl::io_context ctx;
    const auto initiating_function = [&](auto&& token) {
      return asio_impl::async_initiate<decltype(token), void()>(
        [&](auto&& h) {
          asio_impl::post(ctx.get_executor(), [h = std::move(h)]() noexcept { });
          throw std::logic_error("Test");
        },
        token);
    };
    auto op = connect_shared(initiating_function(completion_token), expect_error_receiver_ex(ex));
    CHECK(!ctx.poll());
    CHECK(ctx.stopped());
    ctx.restart();
    start_shared(std::move(op));
    CHECK(!ex);
    CHECK(ctx.poll());
    CHECK(ctx.stopped());
    CHECK(ex);
  }

  TEST_CASE(
    "When there are two parallel paths of asynchronous execution, and "
    "one of them throws an exception, and the other is abandoned, the "
    "operation completes with the thrown exception",
    "[asioexec][completion_token]") {
    for (unsigned u = 0; u < 2; ++u) {
      std::exception_ptr ex;
      asio_impl::io_context ctx;
      const auto initiating_function = [&](auto&& token) {
        return asio_impl::async_initiate<decltype(token), void()>(
          [&](auto&& h) {
            auto ptr = std::make_shared<std::remove_cvref_t<decltype(h)>>(
              std::forward<decltype(h)>(h));
            const auto ex = asio_impl::get_associated_executor(*ptr, ctx.get_executor());
            if (u) {
              asio_impl::post(ex, [ptr]() { throw std::logic_error("Test"); });
              asio_impl::post(ex, [ptr = std::move(ptr)]() noexcept { });
            } else {
              asio_impl::post(ex, [ptr]() noexcept { });
              asio_impl::post(ex, [ptr = std::move(ptr)]() { throw std::logic_error("Test"); });
            }
          },
          token);
      };
      auto op = connect_shared(initiating_function(completion_token), expect_error_receiver_ex(ex));
      CHECK(!ctx.poll());
      CHECK(ctx.stopped());
      ctx.restart();
      start_shared(std::move(op));
      CHECK(!ex);
      CHECK(ctx.poll_one());
      CHECK(!ctx.stopped());
      CHECK(!ex);
      CHECK(ctx.poll_one());
      CHECK(ctx.stopped());
      CHECK(ex);
    }
  }

  TEST_CASE(
    "When the Asio signature suggests the operation will send an "
    "rvalue, but a const lvalue is sent, and the decay-copy throws, that "
    "exception is sent as an error in completing the operation",
    "[asioexec][completion_token]") {
    class obj {
      bool throw_;
     public:
      explicit obj(bool t = false) noexcept
        : throw_(t) {
      }

      obj(obj&&) = default;

      obj(const obj& other)
        : throw_(other.throw_) {
        if (throw_) {
          throw std::logic_error("Test");
        }
      }

      obj& operator=(const obj&) = delete;

      bool operator==(const obj& rhs) const noexcept {
        return throw_ == rhs.throw_;
      }
    };

    const obj expected;
    std::exception_ptr ex;
    const auto initiating_function = [](obj o, auto&& token) {
      return asio_impl::async_initiate<decltype(token), void(obj)>(
        [o = std::move(o)](auto&& h) {
          //  Lambda isn't mutable and there's no std::move so o's type is
          //  const obj&
          std::invoke(std::forward<decltype(h)>(h), o);
        },
        token);
    };
    auto a =
      connect_shared(initiating_function(obj{}, completion_token), expect_value_receiver(expected));
    auto b = connect_shared(
      initiating_function(obj(true), completion_token), expect_error_receiver_ex(ex));
    start_shared(std::move(a));
    CHECK(!ex);
    start_shared(std::move(b));
    CHECK(ex);
  }

  struct value_category_receiver {
    using receiver_concept = receiver_t;

    void set_value(std::mutex&&) && noexcept {
      CHECK(kind_ == kind::none);
      kind_ = kind::rvalue;
    }

    void set_value(std::mutex&) && noexcept {
      CHECK(kind_ == kind::none);
      kind_ = kind::mutable_lvalue;
    }

    void set_value(const std::mutex&) && noexcept {
      CHECK(kind_ == kind::none);
      kind_ = kind::const_lvalue;
    }

    void set_error(std::exception_ptr) && noexcept {
      CHECK(kind_ == kind::none);
      kind_ = kind::error;
    }

    void set_stopped() && noexcept {
      CHECK(kind_ == kind::none);
      kind_ = kind::stopped;
    }
    enum class kind {
      none,
      rvalue,
      mutable_lvalue,
      const_lvalue,
      error,
      stopped
    };

    constexpr explicit value_category_receiver(kind& k) noexcept
      : kind_(k) {
    }
   private:
    kind& kind_;
  };

  TEST_CASE(
    "When the operation declares separate rvalue and lvalue completion signatures they are "
    "appropriately passed through",
    "[asioexec][completion_token]") {
    const auto initiating_function = [](const bool rvalue, auto&& token) {
      return asio_impl::async_initiate<decltype(token), void(std::mutex&&), void(std::mutex&)>(
        [rvalue](auto&& h) {
          std::mutex m;
          if (rvalue) {
            std::invoke(std::forward<decltype(h)>(h), std::move(m));
          } else {
            std::invoke(std::forward<decltype(h)>(h), m);
          }
        },
        token);
    };
    value_category_receiver::kind rvalue_kind{value_category_receiver::kind::none};
    value_category_receiver::kind lvalue_kind{value_category_receiver::kind::none};
    auto rvalue = connect_shared(
      initiating_function(true, completion_token), value_category_receiver(rvalue_kind));
    auto lvalue = connect_shared(
      initiating_function(false, completion_token), value_category_receiver(lvalue_kind));
    CHECK(rvalue_kind == value_category_receiver::kind::none);
    start_shared(std::move(rvalue));
    CHECK(rvalue_kind == value_category_receiver::kind::rvalue);
    CHECK(lvalue_kind == value_category_receiver::kind::none);
    start_shared(std::move(lvalue));
    CHECK(lvalue_kind == value_category_receiver::kind::mutable_lvalue);
  }

  TEST_CASE(
    "When the operation declares separate rvalue and const lvalue completion signatures they are "
    "appropriately passed through even if the lvalue is sent mutable",
    "[asioexec][completion_token]") {
    const auto initiating_function = [](const bool rvalue, auto&& token) {
      return asio_impl::async_initiate<decltype(token), void(std::mutex&&), void(const std::mutex&)>(
        [rvalue](auto&& h) {
          std::mutex m;
          if (rvalue) {
            std::invoke(std::forward<decltype(h)>(h), std::move(m));
          } else {
            std::invoke(std::forward<decltype(h)>(h), m);
          }
        },
        token);
    };
    value_category_receiver::kind rvalue_kind{value_category_receiver::kind::none};
    value_category_receiver::kind lvalue_kind{value_category_receiver::kind::none};
    auto rvalue = connect_shared(
      initiating_function(true, completion_token), value_category_receiver(rvalue_kind));
    auto lvalue = connect_shared(
      initiating_function(false, completion_token), value_category_receiver(lvalue_kind));
    CHECK(rvalue_kind == value_category_receiver::kind::none);
    start_shared(std::move(rvalue));
    CHECK(rvalue_kind == value_category_receiver::kind::rvalue);
    CHECK(lvalue_kind == value_category_receiver::kind::none);
    start_shared(std::move(lvalue));
    CHECK(lvalue_kind == value_category_receiver::kind::const_lvalue);
  }

  TEST_CASE(
    "Lvalue invocation of the completion handler is supported (for compatibility with Boost.Beast "
    "perhaps among others)",
    "[asioexec][completion_token]") {
    const auto initiating_function = [](auto&& token) {
      return asio_impl::async_initiate<decltype(token), void()>([](auto&& h) { h(); }, token);
    };
    auto op = ::STDEXEC::connect(initiating_function(completion_token), expect_value_receiver{});
    ::STDEXEC::start(op);
  }

  TEST_CASE(
    "Cancellation is delivered thread safely even when emitted thread unsafely",
    "[asioexec][use_sender]") {
    bool cancelled = false;
    bool thread_safe = false;
    std::mutex m;
    asio_impl::io_context ctx;
    std::barrier barrier(2);
    const auto initiating_function = [&]<typename CompletionToken>(CompletionToken&& token) {
      return asio_impl::async_initiate<CompletionToken, void()>(
        [&](auto h) {
          asio_impl::get_associated_cancellation_slot(h).assign([&](auto&&...) noexcept {
            const std::unique_lock l(m, std::try_to_lock);
            thread_safe = bool(l);
            cancelled = true;
          });
          const auto ex = asio_impl::get_associated_executor(h, ctx.get_executor());
          asio_impl::post(ex, [&, h = std::move(h)]() mutable {
            auto local = std::move(h);
            (void) local;
            const std::lock_guard l(m);
            barrier.arrive_and_wait();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
          });
        },
        token);
    };
    ::STDEXEC::inplace_stop_source source;
    auto sender = initiating_function(completion_token);
    {
      auto ptr = connect_shared(
        std::move(sender),
        expect_stopped_receiver(::STDEXEC::prop{::STDEXEC::get_stop_token, source.get_token()}));
      std::thread t([&]() noexcept {
        barrier.arrive_and_wait();
        source.request_stop();
      });
      start_shared(std::move(ptr));
      CHECK(ctx.run() != 0);
      //  Just in case
      (void) barrier.arrive();
      t.join();
      CHECK(cancelled);
      CHECK(thread_safe);
    }
  }

  TEST_CASE("Upon completion the stop token is no longer in use", "[asioexec][use_sender]") {
    bool cancelled = false;
    ::STDEXEC::inplace_stop_source source;
    asio_impl::io_context ctx;
    const auto initiating_function = [&]<typename CompletionToken>(CompletionToken&& token) {
      return asio_impl::async_initiate<CompletionToken, void()>(
        [&](auto h) {
          asio_impl::get_associated_cancellation_slot(h)
            .assign([&](auto&&...) noexcept { cancelled = true; });
          const auto ex = asio_impl::get_associated_executor(h, ctx.get_executor());
          asio_impl::post(ex, std::move(h));
        },
        token);
    };
    auto sender = initiating_function(completion_token);
    auto op = ::STDEXEC::connect(
      std::move(sender),
      expect_void_receiver(::STDEXEC::prop{::STDEXEC::get_stop_token, source.get_token()}));
    ::STDEXEC::start(op);
    CHECK(ctx.run() != 0);
    CHECK(ctx.stopped());
    CHECK(!cancelled);
    source.request_stop();
    CHECK(!cancelled);
  }

  TEST_CASE("Upon abandonment the stop token is no longer in use", "[asioexec][use_sender]") {
    bool cancelled = false;
    ::STDEXEC::inplace_stop_source source;
    asio_impl::io_context ctx;
    const auto initiating_function = [&]<typename CompletionToken>(CompletionToken&& token) {
      return asio_impl::async_initiate<CompletionToken, void()>(
        [&](auto h) {
          asio_impl::get_associated_cancellation_slot(h)
            .assign([&](auto&&...) noexcept { cancelled = true; });
          const auto ex = asio_impl::get_associated_executor(h, ctx.get_executor());
          asio_impl::post(ex, [h = std::move(h)]() mutable {
            auto local = std::move(h);
            (void) local;
          });
        },
        token);
    };
    auto sender = initiating_function(completion_token);
    auto op = ::STDEXEC::connect(
      std::move(sender),
      expect_stopped_receiver(::STDEXEC::prop{::STDEXEC::get_stop_token, source.get_token()}));
    ::STDEXEC::start(op);
    CHECK(ctx.run() != 0);
    CHECK(ctx.stopped());
    CHECK(!cancelled);
    source.request_stop();
    CHECK(!cancelled);
  }

  TEST_CASE(
    "Upon abandonment within the initiating function the stop token is no longer in use",
    "[asioexec][use_sender]") {
    bool cancelled = false;
    ::STDEXEC::inplace_stop_source source;
    asio_impl::io_context ctx;
    const auto initiating_function = [&]<typename CompletionToken>(CompletionToken&& token) {
      return asio_impl::async_initiate<CompletionToken, void()>(
        [&](auto h) {
          asio_impl::get_associated_cancellation_slot(h)
            .assign([&](auto&&...) noexcept { cancelled = true; });
        },
        token);
    };
    auto sender = initiating_function(completion_token);
    auto op = ::STDEXEC::connect(
      std::move(sender),
      expect_stopped_receiver(::STDEXEC::prop{::STDEXEC::get_stop_token, source.get_token()}));
    ::STDEXEC::start(op);
    CHECK(ctx.run() == 0);
    CHECK(ctx.stopped());
    CHECK(!cancelled);
    source.request_stop();
    CHECK(!cancelled);
  }

  TEST_CASE("Upon exception the stop token is no longer in use", "[asioexec][use_sender]") {
    bool cancelled = false;
    std::exception_ptr ex;
    ::STDEXEC::inplace_stop_source source;
    asio_impl::io_context ctx;
    const auto initiating_function = [&]<typename CompletionToken>(CompletionToken&& token) {
      return asio_impl::async_initiate<CompletionToken, void()>(
        [&](auto h) {
          asio_impl::get_associated_cancellation_slot(h)
            .assign([&](auto&&...) noexcept { cancelled = true; });
          const auto ex = asio_impl::get_associated_executor(h, ctx.get_executor());
          asio_impl::post(ex, [h = std::move(h)]() mutable { throw std::logic_error("Test"); });
        },
        token);
    };
    auto sender = initiating_function(completion_token);
    auto op = ::STDEXEC::connect(
      std::move(sender),
      expect_error_receiver_ex(::STDEXEC::prop{::STDEXEC::get_stop_token, source.get_token()}, ex));
    ::STDEXEC::start(op);
    CHECK(!ex);
    CHECK(ctx.run() != 0);
    CHECK(ctx.stopped());
    CHECK(!cancelled);
    CHECK(ex);
    source.request_stop();
    CHECK(!cancelled);
  }

  TEST_CASE(
    "Upon exception from the initiating function the stop token is no longer in use",
    "[asioexec][use_sender]") {
    bool cancelled = false;
    std::exception_ptr ex;
    ::STDEXEC::inplace_stop_source source;
    asio_impl::io_context ctx;
    const auto initiating_function = [&]<typename CompletionToken>(CompletionToken&& token) {
      return asio_impl::async_initiate<CompletionToken, void()>(
        [&](auto h) {
          asio_impl::get_associated_cancellation_slot(h)
            .assign([&](auto&&...) noexcept { cancelled = true; });
          throw std::logic_error("Test");
        },
        token);
    };
    auto sender = initiating_function(completion_token);
    auto op = ::STDEXEC::connect(
      std::move(sender),
      expect_error_receiver_ex(::STDEXEC::prop{::STDEXEC::get_stop_token, source.get_token()}, ex));
    ::STDEXEC::start(op);
    CHECK(ex);
    CHECK(!cancelled);
    source.request_stop();
    CHECK(!cancelled);
  }

  TEST_CASE(
    "When the initiating function starts an operation and then throws an exception the operation "
    "which was started may be cancelled, completion is deferred thereuntil, and the thrown "
    "exception is sent on the error channel",
    "[asioexec][use_sender]") {
    std::exception_ptr ex;
    ::STDEXEC::inplace_stop_source source;
    asio_impl::io_context ctx;
    asio_impl::system_timer t(ctx);
    const auto initiating_function = [&]<typename CompletionToken>(CompletionToken&& token) {
      return asio_impl::async_initiate<CompletionToken, void()>(
        [&](auto h) {
          t.expires_after(std::chrono::years(1));
          const auto ex = asio_impl::get_associated_executor(h, t.get_executor());
          const auto slot = asio_impl::get_associated_cancellation_slot(h);
          t.async_wait(
            asio_impl::bind_executor(
              ex, asio_impl::bind_cancellation_slot(slot, [h = std::move(h)](auto&&...) mutable {
                std::move(h)();
              })));
          throw std::logic_error("Test");
        },
        token);
    };
    auto sender = initiating_function(completion_token);
    auto ptr = connect_shared(
      std::move(sender),
      expect_error_receiver_ex(::STDEXEC::prop{::STDEXEC::get_stop_token, source.get_token()}, ex));
    start_shared(std::move(ptr));
    CHECK(ctx.poll() == 0);
    CHECK(!ctx.stopped());
    CHECK(!ex);
    source.request_stop();
    CHECK(ctx.poll() != 0);
    CHECK(ctx.stopped());
    CHECK(ex);
  }

  TEST_CASE(
    "I/O objects may be transformed to use senders as their default vocabulary with only minimal "
    "transformations (i.e. no error adaptation)",
    "[asioexec][completion_token]") {
    bool invoked = false;
    asio_impl::io_context ctx;
    auto t = completion_token.as_default_on(asio_impl::system_timer(ctx));
    static_assert(
      std::is_same_v<decltype(t), completion_token_t::as_default_on_t<asio_impl::system_timer>>);
    t.expires_after(std::chrono::milliseconds(5));
    auto op = ::STDEXEC::connect(
      t.async_wait() | ::STDEXEC::then([](auto ec) { CHECK(!ec); }),
      expect_void_receiver_ex(invoked));
    ::STDEXEC::start(op);
    CHECK(ctx.run() != 0);
    CHECK(ctx.stopped());
  }

  TEST_CASE(
    "Substitution into async_result<completion_token, ...>::initiate is SFINAE-friendly",
    "[asioexec][completion_token]") {
    asio_impl::io_context ctx;
    asio_impl::ip::tcp::socket socket(ctx);
    asio_impl::streambuf buf;
    //  With a SFINAE-unfriendly async_result<...>::initiate the below line doesn't compile because there's a hard compilation error trying to consider the async_read overload for dynamic buffers
    //
    //  See: https://github.com/NVIDIA/stdexec/issues/1684
    auto sender = asio_impl::async_read(socket, buf, completion_token);
    auto op = ::STDEXEC::connect(
      std::move(sender) | ::STDEXEC::then([](const auto ec, const auto bytes_transferred) {
        CHECK(ec);
        CHECK(!bytes_transferred);
      }),
      expect_void_receiver{});
    ::STDEXEC::start(op);
    CHECK(ctx.run() != 0);
    CHECK(ctx.stopped());
  }

  TEST_CASE(
    "Asio operations which declare completion by value but send a mutable lvalue work",
    "[asioexec][completion_token]") {
    const auto initiating_function = [](auto&& token) {
      return asio_impl::async_initiate<decltype(token), void(int)>(
        [](auto&& h) {
          int i = 5;
          std::forward<decltype(h)>(h)(i);
        },
        token);
    };
    auto op = ::STDEXEC::connect(initiating_function(completion_token), expect_value_receiver{5});
    ::STDEXEC::start(op);
  }

} // namespace
