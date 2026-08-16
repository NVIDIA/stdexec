/*
 * Copyright (c) 2023 Maikel Nadolski
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

#include <exec/single_thread_context.hpp>
#include <exec/when_any.hpp>
#include <numbers>
#include <test_common/receivers.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/senders.hpp>
#include <test_common/type_helpers.hpp>

#include <test_common/catch2.hpp>

namespace ex = STDEXEC;

using namespace STDEXEC;

namespace
{

  struct error_ref_sender
  {
    struct error
    {};

    using sender_concept = ex::sender_tag;

    template <class...>
    static consteval auto get_completion_signatures() noexcept
      -> ex::completion_signatures<ex::set_error_t(error&)>
    {
      return {};
    }

    template <class Receiver>
    struct operation
    {
      Receiver rcvr_;
      error    error_;

      void start() & noexcept
      {
        ex::set_error(static_cast<Receiver&&>(rcvr_), error_);
      }
    };

    template <class Receiver>
    auto connect(Receiver rcvr) && noexcept -> operation<Receiver>
    {
      return {static_cast<Receiver&&>(rcvr), {}};
    }
  };

  struct error_ref_receiver
  {
    using receiver_concept = ex::receiver_tag;

    bool* lvalue_error_;

    void set_value() noexcept {}
    void set_stopped() noexcept {}

    void set_error(error_ref_sender::error&) noexcept
    {
      *lvalue_error_ = true;
    }

    void set_error(error_ref_sender::error&&) noexcept
    {
      *lvalue_error_ = false;
    }

    auto get_env() const noexcept -> ex::env<>
    {
      return {};
    }
  };

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  struct copy_noexcept_move_throws
  {
    copy_noexcept_move_throws() = default;

    copy_noexcept_move_throws(copy_noexcept_move_throws const&) noexcept = default;

    copy_noexcept_move_throws(copy_noexcept_move_throws&&) noexcept(false) {}
  };

  struct throwing_move_error_sender
  {
    using sender_concept = ex::sender_tag;

    template <class...>
    static consteval auto get_completion_signatures() noexcept
      -> ex::completion_signatures<ex::set_error_t(copy_noexcept_move_throws&)>
    {
      return {};
    }

    template <class Receiver>
    struct operation
    {
      Receiver                   rcvr_;
      copy_noexcept_move_throws error_;

      void start() & noexcept
      {
        ex::set_error(static_cast<Receiver&&>(rcvr_), error_);
      }
    };

    template <class Receiver>
    auto connect(Receiver rcvr) && noexcept -> operation<Receiver>
    {
      return {static_cast<Receiver&&>(rcvr), {}};
    }
  };

  template <class Type>
  struct throwing_move_receiver
  {
    using receiver_concept = ex::receiver_tag;

    bool* got_value_;
    bool* got_error_;
    bool* got_exception_;

    void set_value(Type&&) noexcept
    {
      *got_value_ = true;
    }

    void set_error(Type&&) noexcept
    {
      *got_error_ = true;
    }

    void set_error(std::exception_ptr) noexcept
    {
      *got_exception_ = true;
    }

    void set_stopped() noexcept {}

    auto get_env() const noexcept -> ex::env<>
    {
      return {};
    }
  };

  struct error_copy_error
  {};

  struct throwing_error
  {
    explicit throwing_error(bool* throw_on_copy) noexcept
      : throw_on_copy_{throw_on_copy}
    {}

    throwing_error(throwing_error const& other)
      : throw_on_copy_{other.throw_on_copy_}
    {
      if (*throw_on_copy_)
      {
        throw error_copy_error{};
      }
    }

    throwing_error(throwing_error&&) noexcept = default;

    bool* throw_on_copy_;
  };

  struct throwing_error_sender
  {
    using sender_concept = ex::sender_tag;

    bool* throw_on_copy_;

    template <class...>
    static consteval auto get_completion_signatures() noexcept
      -> ex::completion_signatures<ex::set_error_t(throwing_error&)>
    {
      return {};
    }

    template <class Receiver>
    struct operation
    {
      Receiver       rcvr_;
      throwing_error error_;

      void start() & noexcept
      {
        ex::set_error(static_cast<Receiver&&>(rcvr_), error_);
      }
    };

    template <class Receiver>
    auto connect(Receiver rcvr) && noexcept -> operation<Receiver>
    {
      return {static_cast<Receiver&&>(rcvr), throwing_error{throw_on_copy_}};
    }
  };

  struct throwing_error_receiver
  {
    using receiver_concept = ex::receiver_tag;

    bool* got_exception_;

    void set_error(throwing_error) noexcept {}

    void set_error(std::exception_ptr) noexcept
    {
      *got_exception_ = true;
    }

    void set_stopped() noexcept {}

    auto get_env() const noexcept -> ex::env<>
    {
      return {};
    }
  };
#endif  // !STDEXEC_NO_STDCPP_EXCEPTIONS()

  TEST_CASE("when_ny returns a sender", "[adaptors][when_any]")
  {
    auto snd = exec::when_any(ex::just(3), ex::just(0.1415));
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("when_any with environment returns a sender", "[adaptors][when_any]")
  {
    auto snd = exec::when_any(ex::just(3), ex::just(0.1415));
    static_assert(ex::sender_in<decltype(snd), ex::env<>>);
    (void) snd;
  }

  TEST_CASE("when_any simple example", "[adaptors][when_any]")
  {
    auto         snd      = exec::when_any(ex::just(3.0));
    auto         snd1     = std::move(snd) | ex::then([](double y) { return y + 0.1415; });
    double const expected = 3.0 + 0.1415;
    auto         op       = ex::connect(std::move(snd1), expect_value_receiver{expected});
    ex::start(op);
  }

  TEST_CASE("when_any completes with only one sender", "[adaptors][when_any]")
  {
    ex::sender auto snd = exec::when_any(completes_if{false} | ex::then([] { return 1; }),
                                         completes_if{true} | ex::then([] { return 42; }));
    wait_for_value(std::move(snd), 42);

    ex::sender auto snd2 = exec::when_any(completes_if{true} | ex::then([] { return 1; }),
                                          completes_if{false} | ex::then([] { return 42; }));
    wait_for_value(std::move(snd2), 1);
  }

  TEST_CASE("when_any with move-only types", "[adaptors][when_any]")
  {
    ex::sender auto snd = exec::when_any(completes_if{false} | ex::then([] { return movable(1); }),
                                         ex::just(movable(42)));
    wait_for_value(std::move(snd), movable(42));
  }

  TEST_CASE("when_any forwards stop signal", "[adaptors][when_any]")
  {
    stopped_scheduler stop;
    int               result = 42;
    ex::sender auto   snd    = exec::when_any(completes_if{false}, ex::schedule(stop))
                        | ex::then([&result] { result += 1; });
    ex::sync_wait(std::move(snd));
    REQUIRE(result == 42);
  }

  TEST_CASE("nested when_any is stoppable", "[adaptors][when_any]")
  {
    int             result = 41;
    ex::sender auto snd = exec::when_any(exec::when_any(completes_if{false}, completes_if{false}),
                                         completes_if{false},
                                         ex::just(),
                                         completes_if{false})
                        | ex::then([&result] { result += 1; });
    ex::sync_wait(std::move(snd));
    REQUIRE(result == 42);
  }

  TEST_CASE("stop is forwarded", "[adaptors][when_any]")
  {
    int             result = 41;
    ex::sender auto snd    = exec::when_any(ex::just_stopped(), completes_if{false})
                        | ex::upon_stopped([&result] { result += 1; });
    ex::sync_wait(std::move(snd));
    REQUIRE(result == 42);
  }

  TEST_CASE("when_any is thread-safe", "[adaptors][when_any]")
  {
    exec::single_thread_context ctx1;
    exec::single_thread_context ctx2;
    exec::single_thread_context ctx3;

    auto sch1 = ex::schedule(ctx1.get_scheduler());
    auto sch2 = ex::schedule(ctx2.get_scheduler());
    auto sch3 = ex::schedule(ctx3.get_scheduler());

    int result = 41;

    ex::sender auto snd =
      exec::when_any(sch1 | ex::let_value([] { return exec::when_any(completes_if{false}); }),
                     sch2 | ex::let_value([] { return completes_if{false}; }),
                     sch3 | ex::then([&result] { result += 1; }),
                     completes_if{false});

    ex::sync_wait(std::move(snd));
    REQUIRE(result == 42);
  }

  TEST_CASE("when_any completion signatures", "[adaptors][when_any]")
  {
    struct move_throws
    {
      move_throws() = default;

      move_throws(move_throws&&) noexcept(false) {}

      auto operator=(move_throws&&) noexcept(false) -> move_throws&
      {
        return *this;
      }
    };

    auto just = exec::when_any(ex::just());
    static_assert(sender<decltype(just)>);
    static_assert(set_equivalent<completion_signatures_of_t<decltype(just)>,
                                 completion_signatures<set_value_t(), set_stopped_t()>>);

    auto just_string = exec::when_any(ex::just(std::string("foo")));
    static_assert(sender<decltype(just_string)>);
    static_assert(set_equivalent<completion_signatures_of_t<decltype(just_string)>,
                                 completion_signatures<set_value_t(std::string), set_stopped_t()>>);

    auto just_stopped = exec::when_any(ex::just_stopped());
    static_assert(sender<decltype(just_stopped)>);
    static_assert(set_equivalent<completion_signatures_of_t<decltype(just_stopped)>,
                                 completion_signatures<set_stopped_t()>>);

    auto just_then = exec::when_any(ex::just() | ex::then([] { return 42; }));
    static_assert(sender<decltype(just_then)>);
    static_assert(
      set_equivalent<
        completion_signatures_of_t<decltype(just_then)>,
        completion_signatures<set_value_t(int), set_stopped_t(), set_error_t(std::exception_ptr)>>);

    auto just_then_noexcept = exec::when_any(ex::just() | ex::then([]() noexcept { return 42; }));
    static_assert(sender<decltype(just_then_noexcept)>);
    static_assert(set_equivalent<completion_signatures_of_t<decltype(just_then_noexcept)>,
                                 completion_signatures<set_value_t(int), set_stopped_t()>>);

    auto just_move_throws = exec::when_any(ex::just(move_throws{}));
    static_assert(sender<decltype(just_move_throws)>);
    static_assert(set_equivalent<completion_signatures_of_t<decltype(just_move_throws)>,
                                 completion_signatures<set_value_t(move_throws),
                                                       set_stopped_t(),
                                                       set_error_t(std::exception_ptr)>>);

    auto mulitple_senders = exec::when_any(ex::just(std::numbers::pi),
                                           ex::just(std::string()),
                                           ex::just(std::string()),
                                           ex::just() | ex::then([] { return 42; }),
                                           ex::just() | ex::then([] { return 42; }));
    static_assert(sender<decltype(mulitple_senders)>);
    static_assert(set_equivalent<completion_signatures_of_t<decltype(mulitple_senders)>,
                                 completion_signatures<set_value_t(double),
                                                       set_value_t(std::string),
                                                       set_value_t(int),
                                                       set_stopped_t(),
                                                       set_error_t(std::exception_ptr)>>);
    // wait_for_value(std::move(snd), movable(42));
  }

  TEST_CASE("when_any decays error completion arguments", "[adaptors][when_any]")
  {
    auto snd = exec::when_any(error_ref_sender{});
    static_assert(set_equivalent<completion_signatures_of_t<decltype(snd)>,
                                 completion_signatures<set_error_t(error_ref_sender::error),
                                                       set_stopped_t()>>);

    bool lvalue_error = false;
    auto op = ex::connect(std::move(snd), error_ref_receiver{&lvalue_error});
    ex::start(op);
    CHECK_FALSE(lvalue_error);
  }

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  TEST_CASE("when_any reports errors from throwing error decay", "[adaptors][when_any]")
  {
    bool throw_on_copy = false;
    auto snd = exec::when_any(throwing_error_sender{&throw_on_copy});
    static_assert(set_equivalent<completion_signatures_of_t<decltype(snd)>,
                                 completion_signatures<set_error_t(throwing_error),
                                                       set_error_t(std::exception_ptr),
                                                       set_stopped_t()>>);

    bool got_exception = false;
    auto op = ex::connect(std::move(snd), throwing_error_receiver{&got_exception});
    throw_on_copy = true;
    ex::start(op);
    CHECK(got_exception);
  }

  TEST_CASE("when_any reports errors for potentially throwing value moves", "[adaptors][when_any]")
  {
    copy_noexcept_move_throws value;
    auto                    snd = exec::when_any(just_ref{value});
    static_assert(set_equivalent<completion_signatures_of_t<decltype(snd)>,
                                 completion_signatures<set_value_t(copy_noexcept_move_throws),
                                                       set_stopped_t(),
                                                       set_error_t(std::exception_ptr)>>);

    bool got_value     = false;
    bool got_error     = false;
    bool got_exception = false;
    auto op = ex::connect(std::move(snd),
                          throwing_move_receiver<copy_noexcept_move_throws>{
                            &got_value, &got_error, &got_exception});
    ex::start(op);
    CHECK(got_value);
    CHECK_FALSE(got_error);
    CHECK_FALSE(got_exception);
  }

  TEST_CASE("when_any reports errors for potentially throwing error moves", "[adaptors][when_any]")
  {
    auto snd = exec::when_any(throwing_move_error_sender{});
    static_assert(set_equivalent<completion_signatures_of_t<decltype(snd)>,
                                 completion_signatures<set_error_t(copy_noexcept_move_throws),
                                                       set_error_t(std::exception_ptr),
                                                       set_stopped_t()>>);

    bool got_value     = false;
    bool got_error     = false;
    bool got_exception = false;
    auto op = ex::connect(std::move(snd),
                          throwing_move_receiver<copy_noexcept_move_throws>{
                            &got_value, &got_error, &got_exception});
    ex::start(op);
    CHECK_FALSE(got_value);
    CHECK(got_error);
    CHECK_FALSE(got_exception);
  }
#endif  // !STDEXEC_NO_STDCPP_EXCEPTIONS()

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  template <class Receiver>
  struct dup_op
  {
    Receiver rec;

    void start() & noexcept
    {
      STDEXEC::set_error(static_cast<Receiver&&>(rec),
                         std::make_exception_ptr(std::runtime_error("dup")));
    }
  };

  struct dup_sender
  {
    using sender_concept        = STDEXEC::sender_tag;
    using completion_signatures = STDEXEC::completion_signatures<set_value_t(),
                                                                 set_error_t(std::exception_ptr),
                                                                 set_error_t(std::exception_ptr&&)>;

    template <class Receiver>
    auto connect(Receiver rec) const noexcept -> dup_op<Receiver>
    {
      return {static_cast<Receiver&&>(rec)};
    }
  };

  TEST_CASE("when_any - with duplicate completions", "[adaptors][when_any]")
  {
    REQUIRE_THROWS(STDEXEC::sync_wait(exec::when_any(dup_sender{})));
  }
#endif  // !STDEXEC_NO_STDCPP_EXCEPTIONS()
}  // namespace
