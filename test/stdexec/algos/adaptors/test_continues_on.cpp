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
#include <catch2/catch_all.hpp>

#include <stdexec/execution.hpp>

#include <exec/sender_for.hpp>
#include <exec/start_detached.hpp>
#include <exec/static_thread_pool.hpp>
#include <test_common/non_completion_query.hpp>
#include <test_common/receivers.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/type_helpers.hpp>

#if STDEXEC_USE_MODULES()
import std;
#else
#  include <chrono>  // IWYU pragma: keep for chrono_literals
#endif

namespace ex = STDEXEC;

using namespace std::chrono_literals;

namespace
{
  template <class Sender>
  using value_type_of_t =
    ex::value_types_of_t<Sender, ex::env<>, std::type_identity_t, std::type_identity_t>;

  struct potentially_throwing_connect_scheduler
  {
    using scheduler_concept = ex::scheduler_tag;

    struct sender
    {
      using sender_concept        = ex::sender_tag;
      using completion_signatures = ex::completion_signatures<ex::set_value_t()>;

      template <class Receiver>
      struct opstate
      {
        void start() & noexcept
        {
          ex::set_value(static_cast<Receiver &&>(receiver_));
        }

        Receiver receiver_;
      };

      template <class Receiver>
      [[nodiscard, maybe_unused]]
      auto connect(Receiver receiver) const noexcept(false) -> opstate<Receiver>
      {
        return {static_cast<Receiver &&>(receiver)};
      }
    };

    [[nodiscard]]
    auto schedule() const noexcept -> sender
    {
      return {};
    }

    auto
    operator==(potentially_throwing_connect_scheduler const &) const noexcept -> bool = default;
  };

  struct value_type
  {
    int val_;
  };

  struct continues_on_test_domain
  {
    template <exec::sender_for<ex::continues_on_t> Sender>
    static auto transform_sender(STDEXEC::set_value_t, Sender &&, auto const &...)
    {
      return ex::just(value_type{53});
    }
  };

  TEST_CASE("continues_on returns a sender", "[adaptors][continues_on]")
  {
    auto snd = ex::continues_on(ex::just(13), inline_scheduler{});
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("continues_on forwards non-completion queries", "[adaptors][continues_on]")
  {
    auto snd = ex::continues_on(test_common::non_completion_query_sender{}, inline_scheduler{});
    CHECK(test_common::non_completion_query(ex::get_env(snd), 41) == 42);
  }

  TEST_CASE("continues_on with environment returns a sender", "[adaptors][continues_on]")
  {
    auto snd = ex::continues_on(ex::just(13), inline_scheduler{});
    static_assert(ex::sender_in<decltype(snd), ex::env<>>);
    (void) snd;
  }

  TEST_CASE("continues_on is nothrow connectable when the scheduler is", "[adaptors][continues_on]")
  {
    using receiver_t = ex::__receiver_archetype<ex::env<>>;

    auto nothrow = ex::continues_on(ex::just(), inline_scheduler{});
    STATIC_REQUIRE(ex::__nothrow_connectable<decltype(nothrow), receiver_t>);

    auto potentially_throwing = ex::continues_on(ex::just(),
                                                 potentially_throwing_connect_scheduler{});
    STATIC_REQUIRE_FALSE(ex::__nothrow_connectable<decltype(potentially_throwing), receiver_t>);
  }

  TEST_CASE("continues_on simple example", "[adaptors][continues_on]")
  {
    auto snd = ex::continues_on(ex::just(13), inline_scheduler{});
    auto op  = ex::connect(std::move(snd), expect_value_receiver{13});
    ex::start(op);
    // The receiver checks if we receive the right value
  }

  TEST_CASE("continues_on can be piped", "[adaptors][continues_on]")
  {
    // Just transfer a value to the impulse scheduler
    ex::scheduler auto sched = impulse_scheduler{};
    ex::sender auto    snd   = ex::just(13) | ex::continues_on(sched);
    // Start the operation
    int  res{0};
    auto op = ex::connect(std::move(snd), expect_value_receiver_ex{res});
    ex::start(op);

    // The value will be available when the scheduler will execute the next
    // operation
    REQUIRE(res == 0);
    sched.start_next();
    REQUIRE(res == 13);
  }

  TEST_CASE("continues_on calls the receiver when the scheduler dictates",
            "[adaptors][continues_on]")
  {
    int               recv_value{0};
    impulse_scheduler sched;
    auto              snd = ex::continues_on(ex::just(13), sched);
    auto              op  = ex::connect(snd, expect_value_receiver_ex{recv_value});
    ex::start(op);
    // Up until this point, the scheduler didn't start any task; no effect
    // expected
    CHECK(recv_value == 0);

    // Tell the scheduler to start executing one task
    sched.start_next();
    CHECK(recv_value == 13);
  }

  TEST_CASE("continues_on calls the given sender when the scheduler dictates",
            "[adaptors][continues_on]")
  {
    bool called{false};
    auto snd_base = ex::just()
                  | ex::then(
                      [&]() -> int
                      {
                        called = true;
                        return 19;
                      });

    int               recv_value{0};
    impulse_scheduler sched;
    auto              snd = ex::continues_on(std::move(snd_base), sched);
    auto              op  = ex::connect(std::move(snd), expect_value_receiver_ex{recv_value});
    ex::start(op);
    // The sender is started, even if the scheduler hasn't yet triggered
    CHECK(called);
    // ... but didn't send the value to the receiver yet
    CHECK(recv_value == 0);

    // Tell the scheduler to start executing one task
    sched.start_next();

    // Now the base sender is called, and a value is sent to the receiver
    CHECK(called);
    CHECK(recv_value == 19);
  }

  TEST_CASE("continues_on works when changing threads", "[adaptors][continues_on]")
  {
    exec::static_thread_pool pool{2};
    std::atomic<bool>        called{false};
    {
      // lunch some work on the thread pool
      ex::sender auto snd = ex::continues_on(ex::just(), pool.get_scheduler())
                          | ex::then([&] { called.store(true); });
      exec::start_detached(std::move(snd));
    }
    // wait for the work to be executed, with timeout perform a poor-man's sync
    // NOTE: it's a shame that the `join` method in static_thread_pool is not
    // public
    for (int i = 0; i < 1000 && !called.load(); i++)
    {
      std::this_thread::sleep_for(1ms);
    }
    // the work should be executed
    REQUIRE(called);
  }

  TEST_CASE("continues_on can be called with rvalue ref scheduler", "[adaptors][continues_on]")
  {
    auto snd = ex::continues_on(ex::just(13), inline_scheduler{});
    auto op  = ex::connect(std::move(snd), expect_value_receiver{13});
    ex::start(op);
    // The receiver checks if we receive the right value
  }

  TEST_CASE("continues_on can be called with const ref scheduler", "[adaptors][continues_on]")
  {
    inline_scheduler const sched;
    auto                   snd = ex::continues_on(ex::just(13), sched);
    auto                   op  = ex::connect(std::move(snd), expect_value_receiver{13});
    ex::start(op);
    // The receiver checks if we receive the right value
  }

  TEST_CASE("continues_on can be called with ref scheduler", "[adaptors][continues_on]")
  {
    inline_scheduler sched;
    auto             snd = ex::continues_on(ex::just(13), sched);
    auto             op  = ex::connect(std::move(snd), expect_value_receiver{13});
    ex::start(op);
    // The receiver checks if we receive the right value
  }

  TEST_CASE("continues_on forwards set_error calls", "[adaptors][continues_on]")
  {
    error_scheduler<std::exception_ptr> sched{std::exception_ptr{}};
    auto                                snd = ex::continues_on(ex::just(13), sched);
    auto                                op  = ex::connect(std::move(snd), expect_error_receiver{});
    ex::start(op);
    // The receiver checks if we receive an error
  }

  TEST_CASE("continues_on forwards set_error calls of other types", "[adaptors][continues_on]")
  {
    error_scheduler<std::string> sched{std::string{"error"}};
    auto                         snd = ex::continues_on(ex::just(13), sched);
    auto op = ex::connect(std::move(snd), expect_error_receiver{std::string{"error"}});
    ex::start(op);
    // The receiver checks if we receive an error
  }

  TEST_CASE("continues_on forwards set_stopped calls", "[adaptors][continues_on]")
  {
    stopped_scheduler sched{};
    auto              snd = ex::continues_on(ex::just(13), sched);
    auto              op  = ex::connect(std::move(snd), expect_stopped_receiver{});
    ex::start(op);
    // The receiver checks if we receive the stopped signal
  }

  TEST_CASE("continues_on has the values_type corresponding to the given values",
            "[adaptors][continues_on]")
  {
    inline_scheduler sched{};

    check_val_types<ex::__mset<pack<int>>>(ex::continues_on(ex::just(1), sched));
    check_val_types<ex::__mset<pack<int, double>>>(ex::continues_on(ex::just(3, 0.14), sched));
    check_val_types<ex::__mset<pack<int, double, std::string>>>(
      ex::continues_on(ex::just(3, 0.14, std::string{"pi"}), sched));
  }

  TEST_CASE("continues_on keeps error_types from scheduler's sender", "[adaptors][continues_on]")
  {
    inline_scheduler     sched1{};
    error_scheduler      sched2{};
    error_scheduler<int> sched3{43};

    check_err_types<ex::__mset<>>(ex::continues_on(ex::just(1), sched1));
    check_err_types<ex::__mset<std::exception_ptr>>(ex::continues_on(ex::just(2), sched2));
    check_err_types<ex::__mset<int>>(ex::continues_on(ex::just(3), sched3));
  }

  TEST_CASE("continues_on sends an exception_ptr if value types are potentially throwing when "
            "copied",
            "[adaptors][continues_on]")
  {
    inline_scheduler sched{};

    check_err_types<ex::__mset<std::exception_ptr>>(
      ex::continues_on(ex::just(potentially_throwing{}), sched));
  }

  TEST_CASE("continues_on keeps sends_stopped from scheduler's sender", "[adaptors][continues_on]")
  {
    inline_scheduler  sched1{};
    error_scheduler   sched2{};
    stopped_scheduler sched3{};

    check_sends_stopped<false>(ex::continues_on(ex::just(1), sched1));
    check_sends_stopped<true>(ex::continues_on(ex::just(2), sched2));
    check_sends_stopped<true>(ex::continues_on(ex::just(3), sched3));
  }

  TEST_CASE("continues_on keeps updates completion domain for error channel",
            "[adaptors][continues_on]")
  {
    // The following scheduler does not send errors. Any errors are forwarded
    // from the predecessor sender on the scheduler's domain.
    basic_inline_scheduler<continues_on_test_domain> sched{};
    auto snd    = ex::continues_on(ex::just_error(1), sched);
    auto domain = ex::get_env(snd).query(ex::get_completion_domain<ex::set_error_t>);
    static_assert(std::same_as<decltype(domain), continues_on_test_domain>);
  }

  //! A scheduler with runtime identity. Its value completions happen on a scheduler
  //! with the same id. If `ErrorId != 0`, its sender can also complete with an
  //! error, which happens on a scheduler with id `ErrorId`.
  template <class ValueDomain, int ErrorId = 0>
  struct id_scheduler
  {
    using scheduler_concept = ex::scheduler_tag;

    int id_;

    auto operator==(id_scheduler const &) const noexcept -> bool = default;

    [[nodiscard]]
    auto query(ex::get_completion_scheduler_t<ex::set_value_t>) const noexcept -> id_scheduler
    {
      return *this;
    }

    [[nodiscard]]
    auto query(ex::get_completion_scheduler_t<ex::set_error_t>) const noexcept -> id_scheduler
      requires(ErrorId != 0)
    {
      return id_scheduler{id_ + ErrorId};
    }

    [[nodiscard]]
    auto query(ex::get_completion_domain_t<ex::set_value_t>) const noexcept -> ValueDomain
    {
      return {};
    }

    [[nodiscard]]
    auto schedule() const noexcept
    {
      return sender{*this};
    }

    struct attrs
    {
      id_scheduler sched_;

      [[nodiscard]]
      auto query(ex::get_completion_scheduler_t<ex::set_value_t>) const noexcept -> id_scheduler
      {
        return sched_;
      }

      [[nodiscard]]
      auto query(ex::get_completion_scheduler_t<ex::set_error_t>) const noexcept -> id_scheduler
        requires(ErrorId != 0)
      {
        return id_scheduler{sched_.id_ + ErrorId};
      }

      [[nodiscard]]
      auto query(ex::get_completion_domain_t<ex::set_value_t>) const noexcept -> ValueDomain
      {
        return {};
      }
    };

    template <class Receiver>
    struct opstate : immovable
    {
      void start() & noexcept
      {
        ex::set_value(static_cast<Receiver &&>(rcvr_));
      }

      Receiver rcvr_;
    };

    struct sender
    {
      using sender_concept = STDEXEC::sender_tag;
      using completion_signatures =
        std::conditional_t<ErrorId != 0,
                           ex::completion_signatures<ex::set_value_t(), ex::set_error_t(int)>,
                           ex::completion_signatures<ex::set_value_t()>>;

      template <class Receiver>
      [[nodiscard]]
      auto connect(Receiver rcvr) const -> opstate<Receiver>
      {
        return {{}, static_cast<Receiver &&>(rcvr)};
      }

      [[nodiscard]]
      auto get_env() const noexcept -> attrs
      {
        return {sched_};
      }

      id_scheduler sched_;
    };
  };

  template <class Sender, class Tag>
  concept has_completion_scheduler = requires(Sender const &snd) {
    ex::get_completion_scheduler<Tag>(ex::get_env(snd));
  };

  template <class Sender, class Tag>
  concept has_completion_domain = requires(Sender const &snd) {
    ex::get_env(snd).query(ex::get_completion_domain<Tag>);
  };

  TEST_CASE("continues_on: completion scheduler of the value channel is the scheduler",
            "[adaptors][continues_on]")
  {
    id_scheduler<continues_on_test_domain> sched{42};
    auto                                   snd = ex::continues_on(ex::just(1), sched);
    auto csched = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(snd));
    CHECK(csched.id_ == 42);
  }

  TEST_CASE("continues_on: forwarded errors and stopped complete on the scheduler's value "
            "completion scheduler",
            "[adaptors][continues_on]")
  {
    id_scheduler<continues_on_test_domain> sched{42};

    auto err       = ex::continues_on(ex::just_error(1), sched);
    auto err_sched = ex::get_completion_scheduler<ex::set_error_t>(ex::get_env(err));
    CHECK(err_sched.id_ == 42);

    auto stopped       = ex::continues_on(ex::just_stopped(), sched);
    auto stopped_sched = ex::get_completion_scheduler<ex::set_stopped_t>(ex::get_env(stopped));
    CHECK(stopped_sched.id_ == 42);

    // ... and not on the predecessor's completion scheduler
    id_scheduler<continues_on_test_domain> other{7};
    auto                                   on_other = ex::continues_on(ex::just_error(1), other);
    auto                                   snd      = ex::continues_on(std::move(on_other), sched);
    auto snd_sched = ex::get_completion_scheduler<ex::set_error_t>(ex::get_env(snd));
    CHECK(snd_sched.id_ == 42);
  }

  TEST_CASE("continues_on: errors sent only by the scheduler's sender complete on the scheduler's "
            "error completion scheduler",
            "[adaptors][continues_on]")
  {
    id_scheduler<continues_on_test_domain, 100> sched{1};
    auto                                        snd = ex::continues_on(ex::just(), sched);
    auto err_sched = ex::get_completion_scheduler<ex::set_error_t>(ex::get_env(snd));
    CHECK(err_sched.id_ == 101);
    // the value completion scheduler is unaffected
    CHECK(ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(snd)).id_ == 1);
  }

  TEST_CASE("continues_on: no completion scheduler when the schedulers are ambiguous or the "
            "channel is never used",
            "[adaptors][continues_on]")
  {
    // Both the predecessor's errors (forwarded through the value scheduler) and the
    // scheduler's own errors are possible. Which scheduler is used is only known
    // at runtime, so there is no completion scheduler.
    id_scheduler<continues_on_test_domain, 100> err_sched{1};
    auto both = ex::continues_on(ex::just_error(1), err_sched);
    static_assert(!has_completion_scheduler<decltype(both), ex::set_error_t>);
    static_assert(has_completion_scheduler<decltype(both), ex::set_value_t>);

    // Nothing can complete with an error.
    id_scheduler<continues_on_test_domain> sched{1};
    auto                                   none = ex::continues_on(ex::just(), sched);
    static_assert(!has_completion_scheduler<decltype(none), ex::set_error_t>);
    static_assert(!has_completion_scheduler<decltype(none), ex::set_stopped_t>);
  }

  TEST_CASE("continues_on: forwarded stopped completes in the scheduler's value domain",
            "[adaptors][continues_on]")
  {
    id_scheduler<continues_on_test_domain> sched{1};
    auto                                   snd = ex::continues_on(ex::just_stopped(), sched);
    auto domain = ex::get_env(snd).query(ex::get_completion_domain<ex::set_stopped_t>);
    static_assert(std::same_as<decltype(domain), continues_on_test_domain>);
  }

  TEST_CASE("continues_on: there is no error completion domain if nothing can send errors",
            "[adaptors][continues_on]")
  {
    id_scheduler<continues_on_test_domain> sched{1};
    auto                                   snd = ex::continues_on(ex::just(), sched);
    static_assert(!has_completion_domain<decltype(snd), ex::set_error_t>);
    static_assert(has_completion_domain<decltype(snd), ex::set_value_t>);
  }

  TEST_CASE("continues_on can be customized", "[adaptors][continues_on]")
  {
    // The customization will return a different value
    basic_inline_scheduler<continues_on_test_domain> sched;
    auto       snd = ex::continues_on(ex::just(value_type{1}), sched);
    value_type res{0};
    auto       op = ex::connect(std::move(snd), expect_value_receiver_ex{res});
    ex::start(op);
    REQUIRE(res.val_ == 53);
  }

  struct test_domain_A
  {
    template <exec::sender_for<ex::continues_on_t> Sender, class Env>
    [[maybe_unused]]
    auto transform_sender(STDEXEC::set_value_t, Sender &&, Env &&) const
    {
      return ex::just(std::string("hello"));
    }
  };

  struct test_domain_B
  {
    template <exec::sender_for<ex::continues_on_t> Sender, class Env>
    auto transform_sender(STDEXEC::set_value_t, Sender &&, Env &&) const
    {
      return ex::just(std::string("goodbye"));
    }
  };

  TEST_CASE("continues_on late customization is based on the scheduler's domain",
            "[adaptors][continues_on]")
  {
    // The customization will return a different value
    ex::scheduler auto sched_A = basic_inline_scheduler<test_domain_A>{};
    ex::scheduler auto sched_B = basic_inline_scheduler<test_domain_B>{};
    auto               snd     = ex::just(std::string("meow"))  //
             | ex::continues_on(sched_A)                        //
             | ex::continues_on(sched_B);
    std::string res;
    auto        op = ex::connect(std::move(snd), expect_value_receiver_ex{res});
    ex::start(op);
    REQUIRE(res == "goodbye");
  }
}  // namespace
