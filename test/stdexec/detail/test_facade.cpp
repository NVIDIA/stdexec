/*
 * Copyright (c) 2026 NVIDIA Corporation
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

// Tests for the sender facade in <stdexec/__detail/__sender_facade.hpp>, through algorithms
// that are written with it: some for the tests, and the library's own. None of them has
// hand-written completion signatures or hand-written completion attributes.

// Catch2 must come before any potential import statements
#include <catch2/catch_all.hpp>

#include <stdexec/__detail/__sender_facade.hpp>
#include <stdexec/execution.hpp>

#if STDEXEC_USE_MODULES()
import std;
#else
#  include <exception>
#  include <optional>
#  include <string>
#  include <thread>
#  include <tuple>
#  include <type_traits>
#  include <utility>
#  include <variant>
#endif

namespace ex = STDEXEC;

namespace
{
  // The tag of a facade algorithm that is written for the tests
  template <class Algo>
  struct test_tag
  {};
}  // namespace

template <class Algo>
struct STDEXEC::__facade::__algorithm<test_tag<Algo>> : Algo
{};

namespace
{
  template <class Algo, class Data, class... Children>
  using test_sender_t = ex::__facade::__sender<test_tag<Algo>, Data, Children...>;

  using cb = ex::__completion_behavior;

  //////////////////////////////////////////////////////////////////////////////////////////
  // Probe senders whose channels have different attributes. (They are two senders because
  // a sender that completes inline completes on its start scheduler, so it cannot also
  // claim an unrelated domain; stdexec checks this.)
  struct value_domain : ex::default_domain
  {};

  struct error_domain : ex::default_domain
  {};

  // Value completions happen in value_domain; error completions in error_domain.
  struct domain_attrs
  {
    constexpr auto query(ex::get_completion_domain_t<ex::set_value_t>) const noexcept
    {
      return value_domain{};
    }

    constexpr auto query(ex::get_completion_domain_t<ex::set_error_t>) const noexcept
    {
      return error_domain{};
    }
  };

  // Value completions are asynchronous; error completions are inline.
  struct behavior_attrs
  {
    constexpr auto query(ex::__get_completion_behavior_t<ex::set_value_t>) const noexcept
    {
      return cb::__asynchronous;
    }

    constexpr auto query(ex::__get_completion_behavior_t<ex::set_error_t>) const noexcept
    {
      return cb::__inline_completion;
    }
  };

  template <class Attrs>
  struct probe
  {
    using sender_concept = ex::sender_tag;

    template <class Self, class... Env>
    static consteval auto get_completion_signatures()
    {
      return ex::completion_signatures<ex::set_value_t(), ex::set_error_t(int)>();
    }

    template <class Rcvr>
    struct opstate
    {
      using operation_state_concept = ex::operation_state_tag;

      void start() & noexcept
      {
        ex::set_error(static_cast<Rcvr &&>(rcvr), 42);
      }

      Rcvr rcvr;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) const -> opstate<Rcvr>
    {
      return {static_cast<Rcvr &&>(rcvr)};
    }

    auto get_env() const noexcept -> Attrs
    {
      return {};
    }
  };

  using domain_probe   = probe<domain_attrs>;
  using behavior_probe = probe<behavior_attrs>;

  //////////////////////////////////////////////////////////////////////////////////////////
  // Helpers
  template <class Sigs, class... Others>
  inline constexpr bool sigs_subset_v = false;

  template <class... Sigs, class... Others>
  inline constexpr bool sigs_subset_v<ex::completion_signatures<Sigs...>, Others...> =
    (ex::__one_of<Sigs, Others...> && ...);

  template <class Sigs1, class Sigs2>
  inline constexpr bool same_sigs_v = false;

  template <class... Sigs1, class... Sigs2>
  inline constexpr bool
    same_sigs_v<ex::completion_signatures<Sigs1...>, ex::completion_signatures<Sigs2...>> =
      sigs_subset_v<ex::completion_signatures<Sigs1...>, Sigs2...>
      && sigs_subset_v<ex::completion_signatures<Sigs2...>, Sigs1...>;

  template <class Sndr, class... Sigs>
  inline constexpr bool has_sigs_v = same_sigs_v<ex::completion_signatures_of_t<Sndr, ex::env<>>,
                                                 ex::completion_signatures<Sigs...>>;

  template <class Sndr, class Tag, class... Env>
  using domain_of_t = decltype(ex::get_completion_domain<Tag>(ex::get_env(std::declval<Sndr>()),
                                                              std::declval<Env>()...));

  template <class Sndr, class Tag, class... Env>
  inline constexpr auto behavior_of_v = ex::__get_completion_behavior<Tag, Sndr, Env...>();

  template <class Sndr, class Tag, class... Env>
  concept has_completion_scheduler = requires(Sndr const &sndr, Env const &...env) {
    ex::get_completion_scheduler<Tag>(ex::get_env(sndr), env...);
  };

  // Is Domain indeterminate_domain<Ds...> for exactly the given set of domains?
  template <class Domain, class... Ds>
  inline constexpr bool is_indeterminate_of_v = false;

  template <class... Ds1, class... Ds2>
  inline constexpr bool is_indeterminate_of_v<ex::indeterminate_domain<Ds1...>, Ds2...> =
    sizeof...(Ds1) == sizeof...(Ds2) && (ex::__one_of<Ds1, Ds2...> && ...);

  using eptr_sig_t     = ex::set_error_t(std::exception_ptr);
  using run_loop_sch_t = decltype(std::declval<ex::run_loop &>().get_scheduler());
  using start_env_t = decltype(ex::prop{ex::get_start_scheduler, std::declval<run_loop_sch_t>()});
  using stop_env_t  = ex::env<ex::prop<ex::get_stop_token_t, ex::inplace_stop_token>,
                              ex::prop<ex::get_start_scheduler_t, run_loop_sch_t>>;

  // A function that may throw, and one that may not:
  inline constexpr auto may_throw = [](auto...)
  {
    return 42;
  };
  inline constexpr auto no_throw = [](auto...) noexcept
  {
    return 42;
  };

  //////////////////////////////////////////////////////////////////////////////////////////
  // 0. Algorithms written for the tests: t_just(vs...) completes from start(),
  // t_then(sndr, fn) transforms the value completions of its child, and t_schedule(sch)
  // completes from a nested schedule operation.
  struct t_just_algo : ex::__facade::__defaults
  {
    template <class State, class Children>
    static constexpr auto __start(State &state, Children &) noexcept
    {
      return ex::__apply(
        [&]<class... Ts>(Ts &&...vs) noexcept
        { return ex::__facade::__send(state.__rcvr_, ex::set_value, static_cast<Ts &&>(vs)...); },
        std::move(state.__data_));
    }
  };

  struct t_then_algo : ex::__facade::__defaults
  {
    template <class State, class Tag, class... Args>
    static constexpr auto
    __complete(ex::__facade::__child<0>, State &state, Tag tag, Args &&...args) noexcept
    {
      if constexpr (std::is_same_v<Tag, ex::set_value_t>)
        return ex::__facade::__set_value_from<test_tag<t_then_algo>>(state.__rcvr_,
                                                                     std::move(state.__data_),
                                                                     static_cast<Args &&>(args)...);
      else
        return ex::__facade::__send(state.__rcvr_, tag, static_cast<Args &&>(args)...);
    }
  };

  // The key of t_schedule's nested schedule operation
  struct sched_key
  {};

  struct t_schedule_algo;

  template <class Sender, class Receiver>
  struct t_schedule_state
  {
    using scheduler_t  = std::decay_t<ex::__data_of<Sender>>;
    using sched_sndr_t = ex::schedule_result_t<scheduler_t const &>;
    using sched_rcvr_t =
      ex::__facade::__receiver_t<t_schedule_algo, sched_key, t_schedule_state, Sender, Receiver>;

    explicit t_schedule_state(Sender &&sndr, Receiver &&rcvr)
      : __rcvr_(static_cast<Receiver &&>(rcvr))
      , op_(ex::connect(ex::schedule(ex::__facade::__get_data(sndr)), sched_rcvr_t{this}))
    {}

    t_schedule_state(t_schedule_state &&) = delete;

    Receiver                                         __rcvr_;
    ex::connect_result_t<sched_sndr_t, sched_rcvr_t> op_;
  };

  // The completions of the schedule operation are passed on, by the default __complete.
  struct t_schedule_algo : ex::__facade::__defaults
  {
    using ex::__facade::__defaults::__nested_sender;

    template <class Sender, class... Env>
    static constexpr auto __nested_sender(sched_key, Sender const &sndr, Env const &...)
    {
      return ex::schedule(ex::__facade::__get_data(sndr));
    }

    template <class Sender, class Receiver>
    static constexpr auto
    __get_state(Sender &&sndr, Receiver rcvr) -> t_schedule_state<Sender, Receiver>
    {
      return t_schedule_state<Sender, Receiver>(static_cast<Sender &&>(sndr),
                                                static_cast<Receiver &&>(rcvr));
    }

    template <class State, class Children>
    static constexpr auto __start(State &state, Children &) noexcept
    {
      return ex::__facade::__start_nested<sched_key>(state.op_);
    }
  };

  template <class... Ts>
  constexpr auto t_just(Ts... vs)
  {
    return ex::__facade::__sender{test_tag<t_just_algo>(), ex::__tuple{std::move(vs)...}};
  }

  template <class Sndr, class Fn>
  constexpr auto t_then(Sndr sndr, Fn fn)
  {
    return ex::__facade::__sender{test_tag<t_then_algo>(), std::move(fn), std::move(sndr)};
  }

  template <class Sch>
  constexpr auto t_schedule(Sch sch)
  {
    return ex::__facade::__sender{test_tag<t_schedule_algo>(), std::move(sch)};
  }

  using t_just_t     = decltype(t_just(1));
  using t_schedule_t = decltype(t_schedule(std::declval<run_loop_sch_t>()));

  // The completion signatures fall out of the code:
  static_assert(has_sigs_v<t_just_t, ex::set_value_t(int)>);
  static_assert(
    has_sigs_v<decltype(t_then(t_just(1), may_throw)), ex::set_value_t(int), eptr_sig_t>);
  static_assert(has_sigs_v<decltype(t_then(t_just(1), no_throw)), ex::set_value_t(int)>);
  static_assert(has_sigs_v<decltype(t_then(domain_probe{}, no_throw)),
                           ex::set_value_t(int),
                           ex::set_error_t(int)>);

  // A completion from start() happens inline, on the scheduler on which the operation
  // starts.
  static_assert(behavior_of_v<t_just_t, ex::set_value_t, start_env_t> == cb::__inline_completion);
  static_assert(std::is_same_v<decltype(ex::get_completion_scheduler<ex::set_value_t>(
                                 ex::get_env(std::declval<t_just_t>()),
                                 std::declval<start_env_t>())),
                               run_loop_sch_t>);

  // A completion from a child's completion happens where the child completes. When fn may
  // throw, t_then's error completions also come from the child's *value* completions, so
  // they may happen in either domain, inline or asynchronously.
  using t_then_throw_t = decltype(t_then(domain_probe{}, may_throw));
  static_assert(
    std::is_same_v<domain_of_t<t_then_throw_t, ex::set_value_t, ex::env<>>, value_domain>);
  static_assert(is_indeterminate_of_v<domain_of_t<t_then_throw_t, ex::set_error_t, ex::env<>>,
                                      value_domain,
                                      error_domain>);
  static_assert(
    behavior_of_v<decltype(t_then(behavior_probe{}, may_throw)), ex::set_error_t, ex::env<>>
    == (cb::__inline_completion | cb::__asynchronous));

  // When fn cannot throw, the error channel is exactly the child's.
  using t_then_nothrow_t = decltype(t_then(domain_probe{}, no_throw));
  static_assert(
    std::is_same_v<domain_of_t<t_then_nothrow_t, ex::set_error_t, ex::env<>>, error_domain>);
  static_assert(
    behavior_of_v<decltype(t_then(behavior_probe{}, no_throw)), ex::set_error_t, ex::env<>>
    == cb::__inline_completion);

  // A completion from a nested operation's completion happens where that operation
  // completes, after the operation started (here, from start()). A run_loop's schedule
  // sender can be stopped only if the receiver's stop token is stoppable, and the derived
  // signatures track that.
  static_assert(has_sigs_v<t_schedule_t, ex::set_value_t()>);
  static_assert(same_sigs_v<ex::completion_signatures_of_t<t_schedule_t, stop_env_t>,
                            ex::completion_signatures<ex::set_value_t(), ex::set_stopped_t()>>);
  static_assert(behavior_of_v<t_schedule_t, ex::set_value_t, start_env_t> == cb::__asynchronous);
  static_assert(std::is_same_v<decltype(ex::get_completion_scheduler<ex::set_value_t>(
                                 ex::get_env(std::declval<t_schedule_t>()),
                                 std::declval<start_env_t>())),
                               run_loop_sch_t>);
  static_assert(std::is_same_v<decltype(ex::get_completion_scheduler<ex::set_value_t>(
                                 ex::get_env(t_then(std::declval<t_schedule_t>(), no_throw)),
                                 std::declval<start_env_t>())),
                               run_loop_sch_t>);
  static_assert(!has_completion_scheduler<t_schedule_t, ex::set_error_t, start_env_t>);

  // A handler that finds its inputs ill-typed reports a type error, and the sender is not
  // well-formed.
  static_assert(
    !ex::__well_formed_sender<decltype(t_then(t_just(1), [](std::string) { return 0; }))>);

  //////////////////////////////////////////////////////////////////////////////////////////
  // 1. Completion signatures fall out of the code.
  static_assert(has_sigs_v<decltype(ex::just(1)), ex::set_value_t(int)>);
  static_assert(
    has_sigs_v<decltype(ex::then(ex::just(1), may_throw)), ex::set_value_t(int), eptr_sig_t>);
  static_assert(has_sigs_v<decltype(ex::then(ex::just(1), no_throw)), ex::set_value_t(int)>);
  static_assert(has_sigs_v<decltype(ex::upon_error(domain_probe{}, no_throw)),
                           ex::set_value_t(),
                           ex::set_value_t(int)>);

  //////////////////////////////////////////////////////////////////////////////////////////
  // 2. then: when fn may throw, error completions also come from the child's *value*
  // completions, so they may happen in either domain, inline or asynchronously.
  using then_throw_t = decltype(ex::then(domain_probe{}, may_throw));
  static_assert(
    std::is_same_v<domain_of_t<then_throw_t, ex::set_value_t, ex::env<>>, value_domain>);
  static_assert(is_indeterminate_of_v<domain_of_t<then_throw_t, ex::set_error_t, ex::env<>>,
                                      value_domain,
                                      error_domain>);
  static_assert(
    behavior_of_v<decltype(ex::then(behavior_probe{}, may_throw)), ex::set_error_t, ex::env<>>
    == (cb::__inline_completion | cb::__asynchronous));

  // When fn cannot throw, the error channel is exactly the child's.
  using then_nothrow_t = decltype(ex::then(domain_probe{}, no_throw));
  static_assert(
    std::is_same_v<domain_of_t<then_nothrow_t, ex::set_error_t, ex::env<>>, error_domain>);
  static_assert(
    behavior_of_v<decltype(ex::then(behavior_probe{}, no_throw)), ex::set_error_t, ex::env<>>
    == cb::__inline_completion);

  //////////////////////////////////////////////////////////////////////////////////////////
  // 3. upon_error: value completions also come from the child's *error* completions.
  static_assert(
    is_indeterminate_of_v<
      domain_of_t<decltype(ex::upon_error(domain_probe{}, no_throw)), ex::set_value_t, ex::env<>>,
      value_domain,
      error_domain>);
  static_assert(
    behavior_of_v<decltype(ex::upon_error(behavior_probe{}, no_throw)), ex::set_value_t, ex::env<>>
    == (cb::__inline_completion | cb::__asynchronous));

  //////////////////////////////////////////////////////////////////////////////////////////
  // Run-time tests

  // Runs a run_loop on a thread of its own until destroyed
  struct loop_thread
  {
    loop_thread()               = default;
    loop_thread(loop_thread &&) = delete;

    ~loop_thread()
    {
      loop.finish();
      worker.join();
    }

    ex::run_loop loop;
    std::thread  worker{[this] { loop.run(); }};
  };

  // 0, continued.
  TEST_CASE("facade: algorithms written for the tests run", "[detail][facade]")
  {
    auto result1 = ex::sync_wait(t_then(t_just(20), [](int i) { return 2 * i + 1; }));
    REQUIRE(result1.has_value());
    CHECK(std::get<0>(*result1) == 41);

    loop_thread ctx;
    auto        sndr = t_then(t_schedule(ctx.loop.get_scheduler()),
                       [] { return std::this_thread::get_id(); });

    // The completion scheduler is a run-time value read from the sender.
    ex::run_loop loop;
    auto         env = ex::prop{ex::get_start_scheduler, loop.get_scheduler()};
    CHECK(ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr), env)
          == ctx.loop.get_scheduler());

    auto result2 = ex::sync_wait(std::move(sndr));
    REQUIRE(result2.has_value());
    CHECK(std::get<0>(*result2) == ctx.worker.get_id());
  }
}  // namespace
