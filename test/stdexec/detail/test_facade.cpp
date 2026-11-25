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

  // A scheduler whose schedule sender can fail with an int only if its environment says
  // where it starts (that is, has a start scheduler). It shows which environment a schedule
  // operation sees.
  struct start_sensitive_scheduler
  {
    using scheduler_concept = ex::scheduler_tag;

    struct sender
    {
      using sender_concept = ex::sender_tag;

      template <class Self, class Env>
      static consteval auto get_completion_signatures()
      {
        if constexpr (ex::__callable<ex::get_start_scheduler_t, Env const &>)
          return ex::completion_signatures<ex::set_value_t(), ex::set_error_t(int)>();
        else
          return ex::completion_signatures<ex::set_value_t()>();
      }

      template <class Rcvr>
      struct opstate
      {
        using operation_state_concept = ex::operation_state_tag;

        void start() & noexcept
        {
          ex::set_value(static_cast<Rcvr &&>(rcvr));
        }

        Rcvr rcvr;
      };

      template <class Rcvr>
      auto connect(Rcvr rcvr) const -> opstate<Rcvr>
      {
        return {static_cast<Rcvr &&>(rcvr)};
      }
    };

    [[nodiscard]]
    auto schedule() const noexcept -> sender
    {
      return {};
    }

    auto operator==(start_sensitive_scheduler const &) const noexcept -> bool = default;
  };

  // Value completions happen on one scheduler, and error completions on another. (The two
  // are equal, but they are read separately, so nothing says that they are.)
  struct sched_attrs
  {
    constexpr auto query(ex::get_completion_scheduler_t<ex::set_value_t>) const noexcept
    {
      return start_sensitive_scheduler{};
    }

    constexpr auto query(ex::get_completion_scheduler_t<ex::set_error_t>) const noexcept
    {
      return start_sensitive_scheduler{};
    }
  };

  using sched_probe = probe<sched_attrs>;

  // A sender that completes with a value or stopped
  struct stop_probe
  {
    using sender_concept = ex::sender_tag;

    template <class Self, class... Env>
    static consteval auto get_completion_signatures()
    {
      return ex::completion_signatures<ex::set_value_t(), ex::set_stopped_t()>();
    }

    template <class Rcvr>
    struct opstate
    {
      using operation_state_concept = ex::operation_state_tag;

      void start() & noexcept
      {
        ex::set_stopped(static_cast<Rcvr &&>(rcvr));
      }

      Rcvr rcvr;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) const -> opstate<Rcvr>
    {
      return {static_cast<Rcvr &&>(rcvr)};
    }
  };

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
  // 4. when_all: the last child to finish delivers the result, in any channel. So an error
  // can be delivered from another child's value completion, in its domain, and
  // asynchronously, and there is no one scheduler on which errors happen.
  using when_all_d_t = decltype(ex::when_all(domain_probe{}, domain_probe{}));
  using when_all_b_t = decltype(ex::when_all(behavior_probe{}, behavior_probe{}));
  static_assert(has_sigs_v<when_all_d_t, ex::set_value_t(), ex::set_error_t(int)>);
  static_assert(
    std::is_same_v<domain_of_t<when_all_d_t, ex::set_value_t, ex::env<>>, value_domain>);
  static_assert(is_indeterminate_of_v<domain_of_t<when_all_d_t, ex::set_error_t, ex::env<>>,
                                      value_domain,
                                      error_domain>);
  static_assert(behavior_of_v<when_all_b_t, ex::set_error_t, ex::env<>>
                == (cb::__inline_completion | cb::__asynchronous));
  static_assert(!has_completion_scheduler<when_all_b_t, ex::set_error_t, start_env_t>);

  //////////////////////////////////////////////////////////////////////////////////////////
  // 5. continues_on: completions are replayed from a nested schedule operation, so they
  // happen where *it* completes.
  using cont_t = decltype(ex::continues_on(ex::just(1), std::declval<run_loop_sch_t>()));
  // A run_loop's schedule sender can be stopped only if the receiver's stop token is
  // stoppable, and the derived signatures track that:
  static_assert(has_sigs_v<cont_t, ex::set_value_t(int)>);
  static_assert(same_sigs_v<ex::completion_signatures_of_t<cont_t, stop_env_t>,
                            ex::completion_signatures<ex::set_value_t(int), ex::set_stopped_t()>>);
  // "inline, then asynchronous" is asynchronous (not "inline or asynchronous")
  static_assert(behavior_of_v<cont_t, ex::set_value_t, start_env_t> == cb::__asynchronous);

  // The schedule operation starts where the child completed. So one that completes where it
  // starts, as inline_scheduler's does, completes where the child completed: on the start
  // scheduler, after just(1); and after a child that does not say on which scheduler it
  // completes, on no scheduler that is known (not the one on which the operation started),
  // in the domains of the child's completions.
  using cont_inline_t       = decltype(ex::continues_on(ex::just(1), ex::inline_scheduler{}));
  using cont_inline_probe_t = decltype(ex::continues_on(domain_probe{}, ex::inline_scheduler{}));
  static_assert(std::is_same_v<decltype(ex::get_completion_scheduler<ex::set_value_t>(
                                 ex::get_env(std::declval<cont_inline_t>()),
                                 std::declval<start_env_t>())),
                               run_loop_sch_t>);
  static_assert(!has_completion_scheduler<cont_inline_probe_t, ex::set_value_t, start_env_t>);
  static_assert(
    is_indeterminate_of_v<domain_of_t<cont_inline_probe_t, ex::set_value_t, start_env_t>,
                          value_domain,
                          error_domain>);

  // The schedule operation sees the receiver's start scheduler only if it starts there:
  // after a child that completes where it starts, as then(just(1), fn) does, however it
  // completes. After a child that completes elsewhere, it does not.
  using loops_sndr_t =
    decltype(ex::when_all(ex::starts_on(std::declval<run_loop_sch_t>(), ex::just(1)),
                          ex::starts_on(std::declval<run_loop_sch_t>(), ex::just(2))));
  using cont_stays_t = decltype(ex::continues_on(ex::then(ex::just(1), may_throw),
                                                 start_sensitive_scheduler{}));
  using cont_moves_t = decltype(ex::continues_on(std::declval<loops_sndr_t>(),
                                                 start_sensitive_scheduler{}));
  static_assert(
    same_sigs_v<ex::completion_signatures_of_t<cont_stays_t, start_env_t>,
                ex::completion_signatures<ex::set_value_t(int), eptr_sig_t, ex::set_error_t(int)>>);
  static_assert(same_sigs_v<ex::completion_signatures_of_t<cont_moves_t, start_env_t>,
                            ex::completion_signatures<ex::set_value_t(int, int)>>);

  //////////////////////////////////////////////////////////////////////////////////////////
  // 6. Type errors are reported where stdexec reports them. A sender that does not depend
  // on its environment is checked when it is constructed. Otherwise,
  // get_completion_signatures reports the error once it is asked with an environment: by
  // throwing it when constexpr exceptions are available, and by returning it otherwise.
  struct _TEST_ERROR_
  {};

  // A sender that can't say how it completes until it knows its environment. In any
  // environment, needs_env<> completes with set_value_t(int), and needs_env<true> does
  // not type-check.
  template <bool IllTyped = false>
  struct needs_env
  {
    using sender_concept = ex::sender_tag;

    template <class Self, class... Env>
    static consteval auto get_completion_signatures()
    {
      if constexpr (sizeof...(Env) == 0)
        return ex::__throw_dependent_sender_error<Self>();
      else if constexpr (IllTyped)
        return ex::__throw_compile_time_error<ex::_WHAT_(_TEST_ERROR_)>();
      else
        return ex::completion_signatures<ex::set_value_t(int)>();
    }
  };

  using ill_typed = needs_env<true>;
  using bad_fn_t  = decltype([](std::string) { return 0; });

  // 6a. An ill-typed sender that doesn't depend on its environment is rejected when it is
  // constructed: ex::then(ex::just(1), bad_fn_t()) is a hard error. The types can still be
  // named:
  template <class Sndr, class Fn>
  using then_sender_t = ex::then_t::__sender<Fn, Sndr>;
  template <class... Sndrs>
  using when_all_sender_t = ex::__facade::__sender<ex::when_all_t, ex::__, Sndrs...>;
  using just_sender_t     = decltype(ex::just(1));

  static_assert(!ex::__well_formed_sender<then_sender_t<just_sender_t, bad_fn_t>>);
  static_assert(!ex::__well_formed_sender<
                when_all_sender_t<just_sender_t, then_sender_t<just_sender_t, bad_fn_t>>>);

  // 6b. Otherwise the sender can be constructed. These are all dependent senders.
  // An ill-typed child, directly and two levels down:
  using child_error_t  = decltype(ex::then(ill_typed{}, no_throw));
  using nested_error_t = decltype(ex::when_all(ex::just(1),
                                               ex::continues_on(ill_typed{},
                                                                std::declval<run_loop_sch_t>())));
  // The algorithm's own type error, directly and one level down:
  using own_error_t  = decltype(ex::then(needs_env{}, bad_fn_t()));
  using deep_error_t = decltype(ex::when_all(ex::just(1), ex::then(needs_env{}, bad_fn_t())));
  // A type error reported by __get_state (a child with two value completions):
  using state_error_t = decltype(ex::when_all(needs_env{},
                                              ex::upon_error(domain_probe{}, no_throw)));

  static_assert(ex::dependent_sender<child_error_t> && ex::dependent_sender<nested_error_t>
                && ex::dependent_sender<own_error_t> && ex::dependent_sender<deep_error_t>
                && ex::dependent_sender<state_error_t>);

  // None is a sender in env<>, and none has completion attributes there (rather than
  // causing hard errors when asked).
  template <class Sndr>
  concept has_domain_query = requires(ex::env_of_t<Sndr> const &attrs) {
    attrs.query(ex::get_completion_domain_t<ex::set_value_t>(), ex::env<>());
  };

  template <class Sndr>
  inline constexpr bool ill_typed_v = !ex::sender_in<Sndr, ex::env<>> && !has_domain_query<Sndr>;

  static_assert(ill_typed_v<child_error_t> && ill_typed_v<nested_error_t>
                && ill_typed_v<own_error_t> && ill_typed_v<deep_error_t>
                && ill_typed_v<state_error_t>);

  // The errors each case must report in env<>
  using test_error_t     = ex::__mexception<ex::_WHAT_(_TEST_ERROR_)>;
  using then_error_t     = ex::__callable_error_t<ex::then_t, bad_fn_t, int>;
  using when_all_error_t = ex::__mexception<
    ex::_WHAT_(ex::_INVALID_ARGUMENT_),
    ex::_WHERE_(ex::_IN_ALGORITHM_, ex::when_all_t),
    ex::_WHY_(ex::__when_all::_THE_GIVEN_SENDER_CAN_COMPLETE_SUCCESSFULLY_IN_MORE_THAN_ONE_WAY_),
    ex::_TO_FIX_THIS_ERROR_(ex::__when_all::_USE_WHEN_ALL_WITH_VARIANT_INSTEAD_)>;

#if STDEXEC_NO_STDCPP_CONSTEXPR_EXCEPTIONS()
  // get_completion_signatures returns the error reported by the ill-typed part.
  template <class Sndr, class Error, class... Env>
  consteval bool reports_impl()
  {
    return std::is_same_v<decltype(ex::get_completion_signatures<Sndr, Env...>()), Error>;
  }
#else
  // get_completion_signatures throws the error reported by the ill-typed part, as the
  // exception stdexec throws for an __mexception.
  template <class Error>
  struct thrown;

  template <class... What>
  struct thrown<ex::_ERROR_<What...>>
  {
    using type = ex::__sender_type_check_failure<ex::__tuple<>, What...>;
  };

  template <class Sndr, class Error, class... Env>
  consteval bool reports_impl()
  {
    try
    {
      (void) ex::get_completion_signatures<Sndr, Env...>();
      return false;
    }
    catch (typename thrown<Error>::type const &)
    {
      return true;
    }
    catch (...)
    {
      return false;
    }
  }
#endif

  template <class Sndr, class Error, class Env = ex::env<>>
  consteval bool reports()
  {
    return reports_impl<Sndr, Error, Env>();
  }

  template <class Sndr, class Error>
  consteval bool reports_without_env()
  {
    return reports_impl<Sndr, Error>();
  }

  static_assert(reports<ill_typed, test_error_t>());
  static_assert(reports<child_error_t, test_error_t>());
  static_assert(reports<nested_error_t, test_error_t>());
  static_assert(reports<own_error_t, then_error_t>());
  static_assert(reports<deep_error_t, then_error_t>());
  static_assert(reports<state_error_t, when_all_error_t>());

  // The error for results that an algorithm must store but cannot decay-copy names the
  // first such result.
  struct move_only
  {
    move_only()                  = default;
    move_only(move_only &&)      = default;
    move_only(move_only const &) = delete;
  };

  template <class Type>
  using decay_copy_error_t =
    ex::__facade::__effects<ex::__mexception<ex::_WHAT_(ex::_TYPE_IS_NOT_DECAY_COPYABLE_),
                                             ex::_WHERE_(ex::_IN_ALGORITHM_, ex::continues_on_t),
                                             ex::_WITH_TYPE_<Type>>>;

  static_assert(std::is_same_v<decltype(ex::__facade::__decay_copy_error<ex::continues_on_t,
                                                                         int,
                                                                         move_only &,
                                                                         move_only const &>()),
                               decay_copy_error_t<move_only &>>);
  static_assert(std::is_same_v<
                decltype(ex::__facade::__decay_copy_error<ex::continues_on_t, move_only const &>()),
                decay_copy_error_t<move_only const &>>);

  //////////////////////////////////////////////////////////////////////////////////////////
  // 7. An algorithm whose own completions depend on its environment opts in to
  // __depends_on_env, as read_env does. Its senders are then dependent, and so is any
  // sender with one as a child.
  inline constexpr auto identity = [](auto x) noexcept
  {
    return x;
  };

  using read_token_t = decltype(ex::read_env(ex::get_stop_token));
  using then_read_t  = decltype(ex::then(ex::read_env(ex::get_stop_token), identity));

  static_assert(ex::dependent_sender<read_token_t> && ex::dependent_sender<then_read_t>);
  static_assert(
    ex::dependent_sender<decltype(ex::when_all(ex::just(1), ex::read_env(ex::get_stop_token)))>);

  // Without an environment, there are no completion attributes either:
  static_assert(!has_completion_scheduler<read_token_t, ex::set_value_t>);
  static_assert(has_completion_scheduler<read_token_t, ex::set_value_t, start_env_t>);

  // Given an environment, the completions follow, and a parent's do too:
  static_assert(has_sigs_v<read_token_t, ex::set_value_t(ex::never_stop_token)>);
  static_assert(same_sigs_v<ex::completion_signatures_of_t<read_token_t, stop_env_t>,
                            ex::completion_signatures<ex::set_value_t(ex::inplace_stop_token)>>);
  static_assert(has_sigs_v<then_read_t, ex::set_value_t(ex::never_stop_token)>);
  static_assert(same_sigs_v<ex::completion_signatures_of_t<then_read_t, stop_env_t>,
                            ex::completion_signatures<ex::set_value_t(ex::inplace_stop_token)>>);

  // A query the environment can't answer is a type error, reported only once there is an
  // environment.
  using read_sched_t     = decltype(ex::read_env(ex::get_scheduler));
  using no_value_error_t = ex::__mexception<
    ex::__read_env::_THE_CURRENT_EXECUTION_ENVIRONMENT_DOESNT_HAVE_A_VALUE_FOR_THE_GIVEN_QUERY_,
    ex::_WHERE_(ex::_IN_ALGORITHM_, ex::__read_env_t),
    ex::_WITH_QUERY_(ex::get_scheduler_t),
    ex::_WITH_ENVIRONMENT_(ex::env<>)>;

  static_assert(ex::dependent_sender<read_sched_t>);
  static_assert(reports<read_sched_t, no_value_error_t>());

  // Without the opt-in, the walk answers as if for env<> when there is no environment.
  // read_env(get_stop_token) then claims to send a never_stop_token in every environment,
  // and read_env(get_scheduler) is rejected as ill-typed when it is constructed.
  struct read_env_without_opt_in : ex::__read_env::__algo
  {
    template <class Sndr>
    static constexpr bool __depends_on_env = false;
  };

  template <class Query>
  using read_env_without_opt_in_t = test_sender_t<read_env_without_opt_in, Query>;

  static_assert(!ex::dependent_sender<read_env_without_opt_in_t<ex::get_stop_token_t>>);
  static_assert(
    same_sigs_v<ex::completion_signatures_of_t<read_env_without_opt_in_t<ex::get_stop_token_t>>,
                ex::completion_signatures<ex::set_value_t(ex::never_stop_token)>>);
  static_assert(!ex::__well_formed_sender<read_env_without_opt_in_t<ex::get_scheduler_t>>);

  // An algorithm with children can opt in, too. The children are checked first, so a
  // child's type error is still reported without an environment, and the sender is still
  // rejected when it is constructed:
  struct then_depending_on_env : ex::__upon::__algo<ex::then_t, ex::set_value_t>
  {
    template <class Sndr>
    static constexpr bool __depends_on_env = true;
  };

  template <class Child>
  using then_depending_on_env_t =
    test_sender_t<then_depending_on_env, std::decay_t<decltype(no_throw)>, Child>;
  using bad_child_t = then_sender_t<just_sender_t, bad_fn_t>;

  static_assert(ex::dependent_sender<then_depending_on_env_t<just_sender_t>>);
  static_assert(!ex::__well_formed_sender<then_depending_on_env_t<bad_child_t>>);
  static_assert(reports_without_env<then_depending_on_env_t<bad_child_t>, then_error_t>());

  //////////////////////////////////////////////////////////////////////////////////////////
  // 8. on returns to where it started (or, given a closure, to where its child completed).
  // It depends on its environment for that, and it is an error if the environment does not
  // say.
  using on_sndr_t = decltype(ex::on(std::declval<run_loop_sch_t>(), ex::just(1)));
  using on_closure_sndr_t =
    decltype(ex::on(ex::just(1), std::declval<run_loop_sch_t>(), ex::then(identity)));

  static_assert(ex::dependent_sender<on_sndr_t> && ex::dependent_sender<on_closure_sndr_t>);
  static_assert(ex::sender_in<on_sndr_t, start_env_t>);
  static_assert(ex::sender_in<on_closure_sndr_t, start_env_t>);

  using cannot_return_error_t = ex::__mexception<
    ex::_WHAT_(ex::_CANNOT_RESTORE_EXECUTION_CONTEXT_AFTER_ON_),
    ex::_WHY_(ex::_THE_CURRENT_EXECUTION_ENVIRONMENT_DOES_NOT_HAVE_A_START_SCHEDULER_),
    ex::_WHERE_(ex::_IN_ALGORITHM_, ex::on_t),
    ex::_WITH_SENDER_<just_sender_t>,
    ex::_WITH_ENVIRONMENT_(ex::env<>)>;

  static_assert(reports<on_sndr_t, cannot_return_error_t>());

  // on(sndr, sch, closure) runs sndr inside the closure's sender. sndr is checked in the
  // environment it sees if the closure does not change the environment, which is also the
  // environment in which on asks sndr where it completes.
  using bad_closure_child_t = decltype(ex::on(ex::read_env(ex::get_allocator),
                                              std::declval<run_loop_sch_t>(),
                                              ex::then(identity)));
  static_assert(!ex::sender_in<bad_closure_child_t, start_env_t>);

  // on(sch, sndr) is lowered to continues_on(starts_on(sch, sndr), orig_sch), whose
  // schedule operation back starts where starts_on(sch, sndr) completed. So it sees, as its
  // start scheduler, the scheduler on which that completes, if that is known (here, the
  // run_loop), or the receiver's, if that sender completes where it starts. Otherwise it
  // sees none.
  using start_sensitive_env_t = decltype(ex::prop{ex::get_start_scheduler,
                                                  start_sensitive_scheduler{}});
  using on_stays_t            = decltype(ex::on(ex::inline_scheduler{}, ex::just()));
  using on_loop_t             = decltype(ex::on(std::declval<run_loop_sch_t>(), ex::just()));
  using on_moves_t = decltype(ex::on(std::declval<run_loop_sch_t>(), std::declval<loops_sndr_t>()));
  static_assert(same_sigs_v<ex::completion_signatures_of_t<on_stays_t, start_sensitive_env_t>,
                            ex::completion_signatures<ex::set_value_t(), ex::set_error_t(int)>>);
  static_assert(same_sigs_v<ex::completion_signatures_of_t<on_loop_t, start_sensitive_env_t>,
                            ex::completion_signatures<ex::set_value_t(), ex::set_error_t(int)>>);
  static_assert(same_sigs_v<ex::completion_signatures_of_t<on_moves_t, start_sensitive_env_t>,
                            ex::completion_signatures<ex::set_value_t(int, int)>>);
  // `on` completes where that schedule operation completes. Here the start scheduler, and so
  // the scheduler that `on` returns to, is inline_scheduler, so `on` completes where
  // starts_on(sch, sndr) completed: on the run_loop.
  using on_inline_return_t = decltype(ex::starts_on(ex::inline_scheduler{},
                                                    std::declval<on_loop_t>()));
  static_assert(std::is_same_v<decltype(ex::get_completion_scheduler<ex::set_value_t>(
                                 ex::get_env(std::declval<on_inline_return_t>()),
                                 ex::env<>{})),
                               run_loop_sch_t>);

  //////////////////////////////////////////////////////////////////////////////////////////
  // 9. A sender that is lowered to another sender has that sender's completions and
  // attributes. when_all_with_variant(sndrs...) is lowered to when_all(into_variant(sndrs)...),
  // and __transfer_when_all(sch, sndrs...) to continues_on(when_all(sndrs...), sch).
  using variant_of_void_t = std::variant<std::tuple<>>;
  using wawv_d_t          = decltype(ex::when_all_with_variant(domain_probe{}, domain_probe{}));
  using wawv_b_t          = decltype(ex::when_all_with_variant(behavior_probe{}, behavior_probe{}));
  // (into_variant always declares an exception_ptr error.)
  static_assert(has_sigs_v<wawv_d_t,
                           ex::set_value_t(variant_of_void_t, variant_of_void_t),
                           ex::set_error_t(int),
                           eptr_sig_t>);
  static_assert(std::is_same_v<domain_of_t<wawv_d_t, ex::set_value_t, ex::env<>>, value_domain>);
  static_assert(is_indeterminate_of_v<domain_of_t<wawv_d_t, ex::set_error_t, ex::env<>>,
                                      value_domain,
                                      error_domain>);
  static_assert(behavior_of_v<wawv_b_t, ex::set_error_t, ex::env<>>
                == (cb::__inline_completion | cb::__asynchronous));

  // Every completion, an error too, is replayed on the scheduler.
  using transfer_t =
    decltype(ex::__transfer_when_all(std::declval<run_loop_sch_t>(), domain_probe{}, ex::just(1)));
  static_assert(has_sigs_v<transfer_t, ex::set_value_t(int), ex::set_error_t(int)>);
  static_assert(std::is_same_v<decltype(ex::get_completion_scheduler<ex::set_error_t>(
                                 ex::get_env(std::declval<transfer_t>()),
                                 std::declval<start_env_t>())),
                               run_loop_sch_t>);
  static_assert(behavior_of_v<transfer_t, ex::set_value_t, start_env_t> == cb::__asynchronous);
  static_assert(
    ex::__sender_for<ex::transform_sender_result_t<transfer_t, start_env_t>, ex::continues_on_t>);

  //////////////////////////////////////////////////////////////////////////////////////////
  // 10. Completions that a facade sender gets the scheduler of from the same place happen on
  // the same scheduler, even if they are of different kinds, and its parent knows that.
  // When fn may throw, then(t_schedule(sch), fn) completes with a value or an error, both on
  // sch. So upon_error over it completes with a value on sch however it completes; and so
  // does continues_on with a scheduler that completes where it starts, with a value or an
  // error.
  using sched_throw_t = decltype(ex::then(std::declval<t_schedule_t>(), may_throw));
  using upon_sched_t  = decltype(ex::upon_error(std::declval<sched_throw_t>(), no_throw));
  using cont_sched_t  = decltype(ex::continues_on(std::declval<sched_throw_t>(),
                                                 ex::inline_scheduler{}));
  static_assert(has_completion_scheduler<upon_sched_t, ex::set_value_t, start_env_t>);
  static_assert(has_completion_scheduler<cont_sched_t, ex::set_value_t, start_env_t>);
  static_assert(has_completion_scheduler<cont_sched_t, ex::set_error_t, start_env_t>);
  static_assert(
    has_completion_scheduler<decltype(ex::then(std::declval<cont_sched_t>(), may_throw)),
                             ex::set_error_t,
                             start_env_t>);

  // The same goes for stopped completions. continues_on(stop_probe(), sch) replays both
  // kinds of stop_probe's completions on sch, so upon_stopped over it completes with a value
  // on sch.
  using cont_stop_t = decltype(ex::continues_on(stop_probe{}, std::declval<run_loop_sch_t>()));
  static_assert(
    has_completion_scheduler<decltype(ex::upon_stopped(std::declval<cont_stop_t>(), no_throw)),
                             ex::set_value_t,
                             start_env_t>);

  // But completions that it gets the schedulers of from different places may happen on
  // different schedulers, even of the same type. A sched_probe is asked for the schedulers of
  // its value and error completions separately, so then over it gets the schedulers of its
  // value and error completions from different places, and upon_error and continues_on over
  // that have no completion scheduler.
  using then_probe_t = decltype(ex::then(sched_probe{}, no_throw));
  static_assert(has_completion_scheduler<then_probe_t, ex::set_value_t, start_env_t>);
  static_assert(has_completion_scheduler<then_probe_t, ex::set_error_t, start_env_t>);
  static_assert(
    !has_completion_scheduler<decltype(ex::upon_error(std::declval<then_probe_t>(), no_throw)),
                              ex::set_value_t,
                              start_env_t>);
  static_assert(!has_completion_scheduler<decltype(ex::continues_on(std::declval<then_probe_t>(),
                                                                    ex::inline_scheduler{})),
                                          ex::set_value_t,
                                          start_env_t>);

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

  // 5, continued.
  TEST_CASE("facade: a completion scheduler is a runtime value read from the sender",
            "[detail][facade]")
  {
    ex::run_loop            loop1;
    ex::run_loop            loop2;
    ex::inplace_stop_source stop;
    auto                    sndr = ex::continues_on(ex::just(1), loop1.get_scheduler());
    // (The props are named because Clang 16 miscompiles them when they are deduced in the
    // braced initializer of the env.)
    auto stop_prop  = ex::prop{ex::get_stop_token, stop.get_token()};
    auto sched_prop = ex::prop{ex::get_start_scheduler, loop2.get_scheduler()};
    auto env        = ex::env{stop_prop, sched_prop};
    CHECK(ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr), env)
          == loop1.get_scheduler());
    CHECK(ex::get_completion_scheduler<ex::set_stopped_t>(ex::get_env(sndr), env)
          == loop1.get_scheduler());

    // just completes inline, so it completes on the start scheduler.
    auto just_sndr = ex::just(1);
    CHECK(ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(just_sndr), env)
          == loop2.get_scheduler());
  }

  // 8, continued.
  TEST_CASE("facade: on completes on the scheduler it returns to", "[detail][facade]")
  {
    ex::run_loop loop1;
    ex::run_loop loop2;
    auto         env = ex::prop{ex::get_start_scheduler, loop2.get_scheduler()};

    // on(sch, sndr) returns to the start scheduler.
    auto sndr = ex::on(loop1.get_scheduler(), ex::just(1));
    CHECK(ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr), env)
          == loop2.get_scheduler());

    // on(sndr, sch, closure) returns to where sndr completes, which for just is where it
    // starts.
    auto closure_sndr = ex::on(ex::just(1), loop1.get_scheduler(), ex::then(identity));
    CHECK(ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(closure_sndr), env)
          == loop2.get_scheduler());
  }

  // 10, continued.
  TEST_CASE("facade: completions of different kinds can happen on the same scheduler",
            "[detail][facade]")
  {
    ex::run_loop loop1;
    ex::run_loop loop2;
    auto         env  = ex::prop{ex::get_start_scheduler, loop2.get_scheduler()};
    auto         sndr = ex::then(t_schedule(loop1.get_scheduler()), may_throw);

    // upon_error completes with a value where then completes, however it completes.
    auto upon_sndr = ex::upon_error(sndr, no_throw);
    CHECK(ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(upon_sndr), env)
          == loop1.get_scheduler());

    // continues_on's schedule operation starts where then completes, and inline_scheduler's
    // completes there.
    auto cont_sndr = ex::continues_on(sndr, ex::inline_scheduler{});
    CHECK(ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(cont_sndr), env)
          == loop1.get_scheduler());
    CHECK(ex::get_completion_scheduler<ex::set_error_t>(ex::get_env(cont_sndr), env)
          == loop1.get_scheduler());
  }

  // The derived information is enough to run real pipelines.
  TEST_CASE("facade: senders run real pipelines", "[detail][facade]")
  {
    loop_thread ctx;
    auto work   = ex::then(ex::when_all(ex::continues_on(ex::just(20), ctx.loop.get_scheduler()),
                                      ex::just(1)),
                         [](int a, int b) { return 2 * a + b; });
    auto result = ex::sync_wait(std::move(work));
    REQUIRE(result.has_value());
    CHECK(std::get<0>(*result) == 41);
  }

  TEST_CASE("facade: on hops there and back", "[detail][facade]")
  {
    loop_thread ctx;
    auto        here = std::this_thread::get_id();
    auto        work = ex::on(ctx.loop.get_scheduler(),
                       ex::just() | ex::then([] { return std::this_thread::get_id(); }))
              | ex::then([](std::thread::id there)
                         { return std::pair{there, std::this_thread::get_id()}; });
    auto result = ex::sync_wait(std::move(work));
    REQUIRE(result.has_value());
    auto [there, back] = std::get<0>(*result);
    CHECK(there == ctx.worker.get_id());
    CHECK(back == here);
  }

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  TEST_CASE("facade: an exception from a function becomes an error completion", "[detail][facade]")
  {
    auto failing = ex::when_all(ex::just(1), ex::then(ex::just(), []() -> int { throw 42; }));
    int  caught  = 0;
    try
    {
      (void) ex::sync_wait(std::move(failing));
    }
    catch (int i)
    {
      caught = i;
    }
    CHECK(caught == 42);
  }
#endif  // !STDEXEC_NO_STDCPP_EXCEPTIONS()

  // 7, continued.
  TEST_CASE("facade: read_env reads the environment of the receiver it is connected to",
            "[detail][facade]")
  {
    ex::inplace_stop_source stop;
    auto                    env = ex::prop{ex::get_stop_token, stop.get_token()};

    auto result1 = ex::sync_wait(ex::write_env(ex::read_env(ex::get_stop_token), env));
    REQUIRE(result1.has_value());
    CHECK(std::get<0>(*result1) == stop.get_token());

    // As a child:
    auto result2 = ex::sync_wait(
      ex::write_env(ex::then(ex::read_env(ex::get_stop_token), identity), env));
    REQUIRE(result2.has_value());
    CHECK(std::get<0>(*result2) == stop.get_token());
  }

  // 7, continued. write_env(sndr, env): the child sees env joined with the receiver's
  // environment, which it shadows. The child's completions and attributes are those it has
  // in that environment.
  using write_token_t = decltype(ex::write_env(ex::read_env(ex::get_stop_token),
                                               ex::prop{ex::get_stop_token,
                                                        ex::never_stop_token()}));
  using write_start_t = decltype(ex::write_env(ex::just(),
                                               ex::prop{ex::get_start_scheduler,
                                                        std::declval<run_loop_sch_t>()}));

  static_assert(ex::dependent_sender<write_token_t>);
  static_assert(same_sigs_v<ex::completion_signatures_of_t<write_token_t, stop_env_t>,
                            ex::completion_signatures<ex::set_value_t(ex::never_stop_token)>>);
  // just() completes inline, so it completes on the start scheduler that write_env writes:
  static_assert(has_completion_scheduler<write_start_t, ex::set_value_t, ex::env<>>);

  TEST_CASE("facade: write_env's child sees the written environment", "[detail][facade]")
  {
    ex::run_loop loop;
    auto sndr = ex::write_env(ex::just(), ex::prop{ex::get_start_scheduler, loop.get_scheduler()});
    CHECK(ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr), ex::env<>())
          == loop.get_scheduler());

    auto result = ex::sync_wait(
      ex::write_env(ex::read_env(ex::get_stop_token),
                    ex::prop{ex::get_stop_token, ex::never_stop_token()}));
    REQUIRE(result.has_value());
  }

  // 9, continued. connect lowers when_all_with_variant(sndrs...) to
  // when_all(into_variant(sndrs)...), but the facade derives the sender's attributes from
  // what when_all's algorithm does for it. That must be what the lowered sender does.
  // Connecting the sender with its own connect runs that algorithm.
  template <class... Values>
  struct values_receiver
  {
    using receiver_concept = ex::receiver_tag;

    void set_value(Values... values) noexcept
    {
      result_->emplace(std::move(values)...);
    }

    void set_error(std::exception_ptr) noexcept {}

    void set_stopped() noexcept {}

    std::optional<std::tuple<Values...>> *result_;
  };

  TEST_CASE("facade: when_all_with_variant's algorithm does what the lowered sender does",
            "[detail][facade]")
  {
    using int_variant_t    = std::variant<std::tuple<int>>;
    using double_variant_t = std::variant<std::tuple<double>>;
    std::optional<std::tuple<int_variant_t, double_variant_t>> result;
    auto op = ex::when_all_with_variant(ex::just(2), ex::just(3.5))
                .connect(values_receiver<int_variant_t, double_variant_t>{&result});
    ex::start(op);
    REQUIRE(result.has_value());
    CHECK(std::get<0>(*result) == int_variant_t{std::tuple{2}});
    CHECK(std::get<1>(*result) == double_variant_t{std::tuple{3.5}});

    // when_all_with_variant() is just():
    std::optional<std::tuple<>> no_result;
    auto op2 = ex::when_all_with_variant().connect(values_receiver<>{&no_result});
    ex::start(op2);
    CHECK(no_result.has_value());
  }

  // Likewise, connect lowers stopped_as_error(sndr, err) to let_stopped(sndr, [err] { return
  // just_error(err); }). Its algorithm turns the child's stopped completion into the error,
  // where the child stops, as the lowered sender does.
  struct int_error_receiver
  {
    using receiver_concept = ex::receiver_tag;

    void set_value() noexcept {}

    void set_error(int error) noexcept
    {
      *error_ = error;
    }

    void set_stopped() noexcept {}

    int *error_;
  };

  using stopped_as_error_t = decltype(ex::stopped_as_error(ex::just_stopped(), 42));
  static_assert(has_sigs_v<stopped_as_error_t, ex::set_error_t(int)>);
  static_assert(ex::__completes_inline<ex::set_error_t, ex::env_of_t<stopped_as_error_t>>);

  // __transfer_just(sch, vals...) is continues_on(just(vals...), sch) with a tag of its own,
  // so that a domain can customize it, and continues_on's algorithm. connect lowers it by
  // re-tagging it, so that the customizations of continues_on apply.
  using transfer_just_t = decltype(ex::__transfer_just(std::declval<run_loop_sch_t>(), 42));
  static_assert(has_completion_scheduler<transfer_just_t, ex::set_value_t, ex::env<>>);
  static_assert(
    std::same_as<
      ex::tag_of_t<decltype(ex::__transfer_just_t::transform_sender(ex::set_value,
                                                                    std::declval<transfer_just_t>(),
                                                                    ex::env<>()))>,
      ex::continues_on_t>);

  TEST_CASE("facade: __transfer_just has continues_on's algorithm", "[detail][facade]")
  {
    ex::run_loop                   loop;
    std::optional<std::tuple<int>> result;
    auto op = ex::__transfer_just(loop.get_scheduler(), 42).connect(values_receiver<int>{&result});
    ex::start(op);
    loop.finish();
    loop.run();
    REQUIRE(result.has_value());
    CHECK(std::get<0>(*result) == 42);
  }

  TEST_CASE("facade: stopped_as_error's algorithm does what the lowered sender does",
            "[detail][facade]")
  {
    int  error = 0;
    auto op    = ex::stopped_as_error(ex::just_stopped(), 42).connect(int_error_receiver{&error});
    ex::start(op);
    CHECK(error == 42);
  }
}  // namespace
