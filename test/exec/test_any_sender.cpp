/*
 * Copyright (c) 2023 NVIDIA Corporation
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

#include <exec/any_sender_of.hpp>
#include <exec/inline_scheduler.hpp>
#include <exec/when_any.hpp>
#include <exec/static_thread_pool.hpp>

#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>

#include <catch2/catch.hpp>


using namespace stdexec;
using namespace exec;

///////////////////////////////////////////////////////////////////////////////
//                                                             any_receiver_ref

struct tag_t : stdexec::__query<::tag_t> {
  template <class T>
  // BUGBUG ambiguous!
    requires stdexec::tag_invocable<tag_t, T>
  auto operator()(T&& t) const noexcept(stdexec::nothrow_tag_invocable<tag_t, T>)
    -> stdexec::tag_invoke_result_t<tag_t, T> {
    return stdexec::tag_invoke(*this, (T&&) t);
  }
};

inline constexpr ::tag_t get_address;

struct env {
  const void* object_{nullptr};
  in_place_stop_token token_;

  friend const void* tag_invoke(::tag_t, env e) noexcept {
    return e.object_;
  }

  friend in_place_stop_token tag_invoke(get_stop_token_t, const env& e) noexcept {
    return e.token_;
  }
};

struct sink_receiver {
  std::variant<std::monostate, int, std::exception_ptr, set_stopped_t> value_{};

  STDEXEC_DEFINE_CUSTOM(void set_value)(this sink_receiver&& r, set_value_t, int value) noexcept {
    r.value_ = value;
  }

  STDEXEC_DEFINE_CUSTOM(void set_error)(
    this sink_receiver&& r,
    set_error_t,
    std::exception_ptr e) noexcept {
    r.value_ = e;
  }

  STDEXEC_DEFINE_CUSTOM(void set_stopped)(this sink_receiver&& r, set_stopped_t) noexcept {
    r.value_ = stdexec::set_stopped;
  }

  STDEXEC_DEFINE_CUSTOM(env get_env)(this const sink_receiver& r, get_env_t) noexcept {
    return {static_cast<const void*>(&r)};
  }
};

TEST_CASE("any_receiver_ref is constructible from receivers", "[types][any_sender]") {
  using Sigs = completion_signatures<set_value_t(int)>;
  sink_receiver rcvr;
  any_receiver_ref<Sigs> ref{rcvr};
  CHECK(receiver<decltype(ref)>);
  CHECK(receiver_of<decltype(ref), Sigs>);
  CHECK(!receiver_of<decltype(ref), completion_signatures<set_value_t()>>);
  CHECK(std::is_copy_assignable_v<any_receiver_ref<Sigs>>);
  CHECK(std::is_constructible_v<any_receiver_ref<Sigs>, const sink_receiver&>);
  CHECK(!std::is_constructible_v<any_receiver_ref<Sigs>, sink_receiver&&>);
  CHECK(!std::is_constructible_v<
        any_receiver_ref<completion_signatures<set_value_t()>>,
        const sink_receiver&>);
}

TEST_CASE("any_receiver_ref is queryable", "[types][any_sender]") {
  using Sigs = completion_signatures<set_value_t(int)>;
  using receiver_ref = any_receiver_ref<Sigs, get_address.signature<const void*()>>;
  sink_receiver rcvr1{};
  sink_receiver rcvr2{};
  receiver_ref ref1{rcvr1};
  receiver_ref ref2{rcvr2};
  CHECK(get_address(get_env(ref1)) == &rcvr1);
  CHECK(get_address(get_env(ref2)) == &rcvr2);
  {
    receiver_ref copied_ref = ref2;
    CHECK(get_address(get_env(copied_ref)) != &ref2);
    CHECK(get_address(get_env(copied_ref)) == &rcvr2);
    ref1 = copied_ref;
    CHECK(get_address(get_env(ref1)) != &rcvr1);
    CHECK(get_address(get_env(ref1)) != &copied_ref);
    CHECK(get_address(get_env(ref1)) == &rcvr2);
    copied_ref = rcvr1;
    CHECK(get_address(get_env(ref1)) == &rcvr2);
    CHECK(get_address(get_env(copied_ref)) == &rcvr1);
  }
  CHECK(get_address(get_env(ref1)) == &rcvr2);
}

TEST_CASE("any_receiver_ref calls receiver methods", "[types][any_sender]") {
  using Sigs =
    completion_signatures<set_value_t(int), set_error_t(std::exception_ptr), set_stopped_t()>;
  using receiver_ref = any_receiver_ref<Sigs>;
  REQUIRE(receiver_of<receiver_ref, Sigs>);
  sink_receiver value{};
  sink_receiver error{};
  sink_receiver stopped{};

  // Check set value
  CHECK(value.value_.index() == 0);
  receiver_ref ref = value;
  stdexec::set_value((receiver_ref&&) ref, 42);
  CHECK(value.value_.index() == 1);
  CHECK(std::get<1>(value.value_) == 42);
  // Check set error
  CHECK(error.value_.index() == 0);
  ref = error;
  stdexec::set_error((receiver_ref&&) ref, std::make_exception_ptr(42));
  CHECK(error.value_.index() == 2);
  CHECK_THROWS_AS(std::rethrow_exception(std::get<2>(error.value_)), int);
  // Check set stopped
  CHECK(stopped.value_.index() == 0);
  ref = stopped;
  stdexec::set_stopped((receiver_ref&&) ref);
  CHECK(stopped.value_.index() == 3);
}

TEST_CASE("any_receiver_ref is connectable with when_any", "[types][any_sender]") {
  using Sigs = completion_signatures<set_value_t(int), set_stopped_t()>;
  using receiver_ref = any_receiver_ref<Sigs>;
  REQUIRE(receiver_of<receiver_ref, Sigs>);
  sink_receiver rcvr{};
  receiver_ref ref = rcvr;

  auto sndr = when_any(just(42));
  CHECK(rcvr.value_.index() == 0);
  auto op = connect(std::move(sndr), std::move(ref));
  start(op);
  CHECK(rcvr.value_.index() == 1);
  CHECK(std::get<1>(rcvr.value_) == 42);
}

///////////////////////////////////////////////////////////////////////////////
//                                                                any.storage

struct empty_vtable_t {
 private:
  template <class T>
  friend empty_vtable_t*
    tag_invoke(__any::__create_vtable_t, __mtype<empty_vtable_t>, __mtype<T>) noexcept {
    static empty_vtable_t vtable{};
    return &vtable;
  }
};

TEST_CASE("empty storage is movable", "[types][any_sender]") {
  struct foo { };

  using any_unique = __any::__unique_storage_t<empty_vtable_t>;
  any_unique s1{};
  any_unique s2 = foo{};
  static_assert(std::is_move_assignable_v<any_unique>);
  static_assert(!std::is_copy_assignable_v<any_unique>);

  CHECK(s2.__get_vtable());
  CHECK(s1.__get_vtable() != s2.__get_vtable());
  CHECK(s1.__get_object_pointer() == nullptr);
  CHECK(s2.__get_object_pointer() != nullptr);
  // Test SBO
  std::intptr_t obj_ptr = reinterpret_cast<std::intptr_t>(s2.__get_object_pointer());
  std::intptr_t s2_ptr = reinterpret_cast<std::intptr_t>(&s2);
  CHECK(std::abs(s2_ptr - obj_ptr) < std::intptr_t(sizeof(any_unique)));

  s1 = std::move(s2);
  CHECK(s2.__get_vtable());
  CHECK(s1.__get_vtable() != s2.__get_vtable());
  CHECK(s1.__get_object_pointer() != nullptr);
  CHECK(s2.__get_object_pointer() == nullptr);

  s1 = std::move(s2);
  CHECK(s1.__get_object_pointer() == nullptr);
  CHECK(s2.__get_object_pointer() == nullptr);
}

TEST_CASE("empty storage is movable, throwing moves will allocate", "[types][any_sender]") {
  struct move_throws {
    move_throws() = default;

    move_throws(move_throws&&) noexcept(false) {
    }

    move_throws& operator=(move_throws&&) noexcept(false) {
      return *this;
    }
  };

  using any_unique = __any::__unique_storage_t<empty_vtable_t>;
  any_unique s1{};
  any_unique s2 = move_throws{};
  static_assert(std::is_move_assignable_v<any_unique>);
  static_assert(!std::is_copy_assignable_v<any_unique>);

  CHECK(s2.__get_vtable());
  CHECK(s1.__get_vtable() != s2.__get_vtable());
  CHECK(s1.__get_object_pointer() == nullptr);
  CHECK(s2.__get_object_pointer() != nullptr);
  // Test SBO
  std::intptr_t obj_ptr = reinterpret_cast<std::intptr_t>(s2.__get_object_pointer());
  std::intptr_t s2_ptr = reinterpret_cast<std::intptr_t>(&s2);
  CHECK(std::abs(s2_ptr - obj_ptr) >= std::intptr_t(sizeof(any_unique)));

  s1 = std::move(s2);
  CHECK(s2.__get_vtable());
  CHECK(s1.__get_vtable() != s2.__get_vtable());
  CHECK(s1.__get_object_pointer() != nullptr);
  CHECK(s2.__get_object_pointer() == nullptr);

  s1 = std::move(s2);
  CHECK(s1.__get_object_pointer() == nullptr);
  CHECK(s2.__get_object_pointer() == nullptr);
}

///////////////////////////////////////////////////////////////////////////////
//                                                                any_sender

template <class... Ts>
using any_sender_of =
  typename any_receiver_ref<completion_signatures<Ts...>>::template any_sender<>;

TEST_CASE("any sender is a sender", "[types][any_sender]") {
  CHECK(stdexec::sender<any_sender_of<set_value_t()>>);
  CHECK(std::is_move_assignable_v<any_sender_of<set_value_t()>>);
  CHECK(std::is_nothrow_move_assignable_v<any_sender_of<set_value_t()>>);
  CHECK(!std::is_copy_assignable_v<any_sender_of<set_value_t()>>);
  any_sender_of<set_value_t()> sender = just();
}

TEST_CASE("sync_wait works on any_sender_of", "[types][any_sender]") {
  int value = 0;
  any_sender_of<set_value_t()> sender = just(42) | then([&](int v) noexcept { value = v; });
  CHECK(std::same_as<
        completion_signatures_of_t<any_sender_of<set_value_t()>>,
        completion_signatures<set_value_t()>>);
  sync_wait(std::move(sender));
  CHECK(value == 42);
}

TEST_CASE("construct any_sender_of recursively from when_all", "[types][any_sender]") {
  any_sender_of<set_value_t(), set_stopped_t(), set_error_t(std::exception_ptr)> sender = just();
  using sender_t = any_sender_of<set_value_t(), set_stopped_t(), set_error_t(std::exception_ptr)>;
  using when_all_t = decltype(when_all(std::move(sender)));
  static_assert(std::is_constructible_v<sender_t, when_all_t&&>);
}

TEST_CASE("sync_wait returns value", "[types][any_sender]") {
  any_sender_of<set_value_t(int)> sender = just(21) | then([&](int v) noexcept { return 2 * v; });
  CHECK(std::same_as<
        completion_signatures_of_t<any_sender_of<set_value_t(int)>>,
        completion_signatures<set_value_t(int)>>);
  auto [value1] = *sync_wait(std::move(sender));
  CHECK(value1 == 42);
}

template <class... Vals>
using my_sender_of = any_sender_of<set_value_t(Vals)..., set_error_t(std::exception_ptr)>;

TEST_CASE("sync_wait returns value and exception", "[types][any_sender]") {
  my_sender_of<int> sender = just(21) | then([&](int v) { return 2 * v; });
  auto [value] = *sync_wait(std::move(sender));
  CHECK(value == 42);

  sender = just(21) | then([&](int v) {
             throw 420;
             return 2 * v;
           });
  CHECK_THROWS_AS(sync_wait(std::move(sender)), int);
}

TEST_CASE("any_sender is connectable with any_receiver_ref", "[types][any_sender]") {
  using Sigs = completion_signatures<set_value_t(int), set_stopped_t()>;
  using receiver_ref = any_receiver_ref<Sigs>;
  using sender = receiver_ref::any_sender<>;
  REQUIRE(receiver_of<receiver_ref, Sigs>);
  sender sndr = just_stopped();
  {
    sink_receiver rcvr{};
    receiver_ref ref = rcvr;
    auto op = connect(std::move(sndr), std::move(ref));
    CHECK(rcvr.value_.index() == 0);
    start(op);
    CHECK(rcvr.value_.index() == 3);
  }
  sndr = just(42);
  {
    sink_receiver rcvr{};
    receiver_ref ref = rcvr;
    auto op = connect(std::move(sndr), std::move(ref));
    CHECK(rcvr.value_.index() == 0);
    start(op);
    CHECK(rcvr.value_.index() == 1);
  }
  sndr = when_any(just(42));
  {
    sink_receiver rcvr{};
    receiver_ref ref = rcvr;
    auto op = connect(std::move(sndr), std::move(ref));
    CHECK(rcvr.value_.index() == 0);
    start(op);
    CHECK(rcvr.value_.index() == 1);
  }
}

class stopped_token {
 private:
  bool stopped_{true};

  struct __callback_type {
    template <class Fn>
    explicit __callback_type(stopped_token t, Fn&& f) noexcept {
      if (t.stopped_) {
        static_cast<Fn&&>(f)();
      }
    }
  };
 public:
  constexpr stopped_token() noexcept = default;
  explicit constexpr stopped_token(bool stopped) noexcept
    : stopped_{stopped} {}

  template <class>
  using callback_type = __callback_type;

  static std::true_type stop_requested() noexcept {
    return {};
  }

  static std::true_type stop_possible() noexcept {
    return {};
  }

  bool operator==(const stopped_token&) const noexcept = default;
};

template <class Token>
struct stopped_receiver_base {
  Token stop_token_{};
};

template <class Token>
struct stopped_receiver_env {
  const stopped_receiver_base<Token>* receiver_;

  friend Token tag_invoke(get_stop_token_t, const stopped_receiver_env& env) noexcept {
    return env.receiver_->stop_token_;
  }
};

template <class Token>
struct stopped_receiver : stopped_receiver_base<Token> {
  stopped_receiver(Token token, bool expect_stop)
    : stopped_receiver_base<Token>{token}
    , expect_stop_{expect_stop} {
  }

  bool expect_stop_{false};

  template <class... Args>
  STDEXEC_DEFINE_CUSTOM(void set_value)(
    this const stopped_receiver& r,
    set_value_t,
    Args&&...) noexcept {
    CHECK(!r.expect_stop_);
  }

  STDEXEC_DEFINE_CUSTOM(void set_stopped)(this const stopped_receiver& r, set_stopped_t) noexcept {
    CHECK(r.expect_stop_);
  }

  STDEXEC_DEFINE_CUSTOM(stopped_receiver_env get_env)(
    this const stopped_receiver& r,
    get_env_t) noexcept {
    return {&r};
  }
};

template <class Token>
stopped_receiver(Token, bool) -> stopped_receiver<Token>;

static_assert(receiver_of<
              stopped_receiver<in_place_stop_token>,
              completion_signatures<set_value_t(int), set_stopped_t()>>);

TEST_CASE("any_sender - does connect with stop token", "[types][any_sender]") {
  using stoppable_sender = any_sender_of<set_value_t(int), set_stopped_t()>;
  stoppable_sender sender = when_any(just(21));
  in_place_stop_source stop_source{};
  stopped_receiver receiver{stop_source.get_token(), true};
  stop_source.request_stop();
  auto do_check = connect(std::move(sender), std::move(receiver));
  // This CHECKS whether set_value is called
  start(do_check);
}

TEST_CASE("any_sender - does connect with an user-defined stop token", "[types][any_sender]") {
  using stoppable_sender = any_sender_of<set_value_t(int), set_stopped_t()>;
  stoppable_sender sender = when_any(just(21));
  SECTION("stopped true") {
    stopped_token token{true};
    stopped_receiver receiver{token, true};
    auto do_check = connect(std::move(sender), std::move(receiver));
    // This CHECKS whether set_value is called
    start(do_check);
  }
  SECTION("stopped false") {
    stopped_token token{false};
    stopped_receiver receiver{token, false};
    auto do_check = connect(std::move(sender), std::move(receiver));
    // This CHECKS whether set_value is called
    start(do_check);
  }
}

TEST_CASE(
  "any_sender - does connect with stop token if the get_stop_token query is registered with "
  "in_place_stop_token",
  "[types][any_sender]") {
  using Sigs = completion_signatures<set_value_t(int), set_stopped_t()>;
  using receiver_ref =
    any_receiver_ref<Sigs, get_stop_token.signature<in_place_stop_token() noexcept>>;
  using stoppable_sender = receiver_ref::any_sender<>;
  stoppable_sender sender = when_any(just(21));
  in_place_stop_source stop_source{};
  stopped_receiver receiver{stop_source.get_token(), true};
  stop_source.request_stop();
  auto do_check = connect(std::move(sender), std::move(receiver));
  // This CHECKS whether a set_stopped is called
  start(do_check);
}

TEST_CASE(
  "any_sender - does connect with stop token if the get_stop_token query is registered with "
  "never_stop_token",
  "[types][any_sender]") {
  using Sigs = completion_signatures<set_value_t(int), set_stopped_t()>;
  using receiver_ref =
    any_receiver_ref<Sigs, get_stop_token.signature<never_stop_token() noexcept>>;
  using unstoppable_sender = receiver_ref::any_sender<>;
  unstoppable_sender sender = when_any(just(21));
  in_place_stop_source stop_source{};
  stopped_receiver receiver{stop_source.get_token(), false};
  stop_source.request_stop();
  auto do_check = connect(std::move(sender), std::move(receiver));
  // This CHECKS whether a set_stopped is called
  start(do_check);
}

///////////////////////////////////////////////////////////////////////////////
//                                                                any_scheduler

template <auto... Queries>
using my_scheduler = typename any_sender_of<>::any_scheduler<Queries...>;

TEST_CASE("any scheduler with inline_scheduler", "[types][any_sender]") {
  static_assert(scheduler<my_scheduler<>>);
  my_scheduler<> scheduler = exec::inline_scheduler();
  my_scheduler<> copied = scheduler;
  CHECK(copied == scheduler);

  auto sched = schedule(scheduler);
  static_assert(sender<decltype(sched)>);
  std::same_as<my_scheduler<>> auto get_sched = get_completion_scheduler<set_value_t>(
    get_env(sched));
  CHECK(get_sched == scheduler);

  bool called = false;
  sync_wait(std::move(sched) | then([&] { called = true; }));
  CHECK(called);
}

TEST_CASE("queryable any_scheduler with inline_scheduler", "[types][any_sender]") {
  using my_scheduler2 =
    my_scheduler<get_forward_progress_guarantee.signature<forward_progress_guarantee()>>;
  static_assert(scheduler<my_scheduler2>);
  my_scheduler2 scheduler = exec::inline_scheduler();
  my_scheduler2 copied = scheduler;
  CHECK(copied == scheduler);

  auto sched = schedule(scheduler);
  static_assert(sender<decltype(sched)>);
  std::same_as<my_scheduler2> auto get_sched = get_completion_scheduler<set_value_t>(
    get_env(sched));
  CHECK(get_sched == scheduler);

  CHECK(
    get_forward_progress_guarantee(scheduler)
    == get_forward_progress_guarantee(exec::inline_scheduler()));

  bool called = false;
  sync_wait(std::move(sched) | then([&] { called = true; }));
  CHECK(called);
}

TEST_CASE("any_scheduler adds set_value_t() completion sig", "[types][any_scheduler][any_sender]") {
  using scheduler_t = any_sender_of<>::any_scheduler<>;
  using schedule_t = decltype(schedule(std::declval<scheduler_t>()));
  CHECK(
    std::is_same_v<completion_signatures_of_t<schedule_t>, completion_signatures<set_value_t()>>);
  CHECK(scheduler<scheduler_t>);
}

TEST_CASE(
  "any_scheduler uniquely adds set_value_t() completion sig",
  "[types][any_scheduler][any_sender]") {
  using scheduler_t = any_sender_of<set_value_t()>::any_scheduler<>;
  using schedule_t = decltype(schedule(std::declval<scheduler_t>()));
  CHECK(
    std::is_same_v<completion_signatures_of_t<schedule_t>, completion_signatures<set_value_t()>>);
  CHECK(scheduler<scheduler_t>);
}

TEST_CASE(
  "any_scheduler adds set_value_t() completion sig (along with error)",
  "[types][any_sender]") {
  using scheduler_t = any_sender_of<set_error_t(std::exception_ptr)>::any_scheduler<>;
  using schedule_t = decltype(schedule(std::declval<scheduler_t>()));
  CHECK(sender_of<schedule_t, set_value_t()>);
  CHECK(sender_of<schedule_t, set_error_t(std::exception_ptr)>);
}

TEST_CASE(
  "User-defined completion_scheduler<set_value_t> is ignored",
  "[types][any_scheduler][any_sender]") {
  using not_scheduler_t =                                                 //
    any_receiver_ref<completion_signatures<set_value_t()>>                //
    ::any_sender<get_completion_scheduler<set_value_t>.signature<void()>> //
    ::any_scheduler<>;
  CHECK(scheduler<not_scheduler_t>);
}

template <auto... Queries>
using stoppable_scheduler = any_sender_of<set_stopped_t()>::any_scheduler<Queries...>;

TEST_CASE("any scheduler with static_thread_pool", "[types][any_sender]") {
  exec::static_thread_pool pool(1);
  stoppable_scheduler<> scheduler = pool.get_scheduler();
  auto copied = scheduler;
  CHECK(copied == scheduler);

  auto sched = schedule(scheduler);
  static_assert(sender<decltype(sched)>);
  std::same_as<stoppable_scheduler<>> auto get_sched = get_completion_scheduler<set_value_t>(
    get_env(sched));
  CHECK(get_sched == scheduler);

  bool called = false;
  sync_wait(std::move(sched) | then([&] { called = true; }));
  CHECK(called);
}

TEST_CASE("queryable any_scheduler with static_thread_pool", "[types][any_sender]") {
  using my_scheduler =
    stoppable_scheduler<get_forward_progress_guarantee.signature<forward_progress_guarantee()>>;

  exec::static_thread_pool pool(1);
  my_scheduler scheduler = pool.get_scheduler();
  auto copied = scheduler;
  CHECK(copied == scheduler);

  auto sched = schedule(scheduler);
  static_assert(sender<decltype(sched)>);
  std::same_as<my_scheduler> auto get_sched = get_completion_scheduler<set_value_t>(get_env(sched));
  CHECK(get_sched == scheduler);

  CHECK(
    get_forward_progress_guarantee(scheduler)
    == get_forward_progress_guarantee(pool.get_scheduler()));

  bool called = false;
  sync_wait(std::move(sched) | then([&] { called = true; }));
  CHECK(called);
}

TEST_CASE("Scheduler with error handling and set_stopped", "[types][any_scheduler][any_sender]") {
  using receiver_ref =
    any_receiver_ref<completion_signatures<set_stopped_t(), set_error_t(std::exception_ptr)>>;
  using sender_t = receiver_ref::any_sender<>;
  using scheduler_t = sender_t::any_scheduler<>;
  scheduler_t scheduler = exec::inline_scheduler();
  {
    auto op = connect(schedule(scheduler), expect_void_receiver{});
    start(op);
  }
  scheduler = stopped_scheduler();
  {
    auto op = connect(schedule(scheduler), expect_stopped_receiver{});
    start(op);
  }
  scheduler = error_scheduler<>{std::make_exception_ptr(std::logic_error("test"))};
  {
    auto op = connect(schedule(scheduler), expect_error_receiver<>{});
    start(op);
  }
}

TEST_CASE("Schedule Sender lifetime", "[types][any_scheduler][any_sender]") {
  using receiver_ref =
    any_receiver_ref<completion_signatures<set_stopped_t(), set_error_t(std::exception_ptr)>>;
  using sender_t = receiver_ref::any_sender<>;
  using scheduler_t = sender_t::any_scheduler<>;
  scheduler_t scheduler = exec::inline_scheduler();
  auto sched = schedule(scheduler);
  scheduler = stopped_scheduler();
  {
    auto op = connect(schedule(scheduler), expect_stopped_receiver{});
    start(op);
  }
  {
    auto op = connect(std::move(sched), expect_void_receiver{});
    start(op);
  }
}

// A scheduler that counts how many instances are extant.
struct counting_scheduler {
  using __id = counting_scheduler;
  using __t = counting_scheduler;

  static int count;

  counting_scheduler() noexcept {
    ++count;
  }

  counting_scheduler(const counting_scheduler&) noexcept {
    ++count;
  }

  counting_scheduler(counting_scheduler&&) noexcept {
    ++count;
  }

  ~counting_scheduler() {
    --count;
  }

  bool operator==(const counting_scheduler&) const noexcept = default;

 private:
  template <class R>
  struct operation : immovable {
    R recv_;

    STDEXEC_DEFINE_CUSTOM(void start)(this operation& self, ex::start_t) noexcept {
      ex::set_value((R&&) self.recv_);
    }
  };

  struct sender {
    using __id = sender;
    using __t = sender;

    using is_sender = void;
    using completion_signatures = ex::completion_signatures<ex::set_value_t()>;

    template <ex::receiver R>
    friend operation<R> tag_invoke(ex::connect_t, sender self, R r) {
      return {{}, (R&&) r};
    }

    friend auto tag_invoke(ex::get_completion_scheduler_t<ex::set_value_t>, sender) noexcept
      -> counting_scheduler {
      return {};
    }

    STDEXEC_DEFINE_CUSTOM(const sender& get_env)(this const sender& self, ex::get_env_t) noexcept {
      return self;
    }
  };

  friend sender tag_invoke(ex::schedule_t, counting_scheduler) noexcept {
    return {};
  }
};

int counting_scheduler::count = 0;

TEST_CASE(
  "check that any_scheduler cleans up all resources",
  "[types][any_scheduler][any_sender]") {
  using receiver_ref = any_receiver_ref<completion_signatures<set_value_t()>>;
  using sender_t = receiver_ref::any_sender<>;
  using scheduler_t = sender_t::any_scheduler<>;
  {
    scheduler_t scheduler = exec::inline_scheduler{};
    scheduler = counting_scheduler{};
    {
      auto op = connect(schedule(scheduler), expect_value_receiver<>{});
      start(op);
    }
  }
  CHECK(counting_scheduler::count == 0);
}