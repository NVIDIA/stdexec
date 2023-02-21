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
#include <exec/static_thread_pool.hpp>

#include <catch2/catch.hpp>


using namespace stdexec;
using namespace exec;

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

  friend const void* tag_invoke(::tag_t, env e) noexcept {
    return e.object_;
  }
};

struct sink_receiver {
  std::variant<std::monostate, int, std::exception_ptr, set_stopped_t> value_{};

  friend void tag_invoke(set_value_t, sink_receiver&& r, int value) noexcept {
    r.value_ = value;
  }

  friend void tag_invoke(set_error_t, sink_receiver&& r, std::exception_ptr e) noexcept {
    r.value_ = e;
  }

  friend void tag_invoke(set_stopped_t, sink_receiver&& r) noexcept {
    r.value_ = set_stopped;
  }

  friend env tag_invoke(get_env_t, const sink_receiver& r) noexcept {
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
  set_value((receiver_ref&&) ref, 42);
  CHECK(value.value_.index() == 1);
  CHECK(std::get<1>(value.value_) == 42);
  // Check set error
  CHECK(error.value_.index() == 0);
  ref = error;
  set_error((receiver_ref&&) ref, std::make_exception_ptr(42));
  CHECK(error.value_.index() == 2);
  CHECK_THROWS_AS(std::rethrow_exception(std::get<2>(error.value_)), int);
  // Check set stopped
  CHECK(stopped.value_.index() == 0);
  ref = stopped;
  set_stopped((receiver_ref&&) ref);
  CHECK(stopped.value_.index() == 3);
}

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

  CHECK(__any::__get_vtable(s2));
  CHECK(__any::__get_vtable(s1) != __any::__get_vtable(s2));
  CHECK(__any::__get_object_pointer(s1) == nullptr);
  CHECK(__any::__get_object_pointer(s2) != nullptr);
  // Test SBO
  std::intptr_t obj_ptr = reinterpret_cast<std::intptr_t>(__any::__get_object_pointer(s2));
  std::intptr_t s2_ptr = reinterpret_cast<std::intptr_t>(&s2);
  CHECK(std::abs(s2_ptr - obj_ptr) < std::intptr_t(sizeof(any_unique)));

  s1 = std::move(s2);
  CHECK(__any::__get_vtable(s2));
  CHECK(__any::__get_vtable(s1) != __any::__get_vtable(s2));
  CHECK(__any::__get_object_pointer(s1) != nullptr);
  CHECK(__any::__get_object_pointer(s2) == nullptr);

  s1 = std::move(s2);
  CHECK(__any::__get_object_pointer(s1) == nullptr);
  CHECK(__any::__get_object_pointer(s2) == nullptr);
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

  CHECK(__any::__get_vtable(s2));
  CHECK(__any::__get_vtable(s1) != __any::__get_vtable(s2));
  CHECK(__any::__get_object_pointer(s1) == nullptr);
  CHECK(__any::__get_object_pointer(s2) != nullptr);
  // Test SBO
  std::intptr_t obj_ptr = reinterpret_cast<std::intptr_t>(__any::__get_object_pointer(s2));
  std::intptr_t s2_ptr = reinterpret_cast<std::intptr_t>(&s2);
  CHECK(std::abs(s2_ptr - obj_ptr) >= std::intptr_t(sizeof(any_unique)));

  s1 = std::move(s2);
  CHECK(__any::__get_vtable(s2));
  CHECK(__any::__get_vtable(s1) != __any::__get_vtable(s2));
  CHECK(__any::__get_object_pointer(s1) != nullptr);
  CHECK(__any::__get_object_pointer(s2) == nullptr);

  s1 = std::move(s2);
  CHECK(__any::__get_object_pointer(s1) == nullptr);
  CHECK(__any::__get_object_pointer(s2) == nullptr);
}

template <class... Ts>
using any_sender_of =
  typename any_receiver_ref<completion_signatures<Ts...>>::template any_sender<>;

TEST_CASE("any sender is a sender", "[types][any_sender]") {
  CHECK(stdexec::sender<any_sender_of<set_value_t()>>);
  CHECK(std::is_move_assignable_v<any_sender_of<set_value_t()>>);
  CHECK(std::is_nothrow_move_assignable_v<any_sender_of<set_value_t()>>);
  CHECK(!std::is_copy_assignable_v<any_sender_of<set_value_t()>>);
  any_sender_of<set_value_t()> sender = just();
  CHECK(sender);
}

TEST_CASE("sync_wait works on any_sender_of", "[types][any_sender]") {
  int value = 0;
  any_sender_of<set_value_t()> sender = just(42) | then([&](int v) noexcept { value = v; });
  CHECK(std::same_as<
        completion_signatures_of_t<any_sender_of<set_value_t()>>,
        completion_signatures<set_value_t()>>);
  CHECK(sender);
  sync_wait(std::move(sender));
  CHECK(sender); // Should this be: CHECK(!sender); ?
  CHECK(value == 42);
  any_sender_of<set_value_t()> sender2 = std::move(sender);
  CHECK(sender2);
  // Do we unconditionally need this even for small trivially copy assignable senders?
  CHECK(!sender);
}

TEST_CASE("sync_wait returns value", "[types][any_sender]") {
  any_sender_of<set_value_t(int)> sender = just(21) | then([&](int v) noexcept { return 2 * v; });
  CHECK(std::same_as<
        completion_signatures_of_t<any_sender_of<set_value_t(int)>>,
        completion_signatures<set_value_t(int)>>);
  CHECK(sender);
  auto [value1] = *sync_wait(std::move(sender));
  CHECK(value1 == 42);
  // uh-oh? this currently works, do we want this? This is a potential footgun
  // Maybe we reset the internal object pointer if the underlying sender is not trivially copy assignable?
  CHECK(sender);
  auto [value2] = *sync_wait(std::move(sender));
  CHECK(value2 == 42);
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

TEST_CASE("any_scheduler adds set_value_t() completion sig (empty)", "[types][any_sender]") {
  using scheduler_t = any_sender_of<>::any_scheduler<>;
  using schedule_t = decltype(schedule(std::declval<scheduler_t>()));
  CHECK(
    std::is_same_v<completion_signatures_of_t<schedule_t>, completion_signatures<set_value_t()>>);
}

TEST_CASE(
  "any_scheduler adds set_value_t() completion sig (along with error)",
  "[types][any_sender]") {
  using scheduler_t = any_sender_of<set_error_t(std::exception_ptr)>::any_scheduler<>;
  using schedule_t = decltype(schedule(std::declval<scheduler_t>()));
  CHECK(sender_of<schedule_t, set_value_t()>);
  CHECK(sender_of<schedule_t, set_error_t(std::exception_ptr)>);
}

TEST_CASE("any_scheduler adds uniquely set_value_t() completion sig", "[types][any_sender]") {
  using scheduler_t = any_sender_of<set_value_t()>::any_scheduler<>;
  using schedule_t = decltype(schedule(std::declval<scheduler_t>()));
  CHECK(
    std::is_same_v<completion_signatures_of_t<schedule_t>, completion_signatures<set_value_t()>>);
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
