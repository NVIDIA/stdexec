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

struct tag {
  template <class T>
      // BUGBUG ambiguous!
      requires stdexec::tag_invocable<tag, T>
    auto operator()(T&& t) const
      noexcept(stdexec::nothrow_tag_invocable<tag, T>)
      -> stdexec::tag_invoke_result_t<tag, T> {
      return stdexec::tag_invoke(*this, (T&&) t);
    }
};

struct env {
  friend int tag_invoke(tag, env) noexcept {
    return 42;
  }
};

struct sink_receiver {
  template <class... Ts>
    friend void tag_invoke(set_value_t, sink_receiver&&, Ts&&...) noexcept {}
  template <class Err>
    friend void tag_invoke(set_value_t, sink_receiver&&, Err&&) noexcept {}
  friend void tag_invoke(set_stopped_t, sink_receiver&&) noexcept {}
  friend env tag_invoke(get_env_t, const sink_receiver&) noexcept {
    return {};
  }
};

TEST_CASE("any receiver reference", "[types][any_sender]") {

  using Sigs = completion_signatures<set_value_t()>;
  sink_receiver rcvr;
  __any::__rec::__ref<Sigs, tag(int())> ref { rcvr };

  CHECK(tag{}(get_env(ref)) == 42);
}

struct empty_vtable_t {
  private:
  template <class T>
    friend empty_vtable_t* 
    tag_invoke(__any::__create_vtable_t, __mtype<empty_vtable_t>, __mtype<T>) noexcept
    {
      static empty_vtable_t vtable{};
      return &vtable;
    }
};

TEST_CASE("empty storage is movable", "[types][any_sender]") {
  struct foo {};
  using any_unique = __any::__storage_t<__any::__unique_storage<>, empty_vtable_t>;
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
    move_throws(move_throws&&) noexcept(false) {}
    move_throws& operator=(move_throws&&) noexcept(false) { return *this; }
  };
  using any_unique = __any::__storage_t<__any::__unique_storage<>, empty_vtable_t>;
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

TEST_CASE("any receiver copyable storage", "[types][any_sender]") {

  using Sigs = completion_signatures<set_value_t()>;
  sink_receiver rcvr;
  __any::__storage_t<__any::__copyable_storage<>, __t<__any::__rec::__vtable<Sigs, tag(int())>>> vtable_holder(rcvr);
  REQUIRE(__any::__get_vtable(vtable_holder));
  REQUIRE(__any::__get_object_pointer(vtable_holder));
  
  CHECK((*__any::__get_vtable(vtable_holder))(tag{}, __any::__get_object_pointer(vtable_holder)) == 42);

  auto vtable2 = vtable_holder;
  REQUIRE(__any::__get_vtable(vtable2));
  REQUIRE(__any::__get_object_pointer(vtable2));
  CHECK((*__any::__get_vtable(vtable_holder))(tag{}, __any::__get_object_pointer(vtable_holder)) == 42);
  CHECK((*__any::__get_vtable(vtable2))(tag{}, __any::__get_object_pointer(vtable2)) == 42);

  CHECK(__any::__get_object_pointer(vtable2) != __any::__get_object_pointer(vtable_holder));
  CHECK(__any::__get_vtable(vtable2) == __any::__get_vtable(vtable_holder));

  // CHECK(tag{}(get_env(ref)) == 42);
}


TEST_CASE("any sender is a sender", "[types][any_sender]") {
  any_sender_of<set_value_t()> sender = just();
  static_assert(stdexec::sender<decltype(sender)>);
}

TEST_CASE("sync_wait works on any_sender_of", "[types][any_sender]") {
  int value = 0;
  any_sender_of<set_value_t()> sender = just(42) | then([&](int v) noexcept { value = v; });
  sync_wait(std::move(sender));
  CHECK(value == 42);
}


TEST_CASE("sync_wait returns value", "[types][any_sender]") {
  any_sender_of<int> sender = just(21) | then([&](int v) noexcept { return 2 * v; });
  static_assert(std::is_same_v<__any::__any_sender_of_t<int>, set_value_t(int)>);
  static_assert(std::same_as<completion_signatures_of_t<any_sender_of<int>>, completion_signatures<set_value_t(int)>>);
  static_assert(std::same_as<completion_signatures_of_t<any_sender_of<set_value_t(int)>>, completion_signatures<set_value_t(int)>>);
  auto [value] = *sync_wait(std::move(sender));
  CHECK(value == 42);
}

template <class... Vals>
using my_sender_of = any_sender_of<Vals..., set_error_t(std::exception_ptr)>;

// TEST_CASE("sync_wait returns value and exception", "[types][any_sender]") {
//   my_sender_of<int> sender{};// = just(21) | then([&](int v) { return 2 * v; });
//   auto [value] = *sync_wait(std::move(sender));
//   CHECK(value == 42);
// }

// TEST_CASE("any scheduler with inline_scheduler", "[types][any_sender]") {
//   static_assert(scheduler<any_scheduler>);
//   any_scheduler scheduler = exec::inline_scheduler();
//   any_scheduler copied = scheduler;
//   CHECK(copied == scheduler);

//   auto sched = schedule(scheduler);
//   static_assert(sender<decltype(sched)>);
//   std::same_as<any_scheduler> auto get_sched = get_completion_scheduler<set_value_t>(get_env(sched));
//   CHECK(get_sched == scheduler);

//   bool called = false;
//   sync_wait(std::move(sched) | then([&] { called = true; }));
//   CHECK(called);
// }

// TEST_CASE("any scheduler with static_thread_pool", "[types][any_sender]") {
//   using any_scheduler = __add_completion_signatures<exec::any_scheduler, set_stopped_t()>;

//   exec::static_thread_pool pool(1);
//   any_scheduler scheduler = pool.get_scheduler();
//   any_scheduler copied = scheduler;
//   CHECK(copied == scheduler);

//   auto sched = schedule(scheduler);
//   static_assert(sender<decltype(sched)>);
//   std::same_as<any_scheduler> auto get_sched = get_completion_scheduler<set_value_t>(get_env(sched));
//   CHECK(get_sched == scheduler);

//   bool called = false;
//   sync_wait(std::move(sched) | then([&] { called = true; }));
//   CHECK(called);
// }
