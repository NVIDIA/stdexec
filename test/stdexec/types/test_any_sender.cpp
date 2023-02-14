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

#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

using namespace std;
using namespace stdexec;

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


TEST_CASE("any receiver copyable storage", "[types][any_sender]") {

  using Sigs = completion_signatures<set_value_t()>;
  sink_receiver rcvr;
  __any::__storage_t<__any::__copyable_storage<>, __any::__rec::__vtable<Sigs, tag(int())>> vtable_holder(rcvr);
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

  using Sigs = completion_signatures<set_value_t()>;
  __any::__sender::__sender<Sigs, __types<>, __types<>> sender = stdexec::just();

  // CHECK(tag{}(get_env(ref)) == 42);
}
