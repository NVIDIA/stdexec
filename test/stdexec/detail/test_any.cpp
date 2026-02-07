/*
 * Copyright (c) 2025 NVIDIA Corporation
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

#include <stdexec/__detail/__any.hpp>

#include <cstdio>

#include <catch2/catch.hpp> // IWYU pragma: keep

namespace any = STDEXEC::__any;

template <class Base>
struct ifoo : any::interface<ifoo, Base> {
  using ifoo::interface::interface;

  constexpr virtual void foo() {
    any::__value(*this).foo();
  }

  constexpr virtual void cfoo() const {
    any::__value(*this).cfoo();
  }
};

template <class Base>
struct ibar : any::interface<ibar, Base, any::__extends<ifoo, any::__icopyable>> {
  using ibar::interface::interface;

  constexpr virtual void bar() {
    any::__value(*this).bar();
  }
};

template <class Base>
struct ibaz : any::interface<ibaz, Base, any::__extends<ibar>, 5 * sizeof(void *)> {
  using ibaz::interface::interface;

  constexpr ~ibaz() = default;

  constexpr virtual void baz() {
    any::__value(*this).baz();
  }
};

using Small = char;
using Big = char[sizeof(any::__any<ibaz>) + 1];

template <class State>
struct foobar {
  constexpr void foo() {
    STDEXEC_IF_NOT_CONSTEVAL {
      std::printf("foo override, __value = %d\n", __value);
    }
  }

  constexpr void cfoo() const {
    STDEXEC_IF_NOT_CONSTEVAL {
      std::printf("cfoo override, __value = %d\n", __value);
    }
  }

  constexpr void bar() {
    STDEXEC_IF_NOT_CONSTEVAL {
      std::printf("bar override, __value = %d\n", __value);
    }
  }

  constexpr void baz() {
    STDEXEC_IF_NOT_CONSTEVAL {
      std::printf("baz override, __value = %d\n", __value);
    }
  }

  bool operator==(foobar const &other) const noexcept = default;

  int __value = 42;
  State state;
};

static_assert(
  std::derived_from<any::__iabstract<any::__icopyable>, any::__iabstract<any::__imovable>>);
static_assert(std::derived_from<any::__iabstract<ibar>, any::__iabstract<ifoo>>);
static_assert(!std::derived_from<any::__iabstract<ibar>, any::__iabstract<any::__icopyable>>);
static_assert(any::__extension_of<any::__iabstract<ibar>, any::__icopyable>);

// Test the Diamond of Death inheritance problem:
template <class Base>
struct IFoo : any::interface<IFoo, Base, any::__extends<any::__icopyable>> {
  using IFoo::interface::interface;

  constexpr virtual void foo() {
    any::__value(*this).foo();
  }
};

template <class Base>
struct IBar : any::interface<IBar, Base, any::__extends<any::__icopyable>> {
  using IBar::interface::interface;

  constexpr virtual void bar() {
    any::__value(*this).bar();
  }
};

template <class Base>
struct IBaz
  : any::interface<IBaz, Base, any::__extends<IFoo, IBar>> // inherits twice
                                                           // from __icopyable
{
  using IBaz::interface::interface;

  constexpr virtual void baz() {
    any::__value(*this).baz();
  }
};

static_assert(std::derived_from<any::__iabstract<IBaz>, any::__iabstract<IFoo>>);
static_assert(std::derived_from<any::__iabstract<IBaz>, any::__iabstract<any::__icopyable>>);

template <class T>
void test_deadly_diamond_of_death() {
  any::__any<IBaz> m(foobar<T>{});

  m.foo();
  m.bar();
  m.baz();
}

static_assert(any::__iabstract<ifoo>::__buffer_size < any::__iabstract<ibaz>::__buffer_size);

// test constant evaluation works
template <class T>
consteval void test_consteval() {
  any::__any<ibaz> m(foobar<T>{});
  [[maybe_unused]]
  auto x = any::__any_static_cast<foobar<T>>(m);
  x = any::__any_cast<foobar<T>>(m);
  m.foo();
  [[maybe_unused]]
  auto n = m;
  [[maybe_unused]]
  auto p = any::__caddressof(m);

  any::__any<any::__iequality_comparable> a = 42;
  if (a != a)
    throw "error";

  any::__any_ptr<ibaz> pifoo = any::__addressof(m);
  [[maybe_unused]]
  auto y = any::__any_cast<foobar<T>>(pifoo);
}

TEMPLATE_TEST_CASE("basic usage of any::__any", "[detail][any]", foobar<Small>, foobar<Big>) {
#if STDEXEC_CLANG() || (STDEXEC_GCC() && STDEXEC_GCC_VERSION >= 14'03)
  test_consteval<TestType>(); // NOLINT(invalid_consteval_call)
#endif

  any::__any<ibaz> m(foobar<TestType>{});
  REQUIRE(m.__in_situ_() == (sizeof(TestType) <= any::__iabstract<ibaz>::__buffer_size));
  REQUIRE(any::__type(m) == STDEXEC::__mtypeid<foobar<TestType>>);

  m.foo();
  m.bar();
  m.baz();

  any::__any<ifoo> n = std::move(m);
  n.foo();

  m = foobar<TestType>{};

  auto ptr = any::__caddressof(m);
  STDEXEC::__unconst(*ptr).foo();
  // ptr->foo(); // does not compile because it is a const-correctness violation
  ptr->cfoo();
  auto const ptr2 = any::__addressof(m);
  ptr2->foo();
  any::__any_ptr<ifoo> pifoo = ptr2;
  m = *ptr; // assignment from type-erased references is supported

  any::__any<any::__isemiregular> a = 42;
  any::__any<any::__isemiregular> b = 42;
  any::__any<any::__isemiregular> c = 43;
  REQUIRE(a == b);
  REQUIRE(!(a != b));
  REQUIRE(!(a == c));
  REQUIRE(a != c);

  any::__reset(b);
  REQUIRE(!(a == b));
  REQUIRE(a != b);
  REQUIRE(!(b == a));
  REQUIRE(b != a);

  any::__any<any::__iequality_comparable> x = a;
  REQUIRE(x == x);
  REQUIRE(x == a);
  REQUIRE(a == x);
  a = 43;
  REQUIRE(x != a);
  REQUIRE(a != x);

  any::__reset(a);
  REQUIRE(b == a);

  auto z = any::__caddressof(c);
  [[maybe_unused]]
  int const *p = &any::__any_cast<int>(c);
  [[maybe_unused]]
  int const *q = any::__any_cast<int>(z);

  REQUIRE(any::__any_cast<int>(z) == &any::__any_cast<int>(c));

  auto y = any::__addressof(c);
  int *r = any::__any_cast<int>(std::move(y));
  REQUIRE(r == &any::__any_cast<int>(c));

  z = y; // assign non-const ptr to const ptr
  z = &*y;

  REQUIRE(y == z);

  test_deadly_diamond_of_death<TestType>();
}
