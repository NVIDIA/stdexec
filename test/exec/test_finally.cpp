#include "exec/finally.hpp"
#include "exec/materialize.hpp"

#include <catch2/catch.hpp>

using namespace stdexec;

template <class Haystack>
struct __mall_contained_in {
  template <class... Needles>
  using __f = __mand<__mapply<__contains<Needles>, Haystack>...>;
};

template <class Needles, class Haystack>
using __all_contained_in = __mapply<__mall_contained_in<Haystack>, Needles>;

template <class Needles, class Haystack>
using __equivalent = __mand<
  __all_contained_in<Needles, Haystack>,
  std::is_same<__mapply<__msize, Needles>, __mapply<__msize, Haystack>>>;

TEST_CASE("finally is a sender", "[adaptors][finally]") {
  auto s = exec::finally(just(), just());
  STATIC_REQUIRE(sender<decltype(s)>);
}

TEST_CASE("finally is a sender in empty env", "[adaptors][finally]") {
  auto s = exec::finally(just(), just());
  STATIC_REQUIRE(sender_in<decltype(s), empty_env>);
  STATIC_REQUIRE(__v<__equivalent<
                   completion_signatures_of_t<decltype(s), empty_env>,
                   completion_signatures<set_error_t(std::exception_ptr), set_value_t()>>>);
}

TEST_CASE("finally executes the final action", "[adaptors][finally]") {
  bool called = false;
  auto s = exec::finally(just(), just() | then([&called]() noexcept { called = true; }));
  STATIC_REQUIRE(__v<__equivalent<
                   completion_signatures_of_t<decltype(s), empty_env>,
                   completion_signatures<set_error_t(std::exception_ptr), set_value_t()>>>);
  sync_wait(s);
  CHECK(called);
}

TEST_CASE("finally executes the final action and returns integer", "[adaptors][finally]") {
  bool called = false;
  auto s = exec::finally(just(42), just() | then([&called]() noexcept { called = true; }));
  STATIC_REQUIRE(__v<__equivalent<
                   completion_signatures_of_t<decltype(s), empty_env>,
                   completion_signatures<set_error_t(std::exception_ptr), set_value_t(int&&)>>>);
  auto [i] = *sync_wait(s);
  CHECK(called);
  CHECK(i == 42);
}

TEST_CASE("finally does not execute the final action and throws integer", "[adaptors][finally]") {
  bool called = false;

  auto s = exec::finally(
    just(21) | then([](int x) {
      throw 42;
      return x;
    }),
    just() | then([&called]() noexcept { called = true; }));
  STATIC_REQUIRE(__v<__equivalent<
                   completion_signatures_of_t<decltype(s), empty_env>,
                   completion_signatures<set_error_t(std::exception_ptr), set_value_t(int&&)>>>);
  CHECK_THROWS_AS(sync_wait(s), int);
  CHECK(called);
}
