#include "exec/sequence/merge_each.hpp"

#include "exec/sequence/transform_each.hpp"
#include "exec/sequence/ignore_all_values.hpp"
#include "exec/sequence/iterate.hpp"

#include <catch2/catch.hpp>

struct then_each_t {
  template <class Sequence, class Fn>
  auto operator()(Sequence&& sequence, Fn fn) const noexcept {
    return exec::transform_each(static_cast<Sequence&&>(sequence), stdexec::then(fn));
  }

  template <class Fn>
  stdexec::__binder_back<then_each_t, Fn> operator()(Fn fn) const noexcept {
    return {{}, {}, {static_cast<Fn&&>(fn)}};
  }
};

inline constexpr then_each_t then_each;

TEST_CASE("merge_each - with plain senders", "[sequence_senders][merge_each]") {
  int checked = 0;
  SECTION("one just") {
    auto s1 =                             //
      exec::merge_each(stdexec::just(42)) //
      | then_each([&](int x) noexcept {
          CHECK(x == 42);
          ++checked;
        })
      | exec::ignore_all_values();
    stdexec::sync_wait(s1);
    CHECK(checked == 1);
  }
  SECTION("two senders") {
    auto s1 = //
      exec::merge_each(
        stdexec::just(42), //
        stdexec::just(43)) //
      | then_each([&](int x) noexcept {
          CHECK(x == 42 + checked);
          ++checked;
        })
      | exec::ignore_all_values();
    stdexec::sync_wait(s1);
    CHECK(checked == 2);
  }
}

#if STDEXEC_HAS_STD_RANGES()
TEST_CASE("merge_each - with iterate", "[sequence_senders][merge_each]") {
  std::array<int, 3> arr = {1, 2, 3};
  auto view = std::views::all(arr);
  int checked = 0;
  auto s1 =                                                                            //
    exec::iterate(view)                                                                //
    | then_each([=](int x) noexcept { return exec::iterate(std::views::iota(0, x)); }) //
    | exec::merge_each()                                                               //
    | then_each([&](int) noexcept {
        ++checked;
      }) //
    | exec::ignore_all_values();                                                       //
  stdexec::sync_wait(s1);
  CHECK(checked == 6);
}
#endif