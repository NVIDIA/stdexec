#include "exec/sequence/merge_each.hpp"

#include "exec/sequence/transform_each.hpp"
#include "exec/sequence/ignore_all_values.hpp"

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