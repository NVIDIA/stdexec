#include <catch2/catch.hpp>
#include <exec/stop_object.hpp>
#include <exec/async_using.hpp>
#include "test_common/schedulers.hpp"
#include "test_common/receivers.hpp"

namespace ex = stdexec;
using exec::stop_object;
using exec::async_using;
using stdexec::sync_wait;

namespace {
  using handle = typename stop_object::handle;

  TEST_CASE("stop_object unused", "[stop_object][async_object]") {
    auto with_stop_object = [](handle s0) {
      return s0.chain(ex::just(false));
    };
    ex::sender auto snd = async_using(with_stop_object, stop_object{});
    auto r = sync_wait(std::move(snd));
    REQUIRE(r.has_value());
    auto [v] = r.value();
    REQUIRE(v == false);
  }

}