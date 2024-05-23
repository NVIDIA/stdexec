#include <catch2/catch.hpp>
#include <exec/async_object.hpp>
#include <exec/stop_object.hpp>
#include <exec/async_using.hpp>
#include <exec/async_tuple.hpp>

#include "test_common/schedulers.hpp"
#include "test_common/receivers.hpp"

namespace ex = stdexec;
using exec::stop_object;
using exec::async_using;
using stdexec::sync_wait;

namespace {

  TEST_CASE("async_tuple simple", "[stop_object][async_object][async_tuple]") {
    auto with_tuple = [](auto tpl) {
      auto [s0, s1] = tpl.handles();
      return s0.chain(ex::just(false));
    };
    ex::sender auto snd = async_using(
      with_tuple, 
      exec::make_async_tuple(stop_object{}, stop_object{}));
    auto r = sync_wait(std::move(snd));
    REQUIRE(r.has_value());
    auto [v] = r.value();
    REQUIRE(v == false);
  }

}