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

  TEST_CASE("chained stop_object is stopped", "[stop_object][async_object]") {
    auto with_stop_objects = [](handle s0, handle s1) {
      auto with_s1_stop_token = [s0](auto stp) mutable noexcept { 
        REQUIRE(s0.stop_requested() == false);
        REQUIRE(stp.stop_requested() == false);
        s0.request_stop();
        return stp.stop_requested(); 
      };
      auto inside = ex::then(ex::read_env(ex::get_stop_token), with_s1_stop_token);
      return s0.chain(s1.chain(inside));
    };
    ex::sender auto snd = async_using(with_stop_objects, stop_object{}, stop_object{});
    auto r = sync_wait(std::move(snd));
    REQUIRE(r.has_value());
    auto [v] = r.value();
    REQUIRE(v == true);
  }

  } // namespace