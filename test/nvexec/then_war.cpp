#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

#include "nvexec/stream_context.cuh"
#include "common.cuh"

namespace ex = stdexec;

using nvexec::is_on_gpu;

// nvbug/3810154
TEST_CASE("then can preceed a sender with values", "[cuda][stream][adaptors][then]") {
  nvexec::stream_context stream_ctx{};

  auto snd = ex::schedule(stream_ctx.get_scheduler()) 
           | ex::then([]() -> bool { return is_on_gpu(); })
           | a_sender([](bool then_was_on_gpu) -> bool {
               return then_was_on_gpu * is_on_gpu(); // nvbug/3810019
             });
  auto [ok] = stdexec::sync_wait(std::move(snd)).value();
  REQUIRE(ok);
}

