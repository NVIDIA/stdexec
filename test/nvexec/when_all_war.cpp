#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

#include "nvexec/stream/common.cuh"
#include "nvexec/stream_context.cuh"
#include "common.cuh"

namespace ex = stdexec;

using nvexec::is_on_gpu;

TEST_CASE("when_all works with unknown senders", "[cuda][stream][adaptors][when_all]") {
  nvexec::stream_context stream_ctx{};
  auto sch = stream_ctx.get_scheduler();

  auto snd = ex::when_all(
      ex::schedule(sch) | ex::then([]() -> int { return is_on_gpu() * 24; }),
      ex::schedule(sch) | a_sender([]() -> int { return is_on_gpu() * 42; }));
  auto [v1, v2] = stdexec::sync_wait(std::move(snd)).value();

  REQUIRE(v1 == 24);
  REQUIRE(v2 == 42);
}

