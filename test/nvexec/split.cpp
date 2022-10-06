#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

#include "nvexec/stream_context.cuh"
#include "common.cuh"

namespace ex = std::execution;

using nvexec::is_on_gpu;

TEST_CASE("split returns a sender", "[cuda][stream][adaptors][split]") {
  nvexec::stream_context stream_ctx{};
  auto snd = ex::split(ex::schedule(stream_ctx.get_scheduler()));
  STATIC_REQUIRE(ex::sender<decltype(snd)>);
  (void)snd;
}

TEST_CASE("split works", "[cuda][stream][adaptors][split]") {
  nvexec::stream_context stream_ctx{};

  auto fork = ex::schedule(stream_ctx.get_scheduler()) //
            | ex::then([=] {
                return is_on_gpu(); 
              })
            | ex::split();

  auto b1 = fork | ex::then([](bool on_gpu) { return on_gpu * 24; });
  auto b2 = fork | ex::then([](bool on_gpu) { return on_gpu * 42; });

  auto [v1] = std::this_thread::sync_wait(std::move(b1)).value();
  auto [v2] = std::this_thread::sync_wait(std::move(b2)).value();

  REQUIRE(v1 == 24);
  REQUIRE(v2 == 42);
}

