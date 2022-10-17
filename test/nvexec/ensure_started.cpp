#include <atomic>
#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

#include <cuda/atomic>

#include "nvexec/stream/common.cuh"
#include "nvexec/stream_context.cuh"
#include "common.cuh"

namespace ex = std::execution;

using nvexec::is_on_gpu;

TEST_CASE("ensure_started is eager", "[cuda][stream][adaptors][ensure_started]") {
  nvexec::stream_context stream_ctx{};

  flags_storage_t flags_storage{};
  auto flags = flags_storage.get();

  auto snd = 
    ex::ensure_started(
             ex::schedule(stream_ctx.get_scheduler())
           | ex::then([=] {
               if (is_on_gpu()) {
                 flags.set();
               }
             }));
  cudaDeviceSynchronize();

  REQUIRE(flags_storage.all_set_once());

  std::this_thread::sync_wait(std::move(snd));
}

TEST_CASE("ensure_started propagates values", "[cuda][stream][adaptors][ensure_started]") {
  nvexec::stream_context stream_ctx{};

  auto snd1 = 
    ex::ensure_started(
             ex::schedule(stream_ctx.get_scheduler())
           | ex::then([]() -> int {
               return is_on_gpu(); 
             }));

  auto snd2 = std::move(snd1)
            | ex::then([](int val) -> int {
                return val * is_on_gpu();
              });

  auto [v] = std::this_thread::sync_wait(std::move(snd2)).value();

  REQUIRE(v == 1);
}

