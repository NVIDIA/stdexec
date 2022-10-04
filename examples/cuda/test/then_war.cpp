#include <catch2/catch.hpp>
#include <execution.hpp>

#include "schedulers/stream.cuh"
#include "common.cuh"

namespace ex = std::execution;
namespace stream = example::cuda::stream;

using example::cuda::is_on_gpu;

// nvbug/3810154
TEST_CASE("then can preceed a sender with values", "[cuda][stream][adaptors][then]") {
  stream::context_t stream_context{};

  auto snd = ex::schedule(stream_context.get_scheduler()) 
           | ex::then([]() -> bool { return is_on_gpu(); })
           | a_sender([](bool then_was_on_gpu) -> bool {
               return then_was_on_gpu * is_on_gpu(); // nvbug/3810019
             });
  auto [ok] = std::this_thread::sync_wait(std::move(snd)).value();
  REQUIRE(ok);
}

