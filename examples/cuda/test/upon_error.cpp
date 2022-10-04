#include <catch2/catch.hpp>
#include <execution.hpp>

#include "schedulers/stream.cuh"
#include "common.cuh"

namespace ex = std::execution;
namespace stream = example::cuda::stream;

using example::cuda::is_on_gpu;

TEST_CASE("upon_error returns a sender", "[cuda][stream][adaptors][upon_error]") {
  stream::context_t stream_context{};

  auto snd = ex::just_error(42) | //
             ex::transfer(stream_context.get_scheduler()) | //
             ex::upon_error([](int) { });
  STATIC_REQUIRE(ex::sender<decltype(snd)>);
  (void)snd;
}

TEST_CASE("upon_error executes on GPU", "[cuda][stream][adaptors][upon_error]") {
  stream::context_t stream_context{};

  flags_storage_t flags_storage{};
  auto flags = flags_storage.get();

  auto snd = ex::just_error(42) | //
             ex::transfer(stream_context.get_scheduler()) | //
             ex::upon_error([=](int err) { 
               if (is_on_gpu() && err == 42) {
                 flags.set();
               }
             });
  std::this_thread::sync_wait(std::move(snd));

  REQUIRE(flags_storage.all_set_once());
}

