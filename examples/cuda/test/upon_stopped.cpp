#include <catch2/catch.hpp>
#include <execution.hpp>

#include "schedulers/stream.cuh"
#include "common.cuh"

namespace ex = std::execution;
namespace stream = example::cuda::stream;

using example::cuda::is_on_gpu;

TEST_CASE("upon_stopped returns a sender", "[cuda][stream][adaptors][upon_stopped]") {
  stream::context_t stream_context{};

  auto snd = ex::just_stopped() | //
             ex::transfer(stream_context.get_scheduler()) | //
             ex::upon_stopped([] { return ex::just(); });
  STATIC_REQUIRE(ex::sender<decltype(snd)>);
  (void)snd;
}

TEST_CASE("upon_stopped executes on GPU", "[cuda][stream][adaptors][upon_stopped]") {
  stream::context_t stream_context{};

  flags_storage_t flags_storage{};
  auto flags = flags_storage.get();

  auto snd = ex::just_stopped() | //
             ex::transfer(stream_context.get_scheduler()) | //
             ex::upon_stopped([=] { 
               if (is_on_gpu()) {
                 flags.set();
               }
             });
  std::this_thread::sync_wait(std::move(snd));

  REQUIRE(flags_storage.all_set_once());
}

