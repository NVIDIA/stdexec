#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

#include "nvexec/stream_context.cuh"
#include "common.cuh"

namespace ex = stdexec;

using nvexec::is_on_gpu;

TEST_CASE("upon_stopped returns a sender", "[cuda][stream][adaptors][upon_stopped]") {
  nvexec::stream_context stream_ctx{};

  auto snd = ex::just_stopped() | //
             ex::transfer(stream_ctx.get_scheduler()) | //
             ex::upon_stopped([] { return ex::just(); });
  STATIC_REQUIRE(ex::sender<decltype(snd)>);
  (void)snd;
}

TEST_CASE("upon_stopped executes on GPU", "[cuda][stream][adaptors][upon_stopped]") {
  nvexec::stream_context stream_ctx{};

  flags_storage_t flags_storage{};
  auto flags = flags_storage.get();

  auto snd = ex::just_stopped() | //
             ex::transfer(stream_ctx.get_scheduler()) | //
             ex::upon_stopped([=] { 
               if (is_on_gpu()) {
                 flags.set();
               }
             });
  stdexec::sync_wait(std::move(snd));

  REQUIRE(flags_storage.all_set_once());
}

