#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

#include "nvexec/stream_context.cuh"
#include "common.cuh"

#if STDEXEC_NVHPC()

namespace ex = std::execution;

using nvexec::is_on_gpu;

TEST_CASE("let_stopped returns a sender", "[cuda][stream][adaptors][let_stopped]") {
  nvexec::stream_context stream_ctx{};

  auto snd = ex::just_stopped() | //
             ex::transfer(stream_ctx.get_scheduler()) | //
             ex::let_stopped([] { return ex::just(); });
  STATIC_REQUIRE(ex::sender<decltype(snd)>);
  (void)snd;
}

TEST_CASE("let_stopped executes on GPU", "[cuda][stream][adaptors][let_stopped]") {
  nvexec::stream_context stream_ctx{};

  flags_storage_t flags_storage{};
  auto flags = flags_storage.get();

  auto snd = ex::just_stopped() | //
             ex::transfer(stream_ctx.get_scheduler()) | //
             ex::let_stopped([=] { 
               if (is_on_gpu()) {
                 flags.set();
               }

               return ex::just();
             });
  std::this_thread::sync_wait(std::move(snd));

  REQUIRE(flags_storage.all_set_once());
}

#endif

