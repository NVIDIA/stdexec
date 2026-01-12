#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

#include "common.cuh"
#include "nvexec/stream_context.cuh"

namespace ex = STDEXEC;

using nvexec::is_on_gpu;

namespace {

  TEST_CASE("nvexec upon_stopped returns a sender", "[cuda][stream][adaptors][upon_stopped]") {
    nvexec::stream_context stream_ctx{};

    auto snd = ex::just_stopped() | ex::continues_on(stream_ctx.get_scheduler())
             | ex::upon_stopped([] { return ex::just(); });
    STATIC_REQUIRE(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("nvexec upon_stopped executes on GPU", "[cuda][stream][adaptors][upon_stopped]") {
    nvexec::stream_context stream_ctx{};

    flags_storage_t flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::just_stopped() | ex::continues_on(stream_ctx.get_scheduler())
             | ex::upon_stopped([=] {
                 if (is_on_gpu()) {
                   flags.set();
                 }
               });
    STDEXEC::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }
} // namespace
