#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

#include "common.cuh"
#include "nvexec/stream/common.cuh"
#include "nvexec/stream_context.cuh"

namespace ex = STDEXEC;

using nvexec::is_on_gpu;

namespace {

  TEST_CASE("nvexec when_all returns a sender", "[cuda][stream][adaptors][when_all]") {
    nvexec::stream_context stream_ctx{};
    auto snd = ex::when_all(
      ex::schedule(stream_ctx.get_scheduler()), ex::schedule(stream_ctx.get_scheduler()));
    STATIC_REQUIRE(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("nvexec when_all works", "[cuda][stream][adaptors][when_all]") {
    nvexec::stream_context stream_ctx{};

    flags_storage_t<2> flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::when_all(
      ex::schedule(stream_ctx.get_scheduler()) | ex::then([=] {
        if (is_on_gpu()) {
          flags.set(0);
        }
      }),
      ex::schedule(stream_ctx.get_scheduler()) | ex::then([=] {
        if (is_on_gpu()) {
          flags.set(1);
        }
      }));
    STDEXEC::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec when_all returns values", "[cuda][stream][adaptors][when_all]") {
    nvexec::stream_context stream_ctx{};

    auto snd = ex::when_all(
      ex::schedule(stream_ctx.get_scheduler()) | ex::then([] { return is_on_gpu() * 24; }),
      ex::schedule(stream_ctx.get_scheduler()) | ex::then([] { return is_on_gpu() * 42; }));
    auto [v1, v2] = STDEXEC::sync_wait(std::move(snd)).value();

    REQUIRE(v1 == 24);
    REQUIRE(v2 == 42);
  }

  TEST_CASE("nvexec when_all with many senders", "[cuda][stream][adaptors][when_all]") {
    nvexec::stream_context stream_ctx{};

    auto snd = ex::when_all(
      ex::schedule(stream_ctx.get_scheduler()) | ex::then([] { return is_on_gpu() * 1; }),
      ex::schedule(stream_ctx.get_scheduler()) | ex::then([] { return is_on_gpu() * 2; }),
      ex::schedule(stream_ctx.get_scheduler()) | ex::then([] { return is_on_gpu() * 3; }),
      ex::schedule(stream_ctx.get_scheduler()) | ex::then([] { return is_on_gpu() * 4; }),
      ex::schedule(stream_ctx.get_scheduler()) | ex::then([] { return is_on_gpu() * 5; }));
    auto [v1, v2, v3, v4, v5] = STDEXEC::sync_wait(std::move(snd)).value();

    REQUIRE(v1 == 1);
    REQUIRE(v2 == 2);
    REQUIRE(v3 == 3);
    REQUIRE(v4 == 4);
    REQUIRE(v5 == 5);
  }

  TEST_CASE("nvexec when_all works with unknown senders", "[cuda][stream][adaptors][when_all]") {
    nvexec::stream_context stream_ctx{};
    auto sch = stream_ctx.get_scheduler();

    auto snd = ex::when_all(
      ex::schedule(sch) | ex::then([]() -> int { return is_on_gpu() * 24; }),
      ex::schedule(sch) | a_sender([]() -> int { return is_on_gpu() * 42; }));
    auto [v1, v2] = STDEXEC::sync_wait(std::move(snd)).value();

    REQUIRE(v1 == 24);
    REQUIRE(v2 == 42);
  }
} // namespace
