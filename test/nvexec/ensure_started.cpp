#include <atomic>
#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

#include "nvexec/detail/cuda_atomic.cuh"
#include "nvexec/stream/common.cuh"
#include "nvexec/stream_context.cuh"
#include "common.cuh"

namespace ex = stdexec;

using nvexec::is_on_gpu;

namespace {

  TEST_CASE("nvexec ensure_started is eager", "[cuda][stream][adaptors][ensure_started]") {
    nvexec::stream_context stream_ctx{};

    flags_storage_t flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::ensure_started(ex::schedule(stream_ctx.get_scheduler()) | ex::then([=] {
                                    if (is_on_gpu()) {
                                      flags.set();
                                    }
                                  }));
    cudaDeviceSynchronize();

    REQUIRE(flags_storage.all_set_once());

    stdexec::sync_wait(std::move(snd));
  }

  TEST_CASE("nvexec ensure_started propagates values", "[cuda][stream][adaptors][ensure_started]") {
    nvexec::stream_context stream_ctx{};

    auto snd1 = ex::ensure_started(
      ex::schedule(stream_ctx.get_scheduler()) | ex::then([]() -> bool { return is_on_gpu(); }));

    auto snd2 = std::move(snd1)
              | ex::then([](bool prev_on_gpu) -> int { return prev_on_gpu && is_on_gpu(); });

    auto [v] = stdexec::sync_wait(std::move(snd2)).value();

    REQUIRE(v == 1);
  }

  TEST_CASE(
    "ensure_started can preceed a sender without values",
    "[cuda][stream][adaptors][ensure_started]") {
    nvexec::stream_context stream_ctx{};

    flags_storage_t<2> flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::ensure_started(ex::schedule(stream_ctx.get_scheduler()) | ex::then([flags] {
                                    if (is_on_gpu()) {
                                      flags.set(0);
                                    }
                                  }))
             | a_sender([flags] {
                 if (is_on_gpu()) {
                   flags.set(1);
                 }
               });
    stdexec::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE(
    "nvexec ensure_started can succeed a sender",
    "[cuda][stream][adaptors][ensure_started]") {
    SECTION("without values") {
      nvexec::stream_context stream_ctx{};
      flags_storage_t<2> flags_storage{};
      auto flags = flags_storage.get();

      auto snd = ex::ensure_started(ex::schedule(stream_ctx.get_scheduler()) | a_sender([flags] {
                                      if (is_on_gpu()) {
                                        flags.set(1);
                                      }
                                    }))
               | ex::then([flags] {
                   if (is_on_gpu()) {
                     flags.set(0);
                   }
                 });
      stdexec::sync_wait(std::move(snd));

      REQUIRE(flags_storage.all_set_once());
    }

    SECTION("with values") {
      nvexec::stream_context stream_ctx{};
      flags_storage_t flags_storage{};
      auto flags = flags_storage.get();

      auto snd = ex::ensure_started(
                   ex::schedule(stream_ctx.get_scheduler())
                   | a_sender([]() -> bool { return is_on_gpu(); }))
               | ex::then([flags](bool a_sender_was_on_gpu) {
                   if (a_sender_was_on_gpu && is_on_gpu()) {
                     flags.set();
                   }
                 });
      stdexec::sync_wait(std::move(snd)).value();

      REQUIRE(flags_storage.all_set_once());
    }
  }
} // namespace
