#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

#include "exec/inline_scheduler.hpp"
#include "nvexec/stream_context.cuh"
#include "common.cuh"

namespace ex = stdexec;

using nvexec::is_on_gpu;

namespace {

  TEST_CASE(
    "nvexec transfer to stream context returns a sender",
    "[cuda][stream][adaptors][transfer]") {
    nvexec::stream_context stream_ctx{};
    stdexec::inline_scheduler cpu{};
    nvexec::stream_scheduler gpu = stream_ctx.get_scheduler();

    auto snd = ex::schedule(cpu) | ex::continues_on(gpu);
    STATIC_REQUIRE(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE(
    "nvexec transfer from stream context returns a sender",
    "[cuda][stream][adaptors][transfer]") {
    nvexec::stream_context stream_ctx{};

    stdexec::inline_scheduler cpu{};
    nvexec::stream_scheduler gpu = stream_ctx.get_scheduler();

    auto snd = ex::schedule(gpu) | ex::continues_on(cpu);
    STATIC_REQUIRE(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("nvexec transfer changes context to GPU", "[cuda][stream][adaptors][transfer]") {
    nvexec::stream_context stream_ctx{};

    stdexec::inline_scheduler cpu{};
    nvexec::stream_scheduler gpu = stream_ctx.get_scheduler();

    auto snd = ex::schedule(cpu) | ex::then([=] {
                 if (!is_on_gpu()) {
                   return 1;
                 }
                 return 0;
               })
             | ex::continues_on(gpu) | ex::then([=](int val) -> int {
                 if (is_on_gpu() && val == 1) {
                   return 2;
                 }
                 return 0;
               });
    const auto [result] = stdexec::sync_wait(std::move(snd)).value();

    REQUIRE(result == 2);
  }

  TEST_CASE("nvexec transfer changes context from GPU", "[cuda][stream][adaptors][transfer]") {
    nvexec::stream_context stream_ctx{};

    stdexec::inline_scheduler cpu{};
    nvexec::stream_scheduler gpu = stream_ctx.get_scheduler();

    auto snd = ex::schedule(gpu) | ex::then([=] {
                 if (is_on_gpu()) {
                   return 1;
                 }
                 return 0;
               })
             | ex::continues_on(cpu) | ex::then([=](int val) -> int {
                 if (!is_on_gpu() && val == 1) {
                   return 2;
                 }
                 return 0;
               });
    const auto [result] = stdexec::sync_wait(std::move(snd)).value();

    REQUIRE(result == 2);
  }

  TEST_CASE("nvexec transfer_just changes context to GPU", "[cuda][stream][adaptors][transfer]") {
    nvexec::stream_context stream_ctx{};
    nvexec::stream_scheduler gpu = stream_ctx.get_scheduler();

    auto snd = ex::transfer_just(gpu, 42) | ex::then([=](auto i) {
                 if (is_on_gpu() && i == 42) {
                   return true;
                 }
                 return false;
               });
    const auto [result] = stdexec::sync_wait(std::move(snd)).value();

    REQUIRE(result == true);
  }

  TEST_CASE("nvexec transfer_just supports move-only types", "[cuda][stream][adaptors][transfer]") {
    nvexec::stream_context stream_ctx{};
    nvexec::stream_scheduler gpu = stream_ctx.get_scheduler();

    auto snd = ex::transfer_just(gpu, move_only_t{42}) | ex::then([=](move_only_t&& val) noexcept {
                 if (is_on_gpu() && val.contains(42)) {
                   return true;
                 }
                 return false;
               });
    const auto [result] = stdexec::sync_wait(std::move(snd)).value();

    REQUIRE(result == true);
  }

  TEST_CASE("nvexec transfer supports move-only types", "[cuda][stream][adaptors][transfer]") {
    nvexec::stream_context stream_ctx{};

    stdexec::inline_scheduler cpu{};
    nvexec::stream_scheduler gpu = stream_ctx.get_scheduler();

    auto snd = ex::schedule(gpu) | ex::then([] { return move_only_t{42}; }) | ex::continues_on(cpu)
             | ex::then([=](move_only_t val) noexcept {
                 if (!is_on_gpu() && val.contains(42)) {
                   return true;
                 }
                 return false;
               });
    const auto [result] = stdexec::sync_wait(std::move(snd)).value();

    REQUIRE(result == true);
  }
} // namespace
