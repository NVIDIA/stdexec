#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

#include "common.cuh"
#include "nvexec/stream_context.cuh"

#include <cuda/std/span>

namespace ex = STDEXEC;

using nvexec::is_on_gpu;

namespace {

  TEST_CASE("nvexec bulk returns a sender", "[cuda][stream][adaptors][bulk]") {
    nvexec::stream_context stream_ctx{};
    auto snd = ex::bulk(ex::schedule(stream_ctx.get_scheduler()), ex::par, 42, [](int) { });
    STATIC_REQUIRE(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("nvexec bulk executes on GPU", "[cuda][stream][adaptors][bulk]") {
    nvexec::stream_context stream_ctx{};

    flags_storage_t<4> flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::schedule(stream_ctx.get_scheduler()) | ex::bulk(ex::par, 4, [=](int idx) {
                 if (is_on_gpu()) {
                   flags.set(idx);
                 }
               });
    STDEXEC::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec bulk forwards values on GPU", "[cuda][stream][adaptors][bulk]") {
    nvexec::stream_context stream_ctx{};

    flags_storage_t<1024> flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::transfer_just(stream_ctx.get_scheduler(), 42)
             | ex::bulk(ex::par, 1024, [=](int idx, int val) {
                 if (is_on_gpu()) {
                   if (val == 42) {
                     flags.set(idx);
                   }
                 }
               });
    STDEXEC::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec bulk forwards multiple values on GPU", "[cuda][stream][adaptors][bulk]") {
    nvexec::stream_context stream_ctx{};

    flags_storage_t<2> flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::transfer_just(stream_ctx.get_scheduler(), 42, 4.2)
             | ex::bulk(ex::par, 2, [=](int idx, int i, double d) {
                 if (is_on_gpu()) {
                   if (i == 42 && d == 4.2) {
                     flags.set(idx);
                   }
                 }
               });
    const auto [i, d] = STDEXEC::sync_wait(std::move(snd)).value();

    REQUIRE(flags_storage.all_set_once());
    REQUIRE(i == 42);
    REQUIRE(d == 4.2);
  }

  TEST_CASE(
    "bulk forwards values that can be taken by reference on GPU",
    "[cuda][stream][adaptors][bulk]") {
    nvexec::stream_context stream_ctx{};

    flags_storage_t<1024> flags_storage{};
    using flags_t = flags_storage_t<1024>::flags_t;
    auto flags = flags_storage.get();

    auto snd = ex::transfer_just(stream_ctx.get_scheduler(), flags)
             | ex::bulk(ex::par, 1024, [](int idx, const flags_t& flags) {
                 if (is_on_gpu()) {
                   flags.set(idx);
                 }
               });
    [[maybe_unused]] auto [flags_actual] = STDEXEC::sync_wait(std::move(snd)).value();

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec bulk can preceed a sender without values", "[cuda][stream][adaptors][bulk]") {
    nvexec::stream_context stream_ctx{};

    flags_storage_t<3> flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::schedule(stream_ctx.get_scheduler())
             | ex::bulk(
                 ex::par,
                 2,
                 [flags](int idx) {
                   if (is_on_gpu()) {
                     flags.set(idx);
                   }
                 })
             | a_sender([flags] {
                 if (is_on_gpu()) {
                   flags.set(2);
                 }
               });
    STDEXEC::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec bulk can succeed a sender", "[cuda][stream][adaptors][bulk]") {
    SECTION("without values") {
      nvexec::stream_context stream_ctx{};
      flags_storage_t<3> flags_storage{};
      auto flags = flags_storage.get();

      auto snd = ex::schedule(stream_ctx.get_scheduler()) | a_sender([flags] {
                   if (is_on_gpu()) {
                     flags.set(2);
                   }
                 })
               | ex::bulk(ex::par, 2, [flags](int idx) {
                   if (is_on_gpu()) {
                     flags.set(idx);
                   }
                 });
      STDEXEC::sync_wait(std::move(snd));

      REQUIRE(flags_storage.all_set_once());
    }

    SECTION("with values") {
      nvexec::stream_context stream_ctx{};
      flags_storage_t<2> flags_storage{};
      auto flags = flags_storage.get();

      auto snd = ex::schedule(stream_ctx.get_scheduler())
               | a_sender([]() -> bool { return is_on_gpu(); })
               | ex::bulk(ex::par, 2, [flags](int idx, bool a_sender_was_on_gpu) {
                   if (a_sender_was_on_gpu && is_on_gpu()) {
                     flags.set(idx);
                   }
                 });
      STDEXEC::sync_wait(std::move(snd)).value();

      REQUIRE(flags_storage.all_set_once());
    }
  }

  TEST_CASE(
    "nvexec bulk can succeed a sender that sends ref into opstate",
    "[cuda][stream][adaptors][bulk]") {
    nvexec::stream_context ctx;

    double* inout = nullptr;
    const int nelems = 10;
    cudaMallocManaged(&inout, nelems * sizeof(double));

    auto task = STDEXEC::transfer_just(ctx.get_scheduler(), cuda::std::span<double>{inout, nelems})
              | STDEXEC::bulk(
                  ex::par,
                  nelems,
                  [](std::size_t i, cuda::std::span<double> out) { out[i] = (double) i; })
              | STDEXEC::let_value([](cuda::std::span<double> out) { return STDEXEC::just(out); })
              | STDEXEC::bulk(ex::par, nelems, [](std::size_t i, cuda::std::span<double> out) {
                  out[i] = 2.0 * out[i];
                });

    STDEXEC::sync_wait(std::move(task)).value();

    for (int i = 0; i < nelems; ++i) {
      REQUIRE(i * 2 == (int) inout[i]);
    }

    cudaFree(inout);
  }
} // namespace
