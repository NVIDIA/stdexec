#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

#include "nvexec/stream_context.cuh"
#include "common.cuh"

namespace ex = stdexec;

using nvexec::is_on_gpu;

TEST_CASE("bulk returns a sender", "[cuda][stream][adaptors][bulk]") {
  nvexec::stream_context stream_ctx{};
  auto snd = ex::bulk(ex::schedule(stream_ctx.get_scheduler()), 42, [] {});
  STATIC_REQUIRE(ex::sender<decltype(snd)>);
  (void)snd;
}

TEST_CASE("bulk executes on GPU", "[cuda][stream][adaptors][bulk]") {
  nvexec::stream_context stream_ctx{};

  flags_storage_t<4> flags_storage{};
  auto flags = flags_storage.get();

  auto snd = ex::schedule(stream_ctx.get_scheduler()) //
           | ex::bulk(4, [=](int idx) {
               if (is_on_gpu()) {
                 flags.set(idx);
               }
             });
  stdexec::sync_wait(std::move(snd));

  REQUIRE(flags_storage.all_set_once());
}

TEST_CASE("bulk forwards values on GPU", "[cuda][stream][adaptors][bulk]") {
  nvexec::stream_context stream_ctx{};

  flags_storage_t<1024> flags_storage{};
  auto flags = flags_storage.get();

  auto snd = ex::transfer_just(stream_ctx.get_scheduler(), 42) //
           | ex::bulk(1024, [=](int idx, int val) {
               if (is_on_gpu()) {
                 if (val == 42) {
                   flags.set(idx);
                 }
               }
             });
  stdexec::sync_wait(std::move(snd));

  REQUIRE(flags_storage.all_set_once());
}

TEST_CASE("bulk forwards multiple values on GPU", "[cuda][stream][adaptors][bulk]") {
  nvexec::stream_context stream_ctx{};

  flags_storage_t<2> flags_storage{};
  auto flags = flags_storage.get();

  auto snd = ex::transfer_just(stream_ctx.get_scheduler(), 42, 4.2) //
           | ex::bulk(2, [=](int idx, int i, double d) {
               if (is_on_gpu()) {
                 if (i == 42 && d == 4.2) {
                   flags.set(idx);
                 }
               }
             });
  const auto [i, d] = stdexec::sync_wait(std::move(snd)).value();

  REQUIRE(flags_storage.all_set_once());
  REQUIRE(i == 42);
  REQUIRE(d == 4.2);
}

TEST_CASE("bulk can preceed a sender without values", "[cuda][stream][adaptors][bulk]") {
  nvexec::stream_context stream_ctx{};

  flags_storage_t<3> flags_storage{};
  auto flags = flags_storage.get();

  auto snd = ex::schedule(stream_ctx.get_scheduler()) //
           | ex::bulk(2, [flags](int idx) {
               if (is_on_gpu()) {
                 flags.set(idx);
               }
             })
           | a_sender([flags] {
               if (is_on_gpu()) {
                 flags.set(2);
               }
             });
  stdexec::sync_wait(std::move(snd));

  REQUIRE(flags_storage.all_set_once());
}

TEST_CASE("bulk can succeed a sender", "[cuda][stream][adaptors][bulk]") {
  SECTION("without values") {
    nvexec::stream_context stream_ctx{};
    flags_storage_t<3> flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::schedule(stream_ctx.get_scheduler()) //
             | a_sender([flags] {
                 if (is_on_gpu()) {
                   flags.set(2);
                 }
               })
             | ex::bulk(2, [flags](int idx) {
                 if (is_on_gpu()) {
                   flags.set(idx);
                 }
               });
    stdexec::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  SECTION("with values") {
    nvexec::stream_context stream_ctx{};
    flags_storage_t<2> flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::schedule(stream_ctx.get_scheduler()) //
             | a_sender([]() -> bool {
                 return is_on_gpu();
               })
             | ex::bulk(2, [flags](int idx, bool a_sender_was_on_gpu) {
                 if (a_sender_was_on_gpu * is_on_gpu()) {
                   flags.set(idx);
                 }
               });
    stdexec::sync_wait(std::move(snd)).value();

    REQUIRE(flags_storage.all_set_once());
  }
}

