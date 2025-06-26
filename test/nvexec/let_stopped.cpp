#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

#include "nvexec/stream_context.cuh"
#include "common.cuh"

namespace ex = stdexec;

using nvexec::is_on_gpu;

namespace {

  TEST_CASE("nvexec let_stopped returns a sender", "[cuda][stream][adaptors][let_stopped]") {
    nvexec::stream_context stream_ctx{};

    auto snd = ex::just_stopped() | ex::continues_on(stream_ctx.get_scheduler())
             | ex::let_stopped([] { return ex::just(); });
    STATIC_REQUIRE(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("nvexec let_stopped executes on GPU", "[cuda][stream][adaptors][let_stopped]") {
    nvexec::stream_context stream_ctx{};

    flags_storage_t flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::just_stopped() | ex::continues_on(stream_ctx.get_scheduler())
             | ex::let_stopped([=] {
                 if (is_on_gpu()) {
                   flags.set();
                 }

                 return ex::just();
               });
    stdexec::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE(
    "nvexec let_stopped can preceed a sender without values",
    "[cuda][stream][adaptors][let_stopped]") {
    nvexec::stream_context stream_ctx{};

    flags_storage_t<2> flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::just_stopped() | ex::continues_on(stream_ctx.get_scheduler())
             | ex::let_stopped([flags] {
                 if (is_on_gpu()) {
                   flags.set(0);
                 }

                 return ex::just();
               })
             | a_sender([flags] {
                 if (is_on_gpu()) {
                   flags.set(1);
                 }
               });
    stdexec::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec let_stopped can succeed a sender", "[cuda][stream][adaptors][let_stopped]") {
    nvexec::stream_context stream_ctx{};
    nvexec::stream_scheduler sch = stream_ctx.get_scheduler();
    flags_storage_t flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::just_stopped() | ex::continues_on(sch) | a_sender([]() noexcept {})
             | ex::let_stopped([=] {
                 if (is_on_gpu()) {
                   flags.set();
                 }

                 return ex::schedule(sch);
               });

    stdexec::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }
} // namespace
