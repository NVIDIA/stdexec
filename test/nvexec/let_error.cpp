#include <exec/env.hpp>
#include <stdexec/execution.hpp>
#include <test_common/catch2.hpp>

#include "common.cuh"
#include "nvexec/stream_context.cuh"

namespace ex = STDEXEC;

using nvexec::is_on_gpu;

namespace
{

  TEST_CASE("nvexec let_error returns a sender", "[cuda][stream][adaptors][let_error]")
  {
    nvexec::stream_context stream_ctx{};

    auto snd = ex::just_error(42)                            //
             | ex::continues_on(stream_ctx.get_scheduler())  //
             | ex::let_error([](int) { return ex::just(); });
    STATIC_REQUIRE(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("nvexec let_error executes on GPU", "[cuda][stream][adaptors][let_error]")
  {
    nvexec::stream_context stream_ctx{};

    flags_storage_t flags_storage{};
    auto            flags = flags_storage.get();

    auto snd = ex::just_error(42)                            //
             | ex::continues_on(stream_ctx.get_scheduler())  //
             | ex::let_error(
                 [=](int err)
                 {
                   if (is_on_gpu() && err == 42)
                   {
                     flags.set();
                   }

                   return ex::just();
                 })
             | exec::completes_on(stream_ctx.get_scheduler());
    STDEXEC::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec let_error can preceed a sender without values",
            "[cuda][stream][adaptors][let_error]")
  {
    nvexec::stream_context stream_ctx{};

    flags_storage_t<2> flags_storage{};
    auto               flags = flags_storage.get();

    auto snd = ex::just_error(42)                            //
             | ex::continues_on(stream_ctx.get_scheduler())  //
             | ex::let_error(
                 [flags](int err)
                 {
                   if (is_on_gpu() && err == 42)
                   {
                     flags.set(0);
                   }

                   return ex::just();
                 })
             | exec::completes_on(stream_ctx.get_scheduler())
             | a_sender(
                 [flags]
                 {
                   if (is_on_gpu())
                   {
                     flags.set(1);
                   }
                 });
    STDEXEC::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec let_error can succeed a sender", "[cuda][stream][adaptors][let_error]")
  {
    nvexec::stream_context   stream_ctx{};
    nvexec::stream_scheduler sch = stream_ctx.get_scheduler();
    flags_storage_t          flags_storage{};
    auto                     flags = flags_storage.get();

    auto snd = ex::just_error(42)                            //
             | ex::continues_on(stream_ctx.get_scheduler())  //
             | a_sender([]() noexcept {})
             | ex::let_error(
                 [=](int err)
                 {
                   if (is_on_gpu() && err == 42)
                   {
                     flags.set();
                   }

                   return ex::schedule(sch);
                 })
             | exec::completes_on(stream_ctx.get_scheduler());
    STDEXEC::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }
}  // namespace
