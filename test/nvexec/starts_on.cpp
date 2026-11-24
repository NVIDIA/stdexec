#include <stdexec/execution.hpp>
#include <test_common/catch2.hpp>

#include "common.cuh"
#include "nvexec/stream_context.cuh"

namespace ex = STDEXEC;

using nvexec::is_on_gpu;

namespace
{
  // a_sender's receiver is not a stream receiver, so the stream operation inside it
  // completes it from a kernel, and it completes starts_on's receiver from there.

  TEST_CASE("nvexec starts_on starts a sender that completes from a kernel",
            "[cuda][stream][adaptors][starts_on]")
  {
    nvexec::stream_context   stream_ctx{};
    nvexec::stream_scheduler sch = stream_ctx.get_scheduler();
    flags_storage_t          flags_storage{};
    auto                     flags = flags_storage.get();

    auto snd = ex::starts_on(sch,
                             ex::schedule(sch)
                               | a_sender(
                                 [=]() noexcept
                                 {
                                   if (is_on_gpu())
                                   {
                                     flags.set();
                                   }
                                 }))
             | ::_proclaim_completes_on(sch);
    STDEXEC::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec starts_on passes on the values of a sender that completes from a kernel",
            "[cuda][stream][adaptors][starts_on]")
  {
    nvexec::stream_context   stream_ctx{};
    nvexec::stream_scheduler sch = stream_ctx.get_scheduler();

    auto snd = ex::starts_on(sch, ex::schedule(sch) | a_sender([]() noexcept { return 42; }))
             | ::_proclaim_completes_on(sch)
             | ex::then([](int i) { return is_on_gpu() && i == 42; });
    auto [ok] = STDEXEC::sync_wait(std::move(snd)).value();

    REQUIRE(ok);
  }
}  // namespace
