#include <nvexec/nvtx.cuh>
#include <nvexec/stream_context.cuh>
#include <stdexec/execution.hpp>
#include <test_common/catch2.hpp>

namespace ex = STDEXEC;

namespace
{
  TEST_CASE("nvexec nvtx preserves names for lvalue senders", "[cuda][stream][nvtx]")
  {
    nvexec::stream_context stream_ctx{};
    auto                   snd = ex::schedule(stream_ctx.get_scheduler())
                 | nvexec::nvtx::push("test");

    ex::sync_wait(snd);

    CHECK(snd.name_ == "test");
  }
}  // namespace
