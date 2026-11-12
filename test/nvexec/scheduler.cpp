#include <test_common/catch2.hpp>

#include "nvexec/multi_gpu_context.cuh"
#include "nvexec/stream_context.cuh"

namespace
{
  TEST_CASE("nvexec stream scheduler equality includes priority", "[cuda][stream][scheduler]")
  {
    nvexec::stream_context stream_ctx{};

    auto high   = stream_ctx.get_scheduler(nvexec::stream_priority::high);
    auto normal = stream_ctx.get_scheduler(nvexec::stream_priority::normal);
    auto low    = stream_ctx.get_scheduler(nvexec::stream_priority::low);

    CHECK(high == stream_ctx.get_scheduler(nvexec::stream_priority::high));
    CHECK(normal == stream_ctx.get_scheduler(nvexec::stream_priority::normal));
    CHECK(low == stream_ctx.get_scheduler(nvexec::stream_priority::low));
    CHECK_FALSE(high == normal);
    CHECK_FALSE(normal == low);
    CHECK_FALSE(high == low);

    nvexec::stream_context other_stream_ctx{};
    CHECK_FALSE(high == other_stream_ctx.get_scheduler(nvexec::stream_priority::high));
  }

  TEST_CASE("nvexec multi-GPU scheduler equality includes priority", "[cuda][stream][scheduler]")
  {
    nvexec::multi_gpu_stream_context stream_ctx{};

    auto high   = stream_ctx.get_scheduler(nvexec::stream_priority::high);
    auto normal = stream_ctx.get_scheduler(nvexec::stream_priority::normal);
    auto low    = stream_ctx.get_scheduler(nvexec::stream_priority::low);

    CHECK(high == stream_ctx.get_scheduler(nvexec::stream_priority::high));
    CHECK(normal == stream_ctx.get_scheduler(nvexec::stream_priority::normal));
    CHECK(low == stream_ctx.get_scheduler(nvexec::stream_priority::low));
    CHECK_FALSE(high == normal);
    CHECK_FALSE(normal == low);
    CHECK_FALSE(high == low);
  }

  TEST_CASE("nvexec stream schedulers provide scheduler_concept", "[cuda][stream][scheduler]")
  {
    // regression guard for issue #2134: per [exec.sched], schedulers must
    // provide the scheduler_concept nested alias
    STATIC_REQUIRE(
      std::same_as<nvexec::stream_scheduler::scheduler_concept, STDEXEC::scheduler_tag>);
    STATIC_REQUIRE(
      std::same_as<nvexec::multi_gpu_stream_scheduler::scheduler_concept, STDEXEC::scheduler_tag>);

    nvexec::stream_context stream_ctx{};
    STATIC_REQUIRE(std::same_as<decltype(stream_ctx.get_scheduler()), nvexec::stream_scheduler>);
  }
}  // namespace
