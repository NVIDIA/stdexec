#include <stdexec/execution.hpp>
#include <test_common/catch2.hpp>

#include "nvexec/stream_context.cuh"

namespace
{
  TEST_CASE("continues on after just", "[cuda][stream][adaptors][continues_on]")
  {
    nvexec::stream_context ctx;

    auto sndr = STDEXEC::just() | STDEXEC::continues_on(ctx.get_scheduler());

    STDEXEC::sync_wait(std::move(sndr));
  }

  TEST_CASE("continues on after schedule", "[cuda][stream][adaptors][continues_on]")
  {
    nvexec::stream_context ctx;

    auto sndr = STDEXEC::schedule(ctx.get_scheduler()) | STDEXEC::continues_on(ctx.get_scheduler());

    STDEXEC::sync_wait(std::move(sndr));
  }

  TEST_CASE("continues on twice in a row", "[cuda][stream][adaptors][continues_on]")
  {
    nvexec::stream_context ctx;

    auto sndr = STDEXEC::just() | STDEXEC::continues_on(ctx.get_scheduler())
              | STDEXEC::continues_on(ctx.get_scheduler());

    STDEXEC::sync_wait(std::move(sndr));
  }

  TEST_CASE("nvexec sync_wait provides a scheduler", "[cuda][stream][sync_wait]")
  {
    nvexec::stream_context ctx;

    auto sndr = STDEXEC::get_scheduler() | STDEXEC::continues_on(ctx.get_scheduler());

    auto result = STDEXEC::sync_wait(std::move(sndr));

    REQUIRE(result.has_value());
  }
}  // namespace
