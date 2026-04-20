#include <catch2/catch_all.hpp>
#include <stdexec/execution.hpp>

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
}  // namespace
