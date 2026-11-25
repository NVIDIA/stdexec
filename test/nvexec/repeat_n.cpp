#include <stdexec/execution.hpp>
#include <test_common/catch2.hpp>
#include <test_common/type_helpers.hpp>

#include "common.cuh"
#include "nvexec/stream_context.cuh"

namespace ex = STDEXEC;

using nvexec::is_on_gpu;

namespace
{
  struct custom_error
  {};

  TEST_CASE("nvexec repeat_n preserves child errors", "[cuda][stream][adaptors][repeat_n]")
  {
    nvexec::stream_context stream_ctx{};

    auto snd = ex::just_error(custom_error{}) | ex::continues_on(stream_ctx.get_scheduler())
             | exec::repeat_n(1);

    check_err_types<ex::__mset<custom_error, std::exception_ptr, cudaError_t>>(snd);
  }

  TEST_CASE("nvexec repeat_n repeats its child on the GPU", "[cuda][stream][adaptors][repeat_n]")
  {
    nvexec::stream_context stream_ctx{};

    flags_storage_t flags_storage{};
    auto            flags = flags_storage.get();

    auto snd = ex::schedule(stream_ctx.get_scheduler())
             | ex::then(
                 [=]
                 {
                   if (is_on_gpu())
                   {
                     flags.set();
                   }
                 })
             | exec::repeat_n(10);
    STDEXEC::sync_wait(std::move(snd));

    REQUIRE(flags_storage.is_set_n_times(10));
  }

  TEST_CASE("nvexec repeat_n with no repetitions does not start its child",
            "[cuda][stream][adaptors][repeat_n]")
  {
    nvexec::stream_context stream_ctx{};

    flags_storage_t flags_storage{};
    auto            flags = flags_storage.get();

    // Without repetitions, exec::repeat_n would complete where it starts. The stream
    // domain's repeat_n completes on the stream, and sync_wait sees that, because the
    // stream domain customizes repeat_n by re-tagging it.
    auto snd = ex::schedule(stream_ctx.get_scheduler()) | ex::then([=] { flags.set(); })
             | exec::repeat_n(0);
    STDEXEC::sync_wait(std::move(snd));

    REQUIRE(flags_storage.is_set_n_times(0));
  }
}  // namespace
