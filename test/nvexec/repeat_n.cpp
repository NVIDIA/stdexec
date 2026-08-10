#include <stdexec/execution.hpp>
#include <test_common/catch2.hpp>
#include <test_common/type_helpers.hpp>

#include "nvexec/stream_context.cuh"

namespace ex = STDEXEC;

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
}  // namespace
