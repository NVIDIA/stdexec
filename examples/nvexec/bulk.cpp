#include <nvexec/stream_context.cuh>
#include <stdexec/execution.hpp>
#include <exec/inline_scheduler.hpp>

#include <cstdio>

namespace ex = stdexec;

int main() {
  nvexec::stream_context stream_ctx{};

  auto snd = ex::schedule(stream_ctx.get_scheduler()) //
           | ex::let_value([=] __host__ __device__ {
               return ex::just();
             });
  stdexec::sync_wait(std::move(snd));
}

