#include <catch2/catch.hpp>
#include <cstdlib>
#include <stdexec/execution.hpp>

#include "common.cuh"
#include "nvexec/detail/cuda_atomic.cuh" // IWYU pragma: keep
#include "nvexec/stream/common.cuh"
#include "nvexec/stream_context.cuh"

namespace ex = STDEXEC;

using nvexec::is_on_gpu;

namespace {

  TEST_CASE("nvexec start_detached doesn't block", "[cuda][stream][consumers][start_detached]") {
    if (const char* env = std::getenv("CUDA_LAUNCH_BLOCKING")) {
      if (std::strlen(env) >= 1 && env[0] == '1') {
        return; // This test is unable to run when the launch is blocking
      }
    }

    nvexec::stream_context stream_ctx{};

    int* host_flag{};
    int* device_flag{};
    STDEXEC_TRY_CUDA_API(cudaMallocHost(&host_flag, sizeof(int)));
    STDEXEC_TRY_CUDA_API(cudaMallocHost(&device_flag, sizeof(int)));
    *host_flag = *device_flag = 0;

    auto snd = ex::schedule(stream_ctx.get_scheduler()) | ex::then([=] {
                 if (is_on_gpu()) {
                   cuda::atomic_ref<int, cuda::thread_scope_system> host_flag_ref(*host_flag);
                   cuda::atomic_ref<int, cuda::thread_scope_system> device_flag_ref(*device_flag);

                   int iteration{1};
                   while (host_flag_ref.load(cuda::memory_order_relaxed) == 0) {
                     iteration++;
                   }
                   device_flag_ref.store(iteration, cuda::memory_order_relaxed);
                 }
               });

    // then won't complete until we set flag, so if the `start_detached` is blocking, we'll deadlock
    ex::start_detached(std::move(snd));

    cuda::atomic_ref<int, cuda::thread_scope_system> host_flag_ref(*host_flag);
    cuda::atomic_ref<int, cuda::thread_scope_system> device_flag_ref(*device_flag);
    host_flag_ref.store(1, cuda::memory_order_relaxed);

    while (device_flag_ref.load(cuda::memory_order_relaxed) == 0)
      ;

    REQUIRE(device_flag_ref.load(cuda::memory_order_relaxed) > 0);

    STDEXEC_TRY_CUDA_API(cudaFreeHost(host_flag));
    STDEXEC_TRY_CUDA_API(cudaFreeHost(device_flag));
  }
} // namespace
