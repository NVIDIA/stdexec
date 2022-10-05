#include <atomic>
#include <catch2/catch.hpp>
#include <execution.hpp>

// clang doesn't get along with libcu++
#if STDEXEC_NVHPC()

#include <cuda/atomic>

#include "schedulers/detail/common.cuh"
#include "schedulers/stream.cuh"
#include "common.cuh"

namespace ex = std::execution;
namespace stream = example::cuda::stream;

using example::cuda::is_on_gpu;

TEST_CASE("start_detached doesn't block", "[cuda][stream][consumers][start_detached]") {
  stream::context_t stream_context{};

  int *host_flag{};
  int *device_flag{};
  THROW_ON_CUDA_ERROR(cudaMallocHost(&host_flag, sizeof(int)));
  THROW_ON_CUDA_ERROR(cudaMallocHost(&device_flag, sizeof(int)));
  *host_flag = *device_flag = 0;

  auto snd = ex::schedule(stream_context.get_scheduler()) //
           | ex::then([=] {
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

  while (device_flag_ref.load(cuda::memory_order_relaxed) == 0);

  REQUIRE(device_flag_ref.load(cuda::memory_order_relaxed) > 0);

  THROW_ON_CUDA_ERROR(cudaFreeHost(host_flag));
  THROW_ON_CUDA_ERROR(cudaFreeHost(device_flag));
}

#endif
