#include <cstddef>
#include <cuda_runtime_api.h>

static int test_device_allocate_free_calls{};

static cudaError_t
test_device_allocate_cudaMemcpy(void*, void const *, std::size_t, cudaMemcpyKind) noexcept
{
  return cudaErrorInvalidValue;
}

static cudaError_t test_device_allocate_cudaFree(void* ptr) noexcept
{
  ++test_device_allocate_free_calls;
  return ::cudaFree(ptr);
}

#define cudaMemcpy test_device_allocate_cudaMemcpy
#define cudaFree test_device_allocate_cudaFree
#include "nvexec/detail/memory.cuh"
#undef cudaFree
#undef cudaMemcpy

#include <test_common/catch2.hpp>

namespace
{
  TEST_CASE("device allocation frees storage when cudaMemcpy fails",
            "[cuda][stream][memory][device_allocate]")
  {
    test_device_allocate_free_calls = 0;
    cudaError_t status              = cudaSuccess;

    {
      auto ptr = nvexec::_strm::device_allocate<int>(status, 42);

      REQUIRE(status == cudaErrorInvalidValue);
      REQUIRE(ptr == nullptr);
    }

    REQUIRE(test_device_allocate_free_calls == 1);
  }
}  // namespace
