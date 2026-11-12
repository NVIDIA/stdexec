#include <cuda_runtime_api.h>

static int test_stream_pool_create_calls{};
static int test_stream_pool_destroy_calls{};

static cudaError_t test_stream_pool_cudaStreamCreate(cudaStream_t* stream) noexcept
{
  ++test_stream_pool_create_calls;
  *stream = nullptr;
  return cudaErrorMemoryAllocation;
}

static cudaError_t test_stream_pool_cudaStreamDestroy(cudaStream_t) noexcept
{
  ++test_stream_pool_destroy_calls;
  return cudaSuccess;
}

#define cudaStreamCreate test_stream_pool_cudaStreamCreate
#define cudaStreamDestroy test_stream_pool_cudaStreamDestroy
#include "nvexec/stream/common.cuh"
#undef cudaStreamDestroy
#undef cudaStreamCreate

#include <test_common/catch2.hpp>

namespace
{
  TEST_CASE("stream provider does not pool a failed stream", "[cuda][stream][stream_pool]")
  {
    test_stream_pool_create_calls  = 0;
    test_stream_pool_destroy_calls = 0;

    {
      nvexec::_strm::stream_pools_t stream_pools;
      nvexec::_strm::context        context{nullptr, nullptr, &stream_pools, nullptr};

      {
        nvexec::_strm::stream_provider provider{context};
        REQUIRE(provider.status_ == cudaErrorMemoryAllocation);
        CHECK_FALSE(provider.own_stream_.has_value());
      }

      {
        nvexec::_strm::stream_provider provider{context};
        CHECK(provider.status_ == cudaErrorMemoryAllocation);
        CHECK_FALSE(provider.own_stream_.has_value());
      }

      CHECK(test_stream_pool_create_calls == 2);
    }

    CHECK(test_stream_pool_destroy_calls == 0);
  }
}  // namespace
