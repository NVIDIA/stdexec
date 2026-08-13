#include <atomic>
#include <stdexec/execution.hpp>
#include <test_common/catch2.hpp>

#include "common.cuh"
#include "nvexec/detail/cuda_atomic.cuh"
#include "nvexec/stream/common.cuh"
#include "nvexec/stream_context.cuh"

namespace ex = STDEXEC;

using nvexec::is_on_gpu;

namespace
{
  class mapped_flag_t
  {
    int* h_flag_{};
    int* d_flag_{};

   public:
    mapped_flag_t(mapped_flag_t const &)                    = delete;
    mapped_flag_t(mapped_flag_t&&)                          = delete;
    auto operator=(mapped_flag_t const &) -> mapped_flag_t& = delete;
    auto operator=(mapped_flag_t&&) -> mapped_flag_t&       = delete;

    mapped_flag_t()
    {
      STDEXEC_TRY_CUDA_API(cudaHostAlloc(&h_flag_, sizeof(int), cudaHostAllocMapped));
      STDEXEC_TRY_CUDA_API(cudaHostGetDevicePointer(&d_flag_, h_flag_, 0));
      *h_flag_ = 0;
    }

    ~mapped_flag_t()
    {
      STDEXEC_ASSERT_CUDA_API(cudaFreeHost(h_flag_));
    }

    class handle_t
    {
      int* h_flag_{};
      int* d_flag_{};

      handle_t(int* h_flag, int* d_flag)
        : h_flag_(h_flag)
        , d_flag_(d_flag)
      {}

     public:
      __host__ __device__ void set() const
      {
        cuda::atomic_ref<int, cuda::thread_scope_system> flag{*(is_on_gpu() ? d_flag_ : h_flag_)};
        flag.store(1, cuda::memory_order_release);
      }

      friend mapped_flag_t;
    };

    auto get() -> handle_t
    {
      return {h_flag_, d_flag_};
    }

    auto is_set() const -> bool
    {
      cuda::atomic_ref<int, cuda::thread_scope_system> flag{*h_flag_};
      return flag.load(cuda::memory_order_acquire) == 1;
    }
  };

  class lifetime_counter_t
  {
    int  h_counter_storage_{};
    int* h_counter_{&h_counter_storage_};
    int* d_counter_{};

   public:
    lifetime_counter_t(lifetime_counter_t const &)                    = delete;
    lifetime_counter_t(lifetime_counter_t&&)                          = delete;
    auto operator=(lifetime_counter_t const &) -> lifetime_counter_t& = delete;
    auto operator=(lifetime_counter_t&&) -> lifetime_counter_t&       = delete;

    lifetime_counter_t()
    {
      STDEXEC_TRY_CUDA_API(cudaMalloc(&d_counter_, sizeof(int)));
      STDEXEC_TRY_CUDA_API(cudaMemset(d_counter_, 0, sizeof(int)));
    }

    ~lifetime_counter_t()
    {
      STDEXEC_ASSERT_CUDA_API(cudaFree(d_counter_));
    }

    class handle_t
    {
      int* h_counter_{};
      int* d_counter_{};

      handle_t(int* h_counter, int* d_counter)
        : h_counter_(h_counter)
        , d_counter_(d_counter)
      {}

      __host__ __device__ void update(int difference) const
      {
        cuda::std::atomic_ref<int> counter{*(is_on_gpu() ? d_counter_ : h_counter_)};
        counter.fetch_add(difference, cuda::std::memory_order_relaxed);
      }

      friend lifetime_counter_t;
      friend class lifetime_tracer_t;
    };

    auto get() -> handle_t
    {
      return {h_counter_, d_counter_};
    }

    auto alive() const -> int
    {
      int d_counter{};
      STDEXEC_TRY_CUDA_API(cudaMemcpy(&d_counter, d_counter_, sizeof(int), cudaMemcpyDeviceToHost));
      return *h_counter_ + d_counter;
    }
  };

  class lifetime_tracer_t
  {
    lifetime_counter_t::handle_t counter_;

   public:
    lifetime_tracer_t()                          = delete;
    lifetime_tracer_t(lifetime_tracer_t const &) = delete;

    __host__ __device__ explicit lifetime_tracer_t(lifetime_counter_t::handle_t counter)
      : counter_(counter)
    {
      counter_.update(1);
    }

    __host__ __device__ lifetime_tracer_t(lifetime_tracer_t&& other)
      : counter_(other.counter_)
    {
      counter_.update(1);
    }

    __host__ __device__ ~lifetime_tracer_t()
    {
      counter_.update(-1);
    }
  };

  TEST_CASE("nvexec ensure_started is eager", "[cuda][stream][adaptors][ensure_started]")
  {
    nvexec::stream_context stream_ctx{};

    flags_storage_t flags_storage{};
    auto            flags = flags_storage.get();

    auto snd = exec::ensure_started(ex::schedule(stream_ctx.get_scheduler())
                                    | ex::then(
                                      [=]
                                      {
                                        if (is_on_gpu())
                                        {
                                          flags.set();
                                        }
                                      }));
    cudaDeviceSynchronize();

    REQUIRE(flags_storage.all_set_once());

    STDEXEC::sync_wait(std::move(snd));
  }

  TEST_CASE("nvexec ensure_started propagates values", "[cuda][stream][adaptors][ensure_started]")
  {
    nvexec::stream_context stream_ctx{};

    auto snd1 = exec::ensure_started(ex::schedule(stream_ctx.get_scheduler())
                                     | ex::then([]() -> bool { return is_on_gpu(); }));

    auto snd2 = std::move(snd1)
              | ex::then([](bool prev_on_gpu) -> int { return prev_on_gpu && is_on_gpu(); });

    auto [v] = STDEXEC::sync_wait(std::move(snd2)).value();

    REQUIRE(v == 1);
  }

  TEST_CASE("nvexec ensure_started destroys completion storage",
            "[cuda][stream][adaptors][ensure_started]")
  {
    nvexec::stream_context stream_ctx{};
    lifetime_counter_t     counter{};
    auto                   handle = counter.get();

    {
      auto snd = exec::ensure_started(
        ex::schedule(stream_ctx.get_scheduler())
        | a_sender([handle]() -> lifetime_tracer_t { return lifetime_tracer_t{handle}; }));
      auto result = STDEXEC::sync_wait(std::move(snd));
      REQUIRE(result.has_value());
    }

    REQUIRE(counter.alive() == 0);
  }

  TEST_CASE("nvexec ensure_started destroys completion storage when detached",
            "[cuda][stream][adaptors][ensure_started]")
  {
    lifetime_counter_t counter{};

    {
      nvexec::stream_context stream_ctx{};
      auto                   handle      = counter.get();
      auto                   make_tracer = [handle]() -> lifetime_tracer_t
      {
        NV_IF_TARGET(NV_IS_DEVICE,
                     (auto const start = clock64(); while (clock64() - start < 10'000'000){}));
        return lifetime_tracer_t{handle};
      };
      auto predecessor = a_sender(ex::schedule(stream_ctx.get_scheduler()), make_tracer);

      {
        auto snd = exec::ensure_started(std::move(predecessor));
      }

      STDEXEC_TRY_CUDA_API(cudaDeviceSynchronize());
    }

    REQUIRE(counter.alive() == 0);
  }

  TEST_CASE("nvexec ensure_started synchronizes direct stream completion when detached",
            "[cuda][stream][adaptors][ensure_started]")
  {
    int device{};
    STDEXEC_TRY_CUDA_API(cudaGetDevice(&device));

    int can_map_host_memory{};
    STDEXEC_TRY_CUDA_API(
      cudaDeviceGetAttribute(&can_map_host_memory, cudaDevAttrCanMapHostMemory, device));
    if (!can_map_host_memory)
    {
      SKIP("device does not support mapped host memory");
    }

    mapped_flag_t completion{};

    {
      nvexec::stream_context stream_ctx{};
      auto                   flag        = completion.get();
      auto                   predecessor = ex::schedule(stream_ctx.get_scheduler())
                       | ex::then(
                           [flag]() -> int
                           {
                             NV_IF_TARGET(NV_IS_DEVICE,
                                          (auto const start = clock64();
                                           while (clock64() - start < 10'000'000) {} flag.set();));
                             return 42;
                           });

      static_assert(
        nvexec::_strm::stream_sender<decltype(predecessor), nvexec::_strm::_ensure_started::env_t>);

      {
        auto snd = exec::ensure_started(std::move(predecessor));
      }

      bool const completed_before_cleanup = completion.is_set();
      STDEXEC_TRY_CUDA_API(cudaDeviceSynchronize());
      REQUIRE(completed_before_cleanup);
    }
  }

  TEST_CASE("ensure_started can preceed a sender without values",
            "[cuda][stream][adaptors][ensure_started]")
  {
    nvexec::stream_context stream_ctx{};

    flags_storage_t<2> flags_storage{};
    auto               flags = flags_storage.get();

    auto snd = exec::ensure_started(ex::schedule(stream_ctx.get_scheduler())
                                    | ex::then(
                                      [flags]
                                      {
                                        if (is_on_gpu())
                                        {
                                          flags.set(0);
                                        }
                                      }))
             | a_sender(
                 [flags]
                 {
                   if (is_on_gpu())
                   {
                     flags.set(1);
                   }
                 });
    STDEXEC::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec ensure_started can succeed a sender",
            "[cuda][stream][adaptors][ensure_started]")
  {
    SECTION("without values")
    {
      nvexec::stream_context stream_ctx{};
      flags_storage_t<2>     flags_storage{};
      auto                   flags = flags_storage.get();

      auto snd = exec::ensure_started(ex::schedule(stream_ctx.get_scheduler())
                                      | a_sender(
                                        [flags]
                                        {
                                          if (is_on_gpu())
                                          {
                                            flags.set(1);
                                          }
                                        }))
               | ex::then(
                   [flags]
                   {
                     if (is_on_gpu())
                     {
                       flags.set(0);
                     }
                   });
      STDEXEC::sync_wait(std::move(snd));

      REQUIRE(flags_storage.all_set_once());
    }

    SECTION("with values")
    {
      nvexec::stream_context stream_ctx{};
      flags_storage_t        flags_storage{};
      auto                   flags = flags_storage.get();

      auto snd = exec::ensure_started(ex::schedule(stream_ctx.get_scheduler())
                                      | a_sender([]() -> bool { return is_on_gpu(); }))
               | ex::then(
                   [flags](bool a_sender_was_on_gpu)
                   {
                     if (a_sender_was_on_gpu && is_on_gpu())
                     {
                       flags.set();
                     }
                   });
      STDEXEC::sync_wait(std::move(snd)).value();

      REQUIRE(flags_storage.all_set_once());
    }
  }
}  // namespace
