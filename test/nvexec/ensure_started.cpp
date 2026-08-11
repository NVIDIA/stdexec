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
  class lifetime_counter_t
  {
    int  h_counter_storage_{};
    int* h_counter_{&h_counter_storage_};
    int* d_counter_{};

   public:
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
