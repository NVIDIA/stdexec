#include <stdexec/execution.hpp>
#include <test_common/catch2.hpp>

#include "common.cuh"
#include "nvexec/stream_context.cuh"

#include <memory_resource>
#include <new>
#include <optional>
#include <tuple>

namespace
{
  class pinned_memory_resource_t : public std::pmr::memory_resource
  {
    std::size_t allocations_{};
    std::size_t deallocations_{};
    std::size_t fail_after_{};
    bool        fail_enabled_{};

    void* do_allocate(std::size_t bytes, std::size_t) override
    {
      if (fail_enabled_ && allocations_ == fail_after_)
      {
        throw std::bad_alloc();
      }

      void* storage{};
      STDEXEC_TRY_CUDA_API(cudaMallocHost(&storage, bytes));
      ++allocations_;
      return storage;
    }

    void do_deallocate(void* storage, std::size_t, std::size_t) override
    {
      STDEXEC_ASSERT_CUDA_API(cudaFreeHost(storage));
      ++deallocations_;
    }

    auto do_is_equal(std::pmr::memory_resource const & other) const noexcept -> bool override
    {
      return this == &other;
    }

   public:
    auto allocations() const noexcept -> std::size_t
    {
      return allocations_;
    }

    auto deallocations() const noexcept -> std::size_t
    {
      return deallocations_;
    }

    void fail_after(std::size_t allocation) noexcept
    {
      fail_after_   = allocation;
      fail_enabled_ = true;
    }
  };

  struct noop_receiver
  {
    using receiver_concept = STDEXEC::receiver_tag;

    auto get_env() const noexcept -> STDEXEC::env<>
    {
      return {};
    }

    void set_value() noexcept {}

    template <class Error>
    void set_error(Error&&) noexcept
    {}

    void set_stopped() noexcept {}
  };

  struct error_receiver
  {
    using receiver_concept = STDEXEC::receiver_tag;

    cudaError_t* error_;

    auto get_env() const noexcept -> STDEXEC::env<>
    {
      return {};
    }

    void set_value() noexcept {}

    void set_error(cudaError_t error) noexcept
    {
      *error_ = error;
    }

    void set_stopped() noexcept {}
  };

  class destruction_probe_t
  {
    flags_storage_t<>::flags_t flags_;
    bool                       owns_{true};

   public:
    destruction_probe_t()                                               = delete;
    destruction_probe_t(destruction_probe_t const &)                    = delete;
    auto operator=(destruction_probe_t const &) -> destruction_probe_t& = delete;
    auto operator=(destruction_probe_t&&) -> destruction_probe_t&       = delete;

    __host__ __device__ explicit destruction_probe_t(flags_storage_t<>::flags_t flags)
      : flags_(flags)
    {}

    __host__ __device__ destruction_probe_t(destruction_probe_t&& other)
      : flags_(other.flags_)
      , owns_(other.owns_)
    {
      other.owns_ = false;
    }

    __host__ __device__ ~destruction_probe_t()
    {
      if (owns_)
      {
        flags_.set();
      }
    }
  };

  TEST_CASE("continues on after just", "[cuda][stream][adaptors][continues_on]")
  {
    nvexec::stream_context ctx;

    auto sndr = STDEXEC::just() | STDEXEC::continues_on(ctx.get_scheduler());

    STDEXEC::sync_wait(std::move(sndr));
  }

  TEST_CASE("schedule_from frees its task when the operation is not started",
            "[cuda][stream][adaptors][schedule_from]")
  {
    pinned_memory_resource_t pinned_memory;
    nvexec::stream_context   ctx;
    auto                     scheduler = ctx.get_scheduler();
    scheduler.ctx_.pinned_resource_    = &pinned_memory;

    auto sndr = STDEXEC::schedule_from(STDEXEC::schedule(scheduler));
    {
      auto op = STDEXEC::connect(std::move(sndr), noop_receiver{});
      (void) op;
    }

    REQUIRE(pinned_memory.allocations() > 0);
    REQUIRE(pinned_memory.allocations() == pinned_memory.deallocations());
  }

  TEST_CASE("schedule_from frees its task when setup fails",
            "[cuda][stream][adaptors][schedule_from]")
  {
    pinned_memory_resource_t pinned_memory;
    pinned_memory.fail_after(2);

    nvexec::stream_context ctx;
    auto                   scheduler = ctx.get_scheduler();
    scheduler.ctx_.pinned_resource_  = &pinned_memory;

    cudaError_t error = cudaSuccess;
    auto        sndr  = STDEXEC::schedule_from(STDEXEC::schedule(scheduler));
    {
      auto op = STDEXEC::connect(std::move(sndr), error_receiver{&error});
      STDEXEC::start(op);
    }

    REQUIRE(error == cudaErrorMemoryAllocation);
    REQUIRE(pinned_memory.allocations() == 2);
    REQUIRE(pinned_memory.allocations() == pinned_memory.deallocations());
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

  TEST_CASE("nvexec sync_wait provides a scheduler", "[cuda][stream][sync_wait]")
  {
    nvexec::stream_context ctx;

    auto sndr = STDEXEC::get_scheduler() | STDEXEC::continues_on(ctx.get_scheduler());

    auto result = STDEXEC::sync_wait(std::move(sndr));

    REQUIRE(result.has_value());
  }

  TEST_CASE("continues_on destroys host-constructed storage after a CUDA error",
            "[cuda][stream][adaptors][continues_on]")
  {
    int device{};
    STDEXEC_TRY_CUDA_API(cudaGetDevice(&device));

    int concurrent_managed_access{};
    STDEXEC_TRY_CUDA_API(cudaDeviceGetAttribute(&concurrent_managed_access,
                                                cudaDevAttrConcurrentManagedAccess,
                                                device));
    if (!concurrent_managed_access)
    {
      SKIP("device does not support concurrent managed access");
    }

    pinned_memory_resource_t pinned_memory;
    nvexec::stream_context   ctx;
    auto                     scheduler = ctx.get_scheduler();
    scheduler.ctx_.managed_resource_   = &pinned_memory;

    flags_storage_t<> destructions{};
    auto              sndr = STDEXEC::just(destruction_probe_t{destructions.get()})
              | STDEXEC::continues_on(scheduler);

    REQUIRE_THROWS(STDEXEC::sync_wait(std::move(sndr)));
    REQUIRE(destructions.all_set_once());
  }

  template <bool DestroySourceFirst>
  auto continue_on_other_context() -> int
  {
    std::optional<nvexec::stream_context> first{std::in_place};
    std::optional<nvexec::stream_context> second{std::in_place};
    nvexec::stream_context&               src = DestroySourceFirst ? *first : *second;
    nvexec::stream_context&               dst = DestroySourceFirst ? *second : *first;

    auto sndr = STDEXEC::schedule(src.get_scheduler())
              | STDEXEC::then([]() -> int { return nvexec::is_on_gpu() ? 1 : 0; })
              | STDEXEC::continues_on(dst.get_scheduler())
              | STDEXEC::then([](int val) -> int
                              { return nvexec::is_on_gpu() && val == 1 ? 2 : 0; });

    auto [result] = STDEXEC::sync_wait(std::move(sndr)).value();

    first.reset();
    second.reset();
    return result;
  }

  TEST_CASE("continues_on between two stream contexts", "[cuda][stream][adaptors][continues_on]")
  {
    SECTION("source context destroyed first")
    {
      REQUIRE(continue_on_other_context<true>() == 2);
    }

    SECTION("destination context destroyed first")
    {
      REQUIRE(continue_on_other_context<false>() == 2);
    }
  }

  TEST_CASE("continues_on between two stream contexts without values",
            "[cuda][stream][adaptors][continues_on]")
  {
    bool on_gpu = false;
    {
      nvexec::stream_context src;
      nvexec::stream_context dst;

      auto sndr = STDEXEC::schedule(src.get_scheduler())
                | STDEXEC::continues_on(dst.get_scheduler())
                | STDEXEC::then([]() -> bool { return nvexec::is_on_gpu(); });

      on_gpu = std::get<0>(STDEXEC::sync_wait(std::move(sndr)).value());
    }
    REQUIRE(on_gpu);
  }

  // Records the stream each step ran on, readable from host and device.
  struct stream_record
  {
    cudaStream_t* streams{};

    stream_record()
    {
      STDEXEC_TRY_CUDA_API(cudaMallocManaged(&streams, 2 * sizeof(cudaStream_t)));
      streams[0] = nullptr;
      streams[1] = nullptr;
    }

    stream_record(stream_record const &)                    = delete;
    auto operator=(stream_record const &) -> stream_record& = delete;

    ~stream_record()
    {
      STDEXEC_ASSERT_CUDA_API(cudaFree(streams));
    }
  };

  template <class FirstSched, class SecondSched>
  void record_streams_across(FirstSched first, SecondSched second, cudaStream_t* streams)
  {
    auto sndr = STDEXEC::starts_on(first, nvexec::get_stream())
              | STDEXEC::then([streams](cudaStream_t stream) { streams[0] = stream; })
              | STDEXEC::continues_on(second)
              | STDEXEC::let_value([] { return nvexec::get_stream(); })
              // proclaim that value completions will happen on the specified scheduler:
              | exec::write_attrs(
                  STDEXEC::prop{STDEXEC::get_completion_scheduler<STDEXEC::set_value_t>, second})
              | STDEXEC::then([streams](cudaStream_t stream) { streams[1] = stream; });

    STDEXEC::sync_wait(std::move(sndr));
  }

  TEST_CASE("continues_on between two stream contexts switches streams",
            "[cuda][stream][adaptors][continues_on]")
  {
    nvexec::stream_context ctx_0;
    nvexec::stream_context ctx_1;
    stream_record          record;

    record_streams_across(ctx_0.get_scheduler(), ctx_1.get_scheduler(), record.streams);

    REQUIRE(record.streams[0] != nullptr);
    REQUIRE(record.streams[1] != nullptr);
    CHECK(record.streams[0] != record.streams[1]);
  }

  TEST_CASE("continues_on within one stream context keeps the stream",
            "[cuda][stream][adaptors][continues_on]")
  {
    nvexec::stream_context ctx;
    stream_record          record;

    record_streams_across(ctx.get_scheduler(), ctx.get_scheduler(), record.streams);

    REQUIRE(record.streams[0] != nullptr);
    CHECK(record.streams[0] == record.streams[1]);
  }

  TEST_CASE("continues_on to another priority switches streams",
            "[cuda][stream][adaptors][continues_on]")
  {
    int least{};
    int greatest{};
    STDEXEC_TRY_CUDA_API(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    if (least == greatest)
    {
      SKIP("the device has a single stream priority");
    }

    nvexec::stream_context ctx;
    stream_record          record;

    record_streams_across(ctx.get_scheduler(nvexec::stream_priority::low),
                          ctx.get_scheduler(nvexec::stream_priority::high),
                          record.streams);

    REQUIRE(record.streams[0] != nullptr);
    REQUIRE(record.streams[1] != nullptr);
    CHECK(record.streams[0] != record.streams[1]);

    int first{};
    int second{};
    STDEXEC_TRY_CUDA_API(cudaStreamGetPriority(record.streams[0], &first));
    STDEXEC_TRY_CUDA_API(cudaStreamGetPriority(record.streams[1], &second));
    CHECK(first == least);
    CHECK(second == greatest);
  }

  TEST_CASE("continues_on between two stream contexts waits for the earlier work",
            "[cuda][stream][adaptors][continues_on]")
  {
    nvexec::stream_context ctx_0;
    nvexec::stream_context ctx_1;

    int* data{};
    STDEXEC_TRY_CUDA_API(cudaMallocManaged(&data, sizeof(int)));
    *data = 0;

    // The first step is slow, so the second one would read 0 if it did not wait.
    auto sndr = STDEXEC::schedule(ctx_0.get_scheduler())
              | STDEXEC::then(
                  [data]
                  {
                    int volatile spin = 0;
                    for (int i = 0; i < (1 << 24); ++i)
                    {
                      spin = spin + 1;
                    }
                    *data = 42;
                  })
              | STDEXEC::continues_on(ctx_1.get_scheduler())
              | STDEXEC::then([data] { return *data; });

    auto [result] = STDEXEC::sync_wait(std::move(sndr)).value();
    STDEXEC_ASSERT_CUDA_API(cudaFree(data));
    CHECK(result == 42);
  }
}  // namespace
