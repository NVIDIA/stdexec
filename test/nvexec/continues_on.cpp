#include <stdexec/execution.hpp>
#include <test_common/catch2.hpp>

#include "common.cuh"
#include "nvexec/stream_context.cuh"

#include <memory_resource>

namespace
{
  class pinned_memory_resource_t : public std::pmr::memory_resource
  {
    std::size_t allocations_{};
    std::size_t deallocations_{};

    void* do_allocate(std::size_t bytes, std::size_t) override
    {
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

  TEST_CASE("continues_on frees its task when the operation is not started",
            "[cuda][stream][adaptors][continues_on]")
  {
    pinned_memory_resource_t pinned_memory;
    nvexec::stream_context   ctx;
    auto                     scheduler = ctx.get_scheduler();
    scheduler.ctx_.pinned_resource_    = &pinned_memory;

    auto sndr = STDEXEC::just() | STDEXEC::continues_on(scheduler);
    {
      auto op = STDEXEC::connect(std::move(sndr), noop_receiver{});
      (void) op;
    }

    REQUIRE(pinned_memory.allocations() > 0);
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
}  // namespace
