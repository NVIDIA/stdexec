#include <test_common/catch2.hpp>

#include <stdexec/execution.hpp>

#include <exec/env.hpp>

#include <nvexec/stream_context.cuh>

#include "common.cuh"

#include <cuda/std/span>

namespace ex = STDEXEC;

using nvexec::is_on_gpu;

namespace
{

  TEST_CASE("nvexec bulk returns a sender", "[cuda][stream][adaptors][bulk]")
  {
    nvexec::stream_context stream_ctx{};
    auto snd = ex::bulk(ex::schedule(stream_ctx.get_scheduler()), ex::par, 42, [](int) {});
    STATIC_REQUIRE(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("nvexec bulk supports const lvalue senders", "[cuda][stream][adaptors][bulk]")
  {
    nvexec::stream_context stream_ctx{};
    auto const snd = ex::schedule(stream_ctx.get_scheduler()) | ex::bulk(ex::par, 1, [](int) {});

    REQUIRE(ex::sync_wait(snd).has_value());
  }

  TEST_CASE("nvexec bulk executes on GPU", "[cuda][stream][adaptors][bulk]")
  {
    nvexec::stream_context stream_ctx{};

    flags_storage_t<4> flags_storage{};
    auto               flags = flags_storage.get();

    auto snd = ex::schedule(stream_ctx.get_scheduler())
             | ex::bulk(ex::par,
                        4,
                        [=](int idx)
                        {
                          if (is_on_gpu())
                          {
                            flags.set(idx);
                          }
                        });
    ex::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec bulk forwards values on GPU", "[cuda][stream][adaptors][bulk]")
  {
    nvexec::stream_context stream_ctx{};

    flags_storage_t<1024> flags_storage{};
    auto                  flags = flags_storage.get();

    auto snd = ex::just(42) | ex::continues_on(stream_ctx.get_scheduler())
             | ex::bulk(ex::par,
                        1024,
                        [=](int idx, int val)
                        {
                          if (is_on_gpu())
                          {
                            if (val == 42)
                            {
                              flags.set(idx);
                            }
                          }
                        });
    ex::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec bulk forwards multiple values on GPU", "[cuda][stream][adaptors][bulk]")
  {
    nvexec::stream_context stream_ctx{};

    flags_storage_t<2> flags_storage{};
    auto               flags = flags_storage.get();

    auto snd = ex::just(42, 4.2) | ex::continues_on(stream_ctx.get_scheduler())
             | ex::bulk(ex::par,
                        2,
                        [=](int idx, int i, double d)
                        {
                          if (is_on_gpu())
                          {
                            if (i == 42 && d == 4.2)
                            {
                              flags.set(idx);
                            }
                          }
                        });
    auto const [i, d] = ex::sync_wait(std::move(snd)).value();

    REQUIRE(flags_storage.all_set_once());
    REQUIRE(i == 42);
    REQUIRE(d == 4.2);
  }

  TEST_CASE("bulk forwards values that can be taken by reference on GPU",
            "[cuda][stream][adaptors][bulk]")
  {
    nvexec::stream_context stream_ctx{};

    flags_storage_t<1024> flags_storage{};
    using flags_t = flags_storage_t<1024>::flags_t;
    auto flags    = flags_storage.get();

    auto snd = ex::just(flags) | ex::continues_on(stream_ctx.get_scheduler())
             | ex::bulk(ex::par,
                        1024,
                        [](int idx, flags_t const & flags)
                        {
                          if (is_on_gpu())
                          {
                            flags.set(idx);
                          }
                        });
    [[maybe_unused]] auto [flags_actual] = ex::sync_wait(std::move(snd)).value();

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec bulk can preceed a sender without values", "[cuda][stream][adaptors][bulk]")
  {
    nvexec::stream_context stream_ctx{};

    flags_storage_t<3> flags_storage{};
    auto               flags = flags_storage.get();

    auto snd = ex::schedule(stream_ctx.get_scheduler())
             | ex::bulk(ex::par,
                        2,
                        [flags](int idx)
                        {
                          if (is_on_gpu())
                          {
                            flags.set(idx);
                          }
                        })
             | a_sender(
                 [flags]
                 {
                   if (is_on_gpu())
                   {
                     flags.set(2);
                   }
                 });
    ex::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec bulk can succeed a sender", "[cuda][stream][adaptors][bulk]")
  {
    SECTION("without values")
    {
      nvexec::stream_context stream_ctx{};
      flags_storage_t<3>     flags_storage{};
      auto                   flags = flags_storage.get();

      auto snd = ex::schedule(stream_ctx.get_scheduler())
               | a_sender(
                   [flags]
                   {
                     if (is_on_gpu())
                     {
                       flags.set(2);
                     }
                   })
               | ex::bulk(ex::par,
                          2,
                          [flags](int idx)
                          {
                            if (is_on_gpu())
                            {
                              flags.set(idx);
                            }
                          });
      ex::sync_wait(std::move(snd));

      REQUIRE(flags_storage.all_set_once());
    }

    SECTION("with values")
    {
      nvexec::stream_context stream_ctx{};
      flags_storage_t<2>     flags_storage{};
      auto                   flags = flags_storage.get();

      auto snd = ex::schedule(stream_ctx.get_scheduler())
               | a_sender([]() -> bool { return is_on_gpu(); })
               | ex::bulk(ex::par,
                          2,
                          [flags](int idx, bool a_sender_was_on_gpu)
                          {
                            if (a_sender_was_on_gpu && is_on_gpu())
                            {
                              flags.set(idx);
                            }
                          });
      ex::sync_wait(std::move(snd)).value();

      REQUIRE(flags_storage.all_set_once());
    }
  }

  TEST_CASE("nvexec bulk can succeed a sender that sends ref into opstate",
            "[cuda][stream][adaptors][bulk]")
  {
    nvexec::stream_context ctx;

    double*   inout  = nullptr;
    int const nelems = 10;
    cudaMallocManaged(&inout, nelems * sizeof(double));

    auto task = ex::just(cuda::std::span<double>{inout, nelems})
              | ex::continues_on(ctx.get_scheduler())
              | ex::bulk(ex::par,
                         nelems,
                         [](std::size_t i, cuda::std::span<double> out) { out[i] = (double) i; })
              | ex::let_value([](cuda::std::span<double> out) { return ex::just(out); })
              | exec::write_attrs(
                  ex::prop{ex::get_completion_scheduler<ex::set_value_t>, ctx.get_scheduler()})
              | ex::bulk(ex::par,
                         nelems,
                         [](std::size_t i, cuda::std::span<double> out) { out[i] = 2.0 * out[i]; });

    ex::sync_wait(std::move(task)).value();

    for (int i = 0; i < nelems; ++i)
    {
      REQUIRE(i * 2 == (int) inout[i]);
    }

    cudaFree(inout);
  }
}  // namespace
