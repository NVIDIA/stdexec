#include <stdexec/execution.hpp>
#include <test_common/catch2.hpp>

#include "common.cuh"
#include "nvexec/stream_context.cuh"

namespace ex = STDEXEC;

using nvexec::is_on_gpu;

namespace
{

  TEST_CASE("nvexec split returns a sender", "[cuda][stream][adaptors][split]")
  {
    nvexec::stream_context stream_ctx{};
    auto                   snd = exec::split(ex::schedule(stream_ctx.get_scheduler()));
    STATIC_REQUIRE(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("nvexec split works", "[cuda][stream][adaptors][split]")
  {
    nvexec::stream_context stream_ctx{};

    auto fork = ex::schedule(stream_ctx.get_scheduler()) | ex::then([=] { return is_on_gpu(); })
              | exec::split();

    auto b1 = fork | ex::then([](bool on_gpu) { return on_gpu * 24; });
    auto b2 = fork | ex::then([](bool on_gpu) { return on_gpu * 42; });

    auto [v1] = STDEXEC::sync_wait(std::move(b1)).value();
    auto [v2] = STDEXEC::sync_wait(std::move(b2)).value();

    REQUIRE(v1 == 24);
    REQUIRE(v2 == 42);
  }

  TEST_CASE("nvexec split handles pre-cancellation", "[cuda][stream][adaptors][split]")
  {
    nvexec::stream_context  stream_ctx{};
    ex::inplace_stop_source stop_source;
    flags_storage_t         flags_storage{};
    auto                    flags = flags_storage.get();

    stop_source.request_stop();

    auto snd = ex::schedule(stream_ctx.get_scheduler()) | ex::then([flags] { flags.set(); })
             | exec::split() | ex::write_env(ex::prop{ex::get_stop_token, stop_source.get_token()})
             | ex::upon_stopped([] { return 42; });

    auto [value] = STDEXEC::sync_wait(std::move(snd)).value();

    REQUIRE(value == 42);
    REQUIRE(flags_storage.all_unset());
  }

  TEST_CASE("nvexec split can preceed a sender without values", "[cuda][stream][adaptors][split]")
  {
    nvexec::stream_context stream_ctx{};

    flags_storage_t flags_storage{};
    auto            flags = flags_storage.get();

    auto snd = ex::schedule(stream_ctx.get_scheduler()) | exec::split()
             | a_sender(
                 [=]() noexcept
                 {
                   if (is_on_gpu())
                   {
                     flags.set();
                   }
                 });

    STDEXEC::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec split can succeed a sender", "[cuda][stream][adaptors][split]")
  {
    SECTION("without values")
    {
      nvexec::stream_context stream_ctx{};
      flags_storage_t<2>     flags_storage{};
      auto                   flags = flags_storage.get();

      auto snd = ex::schedule(stream_ctx.get_scheduler())
               | a_sender(
                   [flags]
                   {
                     if (is_on_gpu())
                     {
                       flags.set(1);
                     }
                   })
               | exec::split()
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

      auto snd = ex::schedule(stream_ctx.get_scheduler())
               | a_sender([]() -> bool { return is_on_gpu(); }) | exec::split()
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
