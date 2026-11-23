#include <stdexec/execution.hpp>
#include <test_common/catch2.hpp>
#include <test_common/schedulers.hpp>

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

    auto snd = ex::schedule(stream_ctx.get_scheduler())                              //
             | ex::then([flags] { flags.set(); })                                    //
             | exec::split()                                                         //
             | ex::write_env(ex::prop{ex::get_stop_token, stop_source.get_token()})  //
             | ex::then([] { return 0; })                                            //
             | ex::upon_stopped([] { return 42; })
             // upon_stopped completes with a value both when its child does and when its
             // child is stopped, so it does not know one completion scheduler for its values.
             // Proclaim the one they share, for sync_wait:
             | ::_proclaim_completes_on(stream_ctx.get_scheduler());

    auto [value] = STDEXEC::sync_wait(std::move(snd)).value();

    REQUIRE(value == 42);
    REQUIRE(flags_storage.all_unset());
  }

  TEST_CASE("nvexec split: an operation's stop request does not stop the others",
            "[cuda][stream][adaptors][split]")
  {
    nvexec::stream_context stream_ctx{};
    auto                   sch = stream_ctx.get_scheduler();
    impulse_scheduler      gate;  // the shared operation waits for it

    // (The function is noexcept: the results of a stream split must be trivially copyable.)
    auto shared = ex::schedule(gate) | ex::then([]() noexcept { return 42; })
                | ex::continues_on(sch) | exec::split();

    // A's stop has been requested before it starts, so A completes with set_stopped.
    ex::inplace_stop_source stop_a;
    stop_a.request_stop();
    auto [a] = STDEXEC::sync_wait(shared  //
                                  | ex::write_env(ex::prop{ex::get_stop_token, stop_a.get_token()})
                                  | ex::then([](int) { return 1; })
                                  | ex::upon_stopped([] { return 2; })
                                  | ::_proclaim_completes_on(sch))
                 .value();
    CHECK(a == 2);

    // B was not asked to stop, so it receives the shared operation's value.
    std::atomic<bool> b_done{false};
    int               b = 0;
    std::thread       thread_b(
      [&]
      {
        auto [v] = STDEXEC::sync_wait(shared  //
                                      | ex::then([](int v) { return v; })
                                      | ex::upon_stopped([] { return -1; })
                                      | ::_proclaim_completes_on(sch))
                     .value();
        b      = v;
        b_done = true;
      });

    // Release the shared operation once B has started it. (Had B been stopped, it would
    // have finished without waiting for it.)
    while (!b_done.load() && !gate.try_start_next())
    {
      std::this_thread::yield();
    }
    thread_b.join();
    CHECK(b == 42);
  }

  TEST_CASE("nvexec split reports its value scheduler without an environment",
            "[cuda][stream][adaptors][split]")
  {
    nvexec::stream_context stream_ctx{};
    auto                   sch = stream_ctx.get_scheduler();

    // The split sender has been customized already, so it answers as it would in an
    // environment.
    auto snd = exec::split(ex::schedule(sch) | ex::then([]() noexcept { return 42; }));
    CHECK(ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(snd)) == sch);
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
