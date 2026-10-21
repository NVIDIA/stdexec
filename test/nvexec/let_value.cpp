#include <test_common/catch2.hpp>

#include <stdexec/execution.hpp>

#include <exec/env.hpp>

#include <nvexec/stream_context.cuh>

#include "common.cuh"

namespace ex = STDEXEC;

using nvexec::is_on_gpu;

namespace
{

  TEST_CASE("nvexec let_value returns a sender", "[cuda][stream][adaptors][let_value]")
  {
    nvexec::stream_context stream_ctx{};
    auto snd = ex::let_value(ex::schedule(stream_ctx.get_scheduler()), [] { return ex::just(); });
    STATIC_REQUIRE(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("nvexec let_value executes on GPU", "[cuda][stream][adaptors][let_value]")
  {
    nvexec::stream_context stream_ctx{};

    flags_storage_t flags_storage{};
    auto            flags = flags_storage.get();

    auto snd = ex::schedule(stream_ctx.get_scheduler())
             | ex::let_value(
                 [=]
                 {
                   if (is_on_gpu())
                   {
                     flags.set();
                   }
                   return ex::just();
                 })
             | exec::completes_on(stream_ctx.get_scheduler());
    ex::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec let_value accepts values on GPU", "[cuda][stream][adaptors][let_value]")
  {
    nvexec::stream_context stream_ctx{};

    flags_storage_t flags_storage{};
    auto            flags = flags_storage.get();

    auto snd = ex::schedule(stream_ctx.get_scheduler()) | ex::then([]() -> int { return 42; })
             | ex::let_value(
                 [=](int val)
                 {
                   if (is_on_gpu())
                   {
                     if (val == 42)
                     {
                       flags.set();
                     }
                   }
                   return ex::just();
                 })
             | exec::completes_on(stream_ctx.get_scheduler());
    ex::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec let_value accepts multiple values on GPU",
            "[cuda][stream][adaptors][let_value]")
  {
    nvexec::stream_context stream_ctx{};

    flags_storage_t flags_storage{};
    auto            flags = flags_storage.get();

    auto snd = ex::just(42, 4.2) | ex::continues_on(stream_ctx.get_scheduler())
             | ex::let_value(
                 [=](int i, double d)
                 {
                   if (is_on_gpu())
                   {
                     if (i == 42 && d == 4.2)
                     {
                       flags.set();
                     }
                   }
                   return ex::just();
                 })
             | exec::completes_on(stream_ctx.get_scheduler());
    ex::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec let_value returns values on GPU", "[cuda][stream][adaptors][let_value]")
  {
    nvexec::stream_context stream_ctx{};

    auto snd = ex::schedule(stream_ctx.get_scheduler())
             | ex::let_value([=]() { return ex::just(is_on_gpu()); })
             | exec::completes_on(stream_ctx.get_scheduler());
    auto const [result] = ex::sync_wait(std::move(snd)).value();

    REQUIRE(result == 1);
  }

  TEST_CASE("nvexec let_value can preceed a sender without values",
            "[cuda][stream][adaptors][let_value]")
  {
    nvexec::stream_context stream_ctx{};

    flags_storage_t<2> flags_storage{};
    auto               flags = flags_storage.get();

    auto snd = ex::schedule(stream_ctx.get_scheduler())
             | ex::let_value(
                 [flags]
                 {
                   if (is_on_gpu())
                   {
                     flags.set(0);
                   }

                   return ex::just();
                 })
             | exec::completes_on(stream_ctx.get_scheduler())
             | a_sender(
                 [flags]
                 {
                   if (is_on_gpu())
                   {
                     flags.set(1);
                   }
                 });
    ex::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec let_value can succeed a sender", "[cuda][stream][adaptors][let_value]")
  {
    nvexec::stream_context   stream_ctx{};
    nvexec::stream_scheduler sch = stream_ctx.get_scheduler();
    flags_storage_t          flags_storage{};
    auto                     flags = flags_storage.get();

    auto snd = ex::schedule(sch) | a_sender([]() noexcept {})
             | ex::let_value(
                 [=]
                 {
                   if (is_on_gpu())
                   {
                     flags.set();
                   }

                   return ex::schedule(sch);
                 })
             | exec::completes_on(sch);
    ex::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec let_value can read a property", "[cuda][stream][adaptors][let_value]")
  {
    nvexec::stream_context   stream_ctx{};
    nvexec::stream_scheduler sch = stream_ctx.get_scheduler();
    flags_storage_t          flags_storage{};
    auto                     flags = flags_storage.get();

    auto snd = ex::schedule(sch)                                   //
             | ex::let_value([] { return nvexec::get_stream(); })  //
             | exec::completes_on(sch)                             //
             | ex::then(
                 [flags](cudaStream_t stream)
                 {
                   if (is_on_gpu())
                   {
                     flags.set();
                   }
                   return stream;
                 });
    auto [stream] = ex::sync_wait(std::move(snd)).value();
    static_assert(std::same_as<decltype(+stream), cudaStream_t>);

    REQUIRE(flags_storage.all_set_once());
  }
}  // namespace
