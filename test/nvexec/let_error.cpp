#include <stdexec/execution.hpp>

#include <exec/env.hpp>

#include <nvexec/stream_context.cuh>

#include <test_common/catch2.hpp>

#include "common.cuh"

namespace ex = STDEXEC;

using nvexec::is_on_gpu;

namespace
{
  using _attrs_base_t =
    ex::env<ex::prop<ex::get_completion_scheduler_t<ex::set_value_t>, nvexec::stream_scheduler>,
            ex::prop<ex::get_completion_domain_t<ex::set_value_t>, nvexec::stream_domain>>;

  struct _attrs : _attrs_base_t
  {};

  template <class SetTag = ex::set_value_t>
  auto _proclaim_completes_on(nvexec::stream_scheduler sch)
  {
    return exec::write_attrs(_attrs{
      ex::env{ex::prop{ex::get_completion_scheduler<SetTag>, sch},
              ex::prop{ex::get_completion_domain<SetTag>, nvexec::stream_domain{}}}
    });
  }

  TEST_CASE("nvexec let_error returns a sender", "[cuda][stream][adaptors][let_error]")
  {
    nvexec::stream_context stream_ctx{};

    auto snd = ex::just_error(42) | ex::continues_on(stream_ctx.get_scheduler())
             | ex::let_error([](int) { return ex::just(); });
    STATIC_REQUIRE(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("nvexec let_error executes on GPU", "[cuda][stream][adaptors][let_error]")
  {
    nvexec::stream_context stream_ctx{};

    flags_storage_t flags_storage{};
    auto            flags = flags_storage.get();

    auto snd = ex::just_error(42) | ex::continues_on(stream_ctx.get_scheduler())
             | ex::let_error(
                 [=](int err)
                 {
                   if (is_on_gpu() && err == 42)
                   {
                     flags.set();
                   }

                   return ex::just();
                 })
             | ::_proclaim_completes_on(stream_ctx.get_scheduler());
    ex::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec let_error can preceed a sender without values",
            "[cuda][stream][adaptors][let_error]")
  {
    nvexec::stream_context stream_ctx{};

    flags_storage_t<2> flags_storage{};
    auto               flags = flags_storage.get();

    auto snd = ex::just_error(42) | ex::continues_on(stream_ctx.get_scheduler())
             | ex::let_error(
                 [flags, sch = stream_ctx.get_scheduler()](int err)
                 {
                   if (is_on_gpu() && err == 42)
                   {
                     flags.set(0);
                   }

                   return ex::just();
                 })
             | ::_proclaim_completes_on(stream_ctx.get_scheduler())
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

  TEST_CASE("nvexec let_error can succeed a sender", "[cuda][stream][adaptors][let_error]")
  {
    nvexec::stream_context   stream_ctx{};
    nvexec::stream_scheduler sch = stream_ctx.get_scheduler();
    flags_storage_t          flags_storage{};
    auto                     flags = flags_storage.get();

    auto snd = ex::just_error(42) | ex::continues_on(stream_ctx.get_scheduler())
             | a_sender([]() noexcept {})
             | ex::let_error(
                 [=](int err)
                 {
                   if (is_on_gpu() && err == 42)
                   {
                     flags.set();
                   }

                   return ex::schedule(sch);
                 });
    ex::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }
}  // namespace
