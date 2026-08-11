#include <stdexec/execution.hpp>
#include <test_common/catch2.hpp>

#include <type_traits>

#include "common.cuh"
#include "nvexec/stream_context.cuh"

namespace ex = STDEXEC;

using nvexec::is_on_gpu;

namespace
{

  struct move_only_error_handler
  {
    move_only_error_handler()                                = default;
    move_only_error_handler(move_only_error_handler const &) = delete;

    STDEXEC_ATTRIBUTE(host, device)
    move_only_error_handler(move_only_error_handler &&) = default;

    STDEXEC_ATTRIBUTE(host, device) auto operator()(int error) const -> int
    {
      return error;
    }
  };

  static_assert(std::is_trivially_copyable_v<move_only_error_handler>);
  static_assert(!std::is_copy_constructible_v<move_only_error_handler>);

  TEST_CASE("nvexec upon_error returns a sender", "[cuda][stream][adaptors][upon_error]")
  {
    nvexec::stream_context stream_ctx{};

    auto snd = ex::just_error(42) | ex::continues_on(stream_ctx.get_scheduler())
             | ex::upon_error([](int) {});
    STATIC_REQUIRE(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("nvexec upon_error supports move-only function objects",
            "[cuda][stream][adaptors][upon_error]")
  {
    nvexec::stream_context stream_ctx{};

    auto snd = ex::just_error(42) | ex::continues_on(stream_ctx.get_scheduler())
             | ex::upon_error(move_only_error_handler{});
    auto const [result] = STDEXEC::sync_wait(std::move(snd)).value();

    REQUIRE(result == 42);
  }

  TEST_CASE("nvexec upon_error executes on GPU", "[cuda][stream][adaptors][upon_error]")
  {
    nvexec::stream_context stream_ctx{};

    flags_storage_t flags_storage{};
    auto            flags = flags_storage.get();

    auto snd = ex::just_error(42) | ex::continues_on(stream_ctx.get_scheduler())
             | ex::upon_error(
                 [=](int err)
                 {
                   if (is_on_gpu() && err == 42)
                   {
                     flags.set();
                   }
                 });
    STDEXEC::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec upon_error can preceed a sender without values",
            "[cuda][stream][adaptors][upon_error]")
  {
    nvexec::stream_context stream_ctx{};

    flags_storage_t<2> flags_storage{};
    auto               flags = flags_storage.get();

    auto snd = ex::just_error(42) | ex::continues_on(stream_ctx.get_scheduler())
             | ex::upon_error(
                 [=](int err)
                 {
                   if (is_on_gpu() && err == 42)
                   {
                     flags.set(0);
                   }
                 })
             | a_sender(
                 [=]() noexcept
                 {
                   if (is_on_gpu())
                   {
                     flags.set(1);
                   }
                 });
    STDEXEC::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec upon_error can succeed a sender without values",
            "[cuda][stream][adaptors][upon_error]")
  {
    nvexec::stream_context stream_ctx{};

    flags_storage_t flags_storage{};
    auto            flags = flags_storage.get();

    auto snd = ex::just_error(42) | ex::continues_on(stream_ctx.get_scheduler())
             | a_sender([=]() noexcept {})
             | ex::upon_error(
                 [=](int err) noexcept
                 {
                   if (is_on_gpu() && err == 42)
                   {
                     flags.set();
                   }
                 });
    STDEXEC::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }
}  // namespace
