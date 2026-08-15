#include <stdexec/execution.hpp>
#include <test_common/catch2.hpp>
#include <test_common/senders.hpp>
#include <test_common/type_helpers.hpp>

#include <type_traits>

#include "common.cuh"
#include "nvexec/stream_context.cuh"

namespace ex = STDEXEC;

using nvexec::is_on_gpu;

namespace
{
  struct move_only_stopped_handler
  {
    move_only_stopped_handler()                                  = default;
    move_only_stopped_handler(move_only_stopped_handler const &) = delete;

    STDEXEC_ATTRIBUTE(host, device)
    move_only_stopped_handler(move_only_stopped_handler &&) = default;

    STDEXEC_ATTRIBUTE(host, device) auto operator()() const -> int
    {
      return 42;
    }
  };

  static_assert(std::is_trivially_copyable_v<move_only_stopped_handler>);
  static_assert(!std::is_copy_constructible_v<move_only_stopped_handler>);

  struct move_only_result
  {
    STDEXEC_ATTRIBUTE(host, device)
    explicit move_only_result(int value) noexcept
      : value_(value)
    {}

    STDEXEC_ATTRIBUTE(host, device)
    move_only_result(move_only_result &&other) noexcept
      : value_(other.value_)
    {
      other.value_ = 0;
    }

    move_only_result(move_only_result const &) = delete;

    STDEXEC_ATTRIBUTE(host, device)
    ~move_only_result() = default;

    STDEXEC_ATTRIBUTE(host, device)
    auto value() const noexcept -> int
    {
      return value_;
    }

   private:
    int value_;
  };

  TEST_CASE("nvexec upon_stopped advertises CUDA launch errors",
            "[cuda][stream][adaptors][upon_stopped]")
  {
    auto fun = []() noexcept {};
    using sender_t =
      nvexec::_strm::upon_stopped_sender<a_sender_of<ex::set_stopped_t()>, decltype(fun)>;
    sender_t snd{a_sender_of<ex::set_stopped_t()>{}, std::move(fun)};

    check_err_types<ex::__mset<cudaError_t>>(snd);
  }

  TEST_CASE("nvexec upon_stopped returns a sender", "[cuda][stream][adaptors][upon_stopped]")
  {
    nvexec::stream_context stream_ctx{};

    auto snd = ex::just_stopped() | ex::continues_on(stream_ctx.get_scheduler())
             | ex::upon_stopped([] { return ex::just(); });
    STATIC_REQUIRE(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("nvexec upon_stopped executes on GPU", "[cuda][stream][adaptors][upon_stopped]")
  {
    nvexec::stream_context stream_ctx{};

    flags_storage_t flags_storage{};
    auto            flags = flags_storage.get();

    auto snd = ex::just_stopped() | ex::continues_on(stream_ctx.get_scheduler())
             | ex::upon_stopped(
                 [=]
                 {
                   if (is_on_gpu())
                   {
                     flags.set();
                   }
                 });
    STDEXEC::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec upon_stopped supports move-only function objects",
            "[cuda][stream][adaptors][upon_stopped]")
  {
    nvexec::stream_context stream_ctx{};

    auto snd = ex::just_stopped() | ex::continues_on(stream_ctx.get_scheduler())
             | ex::upon_stopped(move_only_stopped_handler{});
    auto const [result] = STDEXEC::sync_wait(std::move(snd)).value();

    REQUIRE(result == 42);
  }

  TEST_CASE("nvexec upon_stopped moves its result", "[cuda][stream][adaptors][upon_stopped]")
  {
    nvexec::stream_context stream_ctx{};

    auto snd = ex::just_stopped() | ex::continues_on(stream_ctx.get_scheduler())
             | ex::upon_stopped([] { return move_only_result{42}; });

    auto [result] = STDEXEC::sync_wait(std::move(snd)).value();

    REQUIRE(result.value() == 42);
  }
}  // namespace
