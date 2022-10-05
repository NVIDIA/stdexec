#include <catch2/catch.hpp>
#include <execution.hpp>

#include "schedulers/stream.cuh"
#include "common.cuh"

#if STDEXEC_NVHPC()
namespace ex = std::execution;
namespace stream = example::cuda::stream;

using example::cuda::is_on_gpu;

TEST_CASE("let_value returns a sender", "[cuda][stream][adaptors][let_value]") {
  stream::context_t stream_context{};
  auto snd = ex::let_value(ex::schedule(stream_context.get_scheduler()), [] { return ex::just(); });
  STATIC_REQUIRE(ex::sender<decltype(snd)>);
  (void)snd;
}

TEST_CASE("let_value executes on GPU", "[cuda][stream][adaptors][let_value]") {
  stream::context_t stream_context{};

  flags_storage_t flags_storage{};
  auto flags = flags_storage.get();

  auto snd = ex::schedule(stream_context.get_scheduler()) //
           | ex::let_value([=] {
               if (is_on_gpu()) {
                 flags.set();
               }
               return ex::just();
             });
  std::this_thread::sync_wait(std::move(snd));

  REQUIRE(flags_storage.all_set_once());
}

TEST_CASE("let_value accepts values on GPU", "[cuda][stream][adaptors][let_value]") {
  stream::context_t stream_context{};

  flags_storage_t flags_storage{};
  auto flags = flags_storage.get();

  auto snd = ex::schedule(stream_context.get_scheduler()) //
           | ex::then([]() -> int { return 42; })
           | ex::let_value([=](int val) {
               if (is_on_gpu()) {
                 if (val == 42) {
                   flags.set();
                 }
               }
               return ex::just();
             });
  std::this_thread::sync_wait(std::move(snd));

  REQUIRE(flags_storage.all_set_once());
}

TEST_CASE("let_value accepts multiple values on GPU", "[cuda][stream][adaptors][let_value]") {
  stream::context_t stream_context{};

  flags_storage_t flags_storage{};
  auto flags = flags_storage.get();

  auto snd = ex::transfer_just(stream_context.get_scheduler(), 42, 4.2) //
           | ex::let_value([=](int i, double d) {
               if (is_on_gpu()) {
                 if (i == 42 && d == 4.2) {
                   flags.set();
                 }
               }
               return ex::just();
             });
  std::this_thread::sync_wait(std::move(snd));

  REQUIRE(flags_storage.all_set_once());
}

TEST_CASE("let_value returns values on GPU", "[cuda][stream][adaptors][let_value]") {
  stream::context_t stream_context{};

  auto snd = ex::schedule(stream_context.get_scheduler()) //
           | ex::let_value([=]() {
               return ex::just(is_on_gpu());
             });
  const auto [result] = std::this_thread::sync_wait(std::move(snd)).value();

  REQUIRE(result == 1);
}

#endif
