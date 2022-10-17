#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

#include "nvexec/stream_context.cuh"
#include "common.cuh"

namespace ex = std::execution;

using nvexec::is_on_gpu;

TEST_CASE("upon_error returns a sender", "[cuda][stream][adaptors][upon_error]") {
  nvexec::stream_context stream_ctx{};

  auto snd = ex::just_error(42) | //
             ex::transfer(stream_ctx.get_scheduler()) | //
             ex::upon_error([](int) { });
  STATIC_REQUIRE(ex::sender<decltype(snd)>);
  (void)snd;
}

TEST_CASE("upon_error executes on GPU", "[cuda][stream][adaptors][upon_error]") {
  nvexec::stream_context stream_ctx{};

  flags_storage_t flags_storage{};
  auto flags = flags_storage.get();

  auto snd = ex::just_error(42) | //
             ex::transfer(stream_ctx.get_scheduler()) | //
             ex::upon_error([=](int err) { 
               if (is_on_gpu() && err == 42) {
                 flags.set();
               }
             });
  std::this_thread::sync_wait(std::move(snd));

  REQUIRE(flags_storage.all_set_once());
}

TEST_CASE("upon_error can preceed a sender without values", "[cuda][stream][adaptors][upon_error]") {
  nvexec::stream_context stream_ctx{};

  flags_storage_t<2> flags_storage{};
  auto flags = flags_storage.get();

  auto snd = ex::just_error(42) | //
             ex::transfer(stream_ctx.get_scheduler()) | //
             ex::upon_error([=](int err) { 
               if (is_on_gpu() && err == 42) {
                 flags.set(0);
               }
             }) |
             a_sender([=]() noexcept {
               if (is_on_gpu()) {
                flags.set(1);
               }
             });
  std::this_thread::sync_wait(std::move(snd));

  REQUIRE(flags_storage.all_set_once());
}

TEST_CASE("upon_error can succeed a sender without values", "[cuda][stream][adaptors][upon_error]") {
  nvexec::stream_context stream_ctx{};

  flags_storage_t flags_storage{};
  auto flags = flags_storage.get();

  // TODO Support other error types
  auto snd = ex::just_error(cudaSuccess) | //
             ex::transfer(stream_ctx.get_scheduler()) | //
             a_sender([=]() noexcept {}) |
             ex::upon_error([=](cudaError_t err) { 
               if (is_on_gpu() && err == cudaSuccess) {
                 flags.set();
               }
             }); 
  std::this_thread::sync_wait(std::move(snd));

  REQUIRE(flags_storage.all_set_once());
}

