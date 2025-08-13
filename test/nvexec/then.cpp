#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

#include "nvexec/stream/common.cuh"
#include "nvexec/stream_context.cuh"
#include "common.cuh"

namespace ex = stdexec;

using nvexec::is_on_gpu;

namespace {

  TEST_CASE("nvexec then returns a sender", "[cuda][stream][adaptors][then]") {
    nvexec::stream_context stream_ctx{};
    auto snd = ex::then(ex::schedule(stream_ctx.get_scheduler()), [] { });
    STATIC_REQUIRE(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("nvexec then executes on GPU", "[cuda][stream][adaptors][then]") {
    nvexec::stream_context stream_ctx{};

    flags_storage_t flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::schedule(stream_ctx.get_scheduler()) | ex::then([=] {
                 if (is_on_gpu()) {
                   flags.set();
                 }
               });
    stdexec::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec then accepts values on GPU", "[cuda][stream][adaptors][then]") {
    nvexec::stream_context stream_ctx{};

    flags_storage_t flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::transfer_just(stream_ctx.get_scheduler(), 42) | ex::then([=](int val) {
                 if (is_on_gpu()) {
                   if (val == 42) {
                     flags.set();
                   }
                 }
               });
    stdexec::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec then accepts multiple values on GPU", "[cuda][stream][adaptors][then]") {
    nvexec::stream_context stream_ctx{};

    flags_storage_t flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::transfer_just(stream_ctx.get_scheduler(), 42, 4.2)
             | ex::then([=](int i, double d) {
                 if (is_on_gpu()) {
                   if (i == 42 && d == 4.2) {
                     flags.set();
                   }
                 }
               });
    stdexec::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec then returns values on GPU", "[cuda][stream][adaptors][then]") {
    nvexec::stream_context stream_ctx{};

    auto snd = ex::schedule(stream_ctx.get_scheduler()) | ex::then([=]() -> int {
                 if (is_on_gpu()) {
                   return 42;
                 }

                 return 0;
               });
    const auto [result] = stdexec::sync_wait(std::move(snd)).value();

    REQUIRE(result == 42);
  }

  TEST_CASE("nvexec then can preceed a sender without values", "[cuda][stream][adaptors][then]") {
    nvexec::stream_context stream_ctx{};

    flags_storage_t<2> flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::schedule(stream_ctx.get_scheduler()) | ex::then([flags] {
                 if (is_on_gpu()) {
                   flags.set(0);
                 }
               })
             | a_sender([flags] {
                 if (is_on_gpu()) {
                   flags.set(1);
                 }
               });
    stdexec::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec then can succeed a sender", "[cuda][stream][adaptors][then]") {
    SECTION("without values") {
      nvexec::stream_context stream_ctx{};
      flags_storage_t<2> flags_storage{};
      auto flags = flags_storage.get();

      auto snd = ex::schedule(stream_ctx.get_scheduler()) | a_sender([flags] {
                   if (is_on_gpu()) {
                     flags.set(1);
                   }
                 })
               | ex::then([flags] {
                   if (is_on_gpu()) {
                     flags.set(0);
                   }
                 });
      stdexec::sync_wait(std::move(snd));

      REQUIRE(flags_storage.all_set_once());
    }

    SECTION("with values") {
      nvexec::stream_context stream_ctx{};
      flags_storage_t flags_storage{};
      auto flags = flags_storage.get();

      auto snd = ex::schedule(stream_ctx.get_scheduler())
               | a_sender([]() -> bool { return is_on_gpu(); })
               | ex::then([flags](bool a_sender_was_on_gpu) {
                   if (a_sender_was_on_gpu && is_on_gpu()) {
                     flags.set();
                   }
                 });
      stdexec::sync_wait(std::move(snd)).value();

      REQUIRE(flags_storage.all_set_once());
    }
  }

  TEST_CASE("nvexec then can succeed a receiverless sender", "[cuda][stream][adaptors][then]") {
    SECTION("without values") {
      nvexec::stream_context stream_ctx{};
      flags_storage_t flags_storage{};
      auto flags = flags_storage.get();

      auto snd = ex::schedule(stream_ctx.get_scheduler()) | a_sender() | ex::then([flags] {
                   if (is_on_gpu()) {
                     flags.set();
                   }
                 });
      stdexec::sync_wait(std::move(snd));

      REQUIRE(flags_storage.all_set_once());
    }

    SECTION("with values") {
      nvexec::stream_context stream_ctx{};
      flags_storage_t flags_storage{};
      auto flags = flags_storage.get();

      auto snd = ex::schedule(stream_ctx.get_scheduler())
               | ex::then([]() -> bool { return is_on_gpu(); }) | a_sender()
               | ex::then([flags](bool a_sender_was_on_gpu) {
                   if (a_sender_was_on_gpu && is_on_gpu()) {
                     flags.set();
                   }
                 });
      stdexec::sync_wait(std::move(snd)).value();

      REQUIRE(flags_storage.all_set_once());
    }
  }

  TEST_CASE(
    "nvexec then can return values of non-trivial types",
    "[cuda][stream][adaptors][then]") {
    nvexec::stream_context stream_ctx{};
    flags_storage_t flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::schedule(stream_ctx.get_scheduler())
             | ex::then([]() -> move_only_t { return move_only_t{42}; })
             | ex::then([flags](move_only_t &&val) {
                 if (val.contains(42)) {
                   flags.set();
                 }
               });
    stdexec::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  class tracer_storage_t {
    int h_counter_storage{};
    int *h_counter_{};
    int *d_counter_{};

   public:
    tracer_storage_t()
      : h_counter_{&h_counter_storage} {
      cudaMalloc(&d_counter_, sizeof(int));
      cudaMemset(d_counter_, 0, sizeof(int));
    }

    ~tracer_storage_t() {
      cudaFree(d_counter_);
    }

    class handle_t {
      int *h_counter_{};
      int *d_counter_{};

      handle_t(int *h_counter, int *d_counter)
        : h_counter_{h_counter}
        , d_counter_{d_counter} {
      }

      __host__ __device__ void diff(int val) {
        cuda::std::atomic_ref<int> ref{*(is_on_gpu() ? d_counter_ : h_counter_)};
        ref.fetch_add(val, cuda::std::memory_order_relaxed);
      }

     public:
      __host__ __device__ void more() {
        diff(+1);
      }

      __host__ __device__ void fewer() {
        diff(-1);
      }

      __host__ auto alive() -> int {
        int d_counter{};
        cudaMemcpy(&d_counter, d_counter_, sizeof(int), cudaMemcpyDeviceToHost);
        return *h_counter_ + d_counter;
      }

      friend tracer_storage_t;
    };

    auto get() -> handle_t {
      return handle_t{h_counter_, d_counter_};
    }
  };

  class tracer_t {
    tracer_storage_t::handle_t handle_;

    void print(const char *msg) {
      if (is_on_gpu()) {
        printf("gpu: %s\n", msg);
      } else {
        printf("cpu: %s\n", msg);
      }
    }

   public:
    tracer_t() = delete;
    tracer_t(const tracer_t &other) = delete;

    __host__ __device__ tracer_t(tracer_storage_t::handle_t handle)
      : handle_(handle) {
      handle_.more();
    }

    __host__ __device__ tracer_t(tracer_t &&other)
      : handle_(other.handle_) {
      handle_.more();
    }

    __host__ __device__ ~tracer_t() {
      handle_.fewer();
    }
  };

  TEST_CASE("nvexec then destructs temporary storage", "[cuda][stream][adaptors][then]") {
    nvexec::stream_context stream_ctx{};

    tracer_storage_t storage;
    tracer_storage_t::handle_t handle = storage.get();

    {
      auto snd = ex::schedule(stream_ctx.get_scheduler())
               | ex::then([handle]() -> tracer_t { return tracer_t{handle}; })
               | ex::then([](tracer_t &&) {});
      stdexec::sync_wait(std::move(snd));
    }

    REQUIRE(handle.alive() == 0);
  }
} // namespace
