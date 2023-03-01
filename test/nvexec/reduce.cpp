#include <catch2/catch.hpp>

#include <range/v3/view/iota.hpp>
#include <range/v3/view/repeat_n.hpp>

#include <execution.hpp>

#include "nvexec/stream_context.cuh"
#include "common.cuh"

namespace ex = stdexec;

TEST_CASE("reduce returns a sender", "[cuda][stream][adaptors][reduce]") {
  constexpr int N = 2048;

  nvexec::stream_context stream{};

  auto snd = ex::transfer_just(
               stream.get_scheduler(),
               ranges::views::repeat_n(1, N))
           | nvexec::reduce();

  STATIC_REQUIRE(ex::sender_of<decltype(snd), ex::set_value_t(int&)>);

  (void)snd;
}

TEST_CASE("reduce binds the range and the function", "[cuda][stream][adaptors][reduce]") {
  constexpr int N = 2048;

  nvexec::stream_context stream{};

  auto snd = ex::schedule(stream.get_scheduler())
           | nvexec::reduce(
               ranges::views::iota(0, N),
               [] (int l, int r) {
                 return std::max(l, r);
               });

  auto [result] = std::this_thread::sync_wait(std::move(snd)).value();

  REQUIRE(result == N - 1);
}

TEST_CASE("reduce binds the range and uses the default function", "[cuda][stream][adaptors][reduce]") {
  constexpr int N = 2048;

  nvexec::stream_context stream{};

  auto snd = ex::schedule(stream.get_scheduler())
           | nvexec::reduce(ranges::views::iota(0, N));

  auto [result] = std::this_thread::sync_wait(std::move(snd)).value();

  REQUIRE(result == (N * (N - 1)) / 2);
}

TEST_CASE("reduce binds the range and takes the function from the predecessor", "[cuda][stream][adaptors][reduce]") {
  constexpr int N = 2048;

  nvexec::stream_context stream{};

  auto snd = ex::schedule(stream.get_scheduler())
           | ex::then([] {
               return [] (int l, int r) {
                 return std::max(l, r);
               };
             })
           | nvexec::reduce(ranges::views::iota(0, N));

  auto [result] = std::this_thread::sync_wait(std::move(snd)).value();

  REQUIRE(result == N - 1);
}

TEST_CASE("reduce takes the range from the predecessor and binds the function", "[cuda][stream][adaptors][reduce]") {
  constexpr int N = 2048;

  nvexec::stream_context stream{};

  auto snd = ex::transfer_just(
               stream.get_scheduler(),
               ranges::views::iota(0, N))
           | nvexec::reduce(
               [] (int l, int r) {
                 return std::max(l, r);
               });

  auto [result] = std::this_thread::sync_wait(std::move(snd)).value();

  REQUIRE(result == N - 1);
}

TEST_CASE("reduce takes the range from the predecessor and uses the default function", "[cuda][stream][adaptors][reduce]") {
  constexpr int N = 2048;

  nvexec::stream_context stream{};

  auto snd = ex::transfer_just(
               stream.get_scheduler(),
               ranges::views::iota(0, N))
           | nvexec::reduce();

  auto [result] = std::this_thread::sync_wait(std::move(snd)).value();

  REQUIRE(result == (N * (N - 1)) / 2);
}

TEST_CASE("reduce takes the range and function from the predecessor", "[cuda][stream][adaptors][reduce]") {
  constexpr int N = 2048;

  nvexec::stream_context stream{};

  auto snd = ex::transfer_just(
               stream.get_scheduler(),
               ranges::views::iota(0, N),
               [] (int l, int r) {
                 return std::max(l, r);
               })
           | nvexec::reduce();

  auto [result] = std::this_thread::sync_wait(std::move(snd)).value();

  REQUIRE(result == (N * (N - 1)) / 2);
}

TEST_CASE("reduce accepts std::vector", "[cuda][stream][adaptors][reduce]") {
  constexpr int N = 2048;

  std::vector<int> input(1, N);

  nvexec::stream_context stream{};

  auto snd = ex::transfer_just(
               stream.get_scheduler(),
               input)
           | nvexec::reduce();

  auto [result] = std::this_thread::sync_wait(std::move(snd)).value();

  REQUIRE(result == N - 1);
}

TEST_CASE("reduce executes on GPU", "[cuda][stream][adaptors][reduce]") {
  constexpr int N = 2048;

  nvexec::stream_context stream{};

  auto snd = ex::transfer_just(
               stream.get_scheduler(),
               ranges::views::repeat_n(1, N))
           | nvexec::reduce(
               [] (int l, int r) {
                 return nvexec::is_on_gpu() ? l + r : 0;
               });

  auto [result] = std::this_thread::sync_wait(std::move(snd)).value();

  REQUIRE(result == N - 1);
}

