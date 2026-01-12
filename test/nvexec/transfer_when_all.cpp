/*
 * Copyright (c) 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>

#include <exec/inline_scheduler.hpp>
#include <nvexec/stream_context.cuh>
#include <stdexec/execution.hpp>

#include <catch2/catch.hpp>

namespace ex = STDEXEC;

namespace {

  TEST_CASE(
    "nvexec transfer_when_all returns a sender",
    "[cuda][stream][adaptors][transfer_when_all]") {
    nvexec::stream_context stream_ctx{};
    auto gpu = stream_ctx.get_scheduler();
    auto snd = ex::transfer_when_all(gpu, ex::just(3), ex::just(0.1415));
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE(
    "nvexec transfer_when_all with environment returns a sender",
    "[cuda][stream][adaptors][transfer_when_all]") {
    nvexec::stream_context stream_ctx{};
    auto gpu = stream_ctx.get_scheduler();
    auto snd = ex::transfer_when_all(gpu, ex::just(3), ex::just(0.1415));
    static_assert(ex::sender_in<decltype(snd), ex::env<>>);
    (void) snd;
  }

  TEST_CASE(
    "nvexec transfer_when_all with no senders",
    "[cuda][stream][adaptors][transfer_when_all]") {
    nvexec::stream_context stream_ctx{};
    auto gpu = stream_ctx.get_scheduler();
    auto snd = ex::transfer_when_all(gpu);
    wait_for_value(std::move(snd));
  }

  TEST_CASE("nvexec transfer_when_all one sender", "[cuda][stream][adaptors][transfer_when_all]") {
    nvexec::stream_context stream_ctx{};
    auto gpu = stream_ctx.get_scheduler();
    auto snd = ex::transfer_when_all(gpu, ex::just(3.1415)); // NOLINT(modernize-use-std-numbers)
    wait_for_value(std::move(snd), 3.1415);                  // NOLINT(modernize-use-std-numbers)
  }

  TEST_CASE("nvexec transfer_when_all two senders", "[cuda][stream][adaptors][transfer_when_all]") {
    nvexec::stream_context stream_ctx{};
    auto gpu = stream_ctx.get_scheduler();
    auto snd1 = ex::transfer_when_all(gpu, ex::just(3), ex::just(0.1415));
    auto snd2 = std::move(snd1) | ex::then([](int x, double y) { return x + y; });
    wait_for_value(std::move(snd2), 3.1415); // NOLINT(modernize-use-std-numbers)
  }

  TEST_CASE(
    "nvexec transfer_when_all two senders on same scheduler",
    "[cuda][stream][adaptors][transfer_when_all]") {
    nvexec::stream_context stream_ctx{};
    auto gpu = stream_ctx.get_scheduler();
    auto snd1 =
      ex::transfer_when_all(gpu, ex::transfer_just(gpu, 3), ex::transfer_just(gpu, 0.1415));
    auto snd2 = std::move(snd1) | ex::then([](int x, double y) { return x + y; });
    wait_for_value(std::move(snd2), 3.1415); // NOLINT(modernize-use-std-numbers)
  }

  TEST_CASE(
    "nvexec transfer_when_all_with_variant returns a sender",
    "[cuda][stream][adaptors][transfer_when_all_with_variant]") {
    nvexec::stream_context stream_ctx{};
    auto gpu = stream_ctx.get_scheduler();
    auto snd = ex::transfer_when_all_with_variant(gpu, ex::just(3), ex::just(0.1415));
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE(
    "nvexec transfer_when_all_with_variant with environment returns a sender",
    "[cuda][stream][adaptors][transfer_when_all_with_variant]") {
    nvexec::stream_context stream_ctx{};
    auto gpu = stream_ctx.get_scheduler();
    auto snd = ex::transfer_when_all_with_variant(gpu, ex::just(3), ex::just(0.1415));
    static_assert(ex::sender_in<decltype(snd), ex::env<>>);
    (void) snd;
  }

  TEST_CASE(
    "nvexec transfer_when_all_with_variant two senders",
    "[cuda][stream][adaptors][transfer_when_all_with_variant]") {
    nvexec::stream_context stream_ctx{};
    auto gpu = stream_ctx.get_scheduler();
    auto snd1 = ex::transfer_when_all_with_variant(gpu, ex::just(3), ex::just(0.1415));
    auto snd2 = std::move(snd1) | ex::then([](auto&&, auto&&) { return 42; });
    wait_for_value(std::move(snd2), 42);
  }

  TEST_CASE(
    "nvexec transfer_when_all_with_variant basic example",
    "[cuda][stream][adaptors][transfer_when_all_with_variant]") {
    nvexec::stream_context stream_ctx{};
    auto gpu = stream_ctx.get_scheduler();
    ex::sender auto snd = ex::transfer_when_all_with_variant(gpu, ex::just(2), ex::just(3.14));
    wait_for_value(
      std::move(snd), std::variant<std::tuple<int>>{2}, std::variant<std::tuple<double>>{3.14});
  }
} // namespace
